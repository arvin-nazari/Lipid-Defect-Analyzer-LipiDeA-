"""Step 4: radius-aware defect classification.

For each frame: loads the leaflet meshes Step 3 saved (triangles + per-
triangle centroid/normal), assigns every bead to its nearby candidate
triangles via a kNN search, then classifies each triangle as a packing
defect or not by sampling coverage within a signed-distance slab along the
triangle normal, symmetric on both sides of the surface. A head-group bead
found anywhere in that slab -- above or below the triangle -- immediately
vetoes the triangle as non-defect.  Among triangles
that survive the veto, low total coverage also marks non-defect, and
everything else is classified by comparing head-bead footprint against a
coverage threshold. A triangle already classified as a defect is further
typed neutral-lipid vs. phospholipid-tail by whichever bead class covers
more of its footprint within a tighter sub-slab.

Output per frame: a defect-only PLY per leaflet, and (if
DEFECT_STORE_DEBUG_NPZ) a raw defect NPZ consumed by Step 5's periodic
deduplication. _DEFECT_TAIL_HEX / _DEFECT_NEUTRAL_HEX below are also
imported directly by cleanup.py to keep defect-type coloring consistent
across steps -- despite the underscore prefix, they are a real cross-file
dependency, not private to this module.
"""


import csv
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import open3d as o3d
from numba import njit, prange, get_num_threads

_DEFECT_TAIL_HEX = "#f218bc"
_DEFECT_NEUTRAL_HEX = "#0f0f0f"


# ==================================================
# PDB parsing
# ==================================================

# Parse one frame's PDB lines into a frame_data dict with an "atoms" list.
def parse_frame(frame_lines):
    """Parse one frame's PDB lines into {"atoms": [...]}.

    A malformed line is skipped with a warning rather than aborting the frame.

    Returns: {"atoms": [dict, ...]}
    """
    frame_data = {"atoms": []}

    for line in frame_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("ATOM"):
            try:
                atom = {
                    "atom_serial": line[6:11].strip(),
                    "atom_name": line[12:16].strip(),
                    "res_name": line[17:21].strip(),
                    "res_id": int(line[22:26]),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "original_line": line,
                }
                frame_data["atoms"].append(atom)
            except (ValueError, IndexError):
                print(f"Warning: Failed to parse ATOM line: {line}")
                continue

    return frame_data



# Parse PDB files from a folder one at a time.
def parse_pdb_files(folder_path, file_pattern="*.pdb"):
    """Yield (frame_idx, frame_data, pdb_file_basename) one frame at a time.

    frame_idx is the first integer substring found in the filename. Files
    without a parseable index are skipped with a warning, not aborted.

    folder_path:  directory containing per-frame PDB files (Step 2's output).
    file_pattern: glob pattern, default "*.pdb".
    Yields: (frame_idx, frame_data, basename) per frame, sorted by frame_idx.
    """
    import glob
    import re

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdb_files = glob.glob(os.path.join(folder_path, file_pattern))
    if not pdb_files:
        raise ValueError(f"No PDB files found in folder: {folder_path} with pattern: {file_pattern}")

    def extract_frame_index(filename):
        match = re.search(r"(\d+)", os.path.basename(filename))
        if match:
            return int(match.group(1))
        raise ValueError(f"Cannot extract numerical frame index from file name: {filename}")

    sorted_pdb_files = sorted(pdb_files, key=extract_frame_index)

    for pdb_file in sorted_pdb_files:
        try:
            frame_idx = extract_frame_index(pdb_file)
        except ValueError as e:
            print(f"Warning: {e}, skipping file {pdb_file}")
            continue

        print(f"Reading PDB file: {pdb_file} as frame {frame_idx}")

        with open(pdb_file, "r") as f:
            frame_lines = [line.rstrip("\n") for line in f if line.strip()]

        frame_data = parse_frame(frame_lines)
        yield frame_idx, frame_data, os.path.basename(pdb_file)

        del frame_lines


# ==================================================
# Bead-to-triangle assignment
# ==================================================

def build_triangle_centroid_kdtree(centroids):
    """Build an Open3D KD-tree over triangle centroids, for kNN bead assignment.

    centroids: (N, 3) array of every triangle's centroid, both leaflets
               concatenated (upper first, then lower -- see n_up elsewhere
               in this file for the split point). Comes from Step 3's saved
               mesh NPZ (all_centroids), not recomputed here.
    Returns: an o3d.geometry.KDTreeFlann ready for search_knn_vector_3d queries.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centroids)
    return o3d.geometry.KDTreeFlann(pcd)



def assign_beads_to_triangles_kNN(
    positions, kdt_tri, k, centroids,
    verts_up, tris_up, verts_lo, tris_lo, n_up
):
    """Assign each bead to its k nearest triangle centroids.

    For every bead position, queries kdt_tri for the k nearest triangle
    centroids and adds that bead as a coverage candidate to ALL k of them
    (not just the single nearest) -- a bead can contribute to more than one
    triangle's coverage sampling downstream in
    classify_all_triangles_radius_aware_parallel_core.

    positions: (N, 3) bead positions for this frame.
    kdt_tri:   KD-tree from build_triangle_centroid_kdtree, over both
               leaflets' triangle centroids concatenated.
    k:         number of nearest triangles each bead is assigned to
               (DEFECT_TRI_K in main.py, user-facing as triangle_kd_tree
               in prep.in).
    Returns: {triangle_index: [bead_index, ...]}, only for triangles that
             received at least one candidate bead.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    verts_up  = np.ascontiguousarray(verts_up,  dtype=np.float64)
    verts_lo  = np.ascontiguousarray(verts_lo,  dtype=np.float64)

    tri_map = {}
    for i, p in enumerate(positions):
        n_found, idxs, _ = kdt_tri.search_knn_vector_3d(p, k)
        if n_found == 0:
            continue

        for t in idxs:
            tri_map.setdefault(int(t), []).append(i)  # assign to ALL k triangles

    return tri_map

# ==================================================
# Debug tools
# ==================================================

def validate_triangle_indices(vertices, triangles, tri_indices, leaflet_name="upper"):
    """Sanity-check a set of defect triangle indices against a leaflet's mesh arrays.

    Confirms every index in tri_indices is in range for triangles, and that
    every vertex id those triangles reference is in range for vertices --
    catches an off-by-one or a stale/mismatched mesh array before it
    propagates silently into the saved defect NPZ or PLY.

    Returns nothing; prints an "OK" line on success, raises AssertionError
    on the first failed check.
    """
    tri_indices = np.asarray(tri_indices, dtype=np.int64)
    if tri_indices.size == 0:
        print(f"[VALIDATE] {leaflet_name}: no defect triangles (empty list) — OK")
        return
    n_tris = len(triangles)
    assert tri_indices.min() >= 0, f"{leaflet_name}: negative triangle index found"
    assert tri_indices.max() < n_tris, (
        f"{leaflet_name}: triangle index {tri_indices.max()} >= number of triangles {n_tris}"
    )
    faces = triangles[tri_indices]
    assert faces.ndim == 2 and faces.shape[1] == 3, (
        f"{leaflet_name}: faces should be (N,3); got {faces.shape}"
    )
    n_verts = len(vertices)
    assert faces.min() >= 0, f"{leaflet_name}: negative vertex id in faces"
    assert faces.max() < n_verts, (
        f"{leaflet_name}: vertex id {faces.max()} >= number of vertices {n_verts}"
    )
    print(f"[VALIDATE] {leaflet_name}: {tri_indices.size} triangle indices OK "
          f"(tris < {n_tris}, vertex ids < {n_verts})")


def write_triangle_bead_map(
    mapping_path,
    frame_idx,
    tri_map_idx,
    all_centroids,
    all_normals,
    positions,
    atoms,
):
    """Append a per-triangle candidate-bead debug map for one frame to disk.

    For every triangle that received at least one candidate bead (from
    assign_beads_to_triangles_kNN), writes one line listing each candidate
    bead's atom serial and its signed projection distance onto the
    triangle's plane along its normal. Not used in the pipeline right now.

    mapping_path:  path to triangle_bead_map.txt; appended to, not overwritten.
    frame_idx:     frame number, written as the first column.
    tri_map_idx:   {triangle_index: [bead_index, ...]} from
                   assign_beads_to_triangles_kNN.
    all_centroids: (N, 3) triangle centroids, both leaflets concatenated.
    all_normals:   (N, 3) triangle normals, same ordering.
    positions:     (N, 3) bead positions for this frame.
    atoms:         frame_data["atoms"] list, indexed by bead index for the
                   atom_serial written into each line.
    """
    # Normalize normals safely.
    nrm_mag = np.linalg.norm(all_normals, axis=1, keepdims=True)
    safe_normals = np.divide(
        all_normals,
        np.where(nrm_mag == 0.0, 1.0, nrm_mag),
        where=True
    )

    lines_for_frame = []

    for t in sorted(tri_map_idx.keys()):
        bead_idx_list = tri_map_idx[t]
        ctr = all_centroids[t]
        nrm = safe_normals[t]

        entries = []
        for bi in bead_idx_list:
            serial = atoms[bi].get("atom_serial", bi)
            p = positions[bi]
            proj = float(np.dot(p - ctr, nrm))   # signed projection
            entries.append(f"{serial}:{proj:.1f}")

        joined = "/".join(entries)
        lines_for_frame.append(f"{frame_idx:<15}{t:<20}{joined}\n")

    # Append results for this frame.
    with open(mapping_path, "a") as f:
        for line in lines_for_frame:
            f.write(line)


def write_radius_debug_log(log_path, frame_idx, results, n_up):
    """Append a lightweight per-triangle classification summary for one frame.

    Legacy debug tool, could be useful later. 

    log_path:  path to the log file; appended to, not overwritten.
    frame_idx: frame number, written into every line.
    results:   the dict returned by classify_all_triangles_radius_aware.
    n_up:      triangle count in the upper leaflet -- used to label each
               combined-index triangle as "upper" or "lower" and to
               recover its per-leaflet local index for the log line.
    """
    mode_names = {
        0: "degenerate_triangle",
        1: "head_above_override",
        2: "below_min_total_coverage",
        3: "classified",
    }
    head_fraction = results["head_fraction"]
    other_fraction = results["other_fraction"]
    total_fraction = results["total_fraction"]
    n_candidates = results["n_candidates"]
    n_used = results["n_used"]
    mode_code = results["mode_code"]
    is_defect = results["is_defect"]

    with open(log_path, "a") as f:
        for tri_idx in range(len(head_fraction)):
            if tri_idx < n_up:
                side = "upper"
                local_tri_idx = tri_idx
            else:
                side = "lower"
                local_tri_idx = tri_idx - n_up
            mode = mode_names.get(int(mode_code[tri_idx]), "unknown")
            f.write(
                f"frame={frame_idx} tri={tri_idx} side={side} "
                f"local_tri={local_tri_idx} "
                f"head_frac={head_fraction[tri_idx]:.5f} "
                f"other_frac={other_fraction[tri_idx]:.5f} "
                f"total_frac={total_fraction[tri_idx]:.5f} "
                f"ncand={int(n_candidates[tri_idx])} nused={int(n_used[tri_idx])} "
                f"mode={mode} defect={int(is_defect[tri_idx])}\n"
            )
# ==================================================
# Atom table loading
# ==================================================

def load_atom_table_csv(csv_path):
    """Load results/atom_info.csv into {(resname, atom_name): {flag, vdw_radius}}.

    This is the lookup Step 4 uses to attach a head/tail/neutral flag and a
    van der Waals radius to every bead in the frame, keyed by exactly the
    same (resname, atom_name) pairs write_atom_info_csv wrote in Step 1.

    csv_path: path to results/atom_info.csv.
    Returns: {(resname, atom_name): {"flag": str, "vdw_radius": float}}
    """
    atom_table = {}
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=",")  # comma, not tab
        for row in reader:
            res = row["resname"].strip()
            name = row["atom_name"].strip()
            flag = row["flag"].strip().lower()
            rA = float(row["vdw_radius"])
            atom_table[(res, name)] = {"flag": flag, "vdw_radius": rA}
    return atom_table


# ==================================================
# Radius-aware defect classification
# ==================================================
# Main idea, applied per triangle across both leaflets at once:
#   1) candidate beads come from assign_beads_to_triangles_kNN, above
#   2) build a local in-plane 2D basis from the triangle's own geometry
#   3) project each candidate bead center onto the triangle plane along
#      the triangle normal
#   4) use the bead's vdw_radius to form a circular footprint on that plane
#   5) sample points across the triangle to estimate head vs. non-head
#      coverage (see classify_all_triangles_radius_aware_parallel_core's
#      own docstring for the veto/coverage/typing rules that follow)
#   6) a triangle already classified as a defect is further typed
#      neutral-lipid vs. phospholipid-tail from the same sampling pass


# ==================================================
# Timing
# ==================================================

@contextmanager
def timed(stage, store, logger=logging):
    """Context manager: time a block and both log it and accumulate it in a dict.

    stage:  label used as both the dict key and the log line's name.
    store:  a dict (typically collections.defaultdict(float)) that
            stage's elapsed seconds are added to -- accumulates across
            repeated calls with the same stage name, does not overwrite.
    logger: anything with an .info() method; defaults to the stdlib
            logging module itself (not a Logger instance), which works
            because logging.info() is a valid module-level function.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        store[stage] += dt
        logger.info(f"⏱  {stage:<36} {dt:9.4f} s")



# ==================================================
# Classification: setup for the Numba kernel
# ==================================================
# The functions below prepare plain numeric/CSR arrays for
# classify_all_triangles_radius_aware_parallel_core's single parallel Numba
# kernel, rather than calling into Numba once per triangle from Python
# Preparation here:
#   1) converts the tri_map_idx dict -> CSR arrays (tri_map_dict_to_csr)
#   2) builds per-triangle local bases as compact numeric arrays
#      (build_triangle_geometry_arrays_numba, next pass)
#   3) the kernel itself runs the full triangle loop in one parallel pass,
#      avoiding a per-bead-candidate `inside` mask allocation


def barycentric_grid(n):
    """Build a triangular grid of barycentric sample points, (w0, w1, w2) per point.

    Used to sample coverage across every triangle's footprint in the
    classification kernel -- the same grid is reused for every triangle,
    since barycentric coordinates are triangle-shape-independent.

    n points per edge gives n*(n+1)/2 total sample points, arranged so
    every point satisfies w0 + w1 + w2 = 1 (with a small negative-w0
    tolerance for floating-point roundoff at the grid's edge).

    n: samples per triangle edge. DEFECT_GRID_SAMPLES in main.py (21 by
       default) controls this -- 21 gives 231 sample points per triangle.
    Returns: (n*(n+1)/2, 3) array of (w0, w1, w2) barycentric coordinates.
    """
    pts = []
    denom = max(1, n - 1)
    for i in range(n):
        for j in range(n - i):
            w1 = i / denom
            w2 = j / denom
            w0 = 1.0 - w1 - w2
            if w0 >= -1e-12:
                pts.append((w0, w1, w2))
    return np.asarray(pts, dtype=np.float64)


def build_atom_arrays(atoms, radius_scale):
    """Convert the per-frame atoms list into compact numeric/string arrays, once.

    The classification kernel is called once per frame but iterates every
    triangle's candidate beads inside a Numba parallel loop -- repeated
    dict lookups on Python atom dicts inside that loop would be far slower
    than indexing plain numpy arrays. This function does every atoms[i].get(...)
    lookup exactly once per frame, up front, and hands the kernel flat arrays.


    atoms:         frame_data["atoms"] list, each dict from write_atom_info_csv's
                    flag/vdw_radius lookup already applied by the caller.
    radius_scale:   uniform multiplier applied to every bead's vdw_radius.
    Returns: dict of arrays, one entry per atom, in the same order as atoms:
        atom_num:          int32, atom serial (falls back to 1-based index
                            if "atom_num" is absent from the atom dict).
        atom_radii:         float64, vdw_radius * radius_scale.
        atom_flag_code:     int8, 1 if flag == "h" (head), else 0.
        atom_neutral_code:  int8, 1 if flag == "n" (neutral lipid), else 0.
        res_name, atom_name: object arrays of the raw strings, kept for
                            optional/debug use -- not read by the kernel itself.
    """
    n_atoms = len(atoms)

    atom_num = np.empty(n_atoms, dtype=np.int32)
    atom_radii = np.empty(n_atoms, dtype=np.float64)
    atom_flag_code = np.zeros(n_atoms, dtype=np.int8)     # 1=head, 0=other
    atom_neutral_code = np.zeros(n_atoms, dtype=np.int8)  # 1=neutral lipid ("n"), 0=not
    res_name = np.empty(n_atoms, dtype=object)
    atom_name = np.empty(n_atoms, dtype=object)

    for i, a in enumerate(atoms):
        atom_num[i] = int(a.get("atom_num", i + 1))
        atom_radii[i] = float(a.get("vdw_radius", 0.0)) * radius_scale

        flag = str(a.get("flag", "")).strip().lower()
        atom_flag_code[i] = 1 if flag == "h" else 0
        atom_neutral_code[i] = 1 if flag == "n" else 0

        res_name[i] = a.get("res_name")
        atom_name[i] = a.get("atom_name")

    return {
        "atom_num": atom_num,
        "atom_radii": atom_radii,
        "atom_flag_code": atom_flag_code,
        "atom_neutral_code": atom_neutral_code,
        "res_name": res_name,
        "atom_name": atom_name,
    }


def tri_map_dict_to_csr(tri_map_idx, n_tri_total):
    """Convert {tri_idx: [atom_idx, ...]} into CSR-style flat arrays for Numba.

    Numba's njit functions can't efficiently consume a Python dict of
    variable-length lists -- this flattens tri_map_idx (from
    assign_beads_to_triangles_kNN) into the same compressed-sparse-row
    layout scipy uses, which the kernel indexes directly.

    tri_map_idx:  {triangle_index: [atom_index, ...]}.
    n_tri_total:  total triangle count across both leaflets (upper + lower).
    Returns: (cand_offsets, flat_candidates) where triangle t's candidate
             atom indices are flat_candidates[cand_offsets[t]:cand_offsets[t+1]].
    """
    counts = np.zeros(n_tri_total, dtype=np.int32)
    for tri_idx, cand in tri_map_idx.items():
        if 0 <= int(tri_idx) < n_tri_total:
            counts[int(tri_idx)] = len(cand)

    cand_offsets = np.empty(n_tri_total + 1, dtype=np.int64)
    cand_offsets[0] = 0
    np.cumsum(counts, out=cand_offsets[1:])

    flat_candidates = np.empty(int(cand_offsets[-1]), dtype=np.int32)
    for tri_idx in range(n_tri_total):
        cand = tri_map_idx.get(tri_idx, [])
        start = cand_offsets[tri_idx]
        for j, atom_idx in enumerate(cand):
            flat_candidates[start + j] = int(atom_idx)

    return cand_offsets, flat_candidates

@njit(cache=True, parallel=True, fastmath=True)
def build_triangle_geometry_arrays_numba(verts_up, tris_up, verts_lo, tris_lo, n_up):
    """Build a local orthonormal 2D basis (origin, u, v, normal) for every triangle.

    Runs once per frame, before classification, across both leaflets in one
    parallel pass (triangle index < n_up is upper, >= n_up is lower, offset
    by n_up). The classification kernel then projects each candidate bead
    onto this basis instead of rebuilding it per bead -- this is the
    "compact numeric arrays" step referenced in this section's banner.

    Basis construction per triangle (vertices a, b, c):
      u = normalize(b - a)                      -- first edge direction
      n = normalize(cross(b - a, c - a))         -- triangle normal
      v = cross(n, u)                            -- completes the orthonormal frame
    a becomes the local origin. u, n are already unit and orthogonal by
    construction, so v needs no further normalization step beyond the
    cross product itself (also unit length).

    local_xy stores the three triangle vertices in this local 2D (u, v)
    plane:
      vertex 0 (a) -> (0, 0)               -- always, by construction
      vertex 1 (b) -> (|b - a|, 0)         -- always on the local x-axis
      vertex 2 (c) -> (dot(c-a, u), dot(c-a, v))
    This lets the classification kernel generate every triangle's sample
    points from one shared barycentric_grid, without storing a full 2D
    sample-point array per triangle.

    Degenerate triangles (a zero-length first edge, or a zero-area cross
    product -- both checked against a 1e-12 tolerance) get degenerate[tri_idx]
    = 1 and are left with all-zero basis vectors; the classification kernel
    checks this flag and skips those triangles entirely (mode_code=0)
    rather than dividing by a near-zero length.

    verts_up, tris_up: upper leaflet vertex and triangle arrays.
    verts_lo, tris_lo: lower leaflet vertex and triangle arrays.
    n_up:              triangle count in the upper leaflet -- the split
                        point between upper and lower in every per-triangle
                        output array here and in the classification kernel.
    Returns: (origins, uvecs, vvecs, nvecs, local_xy, degenerate), each
             indexed by the same combined triangle index (upper first,
             then lower) used everywhere else in Step 4.
    """
    n_lo = tris_lo.shape[0]
    n_tri_total = n_up + n_lo

    origins = np.zeros((n_tri_total, 3), dtype=np.float64)
    uvecs = np.zeros((n_tri_total, 3), dtype=np.float64)
    vvecs = np.zeros((n_tri_total, 3), dtype=np.float64)
    nvecs = np.zeros((n_tri_total, 3), dtype=np.float64)
    local_xy = np.zeros((n_tri_total, 3, 2), dtype=np.float64)
    degenerate = np.zeros(n_tri_total, dtype=np.uint8)

    for tri_idx in prange(n_tri_total):
        if tri_idx < n_up:
            tri = tris_up[tri_idx]
            verts = verts_up
        else:
            tri = tris_lo[tri_idx - n_up]
            verts = verts_lo

        ia = tri[0]
        ib = tri[1]
        ic = tri[2]

        ax = verts[ia, 0]
        ay = verts[ia, 1]
        az = verts[ia, 2]
        bx = verts[ib, 0]
        by = verts[ib, 1]
        bz = verts[ib, 2]
        cx = verts[ic, 0]
        cy = verts[ic, 1]
        cz = verts[ic, 2]

        e1x = bx - ax
        e1y = by - ay
        e1z = bz - az
        e2x = cx - ax
        e2y = cy - ay
        e2z = cz - az

        e1_len = (e1x * e1x + e1y * e1y + e1z * e1z) ** 0.5
        if e1_len < 1e-12:
            degenerate[tri_idx] = 1
            continue

        ux = e1x / e1_len
        uy = e1y / e1_len
        uz = e1z / e1_len

        # n = normalize(cross(e1, e2))
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        n_len = (nx * nx + ny * ny + nz * nz) ** 0.5
        if n_len < 1e-12:
            degenerate[tri_idx] = 1
            continue
        nx /= n_len
        ny /= n_len
        nz /= n_len

        # v = normalize(cross(n, u)); n and u are already unit/orthogonal for valid triangles.
        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux
        v_len = (vx * vx + vy * vy + vz * vz) ** 0.5
        if v_len < 1e-12:
            degenerate[tri_idx] = 1
            continue
        vx /= v_len
        vy /= v_len
        vz /= v_len

        origins[tri_idx, 0] = ax
        origins[tri_idx, 1] = ay
        origins[tri_idx, 2] = az

        uvecs[tri_idx, 0] = ux
        uvecs[tri_idx, 1] = uy
        uvecs[tri_idx, 2] = uz

        vvecs[tri_idx, 0] = vx
        vvecs[tri_idx, 1] = vy
        vvecs[tri_idx, 2] = vz

        nvecs[tri_idx, 0] = nx
        nvecs[tri_idx, 1] = ny
        nvecs[tri_idx, 2] = nz

        # local 2D coordinates of triangle vertices
        # a -> (0,0), b -> (e1_len,0), c -> (dot(e2,u), dot(e2,v))
        local_xy[tri_idx, 0, 0] = 0.0
        local_xy[tri_idx, 0, 1] = 0.0
        local_xy[tri_idx, 1, 0] = e1_len
        local_xy[tri_idx, 1, 1] = 0.0
        local_xy[tri_idx, 2, 0] = e2x * ux + e2y * uy + e2z * uz
        local_xy[tri_idx, 2, 1] = e2x * vx + e2y * vy + e2z * vz

    return origins, uvecs, vvecs, nvecs, local_xy, degenerate


@njit(cache=True, parallel=True, fastmath=True)
def classify_all_triangles_radius_aware_parallel_core(
    positions,
    atom_radii,
    atom_flag_code,
    atom_neutral_code,
    cand_offsets,
    flat_candidates,
    origins,
    uvecs,
    vvecs,
    nvecs,
    local_xy,
    degenerate,
    bary_grid,
    signed_lower,
    signed_upper,
    type_signed_lower,
    type_signed_upper,
    head_threshold,
    min_total_coverage,
):
    """Classify every triangle across both leaflets as defect/non-defect, and type it.

    Runs once per frame, in parallel over every triangle. For each
    triangle, its candidate beads (flat_candidates[cand_offsets[t]:
    cand_offsets[t+1]], from tri_map_dict_to_csr) are tested against a
    signed-distance slab along the triangle normal
    A bead inside the slab has its center projected onto the triangle
    plane and its vdW radius used to test which of the shared
    barycentric_grid sample points fall inside its circular footprint.

    Per-triangle outcome, in priority order:
      1. has_head_above: if any
         head-flagged bead (atom_flag_code == 1) covers at least one
         sample point anywhere in the slab, the triangle is NOT a defect,
         regardless of anything else. mode_code = 1.
      2. total_fraction < min_total_coverage: too few sample points are
         covered by anything (head or other) to classify confidently ->
         NOT a defect. mode_code = 2.
      3. Otherwise: defect if head_fraction <= head_threshold, else not.
         mode_code = 3.
      0. degenerate[tri_idx] == 1 (from build_triangle_geometry_arrays_numba):
         skipped entirely, is_defect stays 0. mode_code = 0.

    "has_head_above" is the variable/mode_code name inherited from an
    earlier above-only version of this rule; the actual check has no sign
    restriction on signed_dist and triggers for a head bead on either side
    of the triangle

    Defect typing (neutral lipid vs. phospholipid tail) runs independently
    of the defect/non-defect decision above, using a separate, typically
    tighter sub-slab (type_signed_lower/type_signed_upper)
    Two-stage tiebreak per triangle:
      a. Primary: whichever of neutral_count / tail_count (sample points
         covered by a neutral-flagged vs. tail/other-flagged bead, within
         the type sub-slab) is larger wins.
      b. Tie (neutral_count == tail_count, including both zero): falls
         back to best_type_code, a z-buffer-style pick of the single
         "highest" eligible bead -- signed_dist is measured outward along
         the triangle normal (Step 3 orients upper-leaflet normals to +z
         and lower-leaflet to -z before Poisson reconstruction), so
         max(signed_dist) among type-slab-eligible beads consistently means
         "furthest above the surface wins; among beads below, the one
         closest to the surface wins" -- one comparison covers both cases.
      defect_type_code: 1 = neutral, 0 = tail. Only meaningful for
      triangles where is_defect == 1; computed for every triangle
      regardless, but callers should only read it alongside is_defect.

    positions:          (N, 3) all bead positions for this frame.
    atom_radii:          (N,) vdW radius per bead (radius_scale already applied).
    atom_flag_code:      (N,) 1 = head, 0 = other.
    atom_neutral_code:   (N,) 1 = neutral lipid, 0 = not.
    cand_offsets, flat_candidates: CSR candidate-bead arrays, from
                         tri_map_dict_to_csr.
    origins, uvecs, vvecs, nvecs, local_xy, degenerate: per-triangle local
                         basis, from build_triangle_geometry_arrays_numba.
    bary_grid:           shared (n*(n+1)/2, 3) barycentric sample grid,
                         from barycentric_grid.
    signed_lower, signed_upper: defect-detection slab bounds (DEFECT_SIGNED_
                         LOWER/UPPER in main.py; -30/+30 Angstrom by default).
    type_signed_lower, type_signed_upper: defect-typing sub-slab bounds
                         (DEFECT_TYPE_SIGNED_LOWER/UPPER in main.py).
    head_threshold:      max allowed head_fraction before a triangle counts
                         as a defect (DEFECT_HEAD_THRESHOLD, 0.50 default).
    min_total_coverage:  minimum total_fraction required to classify at all
                         (DEFECT_MIN_TOTAL_COVERAGE, 0.0 default -- disabled).

    Returns: (is_defect, head_fraction, other_fraction, total_fraction,
              n_candidates, n_after_slab, n_used, mode_code, defect_type_code),
             every array indexed by the same combined triangle index
             (upper first, then lower) used throughout Step 4.
    """
    n_tri_total = cand_offsets.shape[0] - 1
    ns = bary_grid.shape[0]

    is_defect = np.zeros(n_tri_total, dtype=np.uint8)
    head_fraction = np.zeros(n_tri_total, dtype=np.float32)
    other_fraction = np.zeros(n_tri_total, dtype=np.float32)
    total_fraction = np.zeros(n_tri_total, dtype=np.float32)
    n_candidates = np.zeros(n_tri_total, dtype=np.int32)
    n_after_slab = np.zeros(n_tri_total, dtype=np.int32)
    n_used = np.zeros(n_tri_total, dtype=np.int32)
    mode_code = np.zeros(n_tri_total, dtype=np.int8)
    defect_type_code = np.zeros(n_tri_total, dtype=np.int8)
    # mode_code: 0 degenerate, 1 head_above_override, 2 below_min_total_coverage, 3 classified

    for tri_idx in prange(n_tri_total):
        start = cand_offsets[tri_idx]
        end = cand_offsets[tri_idx + 1]
        n_candidates[tri_idx] = end - start

        if degenerate[tri_idx] == 1:
            mode_code[tri_idx] = 0
            continue

        # small per-triangle masks.  With grid_samples=21, ns=231.
        head_mask = np.zeros(ns, dtype=np.uint8)
        other_mask = np.zeros(ns, dtype=np.uint8)
        neutral_mask = np.zeros(ns, dtype=np.uint8)
        tail_mask = np.zeros(ns, dtype=np.uint8)

        ox = origins[tri_idx, 0]
        oy = origins[tri_idx, 1]
        oz = origins[tri_idx, 2]

        ux = uvecs[tri_idx, 0]
        uy = uvecs[tri_idx, 1]
        uz = uvecs[tri_idx, 2]

        vx = vvecs[tri_idx, 0]
        vy = vvecs[tri_idx, 1]
        vz = vvecs[tri_idx, 2]

        nx = nvecs[tri_idx, 0]
        ny = nvecs[tri_idx, 1]
        nz = nvecs[tri_idx, 2]

        # local triangle coordinates. Since vertex 0 is (0,0) and vertex 1 y is 0,
        # sample_x = w1*x1 + w2*x2; sample_y = w2*y2
        x1 = local_xy[tri_idx, 1, 0]
        x2 = local_xy[tri_idx, 2, 0]
        y2 = local_xy[tri_idx, 2, 1]

        has_head_above = 0
        used = 0
        after_slab = 0

        # Depth tiebreak state, consumed by the defect_type_code assignment at
        # the end of this loop body. Tracks the typing-eligible bead with the
        # largest signed_dist seen for this triangle. -1.0e30 is below any
        # reachable value (the slab caps |signed_dist| at 30 Å).
        best_type_depth = -1.0e30
        best_type_code = -1

        for kk in range(start, end):
            atom_idx = flat_candidates[kk]

            cx3 = positions[atom_idx, 0]
            cy3 = positions[atom_idx, 1]
            cz3 = positions[atom_idx, 2]

            rx = cx3 - ox
            ry = cy3 - oy
            rz = cz3 - oz

            signed_dist = rx * nx + ry * ny + rz * nz
            if signed_dist < signed_lower or signed_dist > signed_upper:
                continue

            after_slab += 1

            radius = atom_radii[atom_idx]
            if radius <= 0.0:
                continue
            r2 = radius * radius

            # Project bead center onto triangle plane, then into local 2D basis.
            px = cx3 - signed_dist * nx
            py = cy3 - signed_dist * ny
            pz = cz3 - signed_dist * nz

            prx = px - ox
            pry = py - oy
            prz = pz - oz

            projx = prx * ux + pry * uy + prz * uz
            projy = prx * vx + pry * vy + prz * vz

            flag = atom_flag_code[atom_idx]
            neutral_code = atom_neutral_code[atom_idx]
            any_inside = 0

            # Separate, tighter window for neutral-vs-tail typing only.
            # is_defect/head_mask/other_mask keep using the wide signed_lower/
            # signed_upper slab above, completely untouched.
            in_type_slab = (signed_dist >= type_signed_lower) and (signed_dist <= type_signed_upper)

            for si in range(ns):
                sx = bary_grid[si, 1] * x1 + bary_grid[si, 2] * x2
                sy = bary_grid[si, 2] * y2

                dx = sx - projx
                dy = sy - projy
                if dx * dx + dy * dy <= r2:
                    any_inside = 1
                    if flag == 1:
                        head_mask[si] = 1
                    else:
                        other_mask[si] = 1
                        if in_type_slab:
                            if neutral_code == 1:
                                neutral_mask[si] = 1
                            else:
                                tail_mask[si] = 1
            if any_inside == 1:
                used += 1
                if flag == 1:
                    has_head_above = 1
                elif in_type_slab and signed_dist > best_type_depth:
                    # z-buffer along the triangle normal. signed_dist is
                    # measured outward (Step 3 forces upper-leaflet normals to
                    # +z and lower-leaflet to -z before Poisson), so the single
                    # max(signed_dist) test covers the whole tier rule: above
                    # beats below, highest wins among those above, closest to
                    # the surface wins among those below.
                    best_type_depth = signed_dist
                    best_type_code = 1 if neutral_code == 1 else 0

        n_after_slab[tri_idx] = after_slab
        n_used[tri_idx] = used

        head_count = 0
        other_count = 0
        total_count = 0
        neutral_count = 0
        tail_count = 0
        for si in range(ns):
            h = head_mask[si]
            o = other_mask[si]
            if h == 1:
                head_count += 1
            if o == 1:
                other_count += 1
            if h == 1 or o == 1:
                total_count += 1
            if neutral_mask[si] == 1:
                neutral_count += 1
            if tail_mask[si] == 1:
                tail_count += 1

        hf = head_count / ns
        of = other_count / ns
        tf = total_count / ns

        head_fraction[tri_idx] = hf
        other_fraction[tri_idx] = of
        total_fraction[tri_idx] = tf

        if neutral_count > tail_count:
            defect_type_code[tri_idx] = 1
        elif tail_count > neutral_count:
            defect_type_code[tri_idx] = 0
        elif best_type_code == 1:
            defect_type_code[tri_idx] = 1
        else:
            defect_type_code[tri_idx] = 0

        if has_head_above == 1:
            is_defect[tri_idx] = 0
            mode_code[tri_idx] = 1
        elif tf < min_total_coverage:
            is_defect[tri_idx] = 0
            mode_code[tri_idx] = 2
        else:
            is_defect[tri_idx] = 0 if hf > head_threshold else 1
            mode_code[tri_idx] = 3

    return (
        is_defect,
        head_fraction,
        other_fraction,
        total_fraction,
        n_candidates,
        n_after_slab,
        n_used,
        mode_code,
        defect_type_code,
    )


def classify_all_triangles_radius_aware(
    tri_map_idx,
    atoms,
    positions,
    verts_up,
    tris_up,
    verts_lo,
    tris_lo,
    n_up,
    signed_lower,
    signed_upper,
    type_signed_lower,
    type_signed_upper,
    head_threshold,
    radius_scale,
    grid_samples,
    min_total_coverage,
):
    """Run the full Step 4 classification for one frame: setup, kernel, unpack.

    The Python-level orchestrator called once per frame from helpers.py's
    run_step4_defect_finding. Controls the functions in this step. 

    tri_map_idx:  {triangle_index: [atom_index, ...]}, from
                  assign_beads_to_triangles_kNN.
    atoms:        frame_data["atoms"] list, with flag/vdw_radius already
                  attached by the caller from load_atom_table_csv.
    positions:    (N, 3) bead positions for this frame.
    verts_up, tris_up, verts_lo, tris_lo: leaflet mesh arrays from Step 3.
    n_up:         triangle count in the upper leaflet.
    signed_lower, signed_upper, type_signed_lower, type_signed_upper,
    head_threshold, radius_scale, grid_samples, min_total_coverage:
                  passed straight through to the kernel -- see
                  classify_all_triangles_radius_aware_parallel_core's
                  docstring for what each controls.

    Returns: a dict with:
        defect_up, defect_lo: int32 arrays of triangle indices classified
                  as defects, local to each leaflet (already offset back
                  from the kernel's combined indexing).
        is_defect, head_fraction, other_fraction, total_fraction,
        n_candidates, n_after_slab, n_used, mode_code, defect_type_code:
                  the kernel's raw per-triangle output arrays, still in
                  combined (upper-then-lower) indexing -- unlike defect_up/
                  defect_lo, these are NOT split by leaflet.
        cand_offsets, flat_candidates: the CSR arrays built here, returned
                  for any downstream debug use.
        atom_num, atom_flag_code: kept for optional/debug tooling only --
                  not saved to the defect NPZ by default (see save_radius_npz).
    """
    n_tri_total = int(n_up) + len(tris_lo)

    # Make all numeric arrays contiguous/typed once, before entering Numba.
    verts_up = np.ascontiguousarray(verts_up, dtype=np.float64)
    tris_up = np.ascontiguousarray(tris_up, dtype=np.int32)
    verts_lo = np.ascontiguousarray(verts_lo, dtype=np.float64)
    tris_lo = np.ascontiguousarray(tris_lo, dtype=np.int32)
    positions = np.ascontiguousarray(positions, dtype=np.float64)

    bary_grid = barycentric_grid(grid_samples)
    atom_arrays = build_atom_arrays(atoms, radius_scale)
    cand_offsets, flat_candidates = tri_map_dict_to_csr(tri_map_idx, n_tri_total)

    origins, uvecs, vvecs, nvecs, local_xy, degenerate = build_triangle_geometry_arrays_numba(
        verts_up, tris_up, verts_lo, tris_lo, int(n_up)
    )

    (
        is_defect,
        head_fraction,
        other_fraction,
        total_fraction,
        n_candidates,
        n_after_slab,
        n_used,
        mode_code,
        defect_type_code,
    ) = classify_all_triangles_radius_aware_parallel_core(
        positions=positions,
        atom_radii=atom_arrays["atom_radii"],
        atom_flag_code=atom_arrays["atom_flag_code"],
        atom_neutral_code=atom_arrays["atom_neutral_code"],
        cand_offsets=cand_offsets,
        flat_candidates=flat_candidates,
        origins=origins,
        uvecs=uvecs,
        vvecs=vvecs,
        nvecs=nvecs,
        local_xy=local_xy,
        degenerate=degenerate,
        bary_grid=bary_grid,
        signed_lower=float(signed_lower),
        signed_upper=float(signed_upper),
        type_signed_lower=float(type_signed_lower),
        type_signed_upper=float(type_signed_upper),
        head_threshold=float(head_threshold),
        min_total_coverage=float(min_total_coverage),
    )

    defect_global = np.flatnonzero(is_defect).astype(np.int32)
    defect_up = defect_global[defect_global < n_up].astype(np.int32)
    defect_lo = (defect_global[defect_global >= n_up] - int(n_up)).astype(np.int32)

    return {
        "defect_up": defect_up,
        "defect_lo": defect_lo,
        "is_defect": is_defect.astype(np.uint8),
        "head_fraction": head_fraction,
        "other_fraction": other_fraction,
        "total_fraction": total_fraction,
        "n_candidates": n_candidates,
        "n_after_slab": n_after_slab,
        "n_used": n_used,
        "mode_code": mode_code,
        "defect_type_code": defect_type_code.astype(np.int8),
        "cand_offsets": cand_offsets,
        "flat_candidates": flat_candidates,
        # Keep atom arrays here only for optional/debug tooling.  They are not saved by default.
        "atom_num": atom_arrays["atom_num"],
        "atom_flag_code": atom_arrays["atom_flag_code"],
    }



# ==================================================
# Save
# ==================================================

def save_radius_npz(out_dir, frame_idx, verts_up, tris_up, verts_lo, tris_lo, n_up, results):
    """Save one frame's Step 4 output to results/defect/raw/npz/frame_####_radius_defects.npz.

    This is the file get_completed_defect_frames (helpers.py) checks for
    resume, and the file Step 5's periodic deduplication reads as input.
    Only written when DEFECT_STORE_DEBUG_NPZ is True in main.py (the
    default) -- see run_step4_defect_finding's resume-gating logic, which
    ties resume itself to this constant.

    out_dir:   directory to write into; created if missing.
    frame_idx: used to build the output filename, zero-padded to 4 digits.
    verts_up, tris_up, verts_lo, tris_lo, n_up: leaflet mesh arrays from
               Step 3, saved verbatim alongside the classification results
               so Step 5 doesn't need to re-read the Step 3 mesh NPZ.
    results:   the dict returned by classify_all_triangles_radius_aware.
    Returns: the output file path as a string.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"frame_{frame_idx:04d}_radius_defects.npz")
    np.savez_compressed(
        out_path,
        verts_up=np.asarray(verts_up, dtype=np.float32),
        tris_up=np.asarray(tris_up, dtype=np.int32),
        verts_lo=np.asarray(verts_lo, dtype=np.float32),
        tris_lo=np.asarray(tris_lo, dtype=np.int32),
        n_up=np.int32(n_up),
        defect_up=np.asarray(results["defect_up"], dtype=np.int32),
        defect_lo=np.asarray(results["defect_lo"], dtype=np.int32),
        head_fraction=np.asarray(results["head_fraction"], dtype=np.float32),
        other_fraction=np.asarray(results["other_fraction"], dtype=np.float32),
        total_fraction=np.asarray(results["total_fraction"], dtype=np.float32),
        n_candidates=np.asarray(results["n_candidates"], dtype=np.int32),
        n_after_slab=np.asarray(results["n_after_slab"], dtype=np.int32),
        n_used=np.asarray(results["n_used"], dtype=np.int32),
        mode_code=np.asarray(results["mode_code"], dtype=np.int8),
        defect_type_code=np.asarray(results["defect_type_code"], dtype=np.int8),
        is_defect=np.asarray(results["is_defect"], dtype=np.uint8),
    )
    return out_path


# ==================================================
# PLY output
# ==================================================
def _hex_to_rgb01(hex_code):
    """Convert a "#RRGGBB" hex color string to an (R, G, B) array in 0..1 range.

    Open3D vertex colors expect floats in [0, 1], not the 0..255 integer
    range hex strings encode.
    """
    hex_code = hex_code.lstrip("#")
    return np.array([int(hex_code[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def save_defects_only_ply(
    vertices,
    triangles,
    defect_indices,
    frame_idx,
    output_dir,
    prefix,
    defect_type_code=None,
    defect_groups=None,
    global_offset=0,
):
    """Write a PLY containing only the defect triangles for one leaflet.


    vertices, triangles: full leaflet mesh arrays (not pre-subset to defects).
    defect_indices:      triangle indices (local to this leaflet) to keep.
                          An empty array writes a valid but triangle-less PLY,
                          not an error.
    frame_idx, output_dir, prefix: used to build the output filename:
                          {prefix}_defects_frame_{frame_idx:04d}.ply
    defect_type_code:    combined-index (upper-then-lower) array from
                          classify_all_triangles_radius_aware; global_offset
                          is added to defect_indices to look up each
                          triangle's type in this combined array. If None,
                          every triangle is colored as type 0 (tail).
    global_offset:        0 for the upper leaflet, n_up for the lower leaflet
                          -- converts this leaflet's local defect_indices into
                          defect_type_code's combined indexing.
    Returns: the output file path as a string.
    """
    os.makedirs(output_dir, exist_ok=True)
    defect_indices = np.asarray(defect_indices, dtype=np.int64)
    triangles = np.asarray(triangles, dtype=np.int32)
    vertices = np.asarray(vertices, dtype=np.float64)

    out_path = os.path.join(output_dir, f"{prefix}_defects_frame_{frame_idx:04d}.ply")
    mesh = o3d.geometry.TriangleMesh()

    if defect_indices.size == 0:
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(np.empty((0, 3), dtype=np.int32))
        o3d.io.write_triangle_mesh(out_path, mesh)
        return out_path

    faces = triangles[defect_indices]
    uniq, inv = np.unique(faces.reshape(-1), return_inverse=True)
    sub_verts = vertices[uniq]
    sub_tris = inv.reshape(-1, 3).astype(np.int32)

    mesh.vertices = o3d.utility.Vector3dVector(sub_verts)
    mesh.triangles = o3d.utility.Vector3iVector(sub_tris)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    if defect_type_code is not None:
        types = np.asarray(defect_type_code)[global_offset + defect_indices]
    else:
        types = np.zeros(len(defect_indices), dtype=np.int8)

    rgb_tail = _hex_to_rgb01(_DEFECT_TAIL_HEX)
    rgb_neutral = _hex_to_rgb01(_DEFECT_NEUTRAL_HEX)

    vcols = np.zeros((len(sub_verts), 3), dtype=np.float64)
    counts = np.zeros(len(sub_verts), dtype=np.int32)
    for i in range(len(defect_indices)):
        c = rgb_neutral if types[i] == 1 else rgb_tail
        for v in sub_tris[i]:
            vcols[v] += c
            counts[v] += 1
    nz = counts > 0
    vcols[nz] /= counts[nz][:, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vcols)

    o3d.io.write_triangle_mesh(out_path, mesh, write_vertex_colors=True)
    return out_path


