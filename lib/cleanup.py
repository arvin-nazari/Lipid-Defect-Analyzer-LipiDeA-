"""Step 5 (periodic deduplication) and Step 6 (bridge pruning).

Two entry points, both called from helpers.py:
  run_periodic_deduplication  Step 5. Raw connected-component clustering of
                               Step 4's per-triangle defect mask, then
                               removal of duplicate clusters that straddle
                               the periodic tiling boundary.

  run_pruning                 Step 6. Reclusters Step 5's deduplicated
                               output with edge-metric tracking, optionally
                               removes graph-bridge edges (thin necks
                               connecting otherwise-separate defect
                               regions), and writes the final pruned
                               clusters both leaflets need for Step 7.

"""

import os
import re
import glob
import io
import csv
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from . import progress

import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None

from .defects import _DEFECT_TAIL_HEX, _DEFECT_NEUTRAL_HEX


# ==================================================
# Shared helpers (used by both Step 5 and Step 6)
# ==================================================

def frame_index_from_path(p):
    """Recover a frame index from any NPZ filename this pipeline writes.

    Raises ValueError if no digits are found at all.
    """
    base = os.path.basename(p)
    patterns = [
        r"frame_(\d+)_clusters_both\.npz$",
        r"frame_(\d+)_(?:radius_)?defects\.npz$",
        r"frame_(\d+)",
        r"(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, base)
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot parse frame index from: {p}")


def triangle_area(vertices, tri):
    """Area of one triangle, given its 3 vertex indices into vertices."""
    a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def triangle_centroid(vertices, tri):
    """Centroid (mean of 3 vertices) of one triangle."""
    return vertices[tri].mean(axis=0)


def cluster_area(vertices, tris, tri_list):
    """Total area of a cluster: sum of triangle_area over every triangle in tri_list."""
    return float(sum(triangle_area(vertices, tris[int(t)]) for t in tri_list))


def cluster_centroid(vertices, tris, tri_list):
    """Area-weighted centroid of a cluster of triangles.

    Each triangle's own centroid contributes proportionally to its area, so
    a cluster's centroid isn't skewed by many tiny triangles next to one
    large one. Falls back to the first triangle's plain centroid if total
    area is numerically zero (a degenerate cluster).
    """
    totA = 0.0
    ctr = np.zeros(3, dtype=np.float64)
    for t in tri_list:
        tri = tris[int(t)]
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        A = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        ctr += A * (a + b + c) / 3.0
        totA += A
    if totA <= 1e-30:
        return vertices[tris[int(tri_list[0])]].mean(axis=0)
    return ctr / totA


def flatten_clusters_to_tri_ids(clusters):
    """Flatten a list of clusters (each a list of triangle ids) into one sorted, deduplicated array."""
    return np.asarray(sorted({int(t) for cl in clusters for t in cl}), dtype=np.int32)


def build_edge_adjacency(tris, tri_ids):
    """Build a triangle adjacency map from shared edges, for the given triangle subset.

    Two triangles are adjacent if they share an edge (two vertex ids).
    Used by Step 5's raw connected-component clustering
    (compute_clusters_raw) -- Step 6 uses its own edge-metric-tracking
    variant, build_edge_adjacency_pruned, instead of this one.

    Returns: {triangle_id: set(adjacent_triangle_ids)}, one entry per id in tri_ids.
    """
    edge_to_tris = defaultdict(list)

    for t in tri_ids:
        i, j, k = tris[int(t)]
        edge_to_tris[tuple(sorted((int(i), int(j))))].append(int(t))
        edge_to_tris[tuple(sorted((int(j), int(k))))].append(int(t))
        edge_to_tris[tuple(sorted((int(k), int(i))))].append(int(t))

    neighbors = {int(t): set() for t in tri_ids}

    for _, ts in edge_to_tris.items():
        if len(ts) < 2:
            continue
        for a in ts:
            for b in ts:
                if a != b:
                    neighbors[a].add(b)

    return neighbors


def connected_components(neighbors, tri_ids):
    """Find connected components (clusters) via breadth-first search over an adjacency map.

    neighbors: {triangle_id: set(adjacent_ids)}, from build_edge_adjacency
               or build_edge_adjacency_pruned (post-bridge-removal).
    tri_ids:   the triangle ids to cluster -- every id in here gets placed
               into exactly one returned cluster, even if it has no
               neighbors (a cluster of size 1).
    Returns: list of clusters, each a list of triangle ids. Order is
             insertion order (by first occurrence in tri_ids), not size --
             callers sort by size afterward if that's what they need.
    """
    tri_set = set(int(t) for t in tri_ids)
    visited = set()
    clusters = []

    for t0 in tri_ids:
        t0 = int(t0)
        if t0 in visited:
            continue

        q = deque([t0])
        visited.add(t0)
        cluster = [t0]

        while q:
            t = q.popleft()
            for nb in neighbors.get(t, ()):
                if nb in tri_set and nb not in visited:
                    visited.add(nb)
                    q.append(nb)
                    cluster.append(nb)

        clusters.append(cluster)

    return clusters


def remap_submesh(vertices, tris, tri_ids):
    """Extract a standalone submesh (own vertex/triangle arrays) for a set of triangle ids.

    Vertices referenced by tri_ids are pulled out and re-indexed from 0, so
    the returned sub_tris only references sub_verts, not the original full
    mesh's vertex array -- needed before writing an isolated PLY or passing
    to Open3D visualization.

    Returns (np.zeros((0,3)), np.zeros((0,3), int32), np.zeros((0,), int32))
    for an empty tri_ids rather than raising. 

    Returns: (sub_verts, sub_tris, unique_vertex_ids) -- unique_vertex_ids
             maps each sub_verts row back to its index in the original
             vertices array.
    """
    tri_ids = np.asarray(tri_ids, dtype=np.int64)

    if tri_ids.size == 0:
        return (
            np.zeros((0, 3), dtype=vertices.dtype),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    faces = tris[tri_ids]
    unique_vids, inv = np.unique(faces.reshape(-1), return_inverse=True)
    sub_verts = vertices[unique_vids]
    sub_tris = inv.reshape(-1, 3).astype(np.int32)
    return sub_verts, sub_tris, unique_vids.astype(np.int32)


def color_mesh_by_clusters(submesh, sub_tris, clusters_local_tri_ids):
    """Color an Open3D mesh's vertices by which cluster their triangles belong to.
    Used in the visualization by open3d when debugging.

    Triangles with no assigned cluster (id -1, i.e. not part of
    clusters_local_tri_ids) contribute no color to their vertices.

    Mutates submesh.vertex_colors in place and returns submesh, for chaining.
    """
    M = len(sub_tris)
    tri_cluster = np.full(M, -1, dtype=np.int32)

    for cid, cl in enumerate(clusters_local_tri_ids):
        for t in cl:
            tri_cluster[int(t)] = cid

    def cid_color(cid):
        x = (cid * 2654435761) & 0xFFFFFFFF
        r = ((x >> 0) & 255) / 255.0
        g = ((x >> 8) & 255) / 255.0
        b = ((x >> 16) & 255) / 255.0
        return np.array([r, g, b], dtype=np.float64)

    vcols = np.zeros((np.asarray(submesh.vertices).shape[0], 3), dtype=np.float64)
    counts = np.zeros((vcols.shape[0],), dtype=np.int32)

    for t in range(M):
        cid = tri_cluster[t]
        if cid < 0:
            continue
        c = cid_color(cid)
        i, j, k = sub_tris[t]
        for v in (i, j, k):
            vcols[v] += c
            counts[v] += 1

    nz = counts > 0
    vcols[nz] /= counts[nz][:, None]
    submesh.vertex_colors = o3d.utility.Vector3dVector(vcols)
    return submesh


# ==================================================
# Step 5: periodic deduplication
# ==================================================

# --------------------------------------------------
# IO
# --------------------------------------------------
def load_defect_npz(npz_path):
    """Load one Step 4 raw defect NPZ (frame_####_radius_defects.npz).

    npz_path: path to a Step 4 raw defect NPZ.
    Returns: dict with verts_up/tris_up/mask_up/defect_type_up and the
             lower-leaflet equivalents.
    """
    data = np.load(npz_path, allow_pickle=True)

    verts_up = data["verts_up"]
    tris_up = data["tris_up"].astype(np.int32)
    verts_lo = data["verts_lo"]
    tris_lo = data["tris_lo"].astype(np.int32)

    if "defect_mask_up" in data and "defect_mask_lo" in data:
        mask_up = data["defect_mask_up"].astype(np.uint8)
        mask_lo = data["defect_mask_lo"].astype(np.uint8)
    elif "defect_up" in data and "defect_lo" in data:
        mask_up = np.zeros(len(tris_up), dtype=np.uint8)
        mask_lo = np.zeros(len(tris_lo), dtype=np.uint8)
        defect_up = np.asarray(data["defect_up"], dtype=np.int64).ravel()
        defect_lo = np.asarray(data["defect_lo"], dtype=np.int64).ravel()
        defect_up = defect_up[(defect_up >= 0) & (defect_up < len(tris_up))]
        defect_lo = defect_lo[(defect_lo >= 0) & (defect_lo < len(tris_lo))]
        mask_up[defect_up] = 1
        mask_lo[defect_lo] = 1
    else:
        keys = sorted(list(data.keys()))
        raise KeyError(f"Unrecognized defect NPZ format in {npz_path}. Keys found: {keys}")

    if "defect_type_code" not in data:
        raise KeyError(
            f"Missing 'defect_type_code' in {npz_path}. This NPZ predates "
            "defect-type classification -- delete results/defect/raw and "
            "rerun Step 4."
        )
    defect_type_code = np.asarray(data["defect_type_code"], dtype=np.int8)
    defect_type_up = defect_type_code[:len(tris_up)]
    defect_type_lo = defect_type_code[len(tris_up):]

    return {
        "verts_up": verts_up,
        "tris_up": tris_up,
        "mask_up": mask_up,
        "defect_type_up": defect_type_up,
        "verts_lo": verts_lo,
        "tris_lo": tris_lo,
        "mask_lo": mask_lo,
        "defect_type_lo": defect_type_lo,
    }



def collect_dedup_npz_files(npz_dir):
    """Collect raw Step 4 defect NPZ files as input to Step 5 deduplication.

    """
    patterns = [
        os.path.join(npz_dir, "frame_*_defects.npz"),
        os.path.join(npz_dir, "frame_*_radius_defects.npz"),
    ]
    npz_files = []
    for pattern in patterns:
        npz_files.extend(glob.glob(pattern))
    return sorted(set(npz_files), key=frame_index_from_path)

# --------------------------------------------------
# Geometry
# --------------------------------------------------

def wrap_to_periodic_cell(x, y, center_x, center_y, box_lx, box_ly):
    """Wrap an (x, y) point into the periodic box centered at (center_x, center_y).

    Used to bring a cluster centroid from Step 2's 3x3-tiled, cut coordinate
    frame back into a single box_lx x box_ly periodic cell, so two copies
    of the same physical cluster (one from the original tile, one from a
    neighboring tile) land on the same wrapped position and can be matched
    as duplicates in deduplicate_clusters_periodic.

    Returns: (xw, yw), each in [-box_lx/2, box_lx/2) / [-box_ly/2, box_ly/2)
             relative to the given center.
    """
    xr = x - center_x
    yr = y - center_y

    xw = ((xr + box_lx / 2.0) % box_lx) - box_lx / 2.0
    yw = ((yr + box_ly / 2.0) % box_ly) - box_ly / 2.0
    return xw, yw


def estimate_full_center(vertices, tris):
    """Estimate the membrane's xy center from the bounding box of triangle centroids.

    Fallback used by resolve_center when the user hasn't supplied an
    explicit center (DEDUP_FULL_CENTER_X/Y in main.py, both None by
    default). Uses the midpoint of triangle centroids' min/max, not vertex
    positions directly -- centroid-based bounds are less sensitive to a few
    stray boundary vertices than raw vertex bounds would be.

    Returns: (cx, cy) as plain floats.
    """
    tri_xyz = vertices[tris]
    cents = tri_xyz.mean(axis=1)

    cx = 0.5 * (cents[:, 0].min() + cents[:, 0].max())
    cy = 0.5 * (cents[:, 1].min() + cents[:, 1].max())
    return float(cx), float(cy)


def resolve_center(vertices, tris, full_center_x, full_center_y):
    """Resolve the periodic-wrap center to use: user-supplied, or auto-estimated.

    Either coordinate can be independently None -- e.g. a user could pin
    full_center_x but leave full_center_y to auto-estimate. In practice
    both are always None together in the current pipeline
    (DEDUP_FULL_CENTER_X/Y default), so this always falls through to
    estimate_full_center for both.

    Returns: (center_x, center_y) as plain floats, never None.
    """
    center_x = full_center_x
    center_y = full_center_y

    if center_x is None or center_y is None:
        auto_cx, auto_cy = estimate_full_center(vertices, tris)
        if center_x is None:
            center_x = auto_cx
        if center_y is None:
            center_y = auto_cy

    return float(center_x), float(center_y)


# --------------------------------------------------
# Raw clustering
# --------------------------------------------------

def compute_clusters_raw(tris, defect_mask, min_cluster_size=1):
    """Cluster every defect-flagged triangle into connected components, before dedup.

    First stage of Step 5: takes Step 4's per-triangle defect_mask, groups
    defect triangles into clusters via edge adjacency (build_edge_adjacency
    + connected_components), drops clusters smaller than min_cluster_size,
    and sorts the survivors largest-first. Periodic duplicate removal
    (deduplicate_clusters_periodic) runs on this output next.

    tris:             (T, 3) triangle vertex-index array for one leaflet.
    defect_mask:       (T,) 0/1 array from load_defect_npz.
    min_cluster_size:  clusters smaller than this (in triangle count) are
                       dropped here, before dedup even sees them.
    Returns: (raw_defect_tri_ids, raw_clusters) -- an int32 array of every
             defect triangle id, and a list of clusters (each a list of
             triangle ids), largest first. Both empty if defect_mask has no
             1s at all.
    """
    raw_defect_tri_ids = np.where(defect_mask.astype(np.uint8) == 1)[0].astype(np.int32)
    if raw_defect_tri_ids.size == 0:
        return raw_defect_tri_ids, []

    neighbors = build_edge_adjacency(tris, raw_defect_tri_ids)
    raw_clusters = connected_components(neighbors, raw_defect_tri_ids)
    raw_clusters = [cl for cl in raw_clusters if len(cl) >= int(min_cluster_size)]
    raw_clusters = sorted(raw_clusters, key=len, reverse=True)

    return raw_defect_tri_ids, raw_clusters


# --------------------------------------------------
# Periodic deduplication
# --------------------------------------------------

def deduplicate_clusters_periodic(
    vertices,
    tris,
    clusters,
    full_center_x,
    full_center_y,
    periodic_lx,
    periodic_ly,
    match_tol,
):
    """Remove duplicate clusters that are the same physical defect seen through different periodic tiles.

    Step 2 tiled each frame 3x3 before cutting a centered window back out
    (see prep.py), so the same physical membrane defect can appear more
    than once in the mesh -- once per tile copy that survived the cut. This
    wraps every cluster's centroid into one periodic_lx x periodic_ly cell
    (wrap_to_periodic_cell) and keeps the first (largest-area,
    since clusters is expected pre-sorted largest-first) cluster at each
    wrapped position, discarding any later cluster whose wrapped centroid
    falls within match_tol of one already kept.

    match_tol is in the same units as the coordinates (Angstroms) --
    DEDUP_MATCH_TOL in main.py.

    Returns: list of kept clusters (each a list of triangle ids), in the
             same largest-first order deduplication encountered them.
    """
    cluster_info = []

    for cl in clusters:
        ctr = cluster_centroid(vertices, tris, cl)
        area = cluster_area(vertices, tris, cl)

        xw, yw = wrap_to_periodic_cell(
            ctr[0], ctr[1],
            full_center_x, full_center_y,
            periodic_lx, periodic_ly,
        )

        cluster_info.append({
            "triangles": [int(t) for t in cl],
            "centroid": ctr,
            "wrapped": np.array([xw, yw], dtype=np.float64),
            "area": area,
        })

    cluster_info.sort(key=lambda x: x["area"], reverse=True)

    kept = []
    kept_wrapped = []

    for c in cluster_info:
        w = c["wrapped"]
        duplicate = False
        for wk in kept_wrapped:
            if np.linalg.norm(w - wk) < match_tol:
                duplicate = True
                break

        if not duplicate:
            kept.append(c)
            kept_wrapped.append(w)

    return [c["triangles"] for c in kept]


def compute_clusters_dedup(
    vertices,
    tris,
    defect_mask,
    full_center_x,
    full_center_y,
    periodic_lx,
    periodic_ly,
    match_tol,
    min_cluster_size=1,
):
    """Run the full Step 5 pipeline for one leaflet: raw clustering, then periodic dedup.

    First orchestrator combining compute_clusters_raw and
    deduplicate_clusters_periodic 

    Returns: (final_tri_ids, final_clusters, raw_defect_tri_ids, raw_clusters).
             All four are empty/raw_defect_tri_ids-only if there were no
             defect triangles to begin with -- final_tri_ids and
             raw_defect_tri_ids are the same empty array in that case.
    """
    raw_defect_tri_ids, raw_clusters = compute_clusters_raw(
        tris=tris,
        defect_mask=defect_mask,
        min_cluster_size=min_cluster_size,
    )

    if raw_defect_tri_ids.size == 0:
        return raw_defect_tri_ids, [], raw_defect_tri_ids, []

    dedup_clusters = deduplicate_clusters_periodic(
        vertices=vertices,
        tris=tris,
        clusters=raw_clusters,
        full_center_x=full_center_x,
        full_center_y=full_center_y,
        periodic_lx=periodic_lx,
        periodic_ly=periodic_ly,
        match_tol=match_tol,
    )

    final_clusters = sorted(dedup_clusters, key=len, reverse=True)
    final_tri_ids = flatten_clusters_to_tri_ids(final_clusters)

    return final_tri_ids, final_clusters, raw_defect_tri_ids, raw_clusters


# --------------------------------------------------
# Save
# --------------------------------------------------

def _clusters_to_object_array(clusters):
    """Convert a list of clusters (each a list of triangle ids) into a numpy object array.

    NPZ (via np.savez_compressed) can't store a ragged list of
    variable-length arrays directly -- an object-dtype array of int32
    arrays is the standard workaround, read back the same way by
    _normalize_clusters_obj_array in the pruning half of this file.
    """
    return np.array([np.asarray(cl, dtype=np.int32) for cl in clusters], dtype=object)


def save_cluster_npz_both_leaflets(
    out_path,
    d,
    out_up,
    out_lo,
    periodic_lx,
    periodic_ly,
    match_tol,
    min_cluster_size,
):
    """Save one frame's Step 5 output: both leaflets' meshes, raw clusters, and deduplicated clusters.

    This is the file Step 6's collect_npz_files looks for


    out_path: destination NPZ path.
    d:        dict from load_defect_npz -- supplies both leaflets' raw mesh
              arrays and defect_type.
    out_up, out_lo: dicts from process_one_leaflet, one per leaflet.
    periodic_lx, periodic_ly, match_tol, min_cluster_size: the parameters
              this run used, saved alongside the results for provenance.
    """
    np.savez_compressed(
        out_path,
        format_tag=np.array(["dedup_both_leaflets_v1"]),

        verts_up=np.asarray(d["verts_up"], dtype=np.float32),
        tris_up=np.asarray(d["tris_up"], dtype=np.int32),
        defect_mask_up=np.asarray(d["mask_up"], dtype=np.uint8),
        defect_type_up=np.asarray(d["defect_type_up"], dtype=np.int8),
        raw_defect_tri_ids_up=np.asarray(out_up["raw_defect_tri_ids"], dtype=np.int32),
        raw_clusters_up=_clusters_to_object_array(out_up["raw_clusters"]),
        final_defect_tri_ids_up=np.asarray(out_up["final_tri_ids"], dtype=np.int32),
        final_clusters_up=_clusters_to_object_array(out_up["final_clusters"]),
        center_x_up=np.float64(out_up["center_x"]),
        center_y_up=np.float64(out_up["center_y"]),

        verts_lo=np.asarray(d["verts_lo"], dtype=np.float32),
        tris_lo=np.asarray(d["tris_lo"], dtype=np.int32),
        defect_mask_lo=np.asarray(d["mask_lo"], dtype=np.uint8),
        defect_type_lo=np.asarray(d["defect_type_lo"], dtype=np.int8),
        raw_defect_tri_ids_lo=np.asarray(out_lo["raw_defect_tri_ids"], dtype=np.int32),
        raw_clusters_lo=_clusters_to_object_array(out_lo["raw_clusters"]),
        final_defect_tri_ids_lo=np.asarray(out_lo["final_tri_ids"], dtype=np.int32),
        final_clusters_lo=_clusters_to_object_array(out_lo["final_clusters"]),
        center_x_lo=np.float64(out_lo["center_x"]),
        center_y_lo=np.float64(out_lo["center_y"]),

        periodic_lx=np.float64(periodic_lx),
        periodic_ly=np.float64(periodic_ly),
        match_tol=np.float64(match_tol),
        min_cluster_size=np.int32(min_cluster_size),
    )


# --------------------------------------------------
# Per-leaflet processing
# --------------------------------------------------

def process_one_leaflet(
    verts,
    tris,
    mask,
    full_center_x,
    full_center_y,
    periodic_lx,
    periodic_ly,
    match_tol,
    min_cluster_size,
):
    """Run Step 5 (center resolution + raw clustering + periodic dedup) for one leaflet.

    Second orchestrator: resolve_center, then compute_clusters_dedup. Called
    twice per frame by process_and_save_one_file (once per leaflet) and
    once per requested leaflet by visualize_dedup_frame_clusters.

    Returns: dict with center_x, center_y, raw_defect_tri_ids, raw_clusters,
             final_tri_ids, final_clusters.
    """
    center_x, center_y = resolve_center(verts, tris, full_center_x, full_center_y)

    final_tri_ids, final_clusters, raw_defect_tri_ids, raw_clusters = compute_clusters_dedup(
        vertices=verts,
        tris=tris,
        defect_mask=mask,
        full_center_x=center_x,
        full_center_y=center_y,
        periodic_lx=periodic_lx,
        periodic_ly=periodic_ly,
        match_tol=match_tol,
        min_cluster_size=min_cluster_size,
    )

    return {
        "center_x": center_x,
        "center_y": center_y,
        "raw_defect_tri_ids": raw_defect_tri_ids,
        "raw_clusters": raw_clusters,
        "final_tri_ids": final_tri_ids,
        "final_clusters": final_clusters,
    }

def process_and_save_one_file(npz_path, args):
    """Run Step 5 for one frame's raw defect NPZ: both leaflets, then save the combined result.

    Called once per frame by run_periodic_deduplication, either directly
    (serial mode) or via _process_and_save_one_file_capture (parallel mode,
    one call per worker process).

    npz_path: path to one Step 4 raw defect NPZ.
    args:     SimpleNamespace built by run_periodic_deduplication, carrying
              frame_box_xyz, full_center_x/y, match_tol, min_cluster_size,
              and output_npz_dir.
    """
    frame_idx = frame_index_from_path(npz_path)
    lx, ly, _lz = args.frame_box_xyz[frame_idx]

    d = load_defect_npz(npz_path)

    out_up = process_one_leaflet(
        verts=d["verts_up"],
        tris=d["tris_up"],
        mask=d["mask_up"],
        full_center_x=args.full_center_x,
        full_center_y=args.full_center_y,
        periodic_lx=lx,
        periodic_ly=ly,
        match_tol=args.match_tol,
        min_cluster_size=args.min_cluster_size,
    )

    out_lo = process_one_leaflet(
        verts=d["verts_lo"],
        tris=d["tris_lo"],
        mask=d["mask_lo"],
        full_center_x=args.full_center_x,
        full_center_y=args.full_center_y,
        periodic_lx=lx,
        periodic_ly=ly,
        match_tol=args.match_tol,
        min_cluster_size=args.min_cluster_size,
    )

    out_dir = args.output_npz_dir
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"frame_{frame_idx:04d}_clusters_both.npz")

    save_cluster_npz_both_leaflets(
        out_path=out_path,
        d=d,
        out_up=out_up,
        out_lo=out_lo,
        periodic_lx=lx,
        periodic_ly=ly,
        match_tol=args.match_tol,
        min_cluster_size=args.min_cluster_size,
    )

    print(f"NPZ: {npz_path}")
    print(f"Saved: {out_path}")
    print(f"Box (frame {frame_idx}): Lx={lx:.3f}, Ly={ly:.3f} Å")
    print(f"Upper raw clusters: {len(out_up['raw_clusters'])}")
    print(f"Upper after periodic dedup: {len(out_up['final_clusters'])}")
    print(f"Upper final defect triangles: {len(out_up['final_tri_ids'])}")
    print(f"Lower raw clusters: {len(out_lo['raw_clusters'])}")
    print(f"Lower after periodic dedup: {len(out_lo['final_clusters'])}")
    print(f"Lower final defect triangles: {len(out_lo['final_tri_ids'])}")
    print("-" * 60)

def _process_and_save_one_file_capture(npz_path, args):
    """Run one Step-5 deduplication job and return captured console output.

    Keeping worker output captured prevents multiple processes from mixing their
    print statements together in the terminal.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        process_and_save_one_file(npz_path, args)
    return frame_index_from_path(npz_path), buf.getvalue()


# ==================================================
# Step 6: bridge pruning
# ==================================================

# --------------------------------------------------
# IO
# --------------------------------------------------

def _normalize_clusters_obj_array(arr):
    """Convert a saved object-dtype array of clusters back into plain Python lists of ints.

    Reverses _clusters_to_object_array's NPZ-friendly encoding (see that
    function's docstring) -- reading a saved cluster array back out gives
    numpy int32 arrays wrapped in an object array; this flattens each one
    to a plain list[int], which the rest of this file's clustering code
    expects.
    """
    out = []
    for cl in np.asarray(arr, dtype=object):
        out.append([int(x) for x in np.asarray(cl, dtype=np.int32).ravel().tolist()])
    return out


def load_input_npz(npz_path):
    """Load a Step 6 input NPZ, whichever of three formats it turns out to be.

    Tried in order:

      1. Step 5 dedup output (frame_####_clusters_both.npz) -- the normal,
         expected input. Carries both leaflets together, plus the already-
         deduplicated final_defect_tri_ids_up/lo and final_clusters_up/lo,
         which become this function's seed_tri_ids_up/lo and
         seed_clusters_up/lo -- Step 6 reclusters and prunes starting from
         Step 5's deduplicated result, not from scratch.
      2. A raw Step 4 defect NPZ carrying a pre-built mask
         (defect_mask_up/lo) -- lets Step 6 run directly on Step 4 output
         if Step 5 was skipped.
      3. A raw Step 4 defect NPZ carrying a plain triangle-index list
         (defect_up/lo) instead of a mask -- the format defects.py's
         save_radius_npz actually writes; the mask is built here from it.

    This was made to make sure step 6 works regardless of step 5. 

    Formats 2 and 3 both return source_mode="raw_defect", with
    seed_clusters_up/lo empty (no pre-existing clusters to seed from --
    compute_clusters_with_pruning starts raw clustering from seed_tri_ids
    alone in that case) and no defect_type_up/lo key at all, since neither
    raw-format NPZ carries defect typing at the individual-cluster level
    the way Step 5's dedup output does.

    npz_path: path to the NPZ to load.
    Returns: dict; see the three formats above for which keys are present.
             Always has: source_mode, verts_up, tris_up, seed_tri_ids_up,
             seed_clusters_up, mask_up, and the lower-leaflet equivalents.
    Raises: KeyError if the NPZ matches none of the three known formats.
    """
    data = np.load(npz_path, allow_pickle=True)

    # Preferred path: Step 5 dedup output carrying both leaflets together.
    if (
        "verts_up" in data and "tris_up" in data and
        "verts_lo" in data and "tris_lo" in data and
        "final_defect_tri_ids_up" in data and "final_defect_tri_ids_lo" in data
    ):
        return {
            "source_mode": "dedup",
            "verts_up": np.asarray(data["verts_up"]),
            "tris_up": np.asarray(data["tris_up"], dtype=np.int32),
            "seed_tri_ids_up": np.asarray(data["final_defect_tri_ids_up"], dtype=np.int32),
            "seed_clusters_up": _normalize_clusters_obj_array(data["final_clusters_up"]) if "final_clusters_up" in data else [],
            "mask_up": np.asarray(data["defect_mask_up"], dtype=np.uint8) if "defect_mask_up" in data else None,
            "defect_type_up": np.asarray(data["defect_type_up"], dtype=np.int8) if "defect_type_up" in data else None,
            "verts_lo": np.asarray(data["verts_lo"]),
            "tris_lo": np.asarray(data["tris_lo"], dtype=np.int32),
            "seed_tri_ids_lo": np.asarray(data["final_defect_tri_ids_lo"], dtype=np.int32),
            "seed_clusters_lo": _normalize_clusters_obj_array(data["final_clusters_lo"]) if "final_clusters_lo" in data else [],
            "mask_lo": np.asarray(data["defect_mask_lo"], dtype=np.uint8) if "defect_mask_lo" in data else None,
            "defect_type_lo": np.asarray(data["defect_type_lo"], dtype=np.int8) if "defect_type_lo" in data else None,
        }

    # Fallback path: a raw Step 4 defect file.
    verts_up = data["verts_up"]
    tris_up = data["tris_up"].astype(np.int32)
    verts_lo = data["verts_lo"]
    tris_lo = data["tris_lo"].astype(np.int32)

    if "defect_mask_up" in data and "defect_mask_lo" in data:
        mask_up = data["defect_mask_up"].astype(np.uint8)
        mask_lo = data["defect_mask_lo"].astype(np.uint8)
    elif "defect_up" in data and "defect_lo" in data:
        mask_up = np.zeros(len(tris_up), dtype=np.uint8)
        mask_lo = np.zeros(len(tris_lo), dtype=np.uint8)

        defect_up = np.asarray(data["defect_up"], dtype=np.int64).ravel()
        defect_lo = np.asarray(data["defect_lo"], dtype=np.int64).ravel()

        defect_up = defect_up[(defect_up >= 0) & (defect_up < len(tris_up))]
        defect_lo = defect_lo[(defect_lo >= 0) & (defect_lo < len(tris_lo))]

        mask_up[defect_up] = 1
        mask_lo[defect_lo] = 1
    else:
        keys = sorted(list(data.keys()))
        raise KeyError(f"Unrecognized NPZ format in {npz_path}. Keys found: {keys}")

    return {
        "source_mode": "raw_defect",
        "verts_up": verts_up,
        "tris_up": tris_up,
        "seed_tri_ids_up": np.where(mask_up.astype(np.uint8) == 1)[0].astype(np.int32),
        "seed_clusters_up": [],
        "mask_up": mask_up,
        "verts_lo": verts_lo,
        "tris_lo": tris_lo,
        "seed_tri_ids_lo": np.where(mask_lo.astype(np.uint8) == 1)[0].astype(np.int32),
        "seed_clusters_lo": [],
        "mask_lo": mask_lo,
    }

# --------------------------------------------------
# Geometry
# --------------------------------------------------

def _canon_pair(a, b):
    """Order a pair of triangle ids consistently, smaller first.

    Used everywhere an unordered edge (a, b) needs to be a stable dict key
    or set member -- find_graph_bridges and build_edge_adjacency_pruned
    both discover the same edge from either triangle's side and need it to
    hash to the same key regardless of which side found it first.
    """
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)



def find_graph_bridges(neighbors):
    """Find every graph bridge edge in a triangle-adjacency graph via iterative Tarjan DFS.

    A bridge is an edge whose removal splits its connected component into
    two -- in this context, a thin one-triangle-wide neck connecting two
    otherwise-separate defect regions that happen to touch at a single
    shared edge. prune_graph_bridges (below) uses this to optionally cut
    those necks apart, so two defects joined only by a sliver aren't
    counted and measured as one large cluster.

    This is equivalent to the textbook recursive DFS bridge-finding
    algorithm, but it avoids Python's recursion limit for large defect
    clusters with thousands of connected triangles. This is important when
    running pruning with:

        bridge_prune=True (Default)
        bridge_prune_all=True (Default)

    because bridge detection must traverse the full connected triangle graph.

    neighbors: {triangle_id: set(adjacent_triangle_ids)}, from
               build_edge_adjacency_pruned.
    Returns: set of (smaller_id, larger_id) tuples, each an edge that is a
             bridge, via _canon_pair's ordering.
    """
    disc = {}
    low = {}
    parent = {}
    bridges = set()
    time_counter = 0

    for start in list(neighbors.keys()):
        start = int(start)
        if start in disc:
            continue

        parent[start] = None

        # Stack entries are [node, iterator, entered_flag].  We keep the
        # neighbor iterator on the stack so this behaves like recursive DFS
        # without actually using Python recursion.
        stack = [[start, iter(neighbors.get(start, ())), False]]

        while stack:
            u, it, entered = stack[-1]

            if not entered:
                time_counter += 1
                disc[u] = time_counter
                low[u] = time_counter
                stack[-1][2] = True

            try:
                v = int(next(it))
            except StopIteration:
                stack.pop()
                pu = parent.get(u)
                if pu is not None:
                    low[pu] = min(low[pu], low[u])
                    if low[u] > disc[pu]:
                        bridges.add(_canon_pair(pu, u))
                continue

            if v not in disc:
                parent[v] = u
                stack.append([v, iter(neighbors.get(v, ())), False])
            elif parent.get(u) != v:
                low[u] = min(low[u], disc[v])

    return bridges


def build_edge_adjacency_pruned(vertices, tris, tri_ids, return_edge_metrics=False):
    """Build triangle-edge adjacency for Step 6, optionally with per-edge geometric metrics.

    Older version had more conditions, I deleted most of them since it would be
    too complicating. Simple full graph pruning here. 
    Edge metrics (neck_width, area_ratio, centroid_dist) are still computed
    and, if return_edge_metrics=True, returned and later saved to the
    pruned NPZ for diagnostics/output -- but nothing in this pipeline reads
    them back to make a pruning decision.

    vertices, tris:      full leaflet mesh arrays.
    tri_ids:              the (already deduplicated, from Step 5) defect
                          triangle ids to build adjacency over.
    return_edge_metrics:  if True, also return the per-edge metrics dict.
    Returns: neighbors ({triangle_id: set(adjacent_ids)}), and if
             return_edge_metrics, also edge_metrics
             ({(smaller_id, larger_id): {edge_len, neck_width, area_ratio,
             centroid_dist}}) -- only for edges shared by exactly 2
             triangles; a non-manifold edge (shared by >2, or an open
             boundary edge shared by only 1) is silently skipped here.
    """
    tri_ids = np.asarray(tri_ids, dtype=np.int32)

    tri_area = {}
    tri_ctr = {}
    for t in tri_ids:
        t = int(t)
        tri_area[t] = triangle_area(vertices, tris[t])
        tri_ctr[t] = triangle_centroid(vertices, tris[t])

    edge_to_tris = defaultdict(list)
    for t in tri_ids:
        i, j, k = tris[int(t)]
        edge_to_tris[tuple(sorted((int(i), int(j))))].append(int(t))
        edge_to_tris[tuple(sorted((int(j), int(k))))].append(int(t))
        edge_to_tris[tuple(sorted((int(k), int(i))))].append(int(t))

    neighbors = {int(t): set() for t in tri_ids}
    edge_metrics = {}

    for edge, ts in edge_to_tris.items():
        if len(ts) != 2:
            continue

        ta, tb = int(ts[0]), int(ts[1])

        v1, v2 = edge
        p1 = vertices[v1]
        p2 = vertices[v2]
        edge_len = float(np.linalg.norm(p2 - p1))

        Aa = float(tri_area[ta])
        Ab = float(tri_area[tb])

        neck_width = float((2.0 * Aa + 2.0 * Ab) / max(edge_len, 1e-12))
        area_ratio = float(min(Aa, Ab) / max(max(Aa, Ab), 1e-12))
        dcent = float(np.linalg.norm(tri_ctr[ta] - tri_ctr[tb]))

        neighbors[ta].add(tb)
        neighbors[tb].add(ta)
        edge_metrics[_canon_pair(ta, tb)] = {
            "edge_len": edge_len,
            "neck_width": neck_width,
            "area_ratio": area_ratio,
            "centroid_dist": dcent,
        }

    if return_edge_metrics:
        return neighbors, edge_metrics
    return neighbors


# --------------------------------------------------
# Main clustering
# --------------------------------------------------
def prune_graph_bridges(neighbors, prune_all_bridges=True):
    """Optionally remove every graph-bridge edge from an adjacency map.
    Simple pruning. 

    neighbors:          {triangle_id: set(adjacent_ids)}, from
                         build_edge_adjacency_pruned.
    prune_all_bridges:   if False, bridges are found but none are removed
                         (pruned == a copy of the input, unchanged).
    Returns: (pruned, bridges, removed) -- the (possibly) pruned adjacency
             copy, the full set of bridge edges found (regardless of
             prune_all_bridges), and the list of edges actually removed.
    """
    pruned = {int(t): set(map(int, nbs)) for t, nbs in neighbors.items()}
    bridges = find_graph_bridges(pruned)
    removed = []

    if prune_all_bridges:
        for u, v in sorted(bridges):
            pruned[u].discard(v)
            pruned[v].discard(u)
            removed.append((u, v))

    return pruned, bridges, removed


def compute_clusters_with_pruning(
    vertices,
    tris,
    seed_tri_ids,
    min_cluster_size=1, # Default should be 2, 1 is noisy.
    bridge_prune=True,
    bridge_prune_all=True,
):
    """Recluster a leaflet's seed defect triangles and optionally prune graph bridges.

    Flow: validate/dedupe seed_tri_ids -> build edge adjacency with metrics
    -> optionally remove bridge edges -> recompute connected components on
    the (possibly bridge-pruned) adjacency -> drop clusters below
    min_cluster_size -> sort largest-first.

    vertices, tris:     full leaflet mesh arrays.
    seed_tri_ids:        candidate defect triangle ids to cluster; ids
                         outside tris' valid range are silently dropped,
                         duplicates are silently collapsed.
    min_cluster_size:    clusters smaller than this (in triangle count) are
                         dropped, same as Step 5's min_triangle_count.
    bridge_prune:         if False, clustering runs on the full adjacency
                         with no bridge edges removed at all.
    bridge_prune_all:     see prune_graph_bridges -- currently the only
                         supported value when bridge_prune is True.
    Returns: dict with raw_defect_tri_ids, neighbors_before_bridge,
             neighbors_after_bridge, edge_metrics, bridges_found,
             bridges_removed, final_clusters, final_tri_ids. Every field is
             empty/zero-length if seed_tri_ids had no valid triangles.
    """
    raw_defect_tri_ids = np.asarray(seed_tri_ids, dtype=np.int32).ravel()
    raw_defect_tri_ids = raw_defect_tri_ids[(raw_defect_tri_ids >= 0) & (raw_defect_tri_ids < len(tris))]
    raw_defect_tri_ids = np.unique(raw_defect_tri_ids)

    if raw_defect_tri_ids.size == 0:
        return {
            "raw_defect_tri_ids": raw_defect_tri_ids,
            "neighbors_before_bridge": {},
            "neighbors_after_bridge": {},
            "edge_metrics": {},
            "bridges_found": np.zeros((0, 2), dtype=np.int32),
            "bridges_removed": np.zeros((0, 2), dtype=np.int32),
            "final_clusters": [],
            "final_tri_ids": np.zeros((0,), dtype=np.int32),
        }

    neighbors_before_bridge, edge_metrics = build_edge_adjacency_pruned(
        vertices=vertices,
        tris=tris,
        tri_ids=raw_defect_tri_ids,
        return_edge_metrics=True,
    )

    if bridge_prune:
        neighbors_after_bridge, bridges_found, bridges_removed = prune_graph_bridges(
            neighbors=neighbors_before_bridge,
            prune_all_bridges=bridge_prune_all,
        )
    else:
        neighbors_after_bridge = {k: set(v) for k, v in neighbors_before_bridge.items()}
        bridges_found = set()
        bridges_removed = []

    final_clusters = connected_components(neighbors_after_bridge, raw_defect_tri_ids)
    final_clusters = [cl for cl in final_clusters if len(cl) >= int(min_cluster_size)]
    final_clusters = sorted(final_clusters, key=len, reverse=True)
    final_tri_ids = flatten_clusters_to_tri_ids(final_clusters)

    return {
        "raw_defect_tri_ids": raw_defect_tri_ids,
        "neighbors_before_bridge": neighbors_before_bridge,
        "neighbors_after_bridge": neighbors_after_bridge,
        "edge_metrics": edge_metrics,
        "bridges_found": np.asarray(sorted(list(bridges_found)), dtype=np.int32) if len(bridges_found) > 0 else np.zeros((0, 2), dtype=np.int32),
        "bridges_removed": np.asarray(bridges_removed, dtype=np.int32) if len(bridges_removed) > 0 else np.zeros((0, 2), dtype=np.int32),
        "final_clusters": final_clusters,
        "final_tri_ids": final_tri_ids,
    }

# --------------------------------------------------
# Save 
# --------------------------------------------------
def color_mesh_by_defect_type(submesh, sub_tris, tri_type_local):
    """Color an Open3D mesh's vertices by defect type: near-black for neutral, magenta/pink for tail.

    Same accumulate-then-average per-vertex scheme as color_mesh_by_clusters
    above, using the same _DEFECT_NEUTRAL_HEX / _DEFECT_TAIL_HEX constants
    as defects.py's save_defects_only_ply, so a defect's color means the
    same thing whether you're looking at Step 4's raw output or Step 6's
    pruned output.

    tri_type_local: (M,) array aligned with sub_tris -- 1 = neutral,
                    anything else = tail. Already indexed to this submesh's
                    local triangle order by the caller (see
                    save_final_cluster_ply).
    """
    def _hex_to_rgb01(hex_code):
        hex_code = hex_code.lstrip("#")
        return np.array([int(hex_code[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])

    rgb_tail = _hex_to_rgb01(_DEFECT_TAIL_HEX)
    rgb_neutral = _hex_to_rgb01(_DEFECT_NEUTRAL_HEX)

    M = len(sub_tris)
    vcols = np.zeros((np.asarray(submesh.vertices).shape[0], 3), dtype=np.float64)
    counts = np.zeros((vcols.shape[0],), dtype=np.int32)

    for t in range(M):
        c = rgb_neutral if tri_type_local[t] == 1 else rgb_tail
        i, j, k = sub_tris[t]
        for v in (i, j, k):
            vcols[v] += c
            counts[v] += 1

    nz = counts > 0
    vcols[nz] /= counts[nz][:, None]
    submesh.vertex_colors = o3d.utility.Vector3dVector(vcols)
    return submesh



def save_final_cluster_ply(out_path, verts, tris, final_tri_ids, final_clusters, defect_type=None):
    """Write Step 6's final pruned clusters for one leaflet to a PLY, colored by type or by cluster.

    """
    if len(final_tri_ids) == 0:
        return
    if o3d is None:
        raise ImportError("open3d is required to write PLY output but is not installed in this environment.")

    sub_verts, sub_tris, _ = remap_submesh(verts, tris, final_tri_ids)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(sub_verts.astype(np.float64)),
        o3d.utility.Vector3iVector(sub_tris.astype(np.int32)),
    )
    mesh.compute_vertex_normals()

    if defect_type is not None:
        tri_type_local = np.asarray(defect_type)[np.asarray(final_tri_ids, dtype=np.int64)]
        color_mesh_by_defect_type(mesh, sub_tris, tri_type_local)
    else:
        global_to_sub = {int(g): i for i, g in enumerate(final_tri_ids.tolist())}
        clusters_sub = [[global_to_sub[int(t)] for t in cl] for cl in final_clusters]
        color_mesh_by_clusters(mesh, sub_tris, clusters_sub)

    o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=False)


def dict_of_sets_to_object_array(neighbors_dict):
    """Convert a {triangle_id: set(neighbor_ids)} adjacency map into two NPZ-friendly arrays.

    Same object-array workaround as _clusters_to_object_array (ragged data,
    NPZ can't store a dict or variable-length rows directly). Used by
    save_pruned_cluster_npz to persist both the pre- and post-bridge-pruning
    adjacency for diagnostics.

    Returns: (keys, vals) -- keys is a sorted int32 array of triangle ids;
             vals is an object array where vals[i] is the sorted int32
             array of neighbors for keys[i].
    """
    keys = np.array(sorted(neighbors_dict.keys()), dtype=np.int32)
    vals = np.array(
        [np.asarray(sorted(list(neighbors_dict[k])), dtype=np.int32) for k in keys],
        dtype=object
    )
    return keys, vals

def edge_metrics_to_arrays(edge_metrics):
    """Flatten build_edge_adjacency_pruned's edge_metrics dict into parallel arrays for NPZ storage.

    edge_metrics: {(triangle_a, triangle_b): {edge_len, neck_width,
                  area_ratio, centroid_dist}} -- see
                  build_edge_adjacency_pruned's docstring for what each
                  metric means and the note that none of them currently
                  drive any pruning decision; they're saved for diagnostics
                  only.
    Returns: (pairs, edge_len, neck_width, area_ratio, centroid_dist), all
             the same length and in the same (sorted-by-pair) order. All
             empty arrays of the right shape/dtype if edge_metrics is empty.
    """
    if len(edge_metrics) == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    pairs = []
    edge_len = []
    neck_width = []
    area_ratio = []
    centroid_dist = []

    for (a, b), m in sorted(edge_metrics.items()):
        pairs.append((int(a), int(b)))
        edge_len.append(float(m["edge_len"]))
        neck_width.append(float(m["neck_width"]))
        area_ratio.append(float(m["area_ratio"]))
        centroid_dist.append(float(m["centroid_dist"]))

    return (
        np.asarray(pairs, dtype=np.int32),
        np.asarray(edge_len, dtype=np.float32),
        np.asarray(neck_width, dtype=np.float32),
        np.asarray(area_ratio, dtype=np.float32),
        np.asarray(centroid_dist, dtype=np.float32),
    )



def save_pruned_cluster_npz(out_path, verts, tris, defect_mask, results, args, defect_type=None):
    """Save one leaflet's Step 6 output: the full pruning result plus a ready-to-use submesh.

    out_path:     destination NPZ path
                  (frame_####_pruned_clusters_{upper,lower}.npz).
    verts, tris:   full leaflet mesh arrays (not pre-subset).
    defect_mask:   the mask actually used as input to clustering -- may
                  differ from Step 4's original per-triangle mask if
                  seed_tri_ids came from Step 5's already-deduplicated
                  output (see process_and_save_one, which rebuilds this
                  mask from seed_tri_ids before calling here).
    results:       dict from compute_clusters_with_pruning.
    args:          SimpleNamespace from run_pruning, for
                  min_cluster_size/bridge_prune/bridge_prune_all provenance.
    defect_type:   per-triangle neutral/tail array, or None if unavailable
                  (see save_final_cluster_ply's docstring for when that
                  happens); saved as all-zeros in that case, not omitted,
                  so the NPZ schema is consistent either way.
    """
    final_submesh_verts, final_submesh_tris, final_submesh_vertex_ids = remap_submesh(
        verts, tris, results["final_tri_ids"]
    )

    final_clusters_arr = np.array(
        [np.asarray(cl, dtype=np.int32) for cl in results["final_clusters"]],
        dtype=object
    )
    final_cluster_areas = np.asarray(
        [cluster_area(verts, tris, cl) for cl in results["final_clusters"]],
        dtype=np.float32,
    )

    nb_before_keys, nb_before_vals = dict_of_sets_to_object_array(results["neighbors_before_bridge"])
    nb_after_keys, nb_after_vals = dict_of_sets_to_object_array(results["neighbors_after_bridge"])

    edge_pairs, edge_len, neck_width, area_ratio, centroid_dist = edge_metrics_to_arrays(
        results["edge_metrics"]
    )

    np.savez_compressed(
        out_path,
        verts=np.asarray(verts, dtype=np.float32),
        tris=np.asarray(tris, dtype=np.int32),
        defect_mask=np.asarray(defect_mask, dtype=np.uint8),
        defect_type=(np.asarray(defect_type, dtype=np.int8) if defect_type is not None
                     else np.zeros(len(tris), dtype=np.int8)),
        raw_defect_tri_ids=np.asarray(results["raw_defect_tri_ids"], dtype=np.int32),
        neighbors_before_bridge_keys=nb_before_keys,
        neighbors_before_bridge_vals=nb_before_vals,
        neighbors_after_bridge_keys=nb_after_keys,
        neighbors_after_bridge_vals=nb_after_vals,
        edge_metric_pairs=edge_pairs,
        edge_metric_edge_len=edge_len,
        edge_metric_neck_width=neck_width,
        edge_metric_area_ratio=area_ratio,
        edge_metric_centroid_dist=centroid_dist,
        bridges_found=np.asarray(results["bridges_found"], dtype=np.int32),
        bridges_removed=np.asarray(results["bridges_removed"], dtype=np.int32),
        final_defect_tri_ids=np.asarray(results["final_tri_ids"], dtype=np.int32),
        final_clusters=final_clusters_arr,
        final_cluster_areas=final_cluster_areas,
        final_submesh_verts=np.asarray(final_submesh_verts, dtype=np.float32),
        final_submesh_tris=np.asarray(final_submesh_tris, dtype=np.int32),
        final_submesh_vertex_ids=np.asarray(final_submesh_vertex_ids, dtype=np.int32),
        min_cluster_size=np.int32(args.min_cluster_size),
        bridge_prune=np.uint8(1 if args.bridge_prune else 0),
        bridge_prune_all=np.uint8(1 if args.bridge_prune_all else 0),
    )



# --------------------------------------------------
# CSV export
# --------------------------------------------------

def pruned_npz_path_for_frame(output_npz_dir, frame_idx, side_name):
    """Build the expected Step 6 pruned-NPZ path for one frame and leaflet.

    """
    return os.path.join(
        output_npz_dir,
        f"frame_{int(frame_idx):04d}_pruned_clusters_{side_name}.npz",
    )

def expected_pruned_npz_files(input_npz_files, output_npz_dir):
    """Return the exact Step-6 pruned NPZ files expected for this run.

    Built from the input file list, not by scanning output_npz_dir -- this
    is what lets write_clusters_csv_from_pruned_npz below raise a clear
    "expected pruned NPZ was not written" error for a specific missing
    file, rather than silently writing a CSV with a frame missing.

    Always two entries per input frame (upper and lower), regardless of
    whether either leaflet actually ended up with any surviving clusters.
    """
    out = []
    for npz_path in input_npz_files:
        frame_idx = frame_index_from_path(npz_path)
        out.append(pruned_npz_path_for_frame(output_npz_dir, frame_idx, "upper"))
        out.append(pruned_npz_path_for_frame(output_npz_dir, frame_idx, "lower"))
    return out


def frame_and_side_from_pruned_npz_path(npz_path):
    """Parse (frame_idx, side_name) back out of a Step 6 pruned-NPZ filename.

    Inverse of pruned_npz_path_for_frame. Raises ValueError, not a silent
    guess, if the filename doesn't match the expected pattern.
    """
    base = os.path.basename(npz_path)
    m = re.search(r"frame_(\d+)_pruned_clusters_(upper|lower)\.npz$", base)
    if not m:
        raise ValueError(f"Cannot parse pruned frame/leaflet from: {npz_path}")
    return int(m.group(1)), m.group(2)



def write_clusters_csv_from_pruned_npz(pruned_npz_files, csv_out):
    """Write clusters_pruned.csv directly from saved Step 6 NPZ outputs.

    Raises FileNotFoundError immediately if any file in pruned_npz_files
    (from expected_pruned_npz_files) is missing, rather than silently
    writing a CSV with a frame gap.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_out)) or ".", exist_ok=True)

    def sort_key(path):
        frame_idx, side_name = frame_and_side_from_pruned_npz_path(path)
        return (frame_idx, 0 if side_name == "upper" else 1)

    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cluster", "area_angstrom2"])

        for npz_path in sorted(pruned_npz_files, key=sort_key):
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"Expected pruned NPZ was not written: {npz_path}")

            frame_idx, side_name = frame_and_side_from_pruned_npz_path(npz_path)
            with np.load(npz_path, allow_pickle=True) as data:
                if "final_cluster_areas" in data:
                    areas = np.asarray(data["final_cluster_areas"], dtype=np.float64).ravel()
                else:
                    verts = np.asarray(data["verts"])
                    tris = np.asarray(data["tris"], dtype=np.int32)
                    clusters = _normalize_clusters_obj_array(data["final_clusters"])
                    areas = np.asarray(
                        [cluster_area(verts, tris, cl) for cl in clusters],
                        dtype=np.float64,
                    )

            for cid, area in enumerate(areas):
                w.writerow([frame_idx, f"{side_name}:{cid}", f"{float(area):.6f}"])


def _process_and_save_one_capture(npz_path, args):
    """Run one Step-6 pruning job and return captured console output.

    Keeping worker output captured prevents multiple processes from mixing their
    print statements together in the terminal.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        process_and_save_one(npz_path, args)
    return frame_index_from_path(npz_path), buf.getvalue()


# --------------------------------------------------
# Visualization
# --------------------------------------------------
def _selected_visualize_leaflets(leaflet):
    """Normalize a visualize_leaflet setting ("upper"/"lower"/"both") into a list to iterate.

    Used by both run_periodic_deduplication and run_pruning so
    visualize_leaflet="both" (DEDUP_VISUALIZE_LEAFLET /
    PRUNE_VISUALIZE_LEAFLET in main.py) opens one viewer per leaflet
    instead of needing special-case handling at each call site.
    """
    text = str(leaflet or "upper").strip().lower()
    if text == "both":
        return ["upper", "lower"]
    if text in {"upper", "lower"}:
        return [text]
    raise ValueError(
        f"visualize_leaflet must be 'upper', 'lower', or 'both'; got {leaflet!r}"
    )

def _should_visualize_frame(frame_idx, requested_frame, visualized_frames, max_frames):
    """Decide whether to open an Open3D viewer for this frame.

    Two modes, both driven by main.py constants (DEDUP_VISUALIZE_FRAME/
    DEDUP_VISUALIZE_MAX_FRAMES or the PRUNE_ equivalents):

      requested_frame is not None -> show only that exact frame index,
      every other frame is skipped regardless of max_frames.

      requested_frame is None -> show frames in order until
      visualized_frames reaches max_frames (max_frames=None means show
      every frame; max_frames<=0 means show none).
    """
    if requested_frame is not None:
        return int(frame_idx) == int(requested_frame)

    if max_frames is None:
        return True

    max_frames = int(max_frames)
    if max_frames <= 0:
        return False

    return int(visualized_frames) < max_frames

def visualize_dedup_frame_clusters(npz_path, leaflet, show_labels, args):
    """Open an Open3D viewer for one leaflet's Step 5 deduplicated clusters.

    npz_path: path to a raw Step 4 defect NPZ.
    leaflet:  "upper" or "lower" only.
    args:     SimpleNamespace from run_periodic_deduplication.
    """
    if o3d is None:
        raise ImportError("open3d is required for deduplication visualization but is not installed.")

    frame_idx = frame_index_from_path(npz_path)
    lx, ly, _lz = args.frame_box_xyz[frame_idx]

    d = load_defect_npz(npz_path)

    if leaflet == "upper":
        verts, tris, mask = d["verts_up"], d["tris_up"], d["mask_up"]
    elif leaflet == "lower":
        verts, tris, mask = d["verts_lo"], d["tris_lo"], d["mask_lo"]
    else:
        raise ValueError("Deduplication visualization supports leaflet='upper' or 'lower' only.")

    out = process_one_leaflet(
        verts=verts,
        tris=tris,
        mask=mask,
        full_center_x=args.full_center_x,
        full_center_y=args.full_center_y,
        periodic_lx=lx,
        periodic_ly=ly,
        match_tol=args.match_tol,
        min_cluster_size=args.min_cluster_size,
    )

    center_x = out["center_x"]
    center_y = out["center_y"]
    final_clusters = out["final_clusters"]
    final_tri_ids = out["final_tri_ids"]
    raw_clusters = out["raw_clusters"]

    if final_tri_ids.size == 0:
        print(f"No defect triangles in {leaflet} leaflet after deduplication.")
        return

    print(f"NPZ: {npz_path}")
    print(f"Visualization: deduplication | frame={frame_idx} | leaflet={leaflet}")
    print(f"Box (frame {frame_idx}): Lx={lx:.3f}, Ly={ly:.3f} Å")
    print(f"Estimated/used center: ({center_x:.3f}, {center_y:.3f})")
    print(f"Raw clusters: {len(raw_clusters)}")
    print(f"After periodic dedup: {len(final_clusters)}")
    print(f"Final defect triangles: {final_tri_ids.size}")

    sub_verts, sub_tris, _ = remap_submesh(verts, tris, final_tri_ids)
    global_to_sub = {int(g): i for i, g in enumerate(final_tri_ids.tolist())}
    clusters_sub = [[global_to_sub[int(t)] for t in cl] for cl in final_clusters]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(sub_verts.astype(np.float64)),
        o3d.utility.Vector3iVector(sub_tris.astype(np.int32)),
    )
    mesh.compute_vertex_normals()
    color_mesh_by_clusters(mesh, sub_tris, clusters_sub)

    geoms = [mesh]

    if show_labels:
        for cid, cl in enumerate(final_clusters[:200]):
            ctr = cluster_centroid(verts, tris, cl)
            area = cluster_area(verts, tris, cl)
            xw, yw = wrap_to_periodic_cell(
                ctr[0], ctr[1],
                center_x, center_y,
                lx, ly,
            )
            print(
                f"cluster {cid:4d} | size {len(cl):6d} | area {area:.3f} A^2 "
                f"| centroid {ctr} | wrapped_xy [{xw:.3f}, {yw:.3f}]"
            )
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.6)
            sph.translate(ctr)
            sph.compute_vertex_normals()
            sph.paint_uniform_color([1.0, 1.0, 1.0])
            geoms.append(sph)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Deduplicated clusters | frame {frame_idx} | {leaflet}",
    )


def visualize_frame_clusters(npz_path, leaflet, show_labels, args):
    """Open an Open3D viewer for one leaflet's Step 6 pruned clusters.

    This is the pruning-side visualization function. Useful for dubugging but turned
    off in the current version

    npz_path: path to a Step 6 input NPZ (Step 5 dedup output, or a raw
              Step 4 NPZ -- see load_input_npz).
    leaflet:  "upper" or "lower" only -- "both" is handled by the caller
              iterating this function once per side, not by this function.
    args:     SimpleNamespace from run_pruning, carrying min_cluster_size,
              bridge_prune, bridge_prune_all.
    """
    if o3d is None:
        raise ImportError("open3d is required for --visualize but is not installed in this environment.")

    d = load_input_npz(npz_path)

    if leaflet == "upper":
        verts, tris, seed_tri_ids = d["verts_up"], d["tris_up"], d["seed_tri_ids_up"]
    elif leaflet == "lower":
        verts, tris, seed_tri_ids = d["verts_lo"], d["tris_lo"], d["seed_tri_ids_lo"]
    else:
        raise ValueError("Visualization supports leaflet=upper or lower only.")

    results = compute_clusters_with_pruning(
        verts,
        tris,
        seed_tri_ids,
        min_cluster_size=args.min_cluster_size,
        bridge_prune=args.bridge_prune,
        bridge_prune_all=args.bridge_prune_all,
    )

    final_tri_ids = results["final_tri_ids"]
    final_clusters = results["final_clusters"]

    if final_tri_ids.size == 0:
        print("No defect triangles after pruning.")
        return

    print(f"NPZ: {npz_path}")
    print(f"Input defect triangles: {len(results['raw_defect_tri_ids'])}")
    print(f"Bridges found: {len(results['bridges_found'])}")
    print(f"Bridges removed: {len(results['bridges_removed'])}")
    print(f"Final clusters: {len(final_clusters)}")
    print(f"Final defect triangles: {len(final_tri_ids)}")

    sub_verts, sub_tris, _ = remap_submesh(verts, tris, final_tri_ids)
    global_to_sub = {int(g): i for i, g in enumerate(final_tri_ids.tolist())}
    clusters_sub = [[global_to_sub[int(t)] for t in cl] for cl in final_clusters]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(sub_verts),
        o3d.utility.Vector3iVector(sub_tris),
    )
    mesh.compute_vertex_normals()
    color_mesh_by_clusters(mesh, sub_tris, clusters_sub)

    geoms = [mesh]

    if show_labels:
        for cid, cl in enumerate(final_clusters[:200]):
            ctr = cluster_centroid(verts, tris, cl)
            area = cluster_area(verts, tris, cl)
            print(f"cluster {cid:4d} | size {len(cl):6d} | area {area:.3f} A^2 | centroid {ctr}")
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.6)
            sph.translate(ctr)
            sph.compute_vertex_normals()
            sph.paint_uniform_color([1.0, 1.0, 1.0])
            geoms.append(sph)

    o3d.visualization.draw_geometries(geoms)


# --------------------------------------------------
# CLI
# --------------------------------------------------

def collect_npz_files(npz_dir):
    """Collect Step 6 input files: prefers Step 5 dedup output, falls back to raw Step 4 NPZs.

    This is the function actually used by run_pruning.

    When both a dedup file and a raw file exist for the same frame, the
    dedup file wins -- by_frame is only ever set from raw_patterns if that
    frame index isn't already present.

    Returns: list of file paths, one per frame, sorted by frame index.
    """
    dedup_patterns = [
        os.path.join(npz_dir, "cluster", "npz", "frame_*_clusters_both.npz"),
        os.path.join(npz_dir, "ClusterNPZ", "frame_*_clusters_both.npz"),
        os.path.join(npz_dir, "frame_*_clusters_both.npz"),
    ]
    raw_patterns = [
        os.path.join(npz_dir, "frame_*_defects.npz"),
        os.path.join(npz_dir, "frame_*_radius_defects.npz"),
    ]

    by_frame = {}

    for pattern in dedup_patterns:
        for pth in glob.glob(pattern):
            by_frame[frame_index_from_path(pth)] = pth

    for pattern in raw_patterns:
        for pth in glob.glob(pattern):
            fr = frame_index_from_path(pth)
            if fr not in by_frame:
                by_frame[fr] = pth

    return [by_frame[k] for k in sorted(by_frame.keys())]

def _leaflet_payloads(d):
    """Unpack a load_input_npz result into a per-leaflet iteration list.


    d: dict from load_input_npz.
    Returns: [("upper", verts_up, tris_up, seed_tri_ids_up, mask_up, defect_type_up),
              ("lower", verts_lo, tris_lo, seed_tri_ids_lo, mask_lo, defect_type_lo)]
    """
    return [
        ("upper", d["verts_up"], d["tris_up"], d["seed_tri_ids_up"], d.get("mask_up"), d.get("defect_type_up")),
        ("lower", d["verts_lo"], d["tris_lo"], d["seed_tri_ids_lo"], d.get("mask_lo"), d.get("defect_type_lo")),
    ]


def process_and_save_one(npz_path, args):
    """Run Step 6 for one frame: both leaflets, pruning, then save NPZ + PLY for each.

    Called once per frame by run_pruning, either directly (serial mode) or
    via _process_and_save_one_capture (parallel mode, one call per worker
    process).

    mask_to_save is rebuilt from seed_tri_ids here rather than reusing
    d["mask_up"]/d["mask_lo"] directly -- when the input came from Step 5's
    dedup output, seed_tri_ids is already the deduplicated set, and this
    mask should reflect that (post-dedup) triangle set, not Step 4's
    original pre-dedup mask.

    PLY output is skipped (not an error) when either there are zero final
    triangles for this leaflet, or open3d isn't installed -- both are
    logged explicitly so a "missing" PLY isn't mistaken for a crash.

    npz_path: path to one Step 6 input NPZ (from collect_npz_files).
    args:     SimpleNamespace built by run_pruning, carrying
              output_npz_dir/output_upper_dir/output_lower_dir,
              min_cluster_size, bridge_prune, bridge_prune_all.
    """
    d = load_input_npz(npz_path)
    frame_idx = frame_index_from_path(npz_path)

    npz_out_dir = args.output_npz_dir
    ply_upper_dir = args.output_upper_dir
    ply_lower_dir = args.output_lower_dir
    os.makedirs(npz_out_dir, exist_ok=True)
    os.makedirs(ply_upper_dir, exist_ok=True)
    os.makedirs(ply_lower_dir, exist_ok=True)

    for side_name, verts, tris, seed_tri_ids, mask, defect_type in _leaflet_payloads(d):
        results = compute_clusters_with_pruning(
            verts, tris, seed_tri_ids,
            min_cluster_size=args.min_cluster_size,
            bridge_prune=args.bridge_prune,
            bridge_prune_all=args.bridge_prune_all,
        )

        npz_out_path = os.path.join(
            npz_out_dir,
            f"frame_{frame_idx:04d}_pruned_clusters_{side_name}.npz"
        )
        ply_out_dir = ply_upper_dir if side_name == "upper" else ply_lower_dir
        ply_out_path = os.path.join(
            ply_out_dir,
            f"frame_{frame_idx:04d}_pruned_clusters_{side_name}.ply"
        )

        mask_to_save = np.zeros(len(tris), dtype=np.uint8)
        seed_tri_ids = np.asarray(seed_tri_ids, dtype=np.int32)
        seed_tri_ids = seed_tri_ids[(seed_tri_ids >= 0) & (seed_tri_ids < len(tris))]
        mask_to_save[seed_tri_ids] = 1

        save_pruned_cluster_npz(
            out_path=npz_out_path,
            verts=verts,
            tris=tris,
            defect_mask=mask_to_save,
            results=results,
            args=args,
            defect_type=defect_type,
        )

        ply_saved = False
        if len(results["final_tri_ids"]) > 0 and o3d is not None:
            save_final_cluster_ply(
                out_path=ply_out_path,
                verts=verts,
                tris=tris,
                final_tri_ids=results["final_tri_ids"],
                final_clusters=results["final_clusters"],
                defect_type=defect_type,
            )
            ply_saved = True

        print(f"Input NPZ: {npz_path}")
        print(f"Source mode: {d['source_mode']}")
        print(f"Leaflet: {side_name}")
        print(f"Saved NPZ: {npz_out_path}")
        if ply_saved:
            print(f"Saved PLY: {ply_out_path}")
        elif len(results['final_tri_ids']) == 0:
            print("Saved PLY: skipped (no final triangles)")
        else:
            print("Saved PLY: skipped (open3d not available in this environment)")
        print(f"Input seed triangles after dedup: {len(results['raw_defect_tri_ids'])}")
        print(f"Bridges found: {len(results['bridges_found'])}")
        print(f"Bridges removed: {len(results['bridges_removed'])}")
        print(f"Final clusters: {len(results['final_clusters'])}")
        print(f"Final defect triangles: {len(results['final_tri_ids'])}")
        print("-" * 60)

# ==================================================
# Integrated workflow helpers 
# ==================================================

def run_periodic_deduplication(
    npz_dir,
    frame_box_xyz,
    match_tol=5.0,
    min_cluster_size=1,
    full_center_x=None,
    full_center_y=None,
    output_npz_dir=None,
    visualize=False,
    visualize_frame=None,
    visualize_leaflet="upper",
    visualize_labels=False,
    visualize_max_frames=1,
    workers=1,
):
    """Step 5 entry point: raw connected-component clustering + periodic duplicate removal.

    Called once by helpers.py's run_step5_deduplication. Runs
    process_and_save_one_file once per frame (serial if workers<=1,
    otherwise one call per worker process via ProcessPoolExecutor, with
    output captured and printed in the parent process to keep the terminal
    readable). Fails fast, before spawning any worker, if frame_box_xyz
    (box_size.csv) doesn't cover every frame this run needs.

    Visualization (if enabled) always runs in the parent process after all
    workers finish -- Open3D windows opened from a worker process don't
    work reliably, so this is not optional.

    Output:
        <output_npz_dir>/frame_####_clusters_both.npz

    Returns: output_npz_dir (resolved default: npz_dir/cluster/npz).
    """
    from types import SimpleNamespace

    if output_npz_dir is None:
        output_npz_dir = os.path.join(npz_dir, "cluster", "npz")

    args = SimpleNamespace(
        npz_dir=npz_dir,
        output_npz_dir=output_npz_dir,
        min_cluster_size=int(min_cluster_size),
        visualize=bool(visualize),
        frame=visualize_frame,
        leaflet=visualize_leaflet,
        no_labels=not bool(visualize_labels),
        visualize_max_frames=visualize_max_frames,
        full_center_x=full_center_x,
        full_center_y=full_center_y,
        frame_box_xyz=frame_box_xyz,
        match_tol=float(match_tol),
        workers=max(1, int(workers)),
    )

    npz_files = collect_dedup_npz_files(npz_dir)
    if not npz_files:
        raise FileNotFoundError(
            f"No defect NPZ files found in {npz_dir}. "
            "Expected frame_*_defects.npz or frame_*_radius_defects.npz."
        )

    # Fail fast, before any worker is spawned, if box_size.csv doesn't cover
    # every frame this run needs (e.g. a stale cache left over from a
    # different input PDB / frame count).
    for npz_path in npz_files:
        frame_idx = frame_index_from_path(npz_path)
        if frame_idx not in frame_box_xyz:
            raise KeyError(
                f"Frame {frame_idx} ({os.path.basename(npz_path)}) has no matching row in "
                "box_size.csv. Delete results/box_size.csv and rerun to rebuild it."
            )

    progress.set_total(len(npz_files))

    n_workers = min(args.workers, len(npz_files))
    if n_workers <= 1:
        print(f"Step 5 deduplication running in serial mode over {len(npz_files)} frame(s).")
        for npz_path in npz_files:
            process_and_save_one_file(npz_path, args)
            progress.tick()
    else:
        print(
            f"Step 5 deduplication running with {n_workers} worker processes "
            f"over {len(npz_files)} frame(s)."
        )
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_process_and_save_one_file_capture, npz_path, args)
                for npz_path in npz_files
            ]
            for fut in as_completed(futures):
                frame_idx, text = fut.result()
                print(f"[worker done] Step 5 frame {frame_idx}")
                if text:
                    print(text, end="" if text.endswith("\n") else "\n")
                progress.tick()

    # Visualization stays in the parent process. This avoids Open3D windows
    # being opened from multiprocessing workers.
    visualized_frames = 0
    if args.visualize:
        for npz_path in npz_files:
            frame_idx = frame_index_from_path(npz_path)
            if _should_visualize_frame(
                frame_idx=frame_idx,
                requested_frame=args.frame,
                visualized_frames=visualized_frames,
                max_frames=args.visualize_max_frames,
            ):
                for side in _selected_visualize_leaflets(args.leaflet):
                    visualize_dedup_frame_clusters(
                        npz_path=npz_path,
                        leaflet=side,
                        show_labels=not args.no_labels,
                        args=args,
                    )
                visualized_frames += 1

    return output_npz_dir



def run_pruning(
    npz_dir,
    min_cluster_size=1,
    bridge_prune=True,
    bridge_prune_all=True,
    csv_out=None,
    output_npz_dir=None,
    output_upper_dir=None,
    output_lower_dir=None,
    visualize=False,
    visualize_frame=None,
    visualize_leaflet="upper",
    visualize_labels=False,
    visualize_max_frames=1,
    workers=1,
):
    """Step 6 entry point: bridge pruning on Step 5's deduplicated cluster NPZ files.

    Called once by helpers.py's run_step6_pruning. Same worker/visualization
    structure as run_periodic_deduplication -- see that function's docstring
    for the general pattern (serial vs. ProcessPoolExecutor, captured worker
    output, parent-process-only visualization).


    Always writes clusters_pruned.csv from the saved pruned NPZ files after
    all frames finish processing (write_clusters_csv_from_pruned_npz), not
    conditionally -- Step 7's fitting/coverage analysis reads this CSV
    directly.

    Returns: dict with csv, npz_dir, upper_dir, lower_dir -- the resolved
             output locations for this run.
    """
    from types import SimpleNamespace

    if output_npz_dir is None:
        output_npz_dir = os.path.join(npz_dir, "pruned", "npz")
    if output_upper_dir is None:
        output_upper_dir = os.path.join(npz_dir, "pruned", "upper")
    if output_lower_dir is None:
        output_lower_dir = os.path.join(npz_dir, "pruned", "lower")

    args = SimpleNamespace(
        npz_dir=npz_dir,
        output_npz_dir=output_npz_dir,
        output_upper_dir=output_upper_dir,
        output_lower_dir=output_lower_dir,
        min_cluster_size=int(min_cluster_size),
        csv_out=csv_out,
        visualize=bool(visualize),
        frame=visualize_frame,
        leaflet=visualize_leaflet,
        no_labels=not bool(visualize_labels),
        visualize_max_frames=visualize_max_frames,
        bridge_prune=bool(bridge_prune),
        bridge_prune_all=bool(bridge_prune_all),
        workers=max(1, int(workers)),
    )

    npz_files = collect_npz_files(npz_dir)
    if not npz_files:
        raise FileNotFoundError(
            f"No cluster NPZ files found in {npz_dir}. "
            "Expected frame_*_clusters_both.npz files or raw defect NPZs."
        )

    if args.csv_out is None:
        args.csv_out = os.path.join(npz_dir, "clusters_pruned.csv")
    csv_parent = os.path.dirname(args.csv_out)
    if csv_parent:
        os.makedirs(csv_parent, exist_ok=True)

    progress.set_total(len(npz_files))

    n_workers = min(args.workers, len(npz_files))
    if n_workers <= 1:
        print(f"Step 6 pruning running in serial mode over {len(npz_files)} frame(s).")
        for npz_path in npz_files:
            process_and_save_one(npz_path, args)
            progress.tick()
    else:
        print(
            f"Step 6 pruning running with {n_workers} worker processes "
            f"over {len(npz_files)} frame(s)."
        )
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_process_and_save_one_capture, npz_path, args)
                for npz_path in npz_files
            ]
            for fut in as_completed(futures):
                frame_idx, text = fut.result()
                print(f"[worker done] Step 6 frame {frame_idx}")
                if text:
                    print(text, end="" if text.endswith("\n") else "\n")
                progress.tick()

    pruned_npz_files = expected_pruned_npz_files(npz_files, args.output_npz_dir)
    write_clusters_csv_from_pruned_npz(pruned_npz_files, args.csv_out)
    print(f"Wrote CSV from pruned NPZ files: {args.csv_out}")

    # Visualization stays in the parent process. This avoids Open3D windows
    # being opened from multiprocessing workers.  It uses the input cluster NPZ
    # files and recomputes the selected display frame only.
    visualized_frames = 0
    if args.visualize:
        for npz_path in npz_files:
            frame_idx = frame_index_from_path(npz_path)
            if _should_visualize_frame(
                frame_idx=frame_idx,
                requested_frame=args.frame,
                visualized_frames=visualized_frames,
                max_frames=args.visualize_max_frames,
            ):
                for side in _selected_visualize_leaflets(args.leaflet):
                    visualize_frame_clusters(
                        npz_path=npz_path,
                        leaflet=side,
                        show_labels=not args.no_labels,
                        args=args,
                    )
                visualized_frames += 1

    return {
        "csv": args.csv_out,
        "npz_dir": output_npz_dir,
        "upper_dir": output_upper_dir,
        "lower_dir": output_lower_dir,
    }