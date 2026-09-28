"""Step 8: layer classification (monolayer / bilayer).

A resume-friendly post-processor that reads ONLY existing pipeline outputs
(Step-3 mesh NPZ for opposite-leaflet centroids, Step-6 pruned cluster NPZ
for defect clusters). It does not re-run Steps 1-7.

Each Step-6 defect cluster is tagged by how well its triangles appose the
opposite leaflet: a bilayer region has a partner leaflet at roughly the
bilayer thickness; a monolayer region (pore rim, ruptured patch, edge) does
not. This is a strict two-way split -- there is no "mixed" middle category.
Output is one CSV row per cluster plus split PLYs (one file per group per
leaflet).

triangle_areas below is also imported directly by curvature.py (Step 9) --
not private to this module despite living here.
"""


import csv
import glob
import os
import re
from pathlib import Path
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from . import progress

import numpy as np

try:
    import open3d as o3d
except Exception:  # import-safe on headless nodes without Open3D
    o3d = None

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

from .analysis import join_analysis_sections

# ==================================================
# Parallel worker
# ==================================================


def _process_one_layer_frame_side(
    side,
    pruned_path,
    mesh_npz_dir,
    ply_dir,
    thickness,
    thickness_tol,
    max_pair_dist,
    bilayer_frac,
    write_ply,
    solid,
):
    """Run Step 8 layer classification for one (frame, leaflet) pair.

    One unit of work for run_step8_layers -- called either directly
    (serial mode) or via _process_one_layer_frame_side_capture (parallel
    mode, one call per worker process).

    PLY writing is silently skipped if write_ply is False, or if open3d
    isn't installed -- the latter with no warning printed here (unlike
    some other steps' PLY-skip paths), so a run without open3d installed
    produces no layer PLYs with no diagnostic pointing at why.

    Returns: (frame_idx, side, csv_rows, edge_length_or_None,
              thickness_used, counts) for the parent process to aggregate.
              thickness_used is float("nan") when nothing was classified.
    """
    frame_idx = frame_index_from_path(pruned_path)
    mesh_path = os.path.join(str(mesh_npz_dir), f"frame_{frame_idx:04d}_mesh_predefect.npz")
    counts = {"monolayer": 0, "bilayer": 0}

    if not os.path.exists(mesh_path):
        print(f"⚠️ Step 8: no mesh NPZ for frame {frame_idx}; skipping {side}.")
        return frame_idx, side, [], None, float("nan"), counts

    verts, tris, clusters = _load_pruned_clusters(pruned_path)
    if not clusters:
        return frame_idx, side, [], None, float("nan"), counts

    own_c, own_n, opp_c = _load_mesh_centroids_normals(mesh_path, side)
    edge_length = mean_triangle_edge_length(verts, tris)
    areas_all = triangle_areas(verts, tris)

    types, fracs, thick_used = classify_cluster_layers(
        cluster_tri_ids=clusters,
        own_centroids=own_c,
        own_normals=own_n,
        opposite_centroids=opp_c,
        thickness=thickness,
        thickness_tol=thickness_tol,
        max_pair_dist=max_pair_dist,
        bilayer_frac=bilayer_frac,
    )

    csv_rows = []
    grouped = {"monolayer": [], "bilayer": []}
    for cid, (cl, rtype, frac) in enumerate(zip(clusters, types, fracs)):
        area = float(areas_all[cl].sum())
        csv_rows.append((frame_idx, f"{side}:{cid}", rtype,
                          f"{area:.6f}", f"{frac:.4f}"))
        counts[rtype] += 1
        grouped[rtype].append(cl)

    if write_ply and o3d is not None:
        for rtype, cls in grouped.items():
            if not cls:
                continue
            tri_ids = np.concatenate(cls)
            out_ply = ply_dir / rtype / side / f"frame_{frame_idx:04d}_layers_{rtype}_{side}.ply"
            save_triangle_subset_ply(
                str(out_ply), verts, tris, tri_ids, color=solid[rtype]
            )

    return frame_idx, side, csv_rows, edge_length, thick_used, counts


def _process_one_layer_frame_side_capture(*args, **kwargs):
    """Worker wrapper that captures console output (same pattern as Steps 5/6/9)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = _process_one_layer_frame_side(*args, **kwargs)
    return (*result, buf.getvalue())



# ==================================================
# Frame-index / path helpers (match the frame_#### convention used everywhere)
# ==================================================

_FRAME_RE = re.compile(r"frame_(\d+)")


def frame_index_from_path(path):
    """Extract the frame index from any filename matching this pipeline's frame_#### convention.

    Raises ValueError if no frame_#### substring is found.
    """
    m = _FRAME_RE.search(os.path.basename(str(path)))
    if not m:
        raise ValueError(f"Cannot parse frame index from: {path}")
    return int(m.group(1))


def collect_pruned_npz_files(pruned_npz_dir, side):
    """Return every Step-6 pruned cluster NPZ for one leaflet, sorted by frame index.
    """
    files = glob.glob(
        os.path.join(str(pruned_npz_dir), f"frame_*_pruned_clusters_{side}.npz")
    )
    return sorted(set(files), key=frame_index_from_path)


def _side_from_leaflet_key(side):
    """Map "upper"/"lower" to the (own, opposite) NPZ key suffixes used in the Step-3 mesh NPZ.


    Returns: (own_suffix, opposite_suffix) -- e.g. ("up", "lo") for side="upper".
    Raises ValueError for anything other than "upper" or "lower".
    """
    if side == "upper":
        return "up", "lo"
    if side == "lower":
        return "lo", "up"
    raise ValueError(f"side must be 'upper' or 'lower', got {side!r}")


# ==================================================
# Geometry helpers
# ==================================================

def mesh_from_arrays(verts, tris):
    """Build an Open3D TriangleMesh from raw vertex/triangle arrays, with vertex normals computed.

    """
    if o3d is None:
        raise ImportError("open3d is required to build meshes but is not installed.")
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh


def triangle_areas(verts, tris):
    """Return per-triangle area for every triangle in tris, as an (T,) array.

    """
    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    a = verts[tris[:, 1]] - verts[tris[:, 0]]
    b = verts[tris[:, 2]] - verts[tris[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)


def mean_triangle_edge_length(verts, tris):
    """Mean edge length across every triangle in tris, in Angstroms.

    Logged into Step 8's summary text - Dubug tool.

    Returns NaN for an empty triangle array, not zero or an error.
    """
    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    if tris.size == 0:
        return float("nan")
    e0 = np.linalg.norm(verts[tris[:, 1]] - verts[tris[:, 0]], axis=1)
    e1 = np.linalg.norm(verts[tris[:, 2]] - verts[tris[:, 1]], axis=1)
    e2 = np.linalg.norm(verts[tris[:, 0]] - verts[tris[:, 2]], axis=1)
    return float(np.mean(np.concatenate([e0, e1, e2])))



def save_triangle_subset_ply(out_path, verts, tris, tri_ids, color=(0.85, 0.2, 0.2)):
    """Write a solid-colored PLY containing only the given triangles.

    Used to export one file per (region_type, leaflet) group -- e.g. every
    bilayer-tagged triangle in the upper leaflet for one frame, painted a
    single flat color rather than per-cluster colors (contrast with
    cleanup.py's color_mesh_by_clusters, which colors each cluster
    differently within one PLY).

    Returns True if a file was actually written.
    """
    if o3d is None:
        raise ImportError("open3d is required to write PLY output but is not installed.")
    tri_ids = np.asarray(tri_ids, dtype=np.int64)
    if tri_ids.size == 0:
        return False
    faces = np.asarray(tris, dtype=np.int64)[tri_ids]
    uniq, inv = np.unique(faces.reshape(-1), return_inverse=True)
    sub_verts = np.asarray(verts, dtype=np.float64)[uniq]
    sub_tris = inv.reshape(-1, 3).astype(np.int32)
    mesh = mesh_from_arrays(sub_verts, sub_tris)
    mesh.paint_uniform_color(list(color))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=False)
    return True


# ==================================================
# Opposite-leaflet apposition
# ==================================================

def apposition_distances(own_centroids, own_normals, opposite_centroids, max_pair_dist):
    """For each own-leaflet triangle, find the nearest opposite-leaflet triangle and measure apposition distance.

    IMPORTANT -- the search and the reported distance are not the same
    metric. The nearest-neighbor search itself is plain Euclidean
    (scipy's cKDTree, with max_pair_dist as a Euclidean cutoff radius).
    Once that nearest neighbor is found, the value actually returned is
    the component of the vector to it projected onto own_normals -- the
    across-the-membrane distance, not straight-line distance to that
    point. What defines a bilayer is apposition across the membrane, along
    the local normal, so projecting keeps tilted/buckled regions from
    being misclassified by their raw (larger) straight-line distance.

    own_normals=None skips the projection entirely and returns raw
    Euclidean distance instead -- used when normal information isn't
    available or isn't wanted.

    own_centroids, opposite_centroids: (N, 3) / (M, 3) triangle centroid
        arrays for each leaflet.
    own_normals: (N, 3) triangle normals for the own leaflet, or None.
    max_pair_dist: Euclidean search radius (Å) -- pairs farther than this
        are excluded entirely (distance +inf), not just capped.
    Returns: (N,) array of distances, +inf wherever no opposite triangle
             lies within max_pair_dist.
    """
    if cKDTree is None:
        raise ImportError("scipy is required for opposite-leaflet pairing.")
    own_centroids = np.asarray(own_centroids, dtype=np.float64)
    opposite_centroids = np.asarray(opposite_centroids, dtype=np.float64)
    if own_centroids.size == 0 or opposite_centroids.size == 0:
        return np.full(len(own_centroids), np.inf)

    tree = cKDTree(opposite_centroids)
    dist, idx = tree.query(own_centroids, k=1, distance_upper_bound=float(max_pair_dist))

    out = np.full(len(own_centroids), np.inf, dtype=np.float64)
    valid = np.isfinite(dist)
    if own_normals is None:
        out[valid] = dist[valid]
        return out

    own_normals = np.asarray(own_normals, dtype=np.float64)
    nn = np.linalg.norm(own_normals, axis=1, keepdims=True)
    nn[nn == 0.0] = 1.0
    unit = own_normals / nn
    vec = opposite_centroids[idx[valid]] - own_centroids[valid]
    out[valid] = np.abs(np.einsum("ij,ij->i", vec, unit[valid]))
    return out


def classify_cluster_layers(
    cluster_tri_ids,
    own_centroids,
    own_normals,
    opposite_centroids,
    thickness,
    thickness_tol,
    max_pair_dist,
    bilayer_frac,
):
    """Tag each cluster "bilayer" or "monolayer" from its triangles' apposition to the opposite leaflet.

    Strictly two-way -- every cluster gets exactly one of the two labels,
    there is no "mixed" middle category, even for a cluster whose paired
    fraction sits right at the threshold.

    A cluster's paired_fraction is the share of its triangles whose
    apposition_distances value falls within [thickness - thickness_tol,
    thickness + thickness_tol]. A cluster is "bilayer" when paired_fraction
    >= bilayer_frac, else "monolayer".

    thickness=None means auto: the median apposition distance across every
    clustered triangle in this frame/leaflet is used as the working
    thickness for that frame -- computed once per frame from all clusters
    pooled together (not per-cluster), so it's robust to a single small or
    unusual cluster skewing its own estimate. thickness_used is returned so
    callers/logs can report exactly what value was actually applied.

    A cluster with zero triangles, or when thickness_used itself isn't
    finite (e.g. every apposition distance in the frame was +inf), is
    unconditionally tagged "monolayer" with paired_fraction 0.0 -- there's
    no meaningful bilayer signal to test against in either case.

    cluster_tri_ids: list of triangle-id arrays, one per cluster.
    own_centroids, own_normals, opposite_centroids: see apposition_distances.
    thickness: fixed thickness in Å, or None for per-frame auto.
    thickness_tol: tolerance band around thickness, in Å.
    max_pair_dist: passed through to apposition_distances.
    bilayer_frac: paired_fraction threshold in (0, 1].
    Returns: (region_types, paired_fractions, thickness_used) -- the first
             two are lists aligned 1:1 with cluster_tri_ids; thickness_used
             is a single float for the whole frame/leaflet. All three are
             ([], [], NaN) if cluster_tri_ids is empty.
    """
    all_ids = (
        np.concatenate([np.asarray(c, dtype=np.int64).ravel() for c in cluster_tri_ids])
        if cluster_tri_ids else np.zeros(0, dtype=np.int64)
    )
    if all_ids.size == 0:
        return [], [], float("nan")

    d_all = apposition_distances(
        own_centroids[all_ids],
        None if own_normals is None else own_normals[all_ids],
        opposite_centroids,
        max_pair_dist=max_pair_dist,
    )

    if thickness is None:
        finite = d_all[np.isfinite(d_all)]
        thickness_used = float(np.median(finite)) if finite.size else float("nan")
    else:
        thickness_used = float(thickness)

    lo = thickness_used - thickness_tol
    hi = thickness_used + thickness_tol

    types = []
    fracs = []
    pos = 0
    for c in cluster_tri_ids:
        n = int(np.asarray(c).size)
        d = d_all[pos:pos + n]
        pos += n
        if n == 0 or not np.isfinite(thickness_used):
            types.append("monolayer")
            fracs.append(0.0)
            continue
        paired = np.isfinite(d) & (d >= lo) & (d <= hi)
        frac = float(np.count_nonzero(paired)) / float(n)
        fracs.append(frac)
        types.append("bilayer" if frac >= bilayer_frac else "monolayer")
    return types, fracs, thickness_used


# ==================================================
# Loaders
# ==================================================


def _load_pruned_clusters(npz_path):
    """Load a Step-6 pruned cluster NPZ's mesh and final clusters.


    Returns: (verts, tris, final_clusters) -- final_clusters is a list of
             int64 triangle-id arrays, one per cluster.
    """
    with np.load(npz_path, allow_pickle=True) as d:
        verts = np.asarray(d["verts"], dtype=np.float64)
        tris = np.asarray(d["tris"], dtype=np.int64)
        clusters = [np.asarray(cl, dtype=np.int64).ravel() for cl in d["final_clusters"]]
    return verts, tris, clusters


def _load_mesh_centroids_normals(mesh_npz_path, side):
    """Load one leaflet's own triangle centroids/normals plus the opposite leaflet's centroids.

    Reads from the Step-3 mesh NPZ (frame_####_mesh_predefect.npz), using
    _side_from_leaflet_key to translate "upper"/"lower" into that NPZ's
    centroids_up/centroids_lo/normals_up/normals_lo key naming.

    side: "upper" or "lower" -- which leaflet is "own" for this call.
    Returns: (own_centroids, own_normals, opposite_centroids).
    """
    own, opp = _side_from_leaflet_key(side)
    with np.load(mesh_npz_path, allow_pickle=True) as d:
        own_c = np.asarray(d[f"centroids_{own}"], dtype=np.float64)
        own_n = np.asarray(d[f"normals_{own}"], dtype=np.float64)
        opp_c = np.asarray(d[f"centroids_{opp}"], dtype=np.float64)
    return own_c, own_n, opp_c


# ==================================================
# Main engine
# ==================================================
def _build_layers_section(prefix, counts, edge_lengths, thickness_log,
                           thickness, thickness_tol, max_pair_dist, bilayer_frac):
    """Build the 'layer classification' summary section text for _layers.txt."""
    total = sum(counts.values())
    mean_edge = float(np.mean(edge_lengths)) if edge_lengths else float("nan")
    thick_desc = "auto (per-frame median)" if thickness is None else f"{float(thickness):.3f} Å"
    mean_thick = float(np.mean(thickness_log)) if thickness_log else float("nan")

    lines = [
        "=== Layer classification (mono / bilayer) ===",
        f"output prefix = {prefix}",
        f"thickness setting = {thick_desc}",
        f"mean apposition thickness used = {mean_thick:.3f} Å",
        f"thickness tolerance = {float(thickness_tol):.3f} Å",
        f"max pair distance = {float(max_pair_dist):.3f} Å",
        f"bilayer fraction threshold = {float(bilayer_frac):.3f}",
        f"mean triangle edge length = {mean_edge:.3f} Å",
        f"total clusters = {total}",
        f"  bilayer   = {counts['bilayer']}",
        f"  monolayer = {counts['monolayer']}",
    ]
    return "\n".join(lines)


def run_step8_layers(
    mesh_npz_dir,
    pruned_npz_dir,
    output_dir,
    output_prefix,
    thickness,
    thickness_tol,
    max_pair_dist,
    bilayer_frac,
    write_ply=True,
    workers=1,
):
    """Per-cluster monolayer/bilayer tagging + split PLYs. Step 8's single entry point.

    Called from helpers.py's own run_step8_layers(cfg) (a different
    function of the same name in a different module -- accessed there as
    classify_tools.run_step8_layers, not a naming collision, just a
    coincidence worth knowing if you're searching for this name).

    Each (frame, leaflet) pair is independent, so with workers > 1 this
    runs in parallel across them (same pattern as Steps 2/5/6/9).

    Writes:
      <prefix>_layers.csv    frame, side, cluster, region_type, area_angstrom2, paired_fraction
      <prefix>_layers.txt    dashed-section summary
      layers/<bilayer|monolayer>/<upper|lower>/frame_####_layers_<region>_<side>.ply

    Returns: dict with layers_csv, layers_txt, counts ({"monolayer": n,
             "bilayer": n}), and frames_processed.

    """
    output_dir = Path(output_dir)
    ply_dir = output_dir / "layers"
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_ply:
        for rtype in ("bilayer", "monolayer"):
            for side in ("upper", "lower"):
                (ply_dir / rtype / side).mkdir(parents=True, exist_ok=True)

    prefix = Path(str(output_prefix).strip()).name
    if not prefix:
        raise ValueError("output_name_prefix cannot be empty.")

    csv_path = output_dir / f"{prefix}_layers.csv"
    counts = {"monolayer": 0, "bilayer": 0}
    edge_lengths = []
    thickness_log = []

    solid = {"monolayer": (0.85, 0.55, 0.15),
             "bilayer": (0.20, 0.55, 0.85)}

    jobs = [
        (side, pruned_path)
        for side in ("upper", "lower")
        for pruned_path in collect_pruned_npz_files(pruned_npz_dir, side)
    ]

    n_workers = max(1, min(int(workers), len(jobs))) if jobs else 1
    all_rows = []  # list of (frame_idx, side, csv_rows), sorted before writing

    n_workers = max(1, min(int(workers), len(jobs))) if jobs else 1
    all_rows = []  # list of (frame_idx, side, csv_rows), sorted before writing

    progress.set_total(len(jobs))

    if n_workers <= 1:
        print(f"Step 8 layer classification running in serial mode over {len(jobs)} frame/leaflet job(s).")
        for side, pruned_path in jobs:
            frame_idx, side_r, csv_rows, edge_len, thick_used, job_counts = _process_one_layer_frame_side(
                side, pruned_path, mesh_npz_dir, ply_dir,
                thickness, thickness_tol, max_pair_dist, bilayer_frac, write_ply, solid,
            )
            if csv_rows:
                all_rows.append((frame_idx, side_r, csv_rows))
            if edge_len is not None:
                edge_lengths.append(edge_len)
            if np.isfinite(thick_used):
                thickness_log.append(thick_used)
            for k in counts:
                counts[k] += job_counts[k]
            progress.tick()
    else:
        print(
            f"Step 8 layer classification running with {n_workers} worker processes "
            f"over {len(jobs)} frame/leaflet job(s)."
        )
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _process_one_layer_frame_side_capture,
                    side, pruned_path, mesh_npz_dir, ply_dir,
                    thickness, thickness_tol, max_pair_dist, bilayer_frac, write_ply, solid,
                )
                for side, pruned_path in jobs
            ]
            for fut in as_completed(futures):
                frame_idx, side_r, csv_rows, edge_len, thick_used, job_counts, text = fut.result()
                print(f"[worker done] Step 8 frame {frame_idx} ({side_r})")
                if text:
                    print(text, end="" if text.endswith("\n") else "\n")
                if csv_rows:
                    all_rows.append((frame_idx, side_r, csv_rows))
                if edge_len is not None:
                    edge_lengths.append(edge_len)
                if np.isfinite(thick_used):
                    thickness_log.append(thick_used)
                for k in counts:
                    counts[k] += job_counts[k]
                progress.tick()

    n_frames = len(edge_lengths)  # matches original semantics: one count per successful (frame, side) job

    side_rank = {"upper": 0, "lower": 1}
    all_rows.sort(key=lambda item: (item[0], side_rank.get(item[1], 2)))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cluster", "region_type",
                    "area_angstrom2", "paired_fraction"])
        for _, _, csv_rows in all_rows:
            for row in csv_rows:
                w.writerow(row)

    section = _build_layers_section(prefix, counts, edge_lengths, thickness_log,
                                    thickness, thickness_tol, max_pair_dist, bilayer_frac)
    summary_path = output_dir / f"{prefix}_layers.txt"
    summary_path.write_text(join_analysis_sections([section]), encoding="utf-8")

    return {
        "layers_csv": str(csv_path),
        "layers_txt": str(summary_path),
        "counts": counts,
        "frames_processed": n_frames,
    }



