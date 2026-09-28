"""
Step 9: curvature analysis.

Single-fit design. The PyMeshLab quadric fit runs EXACTLY ONCE per
(frame, leaflet), on the Step-3 full-leaflet mesh. Its result is cached to
results/curvature/npz/frame_####_curvature_<side>.npz and every downstream
product is derived from that one file:

  Stage 1 (always) -- compute per-vertex mean curvature H on the full
      leaflet mesh, plus the ColorRamp range PyMeshLab used to paint it, and
      the per-triangle H (vertex average). Cache to NPZ.

  Stage 2 (optional, curvature_full_mesh) -- rebuild the full-mesh heatmap
      PLY straight from the cached verts/tris/H/color. No refit.

  Stage 3 (CSV mandatory, PLY optional) -- read the Step-6 pruned NPZ for
      its final_clusters ONLY (triangle-id arrays). Those ids index the SAME
      triangle array as the Step-3 mesh: Steps 4/5/6 all pass verts/tris
      through verbatim without remapping, so tri id k is the same triangle
      everywhere. Each cluster's curvature is therefore just an area-weighted
      average of H_tri[tri_ids] -- an indexing operation, not a second fit.
      No normalization is applied; the cluster value written to the CSV is
      the raw area-weighted mean curvature H, in the same physical units as
      the full-mesh heatmap it was cut from.

Consistency is structural here, not coincidental: Stage 3 cannot disagree
with Stage 2 because both read the same H_vertex/ramp out of the same NPZ.
"""

import csv
import glob
import io
import os
import re
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
try:
    import open3d as o3d
except Exception:
    o3d = None

import numpy as np
import pymeshlab
from .classify import triangle_areas
from . import progress


# ==================================================
# Frame-index / path helpers 
# ==================================================

_FRAME_RE = re.compile(r"frame_(\d+)")

SIDES = ("upper", "lower")


def frame_index_from_path(path):
    """Extract the frame index from any frame_#### filename this pipeline uses.

    Raises ValueError if no frame_#### substring is found.
    """
    m = _FRAME_RE.search(os.path.basename(str(path)))
    if not m:
        raise ValueError(f"Cannot parse frame index from: {path}")
    return int(m.group(1))


def collect_mesh_npz_files(mesh_npz_dir):
    """Return every Step-3 mesh NPZ, sorted by frame index.

    This is the job driver for Step 9 -- run_step9_curvature builds its
    (frame, side) job list off this list, not off Step 6's pruned NPZs, so
    a frame with no pruned clusters still gets a curvature fit (CSV rows
    are skipped for that frame/leaflet, but the NPZ and heatmap PLY are not).
    """
    files = glob.glob(os.path.join(str(mesh_npz_dir), "frame_*_mesh_predefect.npz"))
    return sorted(set(files), key=frame_index_from_path)


def load_mesh_leaflet(npz_path, side):
    """One leaflet's (verts, tris) from the Step-3 mesh NPZ.

    side: "upper" or "lower" -- selects verts_up/tris_up or verts_lo/tris_lo.
    """
    key_v = "verts_up" if side == "upper" else "verts_lo"
    key_t = "tris_up" if side == "upper" else "tris_lo"
    with np.load(npz_path, allow_pickle=True) as d:
        return (
            np.asarray(d[key_v], dtype=np.float64),
            np.asarray(d[key_t], dtype=np.int32),
        )


def _pruned_path_for(pruned_npz_dir, side, frame_idx):
    """Build the expected Step-6 pruned-NPZ path for one frame/leaflet, or None if it doesn't exist.

    Returning None (rather than raising) for a missing file is deliberate:
    Step 9's job list is driven by the Step-3 mesh, not by what pruned NPZs
    exist, so a frame with no surviving Step-6 clusters is a normal,
    expected case here -- see _process_one_leaflet's handling of a None
    pruned_path.
    """
    p = os.path.join(
        str(pruned_npz_dir), f"frame_{frame_idx:04d}_pruned_clusters_{side}.npz"
    )
    return p if os.path.exists(p) else None


def load_pruned_clusters_only(npz_path):
    """Load only Step-6's final_clusters (triangle-id arrays) from a pruned NPZ.

    Returns: list of int64 triangle-id arrays, one per cluster.
    """
    with np.load(npz_path, allow_pickle=True) as d:
        return [np.asarray(cl, dtype=np.int64).ravel() for cl in d["final_clusters"]]



# ==================================================
# Curvature filter 
# ==================================================

def _run_curvature_filter(verts, tris, method, frame_idx=None, side=None):
    """Run PyMeshLab's curvature filter -> (meshset, H_vertex, face_matrix, updated_verts).

    Repairs non-manifold vertices first

    verts, tris: one leaflet's mesh arrays, from load_mesh_leaflet.
    method:      PyMeshLab curvature method name (e.g. "Quadric Fitting",
                 "Scale Dependent Quadric Fitting") -- passed straight
                 through to compute_curvature_principal_directions_per_vertex.
    frame_idx, side: used only to label warning/log messages; either can be
                 None (the tag falls back to "mesh").
    Returns: (meshset, H_vertex, face_matrix, updated_verts). H_vertex is
             empty (size 0) if verts/tris were empty on input, or if the
             manifold/connectivity check failed after repair -- both cases
             mean "nothing usable was computed," and updated_verts/
             face_matrix reflect whatever state the mesh was actually left
             in (repaired-but-unfit, or the original empty input).
    """
    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int32)

    ms = pymeshlab.MeshSet()
    if verts.size == 0 or tris.size == 0:
        return ms, np.empty((0,), dtype=np.float64), np.empty((0, 3), dtype=np.int32), verts

    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=verts, face_matrix=tris))
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    repaired = ms.current_mesh()
    repaired_verts = repaired.vertex_matrix()
    repaired_tris = repaired.face_matrix()
    tag = f"frame {frame_idx} {side}" if frame_idx is not None else "mesh"

    if o3d is not None:
        m_check = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(repaired_verts),
            o3d.utility.Vector3iVector(repaired_tris),
        )
        if not (m_check.is_edge_manifold() and m_check.is_vertex_manifold()):
            print(f"⚠️  Step 9: {tag}: still non-manifold after vertex-split repair "
                  f"(likely non-manifold edges) -- skipping curvature fit for this leaflet.")
            return ms, np.empty((0,), dtype=np.float64), np.empty((0, 3), dtype=np.int32), repaired_verts

        _, cluster_n_tris, _ = m_check.cluster_connected_triangles()
        if len(cluster_n_tris) > 1:
            print(f"⚠️  Step 9: {tag}: {len(cluster_n_tris)} disconnected component(s) "
                  f"after repair (sizes {sorted(cluster_n_tris)}) -- skipping curvature "
                  f"fit for this leaflet rather than risk another hang.")
            return ms, np.empty((0,), dtype=np.float64), np.empty((0, 3), dtype=np.int32), repaired_verts
    else:
        print("⚠️  Step 9: open3d not available -- non-manifold/disconnected-after-repair "
              "check is disabled; a pathological mesh could still hang the curvature fit.")

    ms.compute_curvature_principal_directions_per_vertex(
        method=method,
        curvcolormethod="Mean Curvature",
        autoclean=False,
    )
    mesh = ms.current_mesh()
    return ms, mesh.vertex_scalar_array(), mesh.face_matrix(), mesh.vertex_matrix()



def ramp_range(H_vertex):
    """(Hmin, Hmax) exactly as PyMeshLab's ColorRamp used them to paint this mesh.

    I.e. the per-vertex quality extremes of the FULL leaflet -- this is
    what makes Stage 2's heatmap and Stage 3's cluster values consistent:
    both derive from this same range, computed once, cached once.

    Returns (0.0, 0.0) for an empty H_vertex, not an error.
    """
    H_vertex = np.asarray(H_vertex, dtype=np.float64)
    if H_vertex.size == 0:
        return 0.0, 0.0
    return float(H_vertex.min()), float(H_vertex.max())



def per_triangle_curvature(face_matrix, H_vertex):
    """Average each triangle's 3 vertex H values into one per-triangle H.

    This is the H_tri consumed by Stage 3's area-weighted cluster mean --
    simple vertex averaging, not a separate fit or filter.
    """
    return H_vertex[face_matrix].mean(axis=1)

# ==================================================
# Stage 1: the curvature NPZ  (results/curvature/npz)
# ==================================================
# Schema v2, faster and doesn't have the hang issue.

_CURVATURE_NPZ_VERSION = 2


def _curvature_npz_path(npz_root, frame_idx, side):
    """Build the expected Stage-1 NPZ path for one frame/leaflet."""
    return Path(npz_root) / f"frame_{frame_idx:04d}_curvature_{side}.npz"


def save_curvature_npz(npz_root, frame_idx, side, verts, tris, H_vertex, H_tri, vertex_color, ramp_lo, ramp_hi):
    """Cache one (frame, leaflet)'s full-mesh curvature result.

    Full-mesh only: no cluster data lives here. Defect clusters are a
    downstream categorization of this field.


    Returns: the output path as a string.
    """
    npz_root = Path(npz_root)
    npz_root.mkdir(parents=True, exist_ok=True)
    out_path = _curvature_npz_path(npz_root, frame_idx, side)

    np.savez_compressed(
        out_path,
        version=np.int32(_CURVATURE_NPZ_VERSION),
        frame=np.int32(frame_idx),
        side=str(side),
        verts=np.asarray(verts, dtype=np.float32),
        tris=np.asarray(tris, dtype=np.int32),
        H_vertex=np.asarray(H_vertex, dtype=np.float32),
        H_tri=np.asarray(H_tri, dtype=np.float32),
        vertex_color=np.asarray(vertex_color, dtype=np.float32),
        ramp_lo=np.float64(ramp_lo),
        ramp_hi=np.float64(ramp_hi),
    )
    return str(out_path)

def load_curvature_npz(npz_root, frame_idx, side):
    """Load the cached Stage-1 result for one frame/leaflet, or None if absent/unreadable/stale.

    Returns: dict with verts, tris, H_vertex, H_tri, vertex_color, ramp_lo,
             ramp_hi -- or None.
    """
    path = _curvature_npz_path(npz_root, frame_idx, side)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            if int(d["version"]) != _CURVATURE_NPZ_VERSION:
                return None
            return {
                "verts": np.asarray(d["verts"], dtype=np.float64),
                "tris": np.asarray(d["tris"], dtype=np.int64),
                "H_vertex": np.asarray(d["H_vertex"], dtype=np.float64),
                "H_tri": np.asarray(d["H_tri"], dtype=np.float64),
                "vertex_color": np.asarray(d["vertex_color"], dtype=np.float64),
                "ramp_lo": float(d["ramp_lo"]),
                "ramp_hi": float(d["ramp_hi"]),
            }
    except Exception as exc:  # corrupt / truncated / stale -> recompute
        print(f"⚠️  Step 9: ignoring unreadable curvature NPZ {path.name} ({exc}).")
        return None


def _compute_curvature_leaflet(mesh_path, npz_root, frame_idx, side, method):
    """Stage 1 for one (frame, leaflet): load from cache, or fit and cache. Returns the result dict.

    Fast path: a valid cached NPZ means _run_curvature_filter is skipped entirely

    Returns None if the mesh is empty, the filter raises, or the filter
    deliberately produced nothing (still non-manifold or disconnected
    after repair -- see _run_curvature_filter) -- all three are "nothing
    to fit or cache" and the caller (_process_one_leaflet) treats them
    identically.

    Returns: the same dict shape as load_curvature_npz, freshly computed
             and already cached to disk if this wasn't a cache hit.
    """
    cached = load_curvature_npz(npz_root, frame_idx, side)
    if cached is not None:
        return cached

    verts, tris = load_mesh_leaflet(mesh_path, side)
    if verts.size == 0 or tris.size == 0:
        print(f"⚠️  Step 9: frame {frame_idx} {side}: empty mesh, skipping.")
        return None

    try:
        ms, H_vertex, face_matrix, repaired_verts = _run_curvature_filter(
            verts, tris, method, frame_idx=frame_idx, side=side
        )
    except Exception as exc:
        print(f"⚠️  Step 9: frame {frame_idx} {side}: repair/curvature filter failed ({exc}); skipping.")
        return None

    if H_vertex.size == 0:
        # Empty input, OR _run_curvature_filter deliberately skipped the fit
        # (still non-manifold, or disconnected, after repair). Either way,
        # nothing consistent to cache or return.
        print(f"⚠️  Step 9: frame {frame_idx} {side}: curvature filter returned nothing.")
        return None

    # H_vertex/face_matrix/vertex_color all came from the SAME final mesh
    # state inside _run_curvature_filter. verts/tris must point at that same
    # repaired mesh -- vertex-split repair can change the vertex count, so
    # the ORIGINAL verts/tris loaded above are a different length as soon as
    # any repair actually fires.
    verts = repaired_verts
    tris = face_matrix

    ramp_lo, ramp_hi = ramp_range(H_vertex)
    H_tri = per_triangle_curvature(face_matrix, H_vertex)
    vertex_color = ms.current_mesh().vertex_color_matrix()

    save_curvature_npz(
        npz_root=npz_root,
        frame_idx=frame_idx,
        side=side,
        verts=verts,
        tris=tris,
        H_vertex=H_vertex,
        H_tri=H_tri,
        vertex_color=vertex_color,
        ramp_lo=ramp_lo,
        ramp_hi=ramp_hi,
    )
    return {
        "verts": np.asarray(verts, dtype=np.float64),
        "tris": np.asarray(tris, dtype=np.int64),
        "H_vertex": np.asarray(H_vertex, dtype=np.float64),
        "H_tri": np.asarray(H_tri, dtype=np.float64),
        "vertex_color": np.asarray(vertex_color, dtype=np.float64),
        "ramp_lo": ramp_lo,
        "ramp_hi": ramp_hi,
    }

# ==================================================
# Stage 2: full-mesh heatmap PLY (rebuilt from the NPZ)
# ==================================================

def save_curvature_heatmap_ply(out_path, verts, tris, H_vertex, vertex_color):
    """Write the full-leaflet heatmap PLY from an already-computed curvature field.
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=np.asarray(verts, dtype=np.float64),
        face_matrix=np.asarray(tris, dtype=np.int32),
        v_scalar_array=np.asarray(H_vertex, dtype=np.float64),
        v_color_matrix=np.asarray(vertex_color, dtype=np.float64),
    ))
    os.makedirs(os.path.dirname(os.path.abspath(str(out_path))) or ".", exist_ok=True)
    ms.save_current_mesh(
        str(out_path),
        save_vertex_color=True,
        save_vertex_quality=True,
    )

# ==================================================
# Stage 3: defect-cluster PLY (subset of the already-colored full mesh)
# ==================================================

def save_colored_triangle_subset_ply(out_path, verts, tris, tri_ids, vertex_quality, vertex_color):
    """Write a subset of triangles as a standalone PLY, carrying over the full-mesh quality/color.

    vertex_quality (raw H) and vertex_color are the FULL leaflet's already-
    computed values, subset down to just the vertices this tri_ids
    selection touches -- not recomputed on the subset. 

    Returns False (writes nothing) for an empty tri_ids, not an error --
    a frame/leaflet can legitimately have zero surviving defect triangles.
    Returns True if a file was actually written.
    """
    tri_ids = np.asarray(tri_ids, dtype=np.int64)
    if tri_ids.size == 0:
        return False

    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    faces = tris[tri_ids]
    uniq, inv = np.unique(faces.reshape(-1), return_inverse=True)
    sub_verts = verts[uniq]
    sub_tris = inv.reshape(-1, 3).astype(np.int32)
    sub_quality = np.asarray(vertex_quality, dtype=np.float64)[uniq]
    sub_color = np.asarray(vertex_color, dtype=np.float64)[uniq]

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=sub_verts,
        face_matrix=sub_tris,
        v_scalar_array=sub_quality,
        v_color_matrix=sub_color,
    ))
    os.makedirs(os.path.dirname(os.path.abspath(str(out_path))) or ".", exist_ok=True)
    ms.save_current_mesh(str(out_path), save_vertex_color=True, save_vertex_quality=True)
    return True


# ==================================================
# Defects CSV (resumable)
# ==================================================
_DEFECTS_CSV_HEADER = ["frame", "cluster", "area_angstrom2", "H"]

def _completed_defect_jobs(csv_path):
    """Return the set of (frame, side) pairs already recorded in an existing defects CSV.

    side is recovered from the "cluster" column's "side:cluster_id" prefix
    (e.g. "upper:3"), same convention as classify.py's/analysis.py's CSVs.

    Returns: set of (frame_idx, side) tuples. Empty set if csv_path doesn't
             exist yet -- not an error, just "nothing completed so far."
    """
    csv_path = Path(csv_path)
    done = set()
    if not csv_path.exists():
        return done
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                frame_idx = int(row[0])
            except ValueError:
                continue
            side = row[1].split(":", 1)[0]
            if side in SIDES:
                done.add((frame_idx, side))
    return done


def _open_defects_csv_for_append(csv_path):
    """Open the combined defects CSV for real-time appends.


    Returns: (file_handle, csv.writer) -- caller is responsible for
             closing file_handle when done (run_step9_curvature does this
             in a finally block).
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = (not csv_path.exists()) or csv_path.stat().st_size == 0
    f = csv_path.open("a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if is_new:
        w.writerow(_DEFECTS_CSV_HEADER)
        f.flush()
        os.fsync(f.fileno())
    return f, w


def _write_defects_rows(f, writer, rows):
    """Append one job's rows to the defects CSV and force them to disk immediately.
    """
    if not rows:
        return
    for frame_idx, side, cluster_idx, area, H_cluster in rows:
        writer.writerow([
            frame_idx,
            f"{side}:{cluster_idx}",
            f"{float(area):.6f}",
            f"{float(H_cluster):.6f}",
        ])
    f.flush()
    os.fsync(f.fileno())


# ==================================================
# Main orchestrator
# ==================================================

def _process_one_leaflet(
    mesh_path,
    pruned_path,
    frame_idx,
    side,
    npz_root,
    full_mesh_root,
    defects_root,
    method,
    write_full_mesh_ply,
    write_defect_ply,
    frame_range,
):
    """Run all of Step 9 for one (frame, leaflet), off a single curvature fit.

    Main orchestrator. Ties everything together.

    Returns: (frame_idx, side, rows) -- rows is a list of
             (frame, side, cluster_idx, area_total, H_cluster) tuples,
             empty if there was nothing to classify for this leaflet.
    """
    cur = _compute_curvature_leaflet(mesh_path, npz_root, frame_idx, side, method)
    if cur is None:
        return frame_idx, side, []

    verts = cur["verts"]
    tris = cur["tris"]
    H_vertex = cur["H_vertex"]
    H_tri = cur["H_tri"]
    vertex_color = cur["vertex_color"]
    Hmin = cur["ramp_lo"]
    Hmax = cur["ramp_hi"]

    in_range = frame_range is None or (frame_range[0] <= frame_idx <= frame_range[1])

    # ---- Stage 2: full-mesh heatmap PLY -------------------------------------
    if write_full_mesh_ply and in_range:
        out_ply = Path(full_mesh_root) / side / f"frame_{frame_idx:04d}_curvature_{side}.ply"
        save_curvature_heatmap_ply(out_ply, verts, tris, H_vertex, vertex_color)
        print(f"Step 9: frame {frame_idx} {side}: full-mesh heatmap -> {out_ply.name}")

    # ---- Stage 3: defect clusters -------------------------------------------
    if pruned_path is None:
        print(f"⚠️  Step 9: frame {frame_idx} {side}: no pruned NPZ; curvature NPZ written, no CSV rows.")
        return frame_idx, side, []

    clusters = load_pruned_clusters_only(pruned_path)
    if not clusters:
        return frame_idx, side, []

    areas_tri = triangle_areas(verts, tris)
    n_tris = len(tris)

    rows = []
    kept_tri_ids = []

    for cluster_idx, tri_ids in enumerate(clusters):
        tri_ids = np.asarray(tri_ids, dtype=np.int64)
        tri_ids = tri_ids[(tri_ids >= 0) & (tri_ids < n_tris)]
        if tri_ids.size == 0:
            continue

        a = areas_tri[tri_ids]
        area_total = float(a.sum())
        if area_total > 0:
            H_cluster = float((H_tri[tri_ids] * a).sum() / area_total)
        else:
            H_cluster = float(H_tri[tri_ids].mean())

        rows.append((frame_idx, side, cluster_idx, area_total, H_cluster))
        kept_tri_ids.append(tri_ids)

    if kept_tri_ids and write_defect_ply and in_range:
        merged = np.concatenate(kept_tri_ids)
        out_ply = Path(defects_root) / side / f"frame_{frame_idx:04d}_defects_{side}.ply"
        save_colored_triangle_subset_ply(out_ply, verts, tris, merged, H_vertex, vertex_color)

    print(
        f"Step 9: frame {frame_idx} {side}: {len(clusters)} cluster(s), "
        f"{len(rows)} classified (H ramp {Hmin:.5f} .. {Hmax:.5f})"
    )
    return frame_idx, side, rows

def _process_one_leaflet_capture(*args, **kwargs):
    """Worker wrapper that captures console output (same pattern as Steps 5/6/8)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        frame_idx, side, rows = _process_one_leaflet(*args, **kwargs)
    return frame_idx, side, rows, buf.getvalue()

# ==================================================
# Main engine
# ==================================================

def run_step9_curvature(
    mesh_npz_dir,
    pruned_npz_dir,
    output_dir,
    method="Scale Dependent Quadric Fitting",
    workers=1,
    write_full_mesh_ply=True,
    write_defect_ply=True,
    frame_range=None,
):
    """Step 9, single-fit. The single entry point, called from helpers.py's run_step9_curvature_cfg.

    Writes:
      <output_dir>/npz/frame_####_curvature_<side>.npz          (always)
      <output_dir>/full_mesh/<side>/frame_####_curvature_<side>.ply
      <output_dir>/defects/<side>/frame_####_defects_<side>.ply
      <output_dir>/defects/defects_curvature.csv                (always)

    Returns: dict with frames_processed, frames_skipped,
             clusters_classified, npz_dir, full_mesh_dir, defects_dir,
             csv_path.

    frames_processed AND frames_skipped BOTH COUNT (FRAME, LEAFLET) JOBS,
    NOT DISTINCT FRAMES -- same counting convention as classify.py's Step 8
    frames_processed. A 100-frame trajectory where both leaflets succeed
    for every frame reports frames_processed=200, not 100. Not renamed
    here to preserve the existing contract with helpers.py's print
    statements, but worth knowing before reading this number as a frame
    count in a paper's methods section.

    Raises FileNotFoundError if mesh_npz_dir has no Step-3 mesh NPZ files
    at all -- this is the fail-fast check classify.py's equivalent
    function (run_step8_layers) is missing.
    """
    output_dir = Path(output_dir)
    npz_root = output_dir / "npz"
    full_mesh_root = output_dir / "full_mesh"
    defects_root = output_dir / "defects"

    npz_root.mkdir(parents=True, exist_ok=True)
    defects_root.mkdir(parents=True, exist_ok=True)  # CSV always lives here
    if write_full_mesh_ply:
        for s in SIDES:
            (full_mesh_root / s).mkdir(parents=True, exist_ok=True)
    if write_defect_ply:
        for s in SIDES:
            (defects_root / s).mkdir(parents=True, exist_ok=True)

    mesh_paths = collect_mesh_npz_files(mesh_npz_dir)
    print(f"Step 9: found {len(mesh_paths)} mesh NPZ frame(s) in {mesh_npz_dir}")
    if not mesh_paths:
        raise FileNotFoundError(
            f"No mesh NPZ files found in {mesh_npz_dir}. "
            "Expected frame_####_mesh_predefect.npz (Step 3)."
        )

    csv_path = defects_root / "defects_curvature.csv"
    done_jobs = _completed_defect_jobs(csv_path)
    if done_jobs:
        print(
            f"Step 9: resume -> {len(done_jobs)} (frame, leaflet) job(s) already "
            f"recorded in {csv_path.name}, skipping those."
        )

    # Build the (frame, side) job list off the Step-3 mesh -- the mesh is the
    # driver now, not the pruned NPZ.
    jobs = []
    n_skipped = 0
    for mesh_path in mesh_paths:
        frame_idx = frame_index_from_path(mesh_path)
        for side in SIDES:
            if (frame_idx, side) in done_jobs:
                n_skipped += 1
                continue
            jobs.append((mesh_path, _pruned_path_for(pruned_npz_dir, side, frame_idx),
                         frame_idx, side))

    csv_file, csv_writer = _open_defects_csv_for_append(csv_path)
    # Resume-skipped jobs count toward the total and are pre-ticked, so the bar
    # reads as a fraction of the whole trajectory rather than of this run's
    # remainder.
    progress.set_total(len(jobs) + n_skipped)
    if n_skipped:
        progress.tick(n_skipped)
    n_done = 0
    n_rows = 0

    try:
        n_workers = max(1, min(int(workers), len(jobs))) if jobs else 1

        if n_workers <= 1:
            print(f"Step 9 running in serial mode over {len(jobs)} (frame, leaflet) job(s).")
            for i, (mesh_path, pruned_path, frame_idx, side) in enumerate(jobs, start=1):
                fr, sd, rows = _process_one_leaflet(
                    mesh_path, pruned_path, frame_idx, side,
                    npz_root, full_mesh_root, defects_root, method,
                    write_full_mesh_ply, write_defect_ply, frame_range,
                )
                _write_defects_rows(csv_file, csv_writer, rows)
                n_done += 1
                n_rows += len(rows)
                print(f"Step 9: [serial {i}/{len(jobs)}] frame {fr} {sd} done (+{len(rows)} rows)")
                progress.tick()
        else:
            print(
                f"Step 9 running with {n_workers} worker processes over "
                f"{len(jobs)} (frame, leaflet) job(s)."
            )
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                futures = [
                    executor.submit(
                        _process_one_leaflet_capture,
                        mesh_path, pruned_path, frame_idx, side,
                        npz_root, full_mesh_root, defects_root, method,
                        write_full_mesh_ply, write_defect_ply, frame_range,
                    )
                    for (mesh_path, pruned_path, frame_idx, side) in jobs
                ]
                print(f"Step 9: submitted {len(futures)} job(s) to the process pool")
                for k, fut in enumerate(as_completed(futures), start=1):
                    fr, sd, rows, text = fut.result()
                    # Parent is the only CSV writer: append + flush + fsync the
                    # instant a job resolves, so a kill never loses finished work.
                    _write_defects_rows(csv_file, csv_writer, rows)
                    n_done += 1
                    n_rows += len(rows)
                    print(f"[worker done] Step 9 frame {fr} {sd} ({k}/{len(futures)} complete)")
                    if text:
                        print(text, end="" if text.endswith("\n") else "\n")
                    progress.tick()
    finally:
        csv_file.close()

    print(
        f"Step 9: done. jobs_processed={n_done} jobs_skipped(resume)={n_skipped} "
        f"new_rows={n_rows}"
    )
    return {
        "frames_processed": n_done,
        "frames_skipped": n_skipped,
        "clusters_classified": n_rows,
        "npz_dir": str(npz_root),
        "full_mesh_dir": str(full_mesh_root),
        "defects_dir": str(defects_root),
        "csv_path": str(csv_path),
    }