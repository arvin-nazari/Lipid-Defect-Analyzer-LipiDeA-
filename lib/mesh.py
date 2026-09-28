"""Step 3: leaflet mesh generation.

For each frame, splits the membrane into upper and lower leaflets via
MDAnalysis's LeafletFinder, reconstructs each leaflet as a triangulated
surface with Open3D's Poisson reconstruction, brings the triangle size to
a user-requested target area (hybrid subdivision + light isotropic
remeshing), trims unsupported edge artifacts, and repairs the result to a
single manifold component before saving.

Output: one .ply mesh per leaflet per frame, read back in by Step 4.
"""

import os
import glob
import re

import numpy as np
import open3d as o3d
import time
from collections import defaultdict, deque

import pymeshlab

try:
    from scipy.spatial import cKDTree, ConvexHull
except Exception:  # scipy is expected through MDAnalysis, but keep a safe fallback
    cKDTree = None
    ConvexHull = None

import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder


# ======================================================================================
# LeafletFinder helper
# ======================================================================================
def leaflet_positions_from_pdb(pdb_path, select, cutoff=15.0, pbc=False):
    """Split one PDB frame into upper/lower leaflet bead positions via LeafletFinder.

    select is an MDAnalysis atom-selection string (built by helpers.py's
    build_leaflet_select from the user's res_name/atom_name in prep.in) that
    should match one representative bead per lipid -- LeafletFinder clusters
    on that selection alone, then every atom in the two largest resulting
    connected components is returned as one leaflet.

    "Upper" vs "lower" is a display label only, chosen by comparing mean z
    of the two leaflets -- it has no effect on which atoms end up in which
    group; LeafletFinder's own connectivity clustering already decided that.

    pdb_path: path to one single-frame PDB (a tiled/cut frame from Step 2).
    select:   MDAnalysis selection string, e.g. "(resname POPC DOPE) and name GL2".
    cutoff:   LeafletFinder distance cutoff in Angstroms.
    pbc:      whether LeafletFinder should apply periodic boundary conditions.
              False by default -- only set True if the PDB carries a valid
              CRYST1 unit cell, or LeafletFinder's neighbor search will be wrong.

    Returns: (upper_positions, lower_positions), each an (N, 3) array.
    Raises:  RuntimeError if the selection matches no atoms, or if
             LeafletFinder finds fewer than two connected components.
    """
    u = mda.Universe(pdb_path)

    # Debug, ensure selection matches atoms.
    ag = u.select_atoms(select)
    print(f"[LeafletFinder] {os.path.basename(pdb_path)}: selected {len(ag)} atoms with: {select}")
    if len(ag) == 0:
        raise RuntimeError(f"Selection matched 0 atoms in {pdb_path}: '{select}'")

    L = LeafletFinder(u, select=select, cutoff=cutoff, pbc=pbc)
    groups = L.groups()  # list of AtomGroups (connected components)

    if len(groups) < 2:
        sizes = [len(g) for g in groups]
        raise RuntimeError(
            f"LeafletFinder found <2 components for {pdb_path}. "
            f"Got sizes={sizes}. Selection='{select}'. Try adjusting cutoff/selection."
        )

    # Take the two largest components as the two leaflets.
    groups_sorted = sorted(groups, key=lambda g: len(g), reverse=True)
    leaf0, leaf1 = groups_sorted[0], groups_sorted[1]

    pos0 = leaf0.positions.copy()
    pos1 = leaf1.positions.copy()

    if pos0[:, 2].mean() >= pos1[:, 2].mean():
        upper_positions, lower_positions = pos0, pos1
    else:
        upper_positions, lower_positions = pos1, pos0

    return upper_positions, lower_positions



# ======================================================================================
# PDB parsing
# ======================================================================================
def parse_frame(frame_lines):
    """Parse one frame's PDB lines into {"atoms": [...]}.

    Each atom dict carries the fixed-width PDB fields plus the original raw
    line (kept for traceability, not otherwise used downstream in Step 3).
    A malformed line is skipped with a warning rather than aborting the
    frame -- consistent with parse_pdb in lib/prep.py.

    Only ATOM records are read, not HETATM. If a force field ever encodes a
    lipid bead as HETATM, it will silently vanish from this frame -- worth
    checking atom_info.csv's row count against the PDB if a system uses a
    non-standard record type.

    frame_lines: list of raw PDB line strings for one frame (no header/CRYST1
                 filtering -- non-ATOM lines are simply skipped).
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


def parse_pdb_files(folder_path, file_pattern="*.pdb"):
    """Yield (frame_idx, frame_data, pdb_file_basename, pdb_file_fullpath) one frame at a time.

    frame_idx is the first integer substring found in the filename (works
    for Step 2's frame_0.pdb, frame_1.pdb, ... naming). Files that don't
    contain a parseable index are skipped with a warning, not aborted.


    folder_path:  directory containing per-frame PDB files.
    file_pattern: glob pattern, default "*.pdb".
    Yields: (frame_idx, frame_data, basename, fullpath) per frame, sorted by
            frame_idx.
    """
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
        yield frame_idx, frame_data, os.path.basename(pdb_file), pdb_file

        del frame_lines


# ======================================================================================
# Mesh creation
# ======================================================================================

def create_leaflet_meshes(
    frame_data=None,
    atom_name_filter=None,
    res_name_filter=None,
    upper_positions=None,
    lower_positions=None,
):
    """Build upper/lower leaflet triangle meshes via Poisson surface reconstruction.


    Poisson reconstruction uses a fixed octree depth of 5 for both leaflets
    regardless of system size.

    After reconstruction, small disconnected fragments (fewer than 100
    triangles) are dropped as reconstruction artifacts -- a fixed threshold,
    not scaled to system size. This is a first-pass cleanup only; the final
    manifold/single-component guarantee comes later from
    repair_leaflet_mesh_manifold, which keeps only the single largest
    component regardless of size.

    Returns: (mesh_upper, mesh_lower, upper_positions, lower_positions).
    An empty mesh/position pair is returned (not raised) if either leaflet
    ends up with zero points, or if a Poisson call itself throws.
    """
    # -------------------------
    # Case 1: LeafletFinder output supplied directly
    # -------------------------
    if upper_positions is not None and lower_positions is not None:
        upper_positions = np.asarray(upper_positions, dtype=float)
        lower_positions = np.asarray(lower_positions, dtype=float)

        if upper_positions.size == 0 or lower_positions.size == 0:
            print("Warning: empty leaflet positions; returning empty meshes.")
            return (
                o3d.geometry.TriangleMesh(),
                o3d.geometry.TriangleMesh(),
                np.array([]),
                np.array([]),
            )

    # -------------------------
    # Case 2: fallback mean-z split (Legacy, switched to LeafletFinder)
    # -------------------------
    else:
        if frame_data is None:
            raise ValueError("Need either (upper_positions, lower_positions) or frame_data.")

        if atom_name_filter is None:
            atom_name_filter = ["C2"]
        if res_name_filter is None:
            res_name_filter = ["PE", "PC", "SAP"]

        target_positions = np.array([
            [atom["x"], atom["y"], atom["z"]]
            for atom in frame_data["atoms"]
            if atom["atom_name"] in atom_name_filter and atom["res_name"] in res_name_filter
        ])

        if target_positions.size == 0:
            print("Warning: No matching atoms found for mesh creation; returning empty meshes.")
            return (
                o3d.geometry.TriangleMesh(),
                o3d.geometry.TriangleMesh(),
                np.array([]),
                np.array([]),
            )

        z_mean = np.mean(target_positions[:, 2])
        upper_positions = target_positions[target_positions[:, 2] >= z_mean]
        lower_positions = target_positions[target_positions[:, 2] < z_mean]

    # ---- Open3D point clouds ----
    pcd_upper = o3d.geometry.PointCloud()
    pcd_upper.points = o3d.utility.Vector3dVector(upper_positions)

    pcd_lower = o3d.geometry.PointCloud()
    pcd_lower.points = o3d.utility.Vector3dVector(lower_positions)

    radius_norms = 100.0
    max_nn = 30
    pcd_upper.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_norms, max_nn=max_nn)
    )
    pcd_lower.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_norms, max_nn=max_nn)
    )

    # Orient normals outward by leaflet: upper points +z, lower points -z.
    normals_upper = np.asarray(pcd_upper.normals)
    normals_upper[normals_upper[:, 2] < 0] *= -1.0
    pcd_upper.normals = o3d.utility.Vector3dVector(normals_upper)

    normals_lower = np.asarray(pcd_lower.normals)
    normals_lower[normals_lower[:, 2] > 0] *= -1.0
    pcd_lower.normals = o3d.utility.Vector3dVector(normals_lower)

    # Poisson reconstruction.
    try:
        mesh_upper = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_upper, depth=5)
        if isinstance(mesh_upper, tuple):
            mesh_upper = mesh_upper[0]
    except Exception as e:
        print(f"Error in Poisson reconstruction for upper mesh: {e}")
        mesh_upper = o3d.geometry.TriangleMesh()

    try:
        mesh_lower = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_lower, depth=5)
        if isinstance(mesh_lower, tuple):
            mesh_lower = mesh_lower[0]
    except Exception as e:
        print(f"Error in Poisson reconstruction for lower mesh: {e}")
        mesh_lower = o3d.geometry.TriangleMesh()

    # First-pass cleanup: drop small disconnected fragments (reconstruction
    # artifacts)
    def _cleanup(mesh):
        if not mesh.has_triangles():
            return mesh
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            clusters, n_tris, _ = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        n_tris = np.asarray(n_tris)
        mask = n_tris[clusters] < 100
        mesh.remove_triangles_by_mask(mask)
        mesh.remove_unreferenced_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.compute_triangle_normals()
        mesh.normalize_normals()
        return mesh

    mesh_upper = _cleanup(mesh_upper)
    mesh_lower = _cleanup(mesh_lower)

    return mesh_upper, mesh_lower, upper_positions, lower_positions

# ======================================================================================
# Diagnostics helpers
# ======================================================================================

def _triangle_area_stats(mesh):
    """Compute per-triangle area statistics for a mesh.

    Used throughout this file to check progress toward a target triangle
    size and to report before/after numbers in verbose logging.

    Returns: dict with n (triangle count), mean, std, cv (coefficient of
    variation = std/mean), min, max -- all zero/0.0 for an empty or
    triangle-less mesh.
    """
    if mesh is None or (not mesh.has_triangles()) or (not mesh.has_vertices()):
        return {"n": 0, "mean": 0.0, "std": 0.0, "cv": 0.0, "min": 0.0, "max": 0.0}

    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    if t.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "cv": 0.0, "min": 0.0, "max": 0.0}

    v0 = v[t[:, 0]]
    v1 = v[t[:, 1]]
    v2 = v[t[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    if areas.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "cv": 0.0, "min": 0.0, "max": 0.0}

    mean = float(areas.mean())
    std = float(areas.std())
    mn = float(areas.min())
    mx = float(areas.max())
    cv = float(std / mean) if mean != 0 else 0.0

    return {"n": int(areas.size), "mean": mean, "std": std, "cv": cv, "min": mn, "max": mx}


def isotropic_remesh(mesh, name="mesh", target_edge_length=None, iterations=5, verbose=True):
    """Redistribute a mesh's triangles to a uniform edge length via PyMeshLab.

    This is what actually lowers coefficient of variation (CV) -- Poisson
    reconstruction alone tends to produce very unevenly sized triangles, and
    isotropic remeshing is the tool that fixes that, at the cost of
    reshaping the surface somewhat compared to the raw Poisson output.

    If target_edge_length is not given, it's derived from the mesh's own
    current mean triangle area, aiming to preserve roughly the current size
    while improving uniformity.

    Falls back to returning the original mesh unchanged if PyMeshLab raises
    for any reason -- a remesh failure on one frame should not abort the run.

    mesh:               Open3D TriangleMesh to remesh.
    target_edge_length: target edge length in Angstroms; auto-derived from
                         current mean triangle area if None.
    iterations:         PyMeshLab remeshing iteration count.
    Returns: a new Open3D TriangleMesh (or the original, on failure).
    """
    if mesh is None or (not mesh.has_vertices()) or (not mesh.has_triangles()):
        if verbose:
            print(f"⚠️  {name}: empty mesh → skipping isotropic remesh.")
        return mesh

    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)

    # Auto target edge length: derived from mean triangle area.
    if target_edge_length is None:
        areas = 0.5 * np.linalg.norm(
            np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1
        )
        # Equilateral triangle with this area -> edge = sqrt(4*A/sqrt(3)).
        target_edge_length = float(np.sqrt(np.mean(areas) * 4.0 / np.sqrt(3.0)))

    area_before = _triangle_area_stats(mesh)
    nT0 = len(t)
    nV0 = len(v)

    t0 = time.perf_counter()

    try:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=v.astype(np.float64),
            face_matrix=t.astype(np.int32),
        ))

        ms.meshing_isotropic_explicit_remeshing(
            iterations=iterations,
            targetlen=pymeshlab.PureValue(target_edge_length),
        )

        out = ms.current_mesh()
        v_out = out.vertex_matrix().astype(np.float64)
        t_out = out.face_matrix().astype(np.int32)

    except Exception as e:
        print(f"⚠️  {name}: isotropic remesh failed ({e}) → returning original mesh.")
        return mesh

    # Rebuild as an Open3D mesh.
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(v_out)
    result.triangles = o3d.utility.Vector3iVector(t_out)
    result.remove_duplicated_vertices()
    result.remove_duplicated_triangles()
    result.remove_non_manifold_edges()
    result.remove_unreferenced_vertices()
    result.compute_vertex_normals()
    result.compute_triangle_normals()

    area_after = _triangle_area_stats(result)
    nV1 = len(np.asarray(result.vertices))
    nT1 = len(np.asarray(result.triangles))
    dt = time.perf_counter() - t0

    if verbose:
        print(f"\n🔧 Isotropic Remesh: {name}")
        print(f"  target_edge_length={target_edge_length:.4g} | iterations={iterations}")
        print(f"  verts: {nV0} → {nV1} | tris: {nT0} → {nT1}")
        print(f"  tri area mean/std/cv: "
              f"{area_before['mean']:.6g}/{area_before['std']:.6g}/{area_before['cv']:.4g}"
              f" → {area_after['mean']:.6g}/{area_after['std']:.6g}/{area_after['cv']:.4g}")
        print(f"  time: {dt:.3f} s")

    return result


# ======================================================================================
# Target triangle size control and subdivision (refinement)
# ======================================================================================

def choose_subdivide_iterations_for_target_area(
    current_mean_area,
    target_triangle_area,
    max_subdivide_iters=6,
):
    """Choose a Loop subdivision iteration count from current vs. target mean area.

    One subdivision iteration splits each triangle into roughly four, so
    mean area is expected to scale as current_area / 4^iterations. The
    chosen count is the nearest integer to log_4(current/target), clamped
    to 0..max_subdivide_iters.

    Returns 0 (no subdivision) if current_mean_area is already at or below
    target_triangle_area, or if either is non-positive.
    """
    current = float(current_mean_area)
    target = float(target_triangle_area)
    if current <= 0 or target <= 0 or current <= target:
        return 0

    max_it = max(0, int(max_subdivide_iters))
    if max_it == 0:
        return 0

    estimated = int(round(np.log(current / target) / np.log(4.0)))
    return int(min(max(estimated, 0), max_it))


def subdivide_mesh_with_diagnostics(mesh, name="mesh", iterations=1, method="loop", verbose=True):
    """Refine a mesh by subdivision -- the opposite of coarsening.

    Increases vertex/triangle counts before final target-size remeshing.
    Each iteration splits every triangle into roughly four, so N iterations
    multiply the triangle count by roughly 4^N.

    Returns: the subdivided mesh (or the original, if empty or iterations <= 0).
    """
    if mesh is None or (not mesh.has_vertices()) or (not mesh.has_triangles()):
        if verbose:
            print(f"⚠️ {name}: empty mesh → skipping subdivision.")
        return mesh

    it = int(iterations)
    if it <= 0:
        if verbose:
            print(f"ℹ️ {name}: subdivision disabled (iterations={iterations}).")
        return mesh

    nV0 = len(np.asarray(mesh.vertices))
    nT0 = len(np.asarray(mesh.triangles))
    area_before = _triangle_area_stats(mesh)

    method = (method or "loop").lower().strip()
    if method == "loop":
        mesh_s = mesh.subdivide_loop(number_of_iterations=it)
    elif method == "midpoint":
        mesh_s = mesh.subdivide_midpoint(number_of_iterations=it)
    else:
        raise ValueError(f"Unknown subdivision method='{method}'. Use: loop, midpoint")

    mesh_s.remove_duplicated_vertices()
    mesh_s.remove_duplicated_triangles()
    mesh_s.remove_non_manifold_edges()
    mesh_s.remove_unreferenced_vertices()
    mesh_s.compute_triangle_normals()
    mesh_s.compute_vertex_normals()

    nV1 = len(np.asarray(mesh_s.vertices))
    nT1 = len(np.asarray(mesh_s.triangles))
    area_after = _triangle_area_stats(mesh_s)

    if verbose:
        incT = (100.0 * (nT1 - nT0) / nT0) if nT0 else 0.0
        print(f"🔺 Subdivision: {name}")
        print(f"  method={method}, iterations={it}")
        print(f"  verts: {nV0} → {nV1} | tris: {nT0} → {nT1} ({incT:.2f}% more)")
        print(f"  tri area mean/std/cv: "
              f"{area_before['mean']:.6g}/{area_before['std']:.6g}/{area_before['cv']:.4g}"
              f" → {area_after['mean']:.6g}/{area_after['std']:.6g}/{area_after['cv']:.4g}")

    return mesh_s

def triangle_area_to_edge_length(target_triangle_area):
    """Convert a target triangle area (Å^2) to edge length (Å) for near-equilateral triangles.

    PyMeshLab's isotropic remeshing asks for a target edge length, while
    prep.in's triangle_size option is easier to reason about as an area.
    For an equilateral triangle: A = sqrt(3)/4 * edge^2, inverted here.
    """
    area = float(target_triangle_area)
    if area <= 0:
        raise ValueError(f"triangle_size/triangle_area must be > 0, got {target_triangle_area!r}")
    return float(np.sqrt(4.0 * area / np.sqrt(3.0)))


def target_triangle_size_with_diagnostics(
    mesh,
    name="mesh",
    target_triangle_area=1.0,
    tolerance=0.20,
    max_cv=None,
    max_iters=2,
    remesh_iters=5,
    verbose=True,
):
    """Remesh toward a target mean triangle area using isotropic remeshing alone.

    The area target is converted to an edge-length target once, then
    isotropic_remesh is called and re-measured; if the resulting mean area
    is still outside tolerance, the edge target is corrected (area scales
    roughly as edge^2, so the correction factor is sqrt(target/measured))
    and remeshing repeats, up to max_iters passes.

    This function is the final size-correction step used by
    hybrid_target_triangle_size_with_diagnostics when subdivision alone
    doesn't land within tolerance. 

    max_cv: optional coefficient-of-variation ceiling; a value <= 0 disables
            the uniformity gate.
    Returns: the remeshed mesh (or the input mesh unchanged, if already
             empty).
    """
    if mesh is None or (not mesh.has_vertices()) or (not mesh.has_triangles()):
        if verbose:
            print(f"⚠️  {name}: empty mesh → skipping target triangle sizing.")
        return mesh

    target_area = float(target_triangle_area)
    if target_area <= 0:
        raise ValueError(f"target_triangle_area must be > 0, got {target_triangle_area!r}")

    tol = float(tolerance)
    if tol < 0:
        raise ValueError(f"triangle_tolerance must be >= 0, got {tolerance!r}")

    # Optional triangle uniformity gate. A value <= 0 disables the CV check.
    max_cv_value = None if max_cv is None else float(max_cv)
    if max_cv_value is not None and max_cv_value <= 0:
        max_cv_value = None

    max_iters = max(1, int(max_iters))
    remesh_iters = max(1, int(remesh_iters))

    target_edge = triangle_area_to_edge_length(target_area)

    if verbose:
        before = _triangle_area_stats(mesh)
        print(f"\n🎯 Target triangle sizing: {name}")
        print(f"  requested mean area={target_area:.6g} Å^2 | tolerance=±{tol * 100:.2f}%")
        if max_cv_value is not None:
            print(f"  max allowed CV={max_cv_value:.4g}")
        print(f"  initial tri area mean/std/cv: {before['mean']:.6g}/{before['std']:.6g}/{before['cv']:.4g}")
        print(f"  initial target_edge_length={target_edge:.6g} Å")

    for attempt in range(1, max_iters + 1):
        mesh = isotropic_remesh(
            mesh,
            name=f"{name} target pass {attempt}",
            target_edge_length=target_edge,
            iterations=remesh_iters,
            verbose=verbose,
        )

        stats = _triangle_area_stats(mesh)
        mean_area = stats["mean"]
        if mean_area <= 0:
            if verbose:
                print(f"⚠️  {name}: mean triangle area is zero after remeshing; stopping.")
            break

        rel_err = (mean_area - target_area) / target_area
        if verbose:
            print(
                f"  pass {attempt}: mean={mean_area:.6g} Å^2, "
                f"rel_error={rel_err * 100:.2f}%, cv={stats['cv']:.4g}"
            )

        mean_ok = abs(rel_err) <= tol
        cv_ok = (max_cv_value is None) or (stats["cv"] <= max_cv_value)

        if mean_ok and cv_ok:
            if verbose:
                print(f"  ✅ {name}: target reached within tolerance and CV gate.")
            break

        if verbose and mean_ok and not cv_ok:
            if attempt < max_iters:
                print(
                    f"  ℹ️  {name}: mean is within tolerance, but CV={stats['cv']:.4g} "
                    f"> max_cv={max_cv_value:.4g}; continuing remesh correction."
                )
            else:
                print(
                    f"  ⚠️  {name}: max correction passes reached. Mean is within tolerance, "
                    f"but CV={stats['cv']:.4g} > max_cv={max_cv_value:.4g}."
                )

        # Area scales approximately with edge_length^2, so correct the edge
        # target using sqrt(target/measured). Only change the edge target when
        # the mean area is not acceptable; if only CV failed, another pass with
        # the same edge target usually improves uniformity without shifting the mean.
        if not mean_ok:
            target_edge *= float(np.sqrt(target_area / mean_area))
            if verbose and attempt < max_iters:
                print(f"  adjusting target_edge_length → {target_edge:.6g} Å")

    return mesh


def _triangle_relative_error(mean_area, target_area):
    """Fractional error of a measured mean area from the target: (mean - target) / target.

    """
    if target_area <= 0:
        return 0.0
    return float((mean_area - target_area) / target_area)


def _print_triangle_stats(prefix, stats):
    """Print one line of _triangle_area_stats' output, prefixed with a label."""
    print(
        f"{prefix}: n={stats['n']}, "
        f"mean/std/cv={stats['mean']:.6g}/{stats['std']:.6g}/{stats['cv']:.4g}, "
        f"min/max={stats['min']:.6g}/{stats['max']:.6g}"
    )

def auto_subdivide_to_target_triangle_size_with_diagnostics(
    mesh,
    name="mesh",
    target_triangle_area=5.0,
    tolerance=0.20,
    max_cv=None,
    max_subdivide_iters=6,
    subdivide_method="loop",
    verbose=True,
):
    """Move toward the target triangle area using Loop subdivision only.

    This is the first stage of the production (hybrid) sizing path -- it
    preserves the Poisson surface's connectivity and shape better than
    jumping straight to isotropic remeshing. It won't hit the target exactly,
    since subdivision changes triangle count in roughly-4x jumps rather than
    continuously; hybrid_target_triangle_size_with_diagnostics is what adds
    the isotropic correction pass afterward if this alone isn't close enough.

    Returns: the subdivided mesh (unchanged if already empty, or if 0
             iterations were chosen).
    """
    if mesh is None or (not mesh.has_vertices()) or (not mesh.has_triangles()):
        if verbose:
            print(f"⚠️  {name}: empty mesh → skipping automatic subdivision.")
        return mesh

    target_area = float(target_triangle_area)
    if target_area <= 0:
        raise ValueError(f"target_triangle_area must be > 0, got {target_triangle_area!r}")

    tol = float(tolerance)
    if tol < 0:
        raise ValueError(f"triangle_tolerance must be >= 0, got {tolerance!r}")

    max_cv_value = None if max_cv is None else float(max_cv)
    if max_cv_value is not None and max_cv_value <= 0:
        max_cv_value = None

    # Automatic subdivision always uses Loop subdivision. Cleaner.
    subdivide_method = "loop"

    before = _triangle_area_stats(mesh)
    n_iter = choose_subdivide_iterations_for_target_area(
        current_mean_area=before["mean"],
        target_triangle_area=target_area,
        max_subdivide_iters=max_subdivide_iters,
    )

    if verbose:
        print(f"\n🎯 Automatic subdivision sizing: {name}")
        print(f"  requested mean area={target_area:.6g} Å^2 | tolerance=±{tol * 100:.2f}%")
        if max_cv_value is not None:
            print(f"  max allowed CV={max_cv_value:.4g}")
        _print_triangle_stats("  before", before)
        predicted = before["mean"] / (4.0 ** n_iter) if n_iter > 0 else before["mean"]
        print(
            f"  chosen subdivision iterations={n_iter} "
            f"(method={subdivide_method}, predicted mean≈{predicted:.6g} Å^2)"
        )

    if n_iter > 0:
        mesh = subdivide_mesh_with_diagnostics(
            mesh,
            name=f"{name} auto-subdivide",
            iterations=n_iter,
            method=subdivide_method,
            verbose=verbose,
        )
    elif verbose:
        print("  subdivision not needed or target is larger than current mean area.")

    after = _triangle_area_stats(mesh)
    rel_err = _triangle_relative_error(after["mean"], target_area)
    if verbose:
        print(
            f"  after subdivision: mean={after['mean']:.6g} Å^2, "
            f"rel_error={rel_err * 100:.2f}%, cv={after['cv']:.4g}"
        )
        mean_ok = abs(rel_err) <= tol
        cv_ok = (max_cv_value is None) or (after["cv"] <= max_cv_value)
        if mean_ok and cv_ok:
            print(f"  ✅ {name}: subdivision reached target within tolerance and CV gate.")
        elif mean_ok and not cv_ok:
            print(
                f"  ℹ️  {name}: subdivision mean is inside tolerance, "
                f"but CV={after['cv']:.4g} > max_cv={max_cv_value:.4g}."
            )
        else:
            print(f"  ℹ️  {name}: subdivision alone is outside mean-area tolerance.")

    return mesh


def hybrid_target_triangle_size_with_diagnostics(
    mesh,
    name="mesh",
    target_triangle_area=5.0,
    tolerance=0.20,
    max_cv=0.25,
    max_subdivide_iters=6,
    subdivide_method="loop",
    remesh_if_needed=True,
    remesh_max_iters=2,
    remesh_iters=5,
    verbose=True,
):
    """The production triangle-sizing path: subdivide first, isotropic-correct only if needed.

    This is the function process_pdb_to_meshes actually calls (its
    triangle_method_n is hardcoded to "hybrid" regardless of what's
    configured -- see that function's docstring).

    Flow:
        Poisson mesh (already cleaned) -> Loop subdivision toward target area
        -> if mean area or CV is still outside tolerance, a light isotropic
           remeshing correction pass via target_triangle_size_with_diagnostics.

    Subdivision-first is deliberate: it preserves the Poisson surface's
    shape better than isotropic remeshing, which is used only as a final
    correction, not as the primary way of reaching the target size.

    Returns: the final mesh, either the subdivided-only result or the
             isotropically-corrected one.
    """
    # Keep the automatic subdivision method fixed to Loop.
    subdivide_method = "loop"

    mesh = auto_subdivide_to_target_triangle_size_with_diagnostics(
        mesh,
        name=name,
        target_triangle_area=target_triangle_area,
        tolerance=tolerance,
        max_cv=max_cv,
        max_subdivide_iters=max_subdivide_iters,
        subdivide_method=subdivide_method,
        verbose=verbose,
    )

    stats = _triangle_area_stats(mesh)
    target_area = float(target_triangle_area)
    rel_err = _triangle_relative_error(stats["mean"], target_area)

    tol = float(tolerance)
    max_cv_value = None if max_cv is None else float(max_cv)
    if max_cv_value is not None and max_cv_value <= 0:
        max_cv_value = None

    mean_ok = abs(rel_err) <= tol
    cv_ok = (max_cv_value is None) or (stats["cv"] <= max_cv_value)

    if mean_ok and cv_ok:
        return mesh

    if not remesh_if_needed:
        if verbose:
            reasons = []
            if not mean_ok:
                reasons.append(f"rel_error={rel_err * 100:.2f}% outside ±{tol * 100:.2f}%")
            if not cv_ok:
                reasons.append(f"CV={stats['cv']:.4g} > max_cv={max_cv_value:.4g}")
            print(
                f"  ℹ️  {name}: light isotropic correction disabled; "
                f"keeping subdivided mesh even though {' and '.join(reasons)}."
            )
        return mesh

    if verbose:
        reasons = []
        if not mean_ok:
            reasons.append(f"rel_error={rel_err * 100:.2f}% outside ±{tol * 100:.2f}%")
        if not cv_ok:
            reasons.append(f"CV={stats['cv']:.4g} > max_cv={max_cv_value:.4g}")
        print(
            f"  🔧 {name}: applying light isotropic correction because "
            f"{' and '.join(reasons)}."
        )

    return target_triangle_size_with_diagnostics(
        mesh,
        name=f"{name} light correction",
        target_triangle_area=target_triangle_area,
        tolerance=tolerance,
        max_cv=max_cv_value,
        max_iters=remesh_max_iters,
        remesh_iters=remesh_iters,
        verbose=verbose,
    )


# ======================================================================================
# Automatic unsupported-edge trimming
# ======================================================================================
def _frame_atom_positions(frame_data):
    """Return every atom/bead coordinate from a parsed PDB frame as an (N, 3) array.

    Feeds the unsupported-edge trim's support check -- an edge/boundary
    triangle is only removed if no atom is found near it, so this is the
    full candidate pool that check searches against.

    """
    atoms = [] if frame_data is None else frame_data.get("atoms", [])
    coords = []
    for atom in atoms:
        try:
            coords.append((float(atom["x"]), float(atom["y"]), float(atom["z"])))
        except (KeyError, TypeError, ValueError):
            continue
    return np.asarray(coords, dtype=np.float64)


def _triangle_centroids_from_mesh(mesh):
    """Return each triangle's centroid (mean of its three vertices) as an (N, 3) array.

    Used to test each candidate edge triangle's local xy position against
    nearby atoms, and to find the corresponding point on the opposite
    leaflet's surface.

    Returns an (0, 3) array for an empty or triangle-less mesh, never None.
    """
    if mesh is None or (not mesh.has_vertices()) or (not mesh.has_triangles()):
        return np.zeros((0, 3), dtype=np.float64)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    t = np.asarray(mesh.triangles, dtype=np.int32)
    if len(t) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return v[t].mean(axis=1)


def _clean_mesh_after_triangle_removal(mesh):
    """Re-clean a mesh in place after triangles were dropped by a boolean mask.

    Removing triangles by mask (as the unsupported-edge trim does) can leave
    orphaned vertices, duplicate geometry, or newly-exposed non-manifold
    edges behind -- this runs the same sequence of Open3D cleanup calls used
    everywhere else in this file, so the result is consistent whichever
    function trimmed the mesh.

    Mutates mesh in place (Open3D's own convention) and returns the same
    object, for chaining.
    """
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh

def _mesh_boundary_shell_mask(tris, n_rings=2):
    """Mark triangles on or near an open mesh boundary, using topological BFS rings.

    A boundary edge is one shared by exactly one triangle (a manifold
    interior edge is shared by exactly two). Every triangle touching a
    boundary edge is ring 0; n_rings controls how many triangle-adjacency
    steps outward from there also get marked. n_rings=0 marks only the
    boundary triangles themselves, no expansion.

    Edges shared by more than two triangles (non-manifold) are treated
    conservatively: every triangle sharing that edge is linked as an
    adjacency neighbor, so the ring expansion doesn't stop short at a
    non-manifold seam.

    tris:    (T, 3) int array of triangle vertex indices.
    n_rings: BFS expansion depth from the boundary; 0 = boundary only.
    Returns: (T,) bool array, True where a triangle counts as boundary/near-boundary.
             All-False for a mesh with no open boundary (e.g. already closed).
    """
    tris = np.asarray(tris, dtype=np.int32)
    n_tri = len(tris)
    mask = np.zeros(n_tri, dtype=bool)
    if n_tri == 0:
        return mask

    edge_to_tris = defaultdict(list)
    neighbors = [set() for _ in range(n_tri)]

    for ti, (a, b, c) in enumerate(tris):
        edge_to_tris[tuple(sorted((int(a), int(b))))].append(ti)
        edge_to_tris[tuple(sorted((int(b), int(c))))].append(ti)
        edge_to_tris[tuple(sorted((int(c), int(a))))].append(ti)

    boundary = set()
    for ts in edge_to_tris.values():
        if len(ts) == 1:
            boundary.add(ts[0])
        elif len(ts) >= 2:
            # Normal manifold case has two triangles per edge. If more are
            # present, connect all of them so the ring expansion remains
            # conservative rather than stopping short at the seam.
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    neighbors[ts[i]].add(ts[j])
                    neighbors[ts[j]].add(ts[i])

    if not boundary:
        return mask

    max_ring = max(0, int(n_rings))
    q = deque((t, 0) for t in boundary)
    visited = set(boundary)
    for t in boundary:
        mask[int(t)] = True

    while q:
        t, depth = q.popleft()
        if depth >= max_ring:
            continue
        for nb in neighbors[t]:
            if nb in visited:
                continue
            visited.add(nb)
            mask[int(nb)] = True
            q.append((nb, depth + 1))

    return mask


def _point_to_polygon_boundary_distance(points_xy, polygon_xy):
    """Return each point's shortest distance to any edge segment of a 2D polygon.

    polygon_xy is treated as a closed loop -- the segment from the last
    vertex back to the first is included automatically. This measures
    distance to the boundary itself, not inside/outside membership; use
    alongside a separate inside/outside test (see _convex_hull_edge_shell_mask)
    when both matter.

    Processed in fixed-size chunks (20000 points) to bound peak memory when
    a leaflet mesh has a very large number of triangle centroids to test at
    once.

    points_xy:   (N, 2) array of query points.
    polygon_xy:  (M, 2) array of polygon vertices, in order.
    Returns: (N,) array of distances. All np.inf if polygon_xy has fewer
             than 2 vertices (not enough to define an edge).
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    polygon_xy = np.asarray(polygon_xy, dtype=np.float64)
    if len(points_xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(polygon_xy) < 2:
        return np.full(len(points_xy), np.inf, dtype=np.float64)

    a = polygon_xy
    b = np.roll(polygon_xy, -1, axis=0)
    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)
    ab2[ab2 <= 1e-30] = 1e-30

    out = np.full(len(points_xy), np.inf, dtype=np.float64)
    chunk = 20000
    for start in range(0, len(points_xy), chunk):
        p = points_xy[start:start + chunk]
        d2_min = np.full(len(p), np.inf, dtype=np.float64)
        for i in range(len(a)):
            ap = p - a[i]
            u = np.sum(ap * ab[i], axis=1) / ab2[i]
            u = np.clip(u, 0.0, 1.0)
            q = a[i] + u[:, None] * ab[i]
            diff = p - q
            d2 = np.sum(diff * diff, axis=1)
            d2_min = np.minimum(d2_min, d2)
        out[start:start + chunk] = np.sqrt(d2_min)
    return out

def _convex_hull_edge_shell_mask(centroids_xy, atom_xy, edge_band_width):
    """Mark centroids outside, or within edge_band_width of, the atom/bead xy convex hull.

    This is the xy-footprint half of "edge" -- a triangle centroid outside
    the physical extent of the membrane's own atoms, or close enough to the
    boundary to be suspect, is a candidate for the unsupported-edge trim.
    (See _mesh_boundary_shell_mask for the other half: topological mesh
    boundary, independent of atom positions.)

    edge_band_width=0.0 (the only value this pipeline ever passes -- see
    _unsupported_edge_trim_one_leaflet) means a centroid is only flagged if
    it is strictly outside the hull or lies exactly on its boundary; no
    inward margin is added.

    centroids_xy:    (N, 2) triangle centroid xy positions to test.
    atom_xy:         (M, 2) atom/bead xy positions defining the footprint;
                     needs >= 3 points for a real hull.
    edge_band_width: extra inward margin from the boundary that also counts
                     as "edge," in the same units as the coordinates (Å).
    Returns: (N,) bool array. All-False if there are too few centroids or
             atoms to define a meaningful hull.
    """
    centroids_xy = np.asarray(centroids_xy, dtype=np.float64)
    atom_xy = np.asarray(atom_xy, dtype=np.float64)
    shell = np.zeros(len(centroids_xy), dtype=bool)
    if len(centroids_xy) == 0 or len(atom_xy) < 3:
        return shell

    edge_band_width = max(0.0, float(edge_band_width))

    if ConvexHull is None:
        # Fallback: use a bbox shell if scipy.spatial.ConvexHull is unavailable.
        mn = atom_xy.min(axis=0)
        mx = atom_xy.max(axis=0)
        x, y = centroids_xy[:, 0], centroids_xy[:, 1]
        outside = (x < mn[0]) | (x > mx[0]) | (y < mn[1]) | (y > mx[1])
        dist_edge = np.minimum.reduce([x - mn[0], mx[0] - x, y - mn[1], mx[1] - y])
        return outside | (dist_edge <= edge_band_width)

    try:
        hull = ConvexHull(atom_xy)
    except Exception:
        return shell

    # ConvexHull equations define "inside" as A*x + b <= 0.
    eq = hull.equations
    signed = centroids_xy @ eq[:, :2].T + eq[:, 2]
    outside = np.any(signed > 1e-9, axis=1)

    polygon = atom_xy[hull.vertices]
    dist_boundary = _point_to_polygon_boundary_distance(centroids_xy, polygon)
    return outside | (dist_boundary <= edge_band_width)


def _unsupported_edge_trim_one_leaflet(mesh, opposite_mesh, atom_positions, name="mesh", verbose=True):
    """
    Remove unsupported artificial overhangs at the mesh edge.

    No user ratios or tolerance knobs are used. The rule is intentionally strict:
      1) only actual edge triangles are checked, and
      2) a checked triangle is removed only when no atom/bead is found locally
         between the current leaflet surface and the opposite leaflet surface.

    "Edge" means either:
      - outside/on the atom-bead xy footprint, with zero extra edge band, or
      - directly touching an open mesh boundary, with zero inward ring expansion.


    mesh:           the leaflet mesh to trim.
    opposite_mesh:  the other leaflet's mesh, used as the far side of the
                    "is there an atom between the two surfaces" test.
    atom_positions: (N, 3) array from _frame_atom_positions -- the full pool
                    of this frame's atom/bead coordinates.
    Returns: the trimmed mesh, or the input mesh unchanged if nothing
             qualified for removal (including every early-exit case below).
    """
    if mesh is None or opposite_mesh is None:
        return mesh
    if (not mesh.has_vertices()) or (not mesh.has_triangles()):
        if verbose:
            print(f"⚠️ {name}: empty mesh → skipping unsupported-edge trim.")
        return mesh
    if (not opposite_mesh.has_vertices()) or (not opposite_mesh.has_triangles()):
        if verbose:
            print(f"⚠️ {name}: opposite mesh is empty → skipping unsupported-edge trim.")
        return mesh

    atom_positions = np.asarray(atom_positions, dtype=np.float64)
    if atom_positions.ndim != 2 or atom_positions.shape[1] != 3 or len(atom_positions) == 0:
        if verbose:
            print(f"⚠️ {name}: no frame atoms/beads available → skipping unsupported-edge trim.")
        return mesh

    if cKDTree is None:
        if verbose:
            print(f"⚠️ {name}: scipy.spatial.cKDTree unavailable → skipping unsupported-edge trim.")
        return mesh

    v = np.asarray(mesh.vertices, dtype=np.float64)
    t = np.asarray(mesh.triangles, dtype=np.int32)
    cent = v[t].mean(axis=1)
    opp_cent = _triangle_centroids_from_mesh(opposite_mesh)
    if len(cent) == 0 or len(opp_cent) == 0:
        return mesh

    # Fixed minimal behavior. These are intentionally not exposed as options.
    # This matches the setting that worked best for the membrane edge artifacts:
    #   support radius = 0 Å
    #   edge band      = 0 Å
    #   z padding      = 0 Å
    #   boundary rings = 0
    near_hull_edge = _convex_hull_edge_shell_mask(
        centroids_xy=cent[:, :2],
        atom_xy=atom_positions[:, :2],
        edge_band_width=0.0,
    )
    near_mesh_boundary = _mesh_boundary_shell_mask(t, n_rings=0)
    near_edge = near_hull_edge | near_mesh_boundary

    candidate_ids = np.where(near_edge)[0]
    if candidate_ids.size == 0:
        if verbose:
            print(f"ℹ️ {name}: no boundary/edge triangles found → unsupported-edge trim skipped.")
        return mesh

    atom_tree = cKDTree(atom_positions[:, :2])
    opp_tree = cKDTree(opp_cent[:, :2])
    _, opp_idx = opp_tree.query(cent[candidate_ids, :2], k=1)

    remove_mask = np.zeros(len(t), dtype=bool)
    unsupported_count = 0

    for local_i, tri_id in enumerate(candidate_ids):
        tri_id = int(tri_id)
        c = cent[tri_id]

        # No support-radius expansion: only atoms/beads at the local xy point
        # are considered. This is intentionally strict for edge-only cleanup.
        atom_ids = atom_tree.query_ball_point(c[:2], r=0.0)
        if not atom_ids:
            unsupported_count += 1
            remove_mask[tri_id] = True
            continue

        oc = opp_cent[int(opp_idx[local_i])]
        zlo = min(c[2], oc[2])
        zhi = max(c[2], oc[2])
        zvals = atom_positions[np.asarray(atom_ids, dtype=np.int64), 2]
        has_between_atom = np.any((zvals >= zlo) & (zvals <= zhi))

        if not has_between_atom:
            unsupported_count += 1
            remove_mask[tri_id] = True

    n_remove = int(remove_mask.sum())
    n_tri = len(t)
    if n_remove == 0:
        if verbose:
            print(
                f"ℹ️ Unsupported-edge trim: {name}\n"
                f"  boundary/edge candidates={candidate_ids.size}, unsupported=0"
            )
        return mesh

    if n_remove >= n_tri:
        if verbose:
            print(f"⚠️ Unsupported-edge trim: {name} would remove ALL triangles → skipping.")
        return mesh

    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_triangles_by_mask(remove_mask)
    out = _clean_mesh_after_triangle_removal(out)

    if verbose:
        print(f"\n🧹 Unsupported-edge trim: {name}")
        print(
            "  rule: remove only actual edge/boundary triangles when no atom/bead "
            "exists between local upper/lower mesh surfaces"
        )
        print(
            f"  boundary/edge candidates={candidate_ids.size} | unsupported={unsupported_count} | "
            f"removed={n_remove}/{n_tri} ({(n_remove / max(1, n_tri)) * 100:.2f}%)"
        )
        print(f"  tris: {n_tri} → {len(np.asarray(out.triangles))}")

    return out


def unsupported_between_mesh_edge_trim(mesh_upper, mesh_lower, frame_data, frame_idx=0, enabled=True, verbose=True):
    """Apply the unsupported-edge trim to both leaflet meshes for one frame.

    enabled=False (or empty atom_positions) returns both input meshes
    completely unchanged, not empty meshes.

    Returns: (mesh_upper_trimmed, mesh_lower_trimmed).
    """
    if not enabled:
        if verbose:
            print("ℹ️ Unsupported-edge trim disabled.")
        return mesh_upper, mesh_lower

    atom_positions = _frame_atom_positions(frame_data)
    if atom_positions.size == 0:
        if verbose:
            print(f"⚠️ frame {frame_idx}: no atoms/beads found → unsupported-edge trim skipped.")
        return mesh_upper, mesh_lower

    mesh_upper_trim = _unsupported_edge_trim_one_leaflet(
        mesh=mesh_upper,
        opposite_mesh=mesh_lower,
        atom_positions=atom_positions,
        name=f"upper frame {frame_idx}",
        verbose=verbose,
    )
    mesh_lower_trim = _unsupported_edge_trim_one_leaflet(
        mesh=mesh_lower,
        opposite_mesh=mesh_upper_trim,
        atom_positions=atom_positions,
        name=f"lower frame {frame_idx}",
        verbose=verbose,
    )

    return mesh_upper_trim, mesh_lower_trim


# ======================================================================================
# Final manifold / connectivity QC
# ======================================================================================

def repair_leaflet_mesh_manifold(mesh, name="mesh", verbose=True):
    """
    Final Step-3 QC pass: repair non-manifold vertices and drop disconnected
    islands, so every leaflet mesh saved to the Step-3 NPZ is a single
    manifold component before any downstream step (4-9) ever reads it.

    Every remeshing helper in this file already calls Open3D's
    remove_non_manifold_edges() after it runs. That only fixes edges shared
    by >2 triangles -- it does nothing for a non-manifold VERTEX (a "bowtie"
    point where two otherwise-separate fan regions touch at one vertex with
    no shared edge), and it does nothing for a boundary-triangle removal that
    pinches off a small disconnected island. Those two cases are exactly what
    Step 9's curvature filter has been skipping leaflets for.

    Fixing both here is safe in a way it is NOT safe in Step 9: Step 3 is the
    source of the face array, so anything dropped or renumbered here becomes
    the mesh every downstream step computes tri_ids against from the start --
    there is no already-computed cluster tri_id to invalidate.

    Order:
      1) PyMeshLab meshing_repair_non_manifold_vertices(vertdispratio=0) --
         topology-only vertex split, the same call already validated in
         lib/curvature.py's _run_curvature_filter. 
      2) Open3D remove_non_manifold_edges() -- cheap idempotent re-check in
         case the vertex split exposed a non-manifold edge.
      3) Open3D cluster_connected_triangles() -- if more than one component
         remains, keep only the largest by triangle count and drop the rest.
      4) Final manifold/connectivity check, logged. Should now always pass;
         if it somehow doesn't, the leaflet is returned as-is with a loud
         warning rather than silently handed to Step 4 broken.
    """
    if mesh is None or (not mesh.has_vertices()) or (not mesh.has_triangles()):
        if verbose:
            print(f"⚠️ {name}: empty mesh → skipping manifold QC.")
        return mesh

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    nV0, nT0 = len(verts), len(tris)

    # 1) Non-manifold vertex repair (PyMeshLab, topology-only split)
    try:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=verts, face_matrix=tris))
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        out = ms.current_mesh()
        verts = out.vertex_matrix().astype(np.float64)
        tris = out.face_matrix().astype(np.int32)
    except Exception as exc:
        if verbose:
            print(f"⚠️ {name}: non-manifold-vertex repair failed ({exc}); continuing with unrepaired mesh.")

    repaired = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )

    # 2) Non-manifold edge re-check
    repaired.remove_non_manifold_edges()
    repaired.remove_unreferenced_vertices()

    # 3) Keep only the largest connected component
    if len(np.asarray(repaired.triangles)) > 0:
        cluster_idx, cluster_n_tris, _ = repaired.cluster_connected_triangles()
        cluster_idx = np.asarray(cluster_idx)
        cluster_n_tris = np.asarray(cluster_n_tris)
        n_components = len(cluster_n_tris)
        if n_components > 1:
            largest = int(np.argmax(cluster_n_tris))
            remove_mask = cluster_idx != largest
            dropped_sizes = sorted(
                (int(n) for i, n in enumerate(cluster_n_tris) if i != largest),
                reverse=True,
            )
            repaired.remove_triangles_by_mask(remove_mask)
            repaired.remove_unreferenced_vertices()
            if verbose:
                print(
                    f"🩹 {name}: dropped {n_components - 1} disconnected island(s) "
                    f"after manifold repair, sizes={dropped_sizes} triangles "
                    f"(kept largest: {int(cluster_n_tris[largest])} triangles)."
                )

    repaired.compute_triangle_normals()
    repaired.compute_vertex_normals()

    # 4) Final check, logged
    is_ok = repaired.is_edge_manifold() and repaired.is_vertex_manifold()
    _, final_n_tris, _ = repaired.cluster_connected_triangles()
    nV1, nT1 = len(np.asarray(repaired.vertices)), len(np.asarray(repaired.triangles))

    if verbose:
        print(f"🩹 Manifold QC: {name}")
        print(f"  verts: {nV0} → {nV1} | tris: {nT0} → {nT1}")
        print(f"  manifold={is_ok} | components={len(final_n_tris)}")

    if not is_ok or len(final_n_tris) != 1:
        print(
            f"⚠️ {name}: STILL non-manifold or multi-component after repair "
            f"(manifold={is_ok}, components={len(final_n_tris)}). "
            "Handing to Step 4 as-is -- Step 9's own skip logic is the fallback."
        )

    return repaired


# ======================================================================================
# Per-frame processing: LeafletFinder wired + automatic triangle sizing + unsupported-edge trim
# ======================================================================================

def process_pdb_to_meshes(
    frame_data,
    output_dir_upper,
    output_dir_lower,
    frame_idx=0,
    pdb_path=None,
    leaflet_select=None,
    leaflet_cutoff=15.0,
    leaflet_pbc=False,
    atom_name_filter=None,
    res_name_filter=None,
    target_triangle_area=5.0,
    triangle_tolerance=0.20,
    triangle_max_cv=0.25,
    triangle_method="hybrid",
    triangle_max_iters=2,
    triangle_remesh_iters=5,
    triangle_max_subdivide_iters=6,
    triangle_auto_subdivide_method="loop",
    triangle_remesh_if_needed=True,
    auto_edge_trim=True,
    manifold_qc=True,
):
    """Run the full Step 3 pipeline for one frame: leaflet split -> mesh -> size -> trim -> QC -> save.

    This is the single entry point helpers.py's run_step3_mesh_generation
    calls per frame. Stage order:

      1. Leaflet split + Poisson reconstruction (create_leaflet_meshes),
         via LeafletFinder when pdb_path + leaflet_select are given
         (always true in the current pipeline -- USE_LEAFLET_FINDER is
         fixed True in main.py), else the frame_data mean-z fallback.
      2. Triangle-size control toward target_triangle_area.
      3. Unsupported-edge trim (unsupported_between_mesh_edge_trim).
      4. Manifold/connectivity QC (repair_leaflet_mesh_manifold), if enabled.
      5. Save both leaflets as .ply.


    Returns: (mesh_upper, mesh_lower), the final Open3D meshes after every
             stage -- the same objects written to upper_path/lower_path.
    """
    os.makedirs(output_dir_upper, exist_ok=True)
    os.makedirs(output_dir_lower, exist_ok=True)

    # Use LeafletFinder if pdb_path + leaflet_select provided; else fallback.
    if pdb_path is not None and leaflet_select is not None:
        up_pos, lo_pos = leaflet_positions_from_pdb(
            pdb_path=pdb_path,
            select=leaflet_select,
            cutoff=leaflet_cutoff,
            pbc=leaflet_pbc,
        )
        mesh_upper, mesh_lower, _, _ = create_leaflet_meshes(
            upper_positions=up_pos,
            lower_positions=lo_pos,
        )
    else:
        mesh_upper, mesh_lower, _, _ = create_leaflet_meshes(
            frame_data=frame_data,
            atom_name_filter=atom_name_filter,
            res_name_filter=res_name_filter,
        )

    # -------------------------------------------------
    # Automatic triangle-size control
    # -------------------------------------------------
    # These are internal code-level settings now. The input file only controls
    # triangle_size and triangle_max_cv.
    triangle_method_n = "hybrid"
    triangle_auto_subdivide_method = "loop"
    triangle_remesh_if_needed = True

    def _apply_triangle_control(mesh, leaflet_name):
        if triangle_method_n == "hybrid":
            return hybrid_target_triangle_size_with_diagnostics(
                mesh,
                name=f"{leaflet_name} frame {frame_idx}",
                target_triangle_area=target_triangle_area,
                tolerance=triangle_tolerance,
                max_cv=triangle_max_cv,
                max_subdivide_iters=triangle_max_subdivide_iters,
                subdivide_method=triangle_auto_subdivide_method,
                remesh_if_needed=triangle_remesh_if_needed,
                remesh_max_iters=triangle_max_iters,
                remesh_iters=triangle_remesh_iters,
            )
        if triangle_method_n in {"subdivide", "subdivide_auto", "auto_subdivide"}:
            return auto_subdivide_to_target_triangle_size_with_diagnostics(
                mesh,
                name=f"{leaflet_name} frame {frame_idx}",
                target_triangle_area=target_triangle_area,
                tolerance=triangle_tolerance,
                max_cv=triangle_max_cv,
                max_subdivide_iters=triangle_max_subdivide_iters,
                subdivide_method=triangle_auto_subdivide_method,
            )
        if triangle_method_n in {"isotropic", "remesh"}:
            return target_triangle_size_with_diagnostics(
                mesh,
                name=f"{leaflet_name} frame {frame_idx}",
                target_triangle_area=target_triangle_area,
                tolerance=triangle_tolerance,
                max_cv=triangle_max_cv,
                max_iters=max(1, int(triangle_max_iters)),
                remesh_iters=max(1, int(triangle_remesh_iters)),
            )
        raise ValueError(
            "triangle_method must be one of: hybrid, subdivide_auto, isotropic"
        )

    mesh_upper = _apply_triangle_control(mesh_upper, "upper")
    mesh_lower = _apply_triangle_control(mesh_lower, "lower")

    # -------------------------------------------------
    # Automatic unsupported-edge trim
    # -------------------------------------------------
    mesh_upper, mesh_lower = unsupported_between_mesh_edge_trim(
        mesh_upper=mesh_upper,
        mesh_lower=mesh_lower,
        frame_data=frame_data,
        frame_idx=frame_idx,
        enabled=auto_edge_trim,
        verbose=True,
    )

    # -------------------------------------------------
    # Final manifold/connectivity QC
    # -------------------------------------------------
    if manifold_qc:
        mesh_upper = repair_leaflet_mesh_manifold(mesh_upper, name=f"upper frame {frame_idx}", verbose=True)
        mesh_lower = repair_leaflet_mesh_manifold(mesh_lower, name=f"lower frame {frame_idx}", verbose=True)

    upper_final_stats = _triangle_area_stats(mesh_upper)
    lower_final_stats = _triangle_area_stats(mesh_lower)
    print("\n📐 Final triangle stats after unsupported-edge trim:")
    _print_triangle_stats(f"  upper frame {frame_idx}", upper_final_stats)
    _print_triangle_stats(f"  lower frame {frame_idx}", lower_final_stats)

    # Save
    upper_path = os.path.join(output_dir_upper, f"upper_mesh_frame_{frame_idx}.ply")
    lower_path = os.path.join(output_dir_lower, f"lower_mesh_frame_{frame_idx}.ply")

    o3d.io.write_triangle_mesh(upper_path, mesh_upper, write_ascii=False)
    o3d.io.write_triangle_mesh(lower_path, mesh_lower, write_ascii=False)

    print(f"  → Saved upper mesh to: {upper_path}")
    print(f"  → Saved lower mesh to: {lower_path}")

    return mesh_upper, mesh_lower


# ======================================================================================
# Resume helper
# ======================================================================================
def get_completed_frames(npz_dir):
    """Return the set of frame indices already processed, for Step 3 resume.

    A frame counts as done if npz_dir contains a file named
    frame_<idx>_mesh_predefect.npz. 

    Returns: set of int frame indices.
    """
    completed = set()
    if not os.path.isdir(npz_dir):
        return completed

    for fname in os.listdir(npz_dir):
        if fname.startswith("frame_") and fname.endswith("_mesh_predefect.npz"):
            try:
                idx = int(fname.split("_")[1])
                completed.add(idx)
            except ValueError:
                pass
    return completed







