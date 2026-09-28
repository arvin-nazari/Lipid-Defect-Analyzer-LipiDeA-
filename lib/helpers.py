"""Pipeline orchestration: one run_stepN_* function per pipeline step.

Called exclusively from main.py's main(), which drives the whole run:
    if should_run_step(N, resume_from_step): run_stepN_...(...)
for N in 1..9, plus run_post_pipeline_cleanup() and write_step_timings_txt()
at the end. This file owns no step's actual algorithm -- each run_stepN_*
function is a thin layer that reads prep.in config, prints a step header,
and calls into the real implementation living in lib/prep.py, lib/mesh.py,
lib/defects.py, lib/cleanup.py, lib/analysis.py, lib/classify.py, or
lib/curvature.py.

Two different import styles are used for those modules, deliberately:
  from .prep import *      } functions from these three land directly in
  from .utilities import * } this module's namespace -- e.g. get_value(...),
  from .mesh import *      } cfg_path(...), process_pdb_to_meshes(...) are
                            } called bare, with no module prefix.
  from . import defects as defect_tools    } these four are namespaced --
  from . import cleanup as cleanup_tools   } defect_tools.xxx(...),
  from . import analysis as analysis_tools } cleanup_tools.xxx(...), etc.
  from . import classify as classify_tools }
  from . import curvature as curvature_tools

Workflow constants (RESULTS_DIR, MESH_NPZ_DIR, DEFECT_TRI_K, and every other
ALL_CAPS name referenced but not defined in this file) are not defined
here -- they live in main.py and are injected into this module's globals by
configure_pipeline_helpers, called once at the start of main.py's main().
"""


from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import logging
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from .prep import *
from .utilities import *
from .mesh import *
from . import defects as defect_tools
from . import cleanup as cleanup_tools
from . import analysis as analysis_tools
from . import classify as classify_tools
from . import curvature as curvature_tools
from . import progress


def apply_user_config_overrides(cfg):
    """Read the small set of user-facing options from prep.in that override a main.py default.

    Only two exist: triangle_kd_tree (Step 4's DEFECT_TRI_K) and cpu_workers/
    workers (CPU_WORKERS, used across Steps 2/5/6/8/9). Called once by
    main.py's main(), before configure_pipeline_helpers exports these
    globals to every other module -- so the override must land here first.
    """
    global DEFECT_TRI_K, CPU_WORKERS

    DEFECT_TRI_K = max(
        1,
        int(get_value(cfg, "triangle_kd_tree", default=DEFECT_TRI_K))
    )

    CPU_WORKERS = max(
        1,
        int(get_value(cfg, "cpu_workers", "workers", default=CPU_WORKERS))
    )


def configure_pipeline_helpers(settings):
    """Expose main.py's internal settings to every function in this module.

    main.py is intentionally the single place where workflow constants
    live. This function is how they reach helpers.py: main.py calls it once
    at startup with a SimpleNamespace built from its own ALL_CAPS globals,
    and every constant referenced-but-undefined throughout this file (e.g.
    RESULTS_DIR, MESH_NPZ_DIR, DEFECT_TRI_K) resolves through this injection.
    """
    globals().update(vars(settings))



# ==================================================
# Mesh input-file settings
# ==================================================

def _strip_optional_quotes(value):
    """Remove one matching pair of surrounding quotes from a config value.

    Lets a prep.in value be written either bare or quoted
    (custom_ff: input/forcefield vs custom_ff: "input/forcefield") without
    the quote characters leaking into the parsed value.
    """
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _selection_list(value):
    """Parse a config value into a list of tokens suitable for an MDAnalysis selection.

    Thin wrapper around as_list (lib/utilities.py) that also strips one
    layer of optional surrounding quotes first, via _strip_optional_quotes.
    """
    items = as_list(_strip_optional_quotes(value))
    return [x for x in items if x]


def build_leaflet_select(res_name_value, atom_name_value):
    """Build the LeafletFinder MDAnalysis selection string from prep.in's res_name and atom_name.

    e.g. res_name="POPC,DOPE,SAPI", atom_name="GL2" ->
         "(resname POPC DOPE SAPI) and name GL2"

    Raises ValueError if either input parses to an empty list -- an empty
    LeafletFinder selection would match zero atoms and fail confusingly
    much later, inside mesh.py's leaflet_positions_from_pdb, rather than
    here with a clear cause.
    """
    res_names = _selection_list(res_name_value)
    atom_names = _selection_list(atom_name_value)

    if not res_names:
        raise ValueError("res_name must contain at least one residue name for LeafletFinder.")
    if not atom_names:
        raise ValueError("atom_name must contain at least one atom/bead name for LeafletFinder.")

    return f"(resname {' '.join(res_names)}) and name {' '.join(atom_names)}"



def parse_arguments(cfg, config_dir, prepared_input_dir):
    """Build the SimpleNamespace of settings Step 3 (mesh generation) needs.

    Mesh-generation settings, default and user-facing.

    input on the returned namespace is the prepared/cut PDB folder Step 2
    produced (prepared_input_dir), not the original multi-frame trajectory
    PDB the user gave as prep.in's `input` key -- easy to confuse since both
    are called "input" in different contexts.
    """
    mesh_input = Path(prepared_input_dir)

    atom_name_value = _strip_optional_quotes(get_value(cfg, "atom_name", default="GL2"))
    res_name_value = _strip_optional_quotes(get_value(cfg, "res_name", default="POPC,DOPE,SAPI"))
    leaflet_select_value = build_leaflet_select(res_name_value, atom_name_value)

    return SimpleNamespace(
        # The prepared/cut PDB folder produced by Step 2, not the original
        # multi-frame trajectory PDB.
        input=str(mesh_input),

        # User-facing mesh selection values.
        atom_name=atom_name_value,
        res_name=res_name_value,

        # LeafletFinder is fixed/internal. The selection is generated from
        # res_name + atom_name above.
        use_leaflet_finder=USE_LEAFLET_FINDER,
        leaflet_select=leaflet_select_value,
        leaflet_cutoff=LEAFLET_CUTOFF,
        leaflet_pbc=LEAFLET_PBC,

        # Automatic triangle-size control. User-facing options:
        #   triangle_size   -> target mean triangle area in Å^2
        #   triangle_max_cv -> triangle-area uniformity gate
        # Everything else is intentionally controlled by the internal constants
        # near the top of this file.
        triangle_size=float(get_value(cfg, "triangle_size", "triangle_area", default=1.0)),
        triangle_max_cv=float(get_value(cfg, "triangle_max_cv", default=0.25)),
        min_triangle_count=max(
            1,
            int(get_value(cfg, "min_triangle_count", default=MIN_TRIANGLE_COUNT_DEFAULT)),
        ),
        triangle_tolerance=TRIANGLE_TOLERANCE,
        triangle_method=TRIANGLE_METHOD,
        triangle_max_iters=TRIANGLE_MAX_ITERS,
        triangle_remesh_iters=TRIANGLE_REMESH_ITERS,
        triangle_max_subdivide_iters=TRIANGLE_MAX_SUBDIVIDE_ITERS,
        triangle_auto_subdivide_method=TRIANGLE_AUTO_SUBDIVIDE_METHOD,
        triangle_remesh_if_needed=TRIANGLE_REMESH_IF_NEEDED,

        # Automatic unsupported-edge trimming.
        auto_edge_trim=AUTO_EDGE_TRIM,
        manifold_qc=MESH_MANIFOLD_QC,

        # Mesh-frame resume is internal and always ON. The user-facing
        # resume_from key now controls pipeline steps, not frame indices.
        resume=MESH_RESUME,
        resume_from=None,
    )


# ==================================================
# Pipeline step-resume helpers
# ==================================================


def parse_resume_from_step(value):
    """Parse a user-facing resume_from value ('step 2', 'mesh generation', or a bare number) into a step number.

    Checks STEP_ALIASES (main.py) first for named forms, then falls back to
    parsing "step N" or a bare N

    Returns 1 (the pipeline start) for an empty or missing value.
    Raises ValueError for anything unrecognized, listing every valid
    "step N" form from PIPELINE_STEPS.
    """
    if value is None or str(value).strip() == "":
        return 1

    text = str(value).strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())

    if text in STEP_ALIASES:
        return STEP_ALIASES[text]

    max_step = max(PIPELINE_STEPS)

    if text.startswith("step "):
        maybe_num = text.split("step ", 1)[1].strip()
        if maybe_num.isdigit() and 1 <= int(maybe_num) <= max_step:
            return int(maybe_num)

    if text.isdigit() and 1 <= int(text) <= max_step:
        return int(text)

    valid = ", ".join(f"step {k}" for k in sorted(PIPELINE_STEPS))
    raise ValueError(f"Invalid resume_from value: {value!r}. Use one of: {valid}.")

def should_run_step(step_number, resume_from_step):
    """True if step_number is at or after the resume point -- the gate every run_stepN_* call in main.py uses."""
    return int(step_number) >= int(resume_from_step)

def print_step_header(step_number, title):
    """Print the boxed 'Step N: Title' banner every run_stepN_* function starts with."""
    print("\n==============================")
    print(f"Step {step_number}: {title}")
    print("==============================")

def get_min_triangle_count(cfg):
    """Read prep.in's min_triangle_count, shared as-is by Steps 5 and 6 (dedup/prune cluster-size floor)."""
    return max(
        1,
        int(get_value(cfg, "min_triangle_count", default=MIN_TRIANGLE_COUNT_DEFAULT)),
    )

def is_segfault_returncode(returncode):
    """True if a subprocess return code indicates it was killed by SIGSEGV.

    Covers all three forms a segfault can surface as, depending on platform
    and how the return code was captured: the raw negative signal number,
    the shell convention of 128+signal, and the specific value 139 (the
    concrete 128+SIGSEGV on Linux, spelled out since SIGSEGV's numeric
    value isn't otherwise guaranteed portable). Used only by
    run_step3_mesh_generation_with_failsafe to decide whether a Step 3
    subprocess crash should trigger a silent restart.
    """
    return returncode in {-signal.SIGSEGV, 128 + signal.SIGSEGV, 139}


# ==================================================
# Step implementations
# ==================================================

# ---------- Step 1: Atom CSV ----------

def run_step1_atom_csv(
    cfg,
    config_path,
    config_dir,
    script_dir,
    input_pdb,
    input_dir,
    output_csv,
    model,
):
    """Step 1: build results/atom_info.csv from the first frame of the trajectory.

    Resolves custom_ff (prep.in) into a type_folder/radius_folder pair --
    either the user's own custom_ff directory (both type and radius files
    expected in the same folder) or the built-in lib/lists/ff/<model>/
    {type,radius} defaults. "off"/"0"/"false"/"f"/"no"/"n" (case-insensitive)
    all mean "use the built-ins"; anything else non-empty is treated as a
    folder path.

    Delegates the actual CSV-building to prep.py's write_atom_info_csv --
    this function is config resolution, validation, and reporting around
    that call.
    """
    print_step_header(1, "Atom CSV")

    pdb_file = find_default_representative_pdb(input_dir, FRAME_PATTERN)
    base = script_dir / "lib" / "lists" / "ff" / model

    custom_ff_value = get_value(cfg, "custom_ff", default="")
    custom_ff_text = str(custom_ff_value).strip()

    if custom_ff_text and custom_ff_text.lower() not in {"0", "false", "f", "no", "n", "off"}:
        custom_ff = True
        custom_ff_folder = cfg_path(custom_ff_text, config_dir)
        type_folder = custom_ff_folder
        radius_folder = custom_ff_folder
    else:
        custom_ff = False
        custom_ff_folder = None
        type_folder = base / "type"
        radius_folder = base / "radius"

    neutral_values = as_list(get_value(cfg, "neutral", default=""))
    neutral_resnames = parse_neutral_resnames(neutral_values, model)

    if not pdb_file.exists():
        raise FileNotFoundError(f"Representative PDB not found: {pdb_file.resolve()}")
    if not type_folder.exists():
        raise FileNotFoundError(f"Type folder not found: {type_folder.resolve()}")
    if not radius_folder.exists():
        raise FileNotFoundError(f"Radius folder not found: {radius_folder.resolve()}")

    counts, unknown_rows, n_pairs, ff_file_used = write_atom_info_csv(
        pdb_file=pdb_file,
        output_csv=output_csv,
        model=model,
        type_folder=type_folder,
        radius_folder=radius_folder,
        neutral_resnames=neutral_resnames,
    )

    print(f"Config file:      {config_path}")
    print(f"Input PDB:        {input_pdb.resolve()}")
    print(f"First frame used: {pdb_file.resolve()}")
    print(f"Output CSV:       {output_csv.resolve()}")
    print(f"Unique pairs:     {n_pairs}")
    print(f"Custom FF:        {custom_ff}")
    if custom_ff_folder is not None:
        print(f"Custom FF folder: {custom_ff_folder.resolve()}")
    print(f"Type folder:      {type_folder.resolve()}")
    print(f"Radius folder:    {radius_folder.resolve()}")
    if ff_file_used:
        print("Martini FF files with radii:")
        for one_ff_file in ff_file_used:
            print(f"  {one_ff_file.resolve()}")

    if neutral_resnames:
        print(f"Neutral lipids:   {', '.join(sorted(neutral_resnames))}")
    else:
        print("Neutral lipids:   none")

    print("\nFlag counts:")
    for flag, n in counts.items():
        print(f"  {flag}: {n}")

    if unknown_rows:
        print(f"\nWARNING: {len(unknown_rows)} pairs had UNKNOWN atom_type.")
        print("First few UNKNOWN pairs:")
        for resname, atom_name in unknown_rows[:20]:
            print(f"  {resname:8s} {atom_name}")



# ---------- Step 2: Tiling and cutting ----------

def run_step2_tiling_and_cutting(input_dir, output_dir, tiled_dir, frame_box_xyz):
    """Step 2: tile each frame 3x3 and cut a centered window back out.

    Fixed geometry (tile count, cut scale) is not user-facing.
    """
    print_step_header(2, "Tiling and cutting")
    print(f"Fixed tiling:      {TILE_NX} x {TILE_NY}")
    print(f"Fixed cut scale:   {CUT_SCALE:.3f}x each frame's box_size.csv box in x/y")
    print("Center rule:       original/tiled/cut use the original atom-coordinate center")
    print("Box source:        per-frame results/box_size.csv (X, Y, Z; atom bounding box)")
    print(f"CPU workers:       {CPU_WORKERS}")
    print(f"Keep tiled files:  {KEEP_TILED}")
    if KEEP_TILED:
        print(f"Tiled folder:      {tiled_dir.resolve()}")

    n_ok, n_total = process_pdb_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        cut_scale=CUT_SCALE,
        nx=TILE_NX,
        ny=TILE_NY,
        pattern=FRAME_PATTERN,
        keep_tiled=KEEP_TILED,
        tiled_dir=tiled_dir,
        workers=CPU_WORKERS,
        frame_box_xyz=frame_box_xyz,
    )

    print(f"\nDone: {n_ok}/{n_total} prepared PDB files written to {output_dir.resolve()}")



# ---------- Step 3: Mesh generation ----------

def run_step3_mesh_generation(cfg, config_dir, output_dir):
    """Step 3: leaflet mesh generation for every prepared frame.

    Runs inside the isolated mesh-worker subprocess (see main.py's
    is_mesh_worker_process check and run_step3_mesh_generation_with_failsafe
    below) -- never called directly from the normal pipeline flow, so a
    PyMeshLab/Open3D segfault here kills only this subprocess, not the
    whole run.

    The progress bar for Step 3 is drawn from here, not from main()'s parent
    process: only this side knows the frame count, and only this side sees a
    frame finish. Every path out of the frame loop ticks exactly once --
    resume-skip, mesh failure, and success alike -- so the bar reaches its
    total on any run that completes.
    """
    print_step_header(3, "Mesh generation")

    args = parse_arguments(cfg, config_dir, output_dir)

    # Fallback selection lists. LeafletFinder is the default, but these are kept
    # for the fallback path inside process_pdb_to_meshes.
    atom_name_filter = _selection_list(args.atom_name)
    res_name_filter = _selection_list(args.res_name)

    upper_mesh_dir = MESH_UPPER_DIR
    lower_mesh_dir = MESH_LOWER_DIR
    npz_dir = MESH_NPZ_DIR

    os.makedirs(upper_mesh_dir, exist_ok=True)
    os.makedirs(lower_mesh_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    if args.use_leaflet_finder and not args.leaflet_select:
        raise ValueError("LeafletFinder selection could not be generated from res_name and atom_name.")

    print(f"📁 Reading PDB frames from: {args.input}")
    print(f"📦 Saving upper meshes to: {upper_mesh_dir}")
    print(f"📦 Saving lower meshes to: {lower_mesh_dir}")
    print(f"💾 Saving per-frame NPZs to: {npz_dir}")
    print(f"LeafletFinder: {'ON' if args.use_leaflet_finder else 'OFF'}")
    if args.use_leaflet_finder:
        print(f"  select='{args.leaflet_select}', cutoff={args.leaflet_cutoff}, pbc={args.leaflet_pbc}")
    else:
        print(f"  atom_name={atom_name_filter}, res_name={res_name_filter}")

    print("Triangle sizing:")
    print(f"  target mean area={args.triangle_size} Å^2")
    if args.triangle_max_cv > 0:
        print(f"  max CV={args.triangle_max_cv}")
    else:
        print("  max CV=disabled")
    print("  internal method=hybrid: loop subdivision + light isotropic correction if needed")
    print("Automatic edge trim:")
    print(f"  unsupported-between-mesh boundary rule={'ON' if args.auto_edge_trim else 'OFF'}")
    print(
        "  remove only actual mesh-boundary triangles when no atom/bead exists "
        "between the local upper/lower mesh surfaces"
    )
    print(
    f"Manifold QC (non-manifold vertex repair + disconnected-island drop): "
    f"{'ON' if args.manifold_qc else 'OFF'}"
    )

    t0 = time.perf_counter()
    completed_frames = get_completed_frames(npz_dir) if args.resume else set()

    # Bar total. The frame loop below iterates a generator, so len() is not
    # available; count the prepared PDBs on disk instead.
    progress.set_total(len(glob.glob(os.path.join(str(args.input), FRAME_PATTERN))))

    for frame_idx, frame_data, pdb_file, pdb_path in parse_pdb_files(args.input):
        if args.resume and frame_idx in completed_frames:
            print(f"⏭️  Frame {frame_idx} already processed — skipping")
            progress.tick()
            continue

        print(f"\n🟦 Processing frame {frame_idx} ({pdb_file})")

        mesh_upper, mesh_lower = process_pdb_to_meshes(
            frame_data=frame_data,
            output_dir_upper=upper_mesh_dir,
            output_dir_lower=lower_mesh_dir,
            frame_idx=frame_idx,

            # LeafletFinder
            pdb_path=(pdb_path if args.use_leaflet_finder else None),
            leaflet_select=(args.leaflet_select if args.use_leaflet_finder else None),
            leaflet_cutoff=args.leaflet_cutoff,
            leaflet_pbc=args.leaflet_pbc,

            # fallback selection
            atom_name_filter=atom_name_filter,
            res_name_filter=res_name_filter,

            # Automatic triangle-size control
            target_triangle_area=args.triangle_size,
            triangle_tolerance=args.triangle_tolerance,
            triangle_max_cv=args.triangle_max_cv,
            triangle_method=args.triangle_method,
            triangle_max_iters=args.triangle_max_iters,
            triangle_remesh_iters=args.triangle_remesh_iters,
            triangle_max_subdivide_iters=args.triangle_max_subdivide_iters,
            triangle_auto_subdivide_method=args.triangle_auto_subdivide_method,
            triangle_remesh_if_needed=args.triangle_remesh_if_needed,

            # automatic unsupported-edge trimming
            auto_edge_trim=args.auto_edge_trim,

            # final manifold/connectivity QC
            manifold_qc=args.manifold_qc,
        )

        if (
            mesh_upper is None
            or mesh_lower is None
            or (not mesh_upper.has_triangles())
            or (not mesh_lower.has_triangles())
        ):
            print(f"⚠️  Frame {frame_idx}: No mesh generated, skipping NPZ.")
            progress.tick()
            continue

        # ----------------
        # Extract arrays (triangle normals + centroids)
        # ----------------
        mesh_upper.compute_triangle_normals()
        verts_up = np.asarray(mesh_upper.vertices)
        tris_up = np.asarray(mesh_upper.triangles)
        normals_up = np.asarray(mesh_upper.triangle_normals)
        cent_up = np.mean(verts_up[tris_up], axis=1)

        mesh_lower.compute_triangle_normals()
        verts_lo = np.asarray(mesh_lower.vertices)
        tris_lo = np.asarray(mesh_lower.triangles)
        normals_lo = np.asarray(mesh_lower.triangle_normals)
        cent_lo = np.mean(verts_lo[tris_lo], axis=1)

        all_centroids = np.vstack([cent_up, cent_lo])
        all_normals = np.vstack([normals_up, normals_lo])
        n_up = np.int32(len(cent_up))

        frame_npz = {
            "verts_up": verts_up.astype(np.float32),
            "tris_up": tris_up.astype(np.int32),
            "normals_up": normals_up.astype(np.float32),
            "centroids_up": cent_up.astype(np.float32),

            "verts_lo": verts_lo.astype(np.float32),
            "tris_lo": tris_lo.astype(np.int32),
            "normals_lo": normals_lo.astype(np.float32),
            "centroids_lo": cent_lo.astype(np.float32),

            "all_centroids": all_centroids.astype(np.float32),
            "all_normals": all_normals.astype(np.float32),
            "n_up": n_up,

            "frame_idx": np.int32(frame_idx),
            "pdb_file": np.bytes_(pdb_file),
        }

        frame_npz_path = os.path.join(npz_dir, f"frame_{frame_idx:04d}_mesh_predefect.npz")
        np.savez(frame_npz_path, **frame_npz)
        print(f"💾 Saved per-frame NPZ → {frame_npz_path}")

        progress.tick()

    print(f"\n⏱️ Mesh processing time: {(time.perf_counter() - t0) / 60:.2f} minutes")
    print("🎉 Mesh generation finished successfully.")

    

def run_step3_mesh_generation_with_failsafe(config_path, script_path, worker_env, max_restarts):
    """Run Step 3 in a child process so a segfault can be restarted instead of killing the whole run.

    Python cannot catch a C/C++ segfault inside the same process -- PyMeshLab
    and Open3D are both compiled extensions capable of segfaulting on a
    pathological mesh. Running Step 3 in a subprocess lets this parent
    process detect that crash (via is_segfault_returncode) from the outside
    and restart it, up to max_restarts times, instead of the whole pipeline
    dying.

    worker_env is set to "1" in the child's environment; main.py's
    is_mesh_worker_process() checks for exactly that, and calls
    run_step3_mesh_generation directly instead of running the normal
    pipeline flow when it's set. Mesh-frame resume is always on internally,
    so a restarted Step 3 skips every frame NPZ already written before the
    crash -- no work is repeated except the one frame that was in flight.

    The child's stdout is handed the run log explicitly in quiet mode. It has
    to be: subprocess inherits the OS-level file descriptor, not the Python
    sys.stdout object lib/progress.py swapped, so redirection in this process
    would otherwise have no effect on the child. progress.log_stream()
    returns None in verbose mode, which is subprocess's own "inherit".
    Stderr is never redirected, so the child's progress bar reaches the
    terminal either way. On a segfault restart the bar restarts from the
    resume point rather than from zero.

    Raises RuntimeError if Step 3 segfaults more than max_restarts times, or
    if it fails with any non-segfault, non-zero return code (that failure
    mode is never retried, since a silent restart could mask a real bug
    behind an apparent success).
    """
    script_path = Path(script_path).resolve()
    attempts = 0

    while True:
        env = os.environ.copy()
        env[worker_env] = "1"

        proc = subprocess.run(
            [sys.executable, str(script_path), str(config_path)],
            env=env,
            stdout=progress.log_stream(),
        )

        if proc.returncode == 0:
            return

        if is_segfault_returncode(proc.returncode) and attempts < int(max_restarts):
            attempts += 1
            # Silent restart: the completed-frame resume logic inside Step 3
            # will skip already written mesh NPZs.
            continue

        if is_segfault_returncode(proc.returncode):
            raise RuntimeError(
                f"Mesh generation segfaulted {attempts + 1} times. "
                "The mesh resume files were kept, but the failing frame likely needs inspection."
            )

        raise RuntimeError(f"Mesh generation failed with return code {proc.returncode}.")

# ---------- Step 4: Defect finding ----------

def defect_radius_npz_path(output_npz_dir, frame_idx):
    """Build the Step-4 raw defect NPZ path for one frame (frame_####_radius_defects.npz)."""
    return os.path.join(
        str(output_npz_dir),
        f"frame_{int(frame_idx):04d}_radius_defects.npz",
    )

def _is_valid_completed_defect_npz(npz_path):
    """Check that a Step-4 NPZ has every key it should, before trusting it for resume.

    Guards against resume skipping a frame whose output file was only
    partially written before an interrupted run -- a truncated or
    still-being-written NPZ would otherwise look "present" and get
    silently treated as complete.

    Returns False for any read failure (missing file, corrupt data,
    wrong format) as well as for a structurally incomplete one -- both
    mean "don't trust this, recompute."
    """
    required_keys = {
        "verts_up", "tris_up",
        "verts_lo", "tris_lo",
        "n_up",
        "defect_up", "defect_lo",
        "is_defect",
    }

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            return required_keys.issubset(set(data.files))
    except Exception:
        return False

def get_completed_defect_frames(npz_dir):
    """Return frame indices that already have a valid, complete Step-4 raw defect NPZ.

    """
    completed = set()
    if not os.path.isdir(npz_dir):
        return completed

    for fname in os.listdir(npz_dir):
        if not (fname.startswith("frame_") and fname.endswith("_radius_defects.npz")):
            continue

        try:
            frame_idx = int(fname.split("_")[1])
        except (IndexError, ValueError):
            continue

        npz_path = os.path.join(npz_dir, fname)
        if _is_valid_completed_defect_npz(npz_path):
            completed.add(frame_idx)
        else:
            print(f"⚠️  Existing Step-4 NPZ is incomplete/corrupt and will be recomputed: {npz_path}")

    return completed


def _prepare_step4_append_log(path, header, resume_enabled):
    """Create one of Step 4's optional debug log files, without clobbering it on a resumed run.
    """
    if resume_enabled and os.path.exists(path) and os.path.getsize(path) > 0:
        return

    with open(path, "w") as f:
        f.write(header)


def run_step4_defect_finding():
    """Step 4: radius-aware defect classification for every mesh NPZ frame.

    Delegates every real computation to defect_tools (lib/defects.py) --
    this function's job is orchestration: resume-frame bookkeeping, the
    optional debug logs, per-frame timing (via defect_tools.timed), and
    printing a running summary.

    Resume is only actually enabled when BOTH DEFECT_RESUME is true AND
    DEFECT_STORE_DEBUG_NPZ is true -- resume depends on reading back the
    raw defect NPZ that DEFECT_STORE_DEBUG_NPZ gates the writing of, so
    resume can't work without it regardless of DEFECT_RESUME's own setting.

    Exits the process (sys.exit(1)) rather than raising if the required
    input directories (DEFECT_INPUT_DIR, DEFECT_NPZ_DIR) don't exist --
    the only place in this file that exits instead of raising, since this
    runs as Step 4 of the full pipeline rather than in an isolated
    subprocess like Step 3.
    """
    print_step_header(4, "Defect finding")
    print(f"Input PDB folder:     {DEFECT_INPUT_DIR}")
    print(f"Mesh NPZ folder:      {DEFECT_NPZ_DIR}")
    print(f"Atom table CSV:       {DEFECT_ATOM_CSV}")
    print(f"Output folder:        {DEFECT_OUTPUT_DIR}")
    print(f"Triangle KD-tree k:  {DEFECT_TRI_K}")

    # logging writes to stderr by default, which is where the progress bar
    # lives. In quiet mode send records to the run log and raise the level to
    # WARNING, so only things worth interrupting a bar for reach the terminal.
    if progress.is_verbose():
        logging.basicConfig(
            level=logging.DEBUG if DEFECT_VERBOSE else logging.INFO,
            format="%(levelname)s: %(message)s",
            force=True,
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
            stream=sys.stdout,
            force=True,
        )

    logging.info(f"Numba threads: {defect_tools.get_num_threads()}")

    if not os.path.isdir(DEFECT_INPUT_DIR):
        logging.error(f"Input PDB folder not found: {DEFECT_INPUT_DIR}")
        sys.exit(1)
    if not os.path.isdir(DEFECT_NPZ_DIR):
        logging.error(f"Mesh NPZ folder not found: {DEFECT_NPZ_DIR}")
        sys.exit(1)

    os.makedirs(DEFECT_OUTPUT_DIR, exist_ok=True)
    ply_up_dir = DEFECT_RAW_UPPER_DIR
    ply_lo_dir = DEFECT_RAW_LOWER_DIR
    debug_npz_dir = DEFECT_RAW_NPZ_DIR
    os.makedirs(ply_up_dir, exist_ok=True)
    os.makedirs(ply_lo_dir, exist_ok=True)
    os.makedirs(debug_npz_dir, exist_ok=True)

    resume_requested = bool(globals().get("DEFECT_RESUME", True))
    resume_enabled = resume_requested and bool(DEFECT_STORE_DEBUG_NPZ)
    completed_defect_frames = get_completed_defect_frames(debug_npz_dir) if resume_enabled else set()

    if resume_requested and not DEFECT_STORE_DEBUG_NPZ:
        print("Step-4 resume: OFF because DEFECT_STORE_DEBUG_NPZ is False")
    else:
        print(f"Step-4 resume:       {'ON' if resume_enabled else 'OFF'}")
    if resume_enabled:
        print(f"Completed frames found: {len(completed_defect_frames)}")

    tri_map_path = os.path.join(DEFECT_OUTPUT_DIR, "triangle_bead_map.txt")
    radius_log_path = os.path.join(DEFECT_OUTPUT_DIR, "radius_coverage_map.txt")
    if not DEFECT_SKIP_TRIANGLE_MAP:
        _prepare_step4_append_log(
            tri_map_path,
            f"{'frame num':<15}{'triangle num':<20}{'bead (atom numbers)'}\n",
            resume_enabled=resume_enabled,
        )
    if not DEFECT_SKIP_RADIUS_LOG:
        _prepare_step4_append_log(
            radius_log_path,
            "# frame/triangle radius-aware coverage log\n",
            resume_enabled=resume_enabled,
        )

    atom_table = defect_tools.load_atom_table_csv(DEFECT_ATOM_CSV)
    # Same generator problem as Step 3: count the prepared PDBs for the total.
    progress.set_total(len(glob.glob(os.path.join(str(DEFECT_INPUT_DIR), FRAME_PATTERN))))
    total_times = defaultdict(float)
    n_frames_done = 0
    t0_total = time.perf_counter()

    for frame_idx, frame_data, pdb_file in defect_tools.parse_pdb_files(DEFECT_INPUT_DIR):
        if resume_enabled and frame_idx in completed_defect_frames:
            out_npz = defect_radius_npz_path(debug_npz_dir, frame_idx)
            print(f"⏭️  Frame {frame_idx} already has Step-4 defect NPZ — skipping ({out_npz})")
            progress.tick()
            continue

        frame_times = defaultdict(float)
        print(f"\n🟦 Processing frame {frame_idx} from file: {pdb_file}")

        with defect_tools.timed("load_frame_npz", frame_times):
            mesh_fname = (
                f"frame_{frame_idx:0{max(0, DEFECT_FRAME_PAD)}d}_mesh_predefect.npz"
                if DEFECT_FRAME_PAD > 0
                else f"frame_{frame_idx}_mesh_predefect.npz"
            )
            frame_npz_path = os.path.join(DEFECT_NPZ_DIR, mesh_fname)
            if not os.path.exists(frame_npz_path):
                logging.warning(f"No mesh NPZ for frame {frame_idx}: {frame_npz_path}; skipping")
                progress.tick()
                continue
            with np.load(frame_npz_path, allow_pickle=True) as frame_npz:
                verts_up = frame_npz["verts_up"]
                tris_up = frame_npz["tris_up"]
                verts_lo = frame_npz["verts_lo"]
                tris_lo = frame_npz["tris_lo"]
                all_centroids = frame_npz["all_centroids"]
                all_normals = frame_npz["all_normals"]
                n_up = int(frame_npz["n_up"])

        with defect_tools.timed("positions_from_pdb_csv", frame_times):
            atoms = frame_data["atoms"]
            for a in atoms:
                key = (a.get("res_name"), a.get("atom_name"))
                info = atom_table.get(key)
                if info is None:
                    raise KeyError(f"Missing (res, atom) in CSV: {key}")
                a["flag"] = info["flag"]
                a["vdw_radius"] = float(info["vdw_radius"])

            positions = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=np.float64)

        with defect_tools.timed("build_triangle_kdtree", frame_times):
            kdt_tri = defect_tools.build_triangle_centroid_kdtree(all_centroids)

        with defect_tools.timed(f"assign_beads_kNN(k={DEFECT_TRI_K})", frame_times):
            tri_map_idx = defect_tools.assign_beads_to_triangles_kNN(
                positions=positions,
                kdt_tri=kdt_tri,
                k=max(1, int(DEFECT_TRI_K)),
                centroids=all_centroids,
                verts_up=verts_up,
                tris_up=tris_up,
                verts_lo=verts_lo,
                tris_lo=tris_lo,
                n_up=n_up,
            )

        if not DEFECT_SKIP_TRIANGLE_MAP:
            with defect_tools.timed("write_triangle_bead_map", frame_times):
                defect_tools.write_triangle_bead_map(
                    tri_map_path,
                    frame_idx,
                    tri_map_idx,
                    all_centroids,
                    all_normals,
                    positions,
                    atoms,
                )

        with defect_tools.timed("radius_aware_classification", frame_times):
            results = defect_tools.classify_all_triangles_radius_aware(
                tri_map_idx=tri_map_idx,
                atoms=atoms,
                positions=positions,
                verts_up=verts_up,
                tris_up=tris_up,
                verts_lo=verts_lo,
                tris_lo=tris_lo,
                n_up=n_up,
                signed_lower=float(DEFECT_SIGNED_LOWER),
                signed_upper=float(DEFECT_SIGNED_UPPER),
                type_signed_lower=float(DEFECT_TYPE_SIGNED_LOWER),
                type_signed_upper=float(DEFECT_TYPE_SIGNED_UPPER),
                head_threshold=float(DEFECT_HEAD_THRESHOLD),
                radius_scale=float(DEFECT_RADIUS_SCALE),
                grid_samples=max(2, int(DEFECT_GRID_SAMPLES)),
                min_total_coverage=float(DEFECT_MIN_TOTAL_COVERAGE),
            )

        if not DEFECT_SKIP_RADIUS_LOG:
            with defect_tools.timed("write_radius_debug_log", frame_times):
                defect_tools.write_radius_debug_log(radius_log_path, frame_idx, results, n_up)

        with defect_tools.timed("validate_triangle_indices", frame_times):
            defect_tools.validate_triangle_indices(verts_up, tris_up, results["defect_up"], "upper")
            defect_tools.validate_triangle_indices(verts_lo, tris_lo, results["defect_lo"], "lower")

        with defect_tools.timed("save_ply_upper", frame_times):
            upper_defect_groups = {int(t): "No Beads" for t in results["defect_up"]}
            defect_tools.save_defects_only_ply(
                verts_up,
                tris_up,
                results["defect_up"],
                frame_idx,
                ply_up_dir,
                "upper_radius",
                defect_type_code=results["defect_type_code"],
                defect_groups=upper_defect_groups,
            )

        with defect_tools.timed("save_ply_lower", frame_times):
            lower_defect_groups = {int(t + n_up): "No Beads" for t in results["defect_lo"]}
            defect_tools.save_defects_only_ply(
                verts_lo,
                tris_lo,
                results["defect_lo"],
                frame_idx,
                ply_lo_dir,
                "lower_radius",
                defect_type_code=results["defect_type_code"],
                defect_groups=lower_defect_groups,
                global_offset=n_up,
            )

        if DEFECT_STORE_DEBUG_NPZ:
            with defect_tools.timed("save_radius_npz", frame_times):
                out_npz = defect_tools.save_radius_npz(
                    out_dir=debug_npz_dir,
                    frame_idx=frame_idx,
                    verts_up=verts_up,
                    tris_up=tris_up,
                    verts_lo=verts_lo,
                    tris_lo=tris_lo,
                    n_up=n_up,
                    results=results,
                )
                logging.info(f"Saved radius NPZ: {out_npz}")

        n_frames_done += 1
        for k, v in frame_times.items():
            total_times[k] += v

        print(
            f"✅ Frame {frame_idx}: defects upper={len(results['defect_up'])}, "
            f"lower={len(results['defect_lo'])}"
        )
        print("— Timing summary (this frame) —")
        for k in sorted(frame_times, key=lambda x: frame_times[x], reverse=True):
            print(f"  {k:<36} {frame_times[k]:9.4f} s")

        # One tick per frame. This sits outside the timing loop above on
        # purpose -- inside it, it fires once per timed section.
        progress.tick()

    total_dt = time.perf_counter() - t0_total
    print("\n🎉 Radius-aware defect pipeline finished.")
    print(f"Frames processed: {n_frames_done}")
    print(f"Wall time total: {total_dt:.4f} s")
    if n_frames_done > 0:
        print("\n=== PER-FRAME AVERAGES ===")
        for k in sorted(total_times, key=lambda x: total_times[x], reverse=True):
            print(f"  {k:<36} {total_times[k] / n_frames_done:9.4f} s/frame")


# ---------- Step 5: Deduplication ----------

def run_step5_deduplication(frame_box_xyz, min_triangle_count):
    """Step 5: raw connected-component clustering of Step 4's defects, then periodic duplicate removal.

    frame_box_xyz's per-frame Lx/Ly range is printed here purely as a
    sanity check for the user -- a suspiciously narrow or wide range can
    be a fast visual clue that box_size.csv is stale or wrong before the
    run gets any further.

    Returns: the cluster NPZ output directory (str), which main.py then
             passes straight into run_step6_pruning as its input.
    """
    print_step_header(5, "Deduplication")
    print(f"Minimum triangle count per cluster: {min_triangle_count}")
    boxes = list(frame_box_xyz.values())
    lxs = [b[0] for b in boxes]
    lys = [b[1] for b in boxes]
    print(
        f"Periodic box source: per-frame results/box_size.csv -> "
        f"Lx min/mean/max = {min(lxs):.3f}/{sum(lxs)/len(lxs):.3f}/{max(lxs):.3f} Å, "
        f"Ly min/mean/max = {min(lys):.3f}/{sum(lys)/len(lys):.3f}/{max(lys):.3f} Å"
    )
    print(f"CPU workers: {CPU_WORKERS}")
    print(f"Open3D visualization: {DEDUP_VISUALIZE}")
    if DEDUP_VISUALIZE:
        print(
            f"  leaflet={DEDUP_VISUALIZE_LEAFLET}, "
            f"frame={DEDUP_VISUALIZE_FRAME}, "
            f"labels={DEDUP_VISUALIZE_LABELS}, "
            f"max_frames={DEDUP_VISUALIZE_MAX_FRAMES}"
        )

    cluster_npz_dir = cleanup_tools.run_periodic_deduplication(
        npz_dir=DEDUP_INPUT_NPZ_DIR,
        output_npz_dir=DEDUP_OUTPUT_NPZ_DIR,
        frame_box_xyz=frame_box_xyz,
        match_tol=DEDUP_MATCH_TOL,
        min_cluster_size=min_triangle_count,
        full_center_x=DEDUP_FULL_CENTER_X,
        full_center_y=DEDUP_FULL_CENTER_Y,
        visualize=DEDUP_VISUALIZE,
        visualize_frame=DEDUP_VISUALIZE_FRAME,
        visualize_leaflet=DEDUP_VISUALIZE_LEAFLET,
        visualize_labels=DEDUP_VISUALIZE_LABELS,
        visualize_max_frames=DEDUP_VISUALIZE_MAX_FRAMES,
        workers=CPU_WORKERS,
    )
    return cluster_npz_dir

# ---------- Step 6: Pruning ----------

def run_step6_pruning(cluster_npz_dir, min_triangle_count):
    """Step 6: bridge pruning on Step 5's deduplicated clusters.

    cluster_npz_dir is Step 5's return value, passed straight through as
    this step's input.
    """
    print_step_header(6, "Pruning")
    print(f"Minimum triangle count per cluster: {min_triangle_count}")
    print(f"Bridge pruning: {PRUNE_BRIDGE_PRUNE}")
    print(f"Bridge prune all: {PRUNE_BRIDGE_PRUNE_ALL}")
    print(f"CPU workers: {CPU_WORKERS}")
    print(f"Open3D visualization: {PRUNE_VISUALIZE}")
    if PRUNE_VISUALIZE:
        print(
            f"  leaflet={PRUNE_VISUALIZE_LEAFLET}, "
            f"frame={PRUNE_VISUALIZE_FRAME}, "
            f"labels={PRUNE_VISUALIZE_LABELS}, "
            f"max_frames={PRUNE_VISUALIZE_MAX_FRAMES}"
        )

    cleanup_tools.run_pruning(
        npz_dir=cluster_npz_dir,
        output_npz_dir=PRUNE_NPZ_DIR,
        output_upper_dir=PRUNE_UPPER_DIR,
        output_lower_dir=PRUNE_LOWER_DIR,
        min_cluster_size=min_triangle_count,
        bridge_prune=PRUNE_BRIDGE_PRUNE,
        bridge_prune_all=PRUNE_BRIDGE_PRUNE_ALL,
        csv_out=PRUNE_CSV_OUT,
        visualize=PRUNE_VISUALIZE,
        visualize_frame=PRUNE_VISUALIZE_FRAME,
        visualize_leaflet=PRUNE_VISUALIZE_LEAFLET,
        visualize_labels=PRUNE_VISUALIZE_LABELS,
        visualize_max_frames=PRUNE_VISUALIZE_MAX_FRAMES,
        workers=CPU_WORKERS,
    )

# ---------- Step 7: Analysis ----------
def _get_required_float(cfg, *keys):
    """Read a required float config value, raising a clear error if missing.

    Used by Step 7's fitting analysis for min_defect_size/max_defect_size/
    min_probability.

    user-chosen window.
    """
    value = get_value(cfg, *keys, default=None)
    if value is None or str(value).strip() == "":
        joined = " / ".join(keys)
        raise ValueError(f"Missing required Step-7 fitting config value: {joined}")
    return float(value)

def _get_required_text(cfg, *keys):
    """Read a required text config value, raising a clear error if missing.
    """
    value = get_value(cfg, *keys, default=None)
    if value is None or str(value).strip() == "":
        joined = " / ".join(keys)
        raise ValueError(f"Missing required Step-7 fitting config value: {joined}")
    return str(value).strip()

# ---------- Step 7: Analysis ----------

def run_step7_fitting(cfg, frame_box_xyz=None):
    """Step 7: five independently-toggleable analyses over results/pruned.csv (Step 6's output).

    Each analysis (fitting, cluster_count, coverage, distribution,
    defect_type_coverage) is gated by its own prep.in on/off key and writes
    its own summary file -- turning one off does not affect the others.

    frame_box_xyz is required only if coverage is on (coverage needs each
    frame's real box area to compute percent-of-surface); every other
    analysis works without it. Passing None while coverage is on raises a
    clear ValueError here rather than failing deep inside coverage's own
    code.
    """
    print_step_header(7, "Analysis")

    # Per-analysis toggles. Add future analyses here with their own on/off key.
    fitting_on = as_bool(get_value(cfg, "fitting", default="on"), default=True)
    cluster_count_on = as_bool(get_value(cfg, "cluster_count", "clusters_per_frame", default="on"), default=True)
    coverage_on = as_bool(get_value(cfg, "coverage", default="on"), default=True)
    distribution_on = as_bool(get_value(cfg, "distribution", default="on"), default=True)
    defect_type_coverage_on = as_bool(get_value(cfg, "defect_type_coverage", default="on"), default=True)
    output_prefix = _get_required_text(
        cfg,
        "output_name_prefix",
        "output name prefix",
        "fit_output_prefix",
        "analysis_output_prefix",
        "out_prefix",
    )

    print(f"Input cluster CSV:  {ANALYSIS_INPUT_CSV}")
    print(f"Output folder:      {ANALYSIS_OUTPUT_DIR}")
    print(f"Output name prefix: {output_prefix}")
    print(f"  fitting:        {'on' if fitting_on else 'off'}")
    print(f"  cluster_count:  {'on' if cluster_count_on else 'off'}")
    print(f"  coverage:       {'on' if coverage_on else 'off'}")
    print(f"  distribution:   {'on' if distribution_on else 'off'}")
    print(f"  defect_type_coverage: {'on' if defect_type_coverage_on else 'off'}")

    ran_any = False
    # Five sub-analyses, each one tick, whether it runs or is toggled off.
    progress.set_total(5)

    # ---- Analysis: fitting ----
    if fitting_on:
        ran_any = True
        min_defect_size = _get_required_float(
            cfg, "min_defect_size", "min defect size",
            "fit_min_defect_size", "analysis_min_defect_size",
        )
        max_defect_size = _get_required_float(
            cfg, "max_defect_size", "max defect size",
            "fit_max_defect_size", "analysis_max_defect_size",
        )
        min_probability = _get_required_float(
            cfg, "min_probability", "min probability",
            "fit_min_probability", "analysis_min_probability",
        )
        if not (0.0 < min_probability <= 1.0):
            raise ValueError(
                "Step-7 min_probability must be a probability between 0 and 1. "
                "Example: min_probability: 1e-4"
            )
        min_probability_power = np.log10(min_probability)

        print(f"\n[fitting] min defect size = {min_defect_size:.4f} Å^2")
        print(f"[fitting] max defect size = {max_defect_size:.4f} Å^2")
        print(f"[fitting] min probability = {min_probability:.6g}")

        summary = analysis_tools.run_simple_fit(
            cluster_csv=ANALYSIS_INPUT_CSV,
            output_dir=ANALYSIS_OUTPUT_DIR,
            output_prefix=output_prefix,
            min_defect_size=min_defect_size,
            max_defect_size=max_defect_size,
            min_probability_power=min_probability_power,
            bin_width=ANALYSIS_BIN_WIDTH,
            dpi=ANALYSIS_DPI,
        )
        print(f"[fitting] pi = {summary['pi_A2']:.4f} Å^2, R^2 = {summary['r2']:.4f}")
        print(f"[fitting] wrote {summary['summary_txt']}")
    else:
        print("\n[fitting] off -> skipped.")
    progress.tick()

    # ---- Analysis: defect clusters per frame ----
    if cluster_count_on:
        ran_any = True
        cc = analysis_tools.run_cluster_count(
            cluster_csv=ANALYSIS_INPUT_CSV,
            output_dir=ANALYSIS_OUTPUT_DIR,
            output_prefix=output_prefix,
        )
        print(f"\n[cluster_count] frames = {cc['n_frames']}, total clusters = {cc['n_clusters_total']}")
        print(f"[cluster_count] wrote {cc['cluster_count_txt']}")
    else:
        print("\n[cluster_count] off -> skipped.")

    progress.tick()

    # ---- Analysis: defect surface coverage per frame ----
    if coverage_on:
        if not frame_box_xyz:
            raise ValueError(
                "coverage analysis needs the per-frame box sizes (frame_box_xyz), "
                "but it was not provided to run_step7_fitting."
            )
        box_area_by_frame = {
            frame: float(lx) * float(ly)
            for frame, (lx, ly, _lz) in frame_box_xyz.items()
        }
        ran_any = True
        cov = analysis_tools.run_coverage(
            cluster_csv=ANALYSIS_INPUT_CSV,
            output_dir=ANALYSIS_OUTPUT_DIR,
            output_prefix=output_prefix,
            box_area_by_frame=box_area_by_frame,
            dpi=ANALYSIS_DPI,
        )
        areas = list(box_area_by_frame.values())
        print(
            f"\n[coverage] box area per leaflet (per-frame): min={min(areas):.4f}, "
            f"mean={sum(areas)/len(areas):.4f}, max={max(areas):.4f} Å^2, "
            f"frames = {cov['n_frames']}"
        )
        print(f"[coverage] wrote {cov['coverage_txt']}")
        print(f"[coverage] wrote {cov['png_plot']}")
        print(f"[coverage] wrote {cov['pdf_plot']}")
    else:
        print("\n[coverage] off -> skipped.")

    progress.tick()

    # ---- Analysis: defect type coverage (neutral lipid vs. tail) ----
    if defect_type_coverage_on:
        ran_any = True
        dt = analysis_tools.run_defect_type_coverage(
            pruned_npz_dir=PRUNE_NPZ_DIR,
            output_dir=ANALYSIS_OUTPUT_DIR,
            output_prefix=output_prefix,
            dpi=ANALYSIS_DPI,
        )
        print(f"\n[defect_type_coverage] frames = {dt['n_frames']}")
        print(f"[defect_type_coverage] neutral = {dt['neutral_pct']:.2f}%, tail = {dt['tail_pct']:.2f}%")
        print(f"[defect_type_coverage] wrote {dt['defect_type_txt']}")
        print(f"[defect_type_coverage] wrote {dt['png_plot']}")
        print(f"[defect_type_coverage] wrote {dt['pdf_plot']}")
    else:
        print("\n[defect_type_coverage] off -> skipped.")

    progress.tick()
    # ---- Analysis: defect size distribution ----
    if distribution_on:
        ran_any = True
        dist = analysis_tools.run_distribution(
            cluster_csv=ANALYSIS_INPUT_CSV,
            output_dir=ANALYSIS_OUTPUT_DIR,
            output_prefix=output_prefix,
            bin_width=ANALYSIS_BIN_WIDTH,
            dpi=ANALYSIS_DPI,
        )
        print(f"\n[distribution] clusters = {dist['n_clusters']}")
        print(f"[distribution] wrote {dist['distribution_txt']}")
        print(f"[distribution] wrote {dist['png_plot']}")
        print(f"[distribution] wrote {dist['pdf_plot']}")
    else:
        print("\n[distribution] off -> skipped.")

    progress.tick()

    if not ran_any:
        print("\nStep 7: every individual analysis is off -> nothing was written.")



def _resolve_subanalysis_fit_bound(cfg, step7_key, caller_name):
    """Resolve a Step 8/9 sub-analysis fit-window bound from Step 7's own key.

    Both layer_fitting (Step 8) and curvature_fitting (Step 9) always reuse
    Step 7's min_defect_size/max_defect_size -- there is no separate
    layer_fit_*/curvature_fit_* override anymore. caller_name only affects
    the wording of the raised error, so each call site still gets a message
    naming itself rather than a generic one.
    """
    raw7 = get_value(cfg, step7_key, default=None)
    if raw7 is not None and str(raw7).strip() != "":
        return float(raw7)
    raise ValueError(
        f"{caller_name} needs Step 7's '{step7_key}' set in prep.in "
        f"({caller_name} always reuses the Step-7 fitting window)."
    )


# ---------- Step 8: Layer classification ----------

def _resolve_subanalysis_min_probability(cfg):
    """Resolve the Step 8/9 sub-analysis probability floor from Step 7's min_probability.

    Shared by both layer_fitting and curvature_fitting -- always reuses
    Step 7's own min_probability, None (no floor) if that isn't set either.
    """
    raw7 = get_value(cfg, "min_probability", default=None)
    if raw7 is not None and str(raw7).strip() != "":
        return float(raw7)
    return None

def _get_optional_float(cfg, *keys, default):
    """Read an optional float config value, falling back to default if absent or empty."""
    value = get_value(cfg, *keys, default=None)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _parse_thickness_setting(cfg):
    """Parse prep.in's layer_thickness: "auto" (per-frame median, the default) or a fixed Å value.

    Returns None for "auto" -- classify.py's classify_cluster_layers treats
    None as the signal to compute a per-frame median itself, rather than
    this function resolving the number.
    """
    raw = get_value(cfg, "layer_thickness", default="auto")
    text = str(raw).strip().lower()
    if text in ("", "auto"):
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"layer_thickness must be 'auto' or a number in Å. Got {raw!r}.")

    

def run_step8_layers(cfg):
    """Step 8: per-cluster monolayer/bilayer tagging, plus its always-on packing-defect-constant sub-analysis.
    """
    print_step_header(8, "Layer classification")

    layer_on = as_bool(get_value(cfg, "layer_analysis", default="on"), default=True)
    if not layer_on:
        print("layer_analysis: off -> skipping Step 8.")
        return

    output_prefix = _get_required_text(
        cfg,
        "output_name_prefix",
        "output name prefix",
        "classification_output_prefix",
        "out_prefix",
    )

    thickness = _parse_thickness_setting(cfg)
    thickness_tol = _get_optional_float(cfg, "layer_thickness_tol", default=CLASSIFY_LAYER_THICKNESS_TOL)
    max_pair_dist = _get_optional_float(cfg, "layer_max_pair_dist", default=CLASSIFY_LAYER_MAX_PAIR_DIST)
    bilayer_frac = _get_optional_float(cfg, "layer_bilayer_frac", default=CLASSIFY_LAYER_BILAYER_FRAC)
    if not (0.0 < bilayer_frac <= 1.0):
        raise ValueError(f"layer_bilayer_frac must be in (0, 1]. Got {bilayer_frac!r}.")

    layer_defect_mesh_on = as_bool(get_value(cfg, "layer_defect_mesh", default="on"), default=True)

    thick_desc = "auto (per-frame median)" if thickness is None else f"{thickness:.3f} Å"
    print(f"Mesh NPZ dir:   {CLASSIFY_MESH_NPZ_DIR}")
    print(f"Pruned NPZ dir: {CLASSIFY_PRUNED_NPZ_DIR}")
    print(f"Output folder:  {CLASSIFY_OUTPUT_DIR}")
    print(f"Output prefix:  {output_prefix}")
    print(f"thickness={thick_desc}, tol={thickness_tol:.3f} Å, "
          f"max_pair={max_pair_dist:.3f} Å, bilayer_frac={bilayer_frac:.3f}")
    print(f"Defect PLY writing: {'on' if layer_defect_mesh_on else 'off'}")

    result = classify_tools.run_step8_layers(
        mesh_npz_dir=CLASSIFY_MESH_NPZ_DIR,
        pruned_npz_dir=CLASSIFY_PRUNED_NPZ_DIR,
        output_dir=CLASSIFY_OUTPUT_DIR,
        output_prefix=output_prefix,
        thickness=thickness,
        thickness_tol=thickness_tol,
        max_pair_dist=max_pair_dist,
        bilayer_frac=bilayer_frac,
        write_ply=layer_defect_mesh_on,
        workers=CPU_WORKERS,
    )
    c = result["counts"]
    print(f"frames={result['frames_processed']}  "
        f"bilayer={c['bilayer']}  monolayer={c['monolayer']}")
    print(f"CSV: {result['layers_csv']}")
    print(f"TXT: {result['layers_txt']}")

    # ---- Sub-analysis: monolayer vs bilayer packing-defect constant fit ----
    # Always runs now (no layer_fitting toggle); always reuses Step 7's
    # min_defect_size/max_defect_size/min_probability.
    fit_min = _resolve_subanalysis_fit_bound(cfg, "min_defect_size", "layer_fitting")
    fit_max = _resolve_subanalysis_fit_bound(cfg, "max_defect_size", "layer_fitting")
    fit_min_prob = _resolve_subanalysis_min_probability(cfg)

    fit_result = analysis_tools.run_defect_constant_fit(
        layers_csv=result["layers_csv"],
        output_dir=Path(CLASSIFY_OUTPUT_DIR) / "layers",
        output_prefix=output_prefix,
        min_defect_size=fit_min,
        max_defect_size=fit_max,
        min_probability=fit_min_prob,
        bin_width=ANALYSIS_BIN_WIDTH,
        dpi=ANALYSIS_DPI,
    )
    print(f"[layer_fitting] fit window = [{fit_min:.3f}, {fit_max:.3f}] Å^2")
    for region, pi in fit_result["pi_by_region"].items():
        print(f"[layer_fitting] {region}: pi = {pi:.4f} Å^2")
    print(f"[layer_fitting] wrote {fit_result['summary_txt']}")
    print(f"[layer_fitting] wrote {fit_result['png_plot']}")
    print(f"[layer_fitting] wrote {fit_result['pdf_plot']}")


# ---------- Step 9: Curvature ----------

def _parse_frame_range(cfg, *keys, default=None):
    """Parse prep.in's curvature_frames "start-end" (e.g. "1990-2000") into an inclusive (start, end) int tuple.

    Returns default (None = all frames) if unset. Raises ValueError for
    anything that isn't two integers separated by a single "-", or where
    start > end.
    """
    raw = get_value(cfg, *keys, default=None)
    if raw is None or str(raw).strip() == "":
        return default
    text = str(raw).strip()
    if "-" not in text:
        raise ValueError(f"{keys[0]} must look like 'start-end' (e.g. '1990-2000'). Got {raw!r}.")
    lo_s, _, hi_s = text.partition("-")
    try:
        lo, hi = int(lo_s.strip()), int(hi_s.strip())
    except ValueError:
        raise ValueError(f"{keys[0]} must be two integers separated by '-'. Got {raw!r}.")
    if lo > hi:
        raise ValueError(f"{keys[0]} start must be <= end. Got {raw!r}.")
    return (lo, hi)


def run_step9_curvature_cfg(cfg):
    """Step 9: curvature analysis, plus its always-on packing-defect-constant-by-H-segment sub-analysis.

    Segment count (CURVATURE_FIT_SEGMENTS) is a main.py constant, not a
    prep.in key.
    """
    print_step_header(9, "Curvature analysis")

    curvature_on = as_bool(get_value(cfg, "curvature_analysis", default="on"), default=True)
    if not curvature_on:
        print("curvature_analysis: off -> skipping Step 9.")
        return

    full_mesh_on = as_bool(get_value(cfg, "curvature_full_mesh", default="on"), default=True)
    defect_mesh_on = as_bool(get_value(cfg, "curvature_defect_mesh", default="on"), default=True)
    frame_range = _parse_frame_range(cfg, "curvature_frames", default=None)

    print(f"Mesh NPZ dir:   {CURVATURE_MESH_NPZ_DIR}")
    print(f"Pruned NPZ dir: {PRUNE_NPZ_DIR}")
    print(f"Output folder:  {CURVATURE_OUTPUT_DIR}")
    print(f"Method:         {CURVATURE_METHOD}")
    print(f"CPU workers:    {CPU_WORKERS}")
    print(f"Full-mesh PLY:  {'on' if full_mesh_on else 'off'}")
    print(f"Defect PLY:     {'on' if defect_mesh_on else 'off'}")
    print(
        "PLY frame range: "
        + ("all frames" if frame_range is None else f"{frame_range[0]}-{frame_range[1]}")
    )
    print("Curvature NPZ + defects CSV are always written (toggles gate PLYs only).")

    result = curvature_tools.run_step9_curvature(
        mesh_npz_dir=CURVATURE_MESH_NPZ_DIR,
        pruned_npz_dir=PRUNE_NPZ_DIR,
        output_dir=CURVATURE_OUTPUT_DIR,
        method=CURVATURE_METHOD,
        workers=CPU_WORKERS,
        write_full_mesh_ply=full_mesh_on,
        write_defect_ply=defect_mesh_on,
        frame_range=frame_range,
    )

    print(f"Step 9: jobs_processed={result['frames_processed']} "
          f"jobs_skipped(resume)={result['frames_skipped']}")
    print(f"Step 9: new cluster rows this run={result['clusters_classified']}")
    print(f"Step 9: curvature NPZ -> {result['npz_dir']}")
    print(f"Step 9: CSV           -> {result['csv_path']}")
    if full_mesh_on:
        print(f"Step 9: full-mesh PLY -> {result['full_mesh_dir']}")
    if defect_mesh_on:
        print(f"Step 9: defect PLY    -> {result['defects_dir']}")

    # ---- Sub-analysis: packing-defect constant fit by H segment ----
    # Always runs now (no curvature_fitting toggle); always reuses Step 7's
    # min_defect_size/max_defect_size/min_probability. Segment count is fixed
    # internally (CURVATURE_FIT_SEGMENTS in main.py), no longer a prep.in key.
    output_prefix = _get_required_text(
        cfg,
        "output_name_prefix",
        "output name prefix",
        "curvature_output_prefix",
        "out_prefix",
    )
    fit_min = _resolve_subanalysis_fit_bound(cfg, "min_defect_size", "curvature_fitting")
    fit_max = _resolve_subanalysis_fit_bound(cfg, "max_defect_size", "curvature_fitting")
    fit_min_prob = _resolve_subanalysis_min_probability(cfg)
    n_segments = max(1, int(CURVATURE_FIT_SEGMENTS))

    seg_result = analysis_tools.run_curvature_segment_fit(
        defects_csv=result["csv_path"],
        output_dir=result["defects_dir"],
        output_prefix=output_prefix,
        min_defect_size=fit_min,
        max_defect_size=fit_max,
        min_probability=fit_min_prob,
        n_segments=n_segments,
        bin_width=ANALYSIS_BIN_WIDTH,
        dpi=ANALYSIS_DPI,
    )
    print(f"[curvature_fitting] fit window = [{fit_min:.3f}, {fit_max:.3f}] Å^2, segments = {n_segments}")
    for seg_key, pi in seg_result["pi_by_segment"].items():
        print(f"[curvature_fitting] {seg_key}: pi = {pi:.4f} Å^2")
    print(f"[curvature_fitting] wrote {seg_result['summary_txt']}")
    print(f"[curvature_fitting] wrote {seg_result['png_plot']}")
    print(f"[curvature_fitting] wrote {seg_result['pdf_plot']}")


# ==================================================
# Post-run cleanup
# ==================================================

def _delete_folder_if_requested(label, path):
    """Delete one generated output folder, if it exists, printing what happened either way.

    Called only for folders whose KEEP_* flag is False -- run_post_pipeline_
    cleanup already filters that before calling here. 
    """
    folder = Path(path)

    if not folder.exists():
        print(f"  {label}: not found → nothing to delete ({folder})")
        return

    if not folder.is_dir():
        print(f"  {label}: path exists but is not a folder → skipped ({folder})")
        return

    try:
        shutil.rmtree(folder)
        print(f"  {label}: deleted ({folder})")
    except OSError as exc:
        print(f"  {label}: WARNING could not delete {folder}: {exc}")


def run_post_pipeline_cleanup():
    """Delete intermediate output folders whose KEEP_* flag (main.py) is False, after a successful run.

    Prints a one-line "nothing to delete" summary instead of a full report
    when every KEEP_* flag is True, so a default run (which keeps
    everything) doesn't print a cleanup section that did nothing.
    """
    cleanup_targets = [
        (
            "prepared PDB files",
            KEEP_PDB_FILES,
            OUTPUT_DIR,
        ),
        (
            "raw defect NPZ files",
            KEEP_DEFECT_RAW_NPZ,
            DEFECT_RAW_NPZ_DIR,
        ),
        (
            "cluster NPZ files",
            KEEP_DEFECT_CLUSTER_NPZ,
            DEDUP_OUTPUT_NPZ_DIR,
        ),
        (
            "pruned NPZ files",
            KEEP_DEFECT_PRUNED_NPZ,
            PRUNE_NPZ_DIR,
        ),
        (
            "mesh NPZ files",
            KEEP_MESH_NPZ,
            MESH_NPZ_DIR,
        ),
        (
            "curvature NPZ files",
            KEEP_CURVATURE_NPZ,
            CURVATURE_NPZ_DIR
        ),
    ]

    if all(keep for _, keep, _ in cleanup_targets):
        print("\nPost-run cleanup: all keep flags are True, so no folders were deleted.")
        return

    print("\n==============================")
    print("Post-run cleanup")
    print("==============================")

    for label, keep, path in cleanup_targets:
        if keep:
            print(f"  {label}: kept ({path})")
        else:
            _delete_folder_if_requested(label, path)



# ==================================================
# Pipeline step timing report
# ==================================================

def write_step_timings_txt(step_times, results_dir):
    """Write one row per pipeline step (seconds and minutes) to results/step_timings.txt.


    step_times: {step_number: elapsed_wall_clock_seconds}, populated only
                for steps that actually ran this session (built up by
                main.py's main() as each should_run_step block executes).
    Returns: the written file's Path.
    """
    out_path = Path(results_dir) / "step_timings.txt"

    header = f"{'step':<5}{'name':<24}{'seconds':>12}{'minutes':>12}"
    lines = [
        "==============================",
        "Pipeline step timings",
        "==============================",
        header,
    ]

    total_seconds = 0.0
    for step_number in sorted(PIPELINE_STEPS):
        name = PIPELINE_STEPS[step_number]
        seconds = float(step_times.get(step_number, 0.0))
        total_seconds += seconds
        lines.append(f"{step_number:<5}{name:<24}{seconds:>12.3f}{seconds / 60.0:>12.3f}")

    lines.append("-" * len(header))
    lines.append(f"{'total':<29}{total_seconds:>12.3f}{total_seconds / 60.0:>12.3f}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote pipeline step timings: {out_path}")
    return out_path