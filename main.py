"""LipiDeA v1.0
By Arvin Nazari and Yu-ming M. Huang

Command for running:
    python main.py *.in

"""

__version__ = "1.0"


from pathlib import Path
import os
import sys
import tempfile
import time
from types import SimpleNamespace

from lib.prep import split_multiframe_pdb, get_or_build_frame_box_csv
from lib.utilities import as_bool, cfg_path, get_value, load_input_file
from lib import progress
from lib.helpers import *

# ==================================================
# CPU workers
# ==================================================
# Process-pool worker count for Steps 2/5/6/7.2/7.3. Step 4's Numba thread count
# is independent (see run_step4_defect_finding in lib/helpers.py -- Numba
# picks its own default).
#
# The user's cpu_workers/workers key in prep.in (applied in
# apply_user_config_overrides, lib/helpers.py) is the priority and is used
# verbatim across Steps 2, 5, 6, 7.2, 7.3. If the user does not set it, the
# default is a flat 4.

CPU_WORKERS = 4

# ==================================================
# Output folders (shared across steps)
# ==================================================
RESULTS_DIR_NAME = "results"
OUTPUT_DIR_NAME = "pdb_files"

OUTPUT_ROOT = ""                 # set by rebase_output_dirs
RESULTS_DIR = RESULTS_DIR_NAME
OUTPUT_DIR = OUTPUT_DIR_NAME
ATOM_CSV = os.path.join(RESULTS_DIR, "atom_info.csv")

MESH_DIR = os.path.join(RESULTS_DIR, "meshes")
MESH_UPPER_DIR = os.path.join(MESH_DIR, "upper")
MESH_LOWER_DIR = os.path.join(MESH_DIR, "lower")
MESH_NPZ_DIR = os.path.join(MESH_DIR, "npz")

FRAME_PATTERN = "frame_*.pdb"
# ==================================================
# Terminal output
# ==================================================
# verbose (prep.in, on/off) picks between the two output modes. off is the
# default for a published release: every step prints a progress bar to the
# terminal and the full text output goes to RUN_LOG instead. on reproduces
# the original print-everything behavior and writes no log.
#
# Unrelated to DEFECT_VERBOSE, which is an internal Step-4 log-level switch.

VERBOSE_DEFAULT = False
RUN_LOG = os.path.join(RESULTS_DIR, "run.log")

# ==================================================
# Step 2: Tiling and cutting
# ==================================================
TILE_NX = 3
TILE_NY = 3
CUT_SCALE = 1.5          # final cut window = 150% of each frame's PDB box in x/y

# Code-level debug option: keep_tiled writes the intermediate 3x3-tiled PDB
# to disk (before the cut window is taken back out of it). Not read from
# prep.in -- change here only.
KEEP_TILED = False
TILED_DIR_NAME = "tiled"


# ==================================================
# Step 3: Mesh generation
# ==================================================

# ---- resume / mesh-crash recovery ----
MESH_RESUME = True
MESH_WORKER_ENV = "PACKING_PIPELINE_MESH_WORKER"
MESH_MAX_SEGFAULT_RESTARTS = 50

# ---- mesh selection (LeafletFinder) ----
USE_LEAFLET_FINDER = True
LEAFLET_CUTOFF = 15.0
LEAFLET_PBC = False

MIN_TRIANGLE_COUNT_DEFAULT = 2

# ---- triangle sizing ----
# The user-facing input only expose:
#   triangle_size   -> target mean triangle area in Å^2
#   triangle_max_cv -> triangle-area uniformity gate
#
TRIANGLE_TOLERANCE = 0.20
TRIANGLE_METHOD = "hybrid"
TRIANGLE_AUTO_SUBDIVIDE_METHOD = "loop"
TRIANGLE_MAX_SUBDIVIDE_ITERS = 6
TRIANGLE_REMESH_IF_NEEDED = True
TRIANGLE_MAX_ITERS = 2
TRIANGLE_REMESH_ITERS = 3

# ---- automatic edge trimming ----
AUTO_EDGE_TRIM = True

# ---- mesh manifold QC ----
MESH_MANIFOLD_QC = True


# ==================================================
# Step 4: Defect finding
# ==================================================
DEFECT_RESUME = True
DEFECT_INPUT_DIR = OUTPUT_DIR
DEFECT_NPZ_DIR = MESH_NPZ_DIR
DEFECT_ATOM_CSV = ATOM_CSV
DEFECT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "defect", "raw")
DEFECT_RAW_UPPER_DIR = os.path.join(DEFECT_OUTPUT_DIR, "upper")
DEFECT_RAW_LOWER_DIR = os.path.join(DEFECT_OUTPUT_DIR, "lower")
DEFECT_RAW_NPZ_DIR = os.path.join(DEFECT_OUTPUT_DIR, "npz")

DEFECT_TRI_K = 48
DEFECT_FRAME_PAD = 4 # Naming 0001, four sig figs
DEFECT_SIGNED_LOWER = -30.0
DEFECT_SIGNED_UPPER = 30.0

DEFECT_TYPE_SIGNED_LOWER = -30.0
DEFECT_TYPE_SIGNED_UPPER = 30.0
DEFECT_HEAD_THRESHOLD = 0.50
DEFECT_RADIUS_SCALE = 1.0
DEFECT_GRID_SAMPLES = 21
DEFECT_MIN_TOTAL_COVERAGE = 0.0

DEFECT_STORE_DEBUG_NPZ = True
DEFECT_VERBOSE = False
DEFECT_SKIP_TRIANGLE_MAP = True
DEFECT_SKIP_RADIUS_LOG = True

# ==================================================
# Step 5: Deduplication
# ==================================================
DEDUP_INPUT_NPZ_DIR = DEFECT_RAW_NPZ_DIR
DEDUP_OUTPUT_NPZ_DIR = os.path.join(RESULTS_DIR, "defect", "cluster", "npz")

DEDUP_MIN_CLUSTER_SIZE = 1
DEDUP_MATCH_TOL = 10.0
DEDUP_FULL_CENTER_X = None #AUTO
DEDUP_FULL_CENTER_Y = None #AUTO

# ---- Open3D visualization DEBUG (Step 5) ----
DEDUP_VISUALIZE = False
DEDUP_VISUALIZE_FRAME = 0        # None = first eligible frame; or set an integer frame number
DEDUP_VISUALIZE_LEAFLET = "both"   # "upper", "lower", or "both"
DEDUP_VISUALIZE_LABELS = True      # True adds centroid spheres + prints cluster labels
DEDUP_VISUALIZE_MAX_FRAMES = 1      # Used only when DEDUP_VISUALIZE_FRAME is None


# ==================================================
# Step 6: Pruning
# ==================================================
# Step 6 uses the cluster NPZ files written by deduplication.
PRUNE_INPUT_NPZ_DIR = DEDUP_OUTPUT_NPZ_DIR
PRUNE_OUTPUT_DIR = os.path.join(RESULTS_DIR, "defect", "pruned")
PRUNE_UPPER_DIR = os.path.join(PRUNE_OUTPUT_DIR, "upper")
PRUNE_LOWER_DIR = os.path.join(PRUNE_OUTPUT_DIR, "lower")
PRUNE_NPZ_DIR = os.path.join(PRUNE_OUTPUT_DIR, "npz")

PRUNE_BRIDGE_PRUNE = True
PRUNE_BRIDGE_PRUNE_ALL = True
PRUNE_CSV_OUT = os.path.join(RESULTS_DIR, "pruned.csv")

# ---- Open3D visualization DEBUG (Step 6) ----
PRUNE_VISUALIZE = False
PRUNE_VISUALIZE_FRAME = 0        # None = first eligible frame; or set an integer frame number
PRUNE_VISUALIZE_LEAFLET = "both"   # "upper", "lower", or "both"
PRUNE_VISUALIZE_LABELS = True      # True adds centroid spheres + prints cluster labels
PRUNE_VISUALIZE_MAX_FRAMES = 1      # Used only when PRUNE_VISUALIZE_FRAME is None

# ==================================================
# Step 7.1: Analysis (fitting)
# ==================================================
# Step 7.1 uses the Step-6 CSV and writes only the requested fit plot + summary.
ANALYSIS_INPUT_CSV = PRUNE_CSV_OUT
ANALYSIS_OUTPUT_DIR = os.path.join(RESULTS_DIR, "fitting")
ANALYSIS_BIN_WIDTH = 1.0
ANALYSIS_DPI = 300

# ==================================================
# Step 7.2: layer classification
# ==================================================
CLASSIFY_MESH_NPZ_DIR = MESH_NPZ_DIR            # Step 3 frame_####_mesh_predefect.npz
CLASSIFY_PRUNED_NPZ_DIR = PRUNE_NPZ_DIR          # Step 6 frame_####_pruned_clusters_<side>.npz
CLASSIFY_OUTPUT_DIR = os.path.join(RESULTS_DIR, "classification")

CLASSIFY_LAYER_THICKNESS_TOL = 8.0    # Å band around apposition distance
CLASSIFY_LAYER_MAX_PAIR_DIST = 40.0   # Å; no opposite triangle within this -> monolayer
CLASSIFY_LAYER_BILAYER_FRAC = 0.5     # cluster is bilayer if >= this paired fraction


# ==================================================
# Step 7.3: curvature analysis
# ==================================================
CURVATURE_MESH_NPZ_DIR = MESH_NPZ_DIR              # Step 3 frame_####_mesh_predefect.npz
CURVATURE_OUTPUT_DIR = os.path.join(RESULTS_DIR, "curvature")
CURVATURE_NPZ_DIR = os.path.join(CURVATURE_OUTPUT_DIR, "npz")

CURVATURE_METHOD = "Scale Dependent Quadric Fitting"

# Number of H segments for the always-on curvature_fitting sub-analysis.
CURVATURE_FIT_SEGMENTS = 5


# ==================================================
# Post-run cleanup flags
# ==================================================
KEEP_PDB_FILES = True
KEEP_MESH_NPZ = True
KEEP_DEFECT_RAW_NPZ = True
KEEP_DEFECT_CLUSTER_NPZ = False
KEEP_DEFECT_PRUNED_NPZ = True
KEEP_CURVATURE_NPZ = True


# ==================================================
# Pipeline steps and resume aliases
# ==================================================
PIPELINE_STEPS = {
    1: "atom csv",
    2: "tiling and cutting",
    3: "mesh generation",
    4: "defect finding",
    5: "deduplication",
    6: "pruning",
    7.1: "fitting",
    7.2: "layer classification",
    7.3: "curvature",
}

STEP_ALIASES = {
    "1": 1, "step 1": 1, "atom": 1, "atom csv": 1, "atom_csv": 1, "csv": 1,
    "2": 2, "step 2": 2, "tiling": 2, "cutting": 2, "tiling and cutting": 2,
    "tile": 2, "cut": 2, "prep": 2,
    "3": 3, "step 3": 3, "mesh": 3, "mesh generation": 3,
    "4": 4, "step 4": 4, "defect": 4, "defects": 4, "defect finding": 4,
    "5": 5, "step 5": 5, "dedup": 5, "deduplication": 5, "periodic deduplication": 5,
    "6": 6, "step 6": 6, "prune": 6, "pruning": 6, "cluster pruning": 6,
    "7.1": 7.1, "step 7.1": 7.1, "fit": 7.1, "fitting": 7.1, "analysis": 7.1, "pi fit": 7.1,
    "7.2": 7.2, "step 7.2": 7.2, "layer classification": 7.2, "layers": 7.2, "classification": 7.2,
    "7.3": 7.3, "step 7.3": 7.3, "curvature": 7.3, "curvature analysis": 7.3,
}


# ==================================================
# Ignored legacy config keys
# ==================================================
IGNORED_CONFIG_KEYS = {
    "nx", "ny", "lx", "ly", "keep_tiled", "tiled_folder", "tiled_dir", "skip_cut",
    "mesh", "mesh_dir", "mesh_input", "resume",
    "smooth", "smooth_method", "smooth_iters", "smooth_lambda", "smooth_mu", "no_smooth_diag",
    "use_leaflet_finder", "leaflet_select", "leaflet_cutoff", "leaflet_pbc",
    "triangle_tolerance", "triangle_method", "triangle_auto_subdivide_method",
    "triangle_max_subdivide_iters", "triangle_remesh_if_needed",
    "triangle_max_iters", "triangle_remesh_iters",
    "edge_cut_x_pct", "edge_cut_y_pct",
    "defect_input", "defect_npz", "defect_list", "defect_output",
    "tri_k", "signed_lower", "signed_upper", "head_threshold",
    "radius_scale", "grid_samples", "store_debug_npz",
    "skip_triangle_map", "skip_radius_log", "threads",
    "npz_dir", "leaflet", "min_cluster_size", "visualize", "frame", "no_labels",
    "full_center_x", "full_center_y", "periodic_lx", "periodic_ly", "match_tol",
    "csv_out", "bridge_prune", "bridge_prune_all",
}


# ==================================================
# Output path rebasing
# ==================================================
# Every path constant defined above, in definition order. Order does not
# matter for correctness (each value is joined onto the base independently,
# not rebuilt from its parent), but keeping it grouped by step makes an
# omission easy to spot.

_REBASE_PATH_GLOBALS = (
    "RESULTS_DIR", "OUTPUT_DIR", "ATOM_CSV",
    "MESH_DIR", "MESH_UPPER_DIR", "MESH_LOWER_DIR", "MESH_NPZ_DIR",
    "RUN_LOG",
    "DEFECT_INPUT_DIR", "DEFECT_NPZ_DIR", "DEFECT_ATOM_CSV",
    "DEFECT_OUTPUT_DIR", "DEFECT_RAW_UPPER_DIR", "DEFECT_RAW_LOWER_DIR",
    "DEFECT_RAW_NPZ_DIR",
    "DEDUP_INPUT_NPZ_DIR", "DEDUP_OUTPUT_NPZ_DIR",
    "PRUNE_INPUT_NPZ_DIR", "PRUNE_OUTPUT_DIR", "PRUNE_UPPER_DIR",
    "PRUNE_LOWER_DIR", "PRUNE_NPZ_DIR", "PRUNE_CSV_OUT",
    "ANALYSIS_INPUT_CSV", "ANALYSIS_OUTPUT_DIR",
    "CLASSIFY_MESH_NPZ_DIR", "CLASSIFY_PRUNED_NPZ_DIR", "CLASSIFY_OUTPUT_DIR",
    "CURVATURE_MESH_NPZ_DIR", "CURVATURE_OUTPUT_DIR", "CURVATURE_NPZ_DIR",
)


def rebase_output_dirs(base_dir):
    """Rewrite every output path constant in this module to sit under base_dir.

    base_dir: folder to write into, i.e. input_pdb.parent.
    Returns: the resolved base folder as a str.
    """
    g = globals()
    base = str(Path(base_dir).resolve())

    for name in _REBASE_PATH_GLOBALS:
        g[name] = os.path.join(base, g[name])

    g["OUTPUT_ROOT"] = base

    # A path constant added above but not listed in _REBASE_PATH_GLOBALS
    # would still be relative here, and its output would silently land in
    # the current working directory instead of next to the input PDB.
    # Fail loudly rather than scatter files.
    stray = sorted(
        name
        for name, value in g.items()
        if name.isupper()
        and not name.endswith("_NAME")
        and isinstance(value, str)
        and (
            value in (RESULTS_DIR_NAME, OUTPUT_DIR_NAME)
            or value.startswith(RESULTS_DIR_NAME + os.sep)
            or value.startswith(OUTPUT_DIR_NAME + os.sep)
        )
    )
    if stray:
        raise RuntimeError(
            "Output path constant(s) missing from _REBASE_PATH_GLOBALS: "
            + ", ".join(stray)
            + ". Add them there so they are rebased with the rest."
        )

    return base

# ==================================================
# Entry point
# ==================================================


def print_welcome_banner():
    banner = r"""
==================================================
 _     _       _______      ___  
| |   (_)     (_)  _  \    / _ \ 
| |    _ _ __  _| | | |___/ /_\ \
| |   | | '_ \| | | | / _ \  _  |
| |___| | |_) | | |/ /  __/ | | |
\_____/_| .__/|_|___/ \___\_| |_/
        | |                      
        |_|                      
        
  LipiDeA {version} -- Lipid Defect Analyzer
  Arvin Nazari and Yu-ming M. Huang
==================================================
""".format(version=__version__)
    progress.notice(banner)

def is_mesh_worker_process():
    """True when this process is the isolated Step-3 mesh worker, re-invoked as a subprocess.

    Set by run_step3_mesh_generation_with_failsafe (lib/helpers.py), which
    launches this same script again with MESH_WORKER_ENV=1 in the child's
    environment. main() checks this at startup and, if true, jumps straight
    to run_step3_mesh_generation instead of running the normal pipeline --
    see main()'s own comment at that check for why (segfault isolation).
    """
    return os.environ.get(MESH_WORKER_ENV, "") == "1"



def main():
    # Usage:
    #   python main.py prep.in
    # If no config file is given, it uses prep.in in the current folder.
    config_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    config_path = Path(config_args[0]) if config_args else Path("prep.in")
    config_path = config_path.resolve()
    config_dir = config_path.parent
    script_dir = Path(__file__).resolve().parent

    cfg = load_input_file(config_path)

    input_pdb = cfg_path(get_value(cfg, "input", required=True), config_dir)
    if input_pdb is None or not input_pdb.is_file():
        raise FileNotFoundError(
            "input must be one multi-frame PDB file, not a folder.\n"
            f"Received: {input_pdb.resolve() if input_pdb is not None else input_pdb}"
        )
    rebase_output_dirs(input_pdb.parent)

    configure_pipeline_helpers(SimpleNamespace(**{
        name: value
        for name, value in globals().items()
        if name.isupper()
    }))

    apply_user_config_overrides(cfg)

    # Output mode. The Step-3 mesh worker is a subprocess whose fd 1 its
    # parent already pointed at the run log, so it draws bars but must not
    # open or swap that log a second time.
    worker_mode = is_mesh_worker_process()
    verbose = as_bool(
        get_value(cfg, "verbose", default=("on" if VERBOSE_DEFAULT else "off")),
        default=VERBOSE_DEFAULT,
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    progress.configure(
        verbose=verbose,
        log_path=(None if worker_mode else RUN_LOG),
        capture_stdout=not worker_mode,
    )

    try:
        if not worker_mode:
            print_welcome_banner()
            progress.notice(f"Output folder: {OUTPUT_ROOT}")
            if not verbose:
                progress.notice(f"verbose: off -> full output goes to {RUN_LOG}")

        model = str(get_value(cfg, "model", default="CG")).upper().strip()

        # input_pdb was resolved, validated, and used as the output base at
        # the top of main() -- see rebase_output_dirs there.
        output_dir = Path(OUTPUT_DIR)
        output_csv = Path(ATOM_CSV)
        tiled_dir = output_dir / TILED_DIR_NAME

        # Mesh worker mode is internal. It is used only by the parent process to
        # isolate Step 3 so a C/C++ segfault does not kill the whole pipeline.
        if worker_mode:
            with progress.step(3, PIPELINE_STEPS[3]):
                run_step3_mesh_generation(cfg, config_dir, output_dir)
            return

        resume_from_step = parse_resume_from_step(get_value(cfg, "resume_from", default=""))
        min_triangle_count = get_min_triangle_count(cfg)

        ignored_keys = sorted(k for k in cfg if k in IGNORED_CONFIG_KEYS)
        if ignored_keys:
            print(
                "[info] Ignoring legacy config key(s): "
                + ", ".join(ignored_keys)
                + ". These settings are fixed/internal in main.py."
            )

        print(f"Resume from: step {resume_from_step} ({PIPELINE_STEPS.get(resume_from_step, 'unknown')})")

        # Canonical per-frame box size (X, Y, Z), cached to results/box_size.csv.
        # Computed regardless of resume_from: Steps 2/5/7 can each be entered
        # directly without Step 1 having run this session, and all three need it.
        frame_box_xyz = get_or_build_frame_box_csv(input_pdb, RESULTS_DIR)

        temp_frames = None
        input_dir = None
        step_times = {}  # {step_number: elapsed_seconds}, only for steps run this session

        if should_run_step(1, resume_from_step) or should_run_step(2, resume_from_step):
            if resume_from_step <= 2:
                temp_frames = tempfile.TemporaryDirectory(prefix="pdb_frames_")
                input_dir = Path(temp_frames.name)
                n_split = split_multiframe_pdb(input_pdb, input_dir, frame_prefix="frame")
                print(f"Split multi-frame PDB into {n_split} temporary frame files: {input_dir}")

        if should_run_step(1, resume_from_step):
            _t0 = time.perf_counter()
            # No frame loop to count: Step 1 reads one frame, so its bar is a
            # spinner rather than a fraction.
            with progress.step(1, PIPELINE_STEPS[1]):
                run_step1_atom_csv(
                    cfg=cfg,
                    config_path=config_path,
                    config_dir=config_dir,
                    script_dir=script_dir,
                    input_pdb=input_pdb,
                    input_dir=input_dir,
                    output_csv=output_csv,
                    model=model,
                )
            step_times[1] = time.perf_counter() - _t0

        if should_run_step(2, resume_from_step):
            if input_dir is None:
                raise RuntimeError("Internal error: Step 2 requires split temporary PDB frames.")
            _t0 = time.perf_counter()
            with progress.step(2, PIPELINE_STEPS[2]):
                run_step2_tiling_and_cutting(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    tiled_dir=tiled_dir,
                    frame_box_xyz=frame_box_xyz,
                )
            step_times[2] = time.perf_counter() - _t0

        if should_run_step(3, resume_from_step):
            _t0 = time.perf_counter()
            # No progress.step here on purpose: Step 3 runs in a subprocess and
            # draws its own bar from inside, where the frame count is known.
            run_step3_mesh_generation_with_failsafe(
                config_path,
                script_path=Path(__file__).resolve(),
                worker_env=MESH_WORKER_ENV,
                max_restarts=MESH_MAX_SEGFAULT_RESTARTS,
            )
            step_times[3] = time.perf_counter() - _t0

        if should_run_step(4, resume_from_step):
            _t0 = time.perf_counter()
            with progress.step(4, PIPELINE_STEPS[4]):
                run_step4_defect_finding()
            step_times[4] = time.perf_counter() - _t0

        cluster_npz_dir = PRUNE_INPUT_NPZ_DIR

        if should_run_step(5, resume_from_step):
            _t0 = time.perf_counter()
            with progress.step(5, PIPELINE_STEPS[5]):
                cluster_npz_dir = run_step5_deduplication(
                    frame_box_xyz=frame_box_xyz,
                    min_triangle_count=min_triangle_count,
                )
            step_times[5] = time.perf_counter() - _t0

        if should_run_step(6, resume_from_step):
            _t0 = time.perf_counter()
            with progress.step(6, PIPELINE_STEPS[6]):
                run_step6_pruning(
                    cluster_npz_dir=cluster_npz_dir,
                    min_triangle_count=min_triangle_count,
                )
            step_times[6] = time.perf_counter() - _t0

        if should_run_step(7.1, resume_from_step):
            _t0 = time.perf_counter()
            with progress.step(7.1, PIPELINE_STEPS[7.1]):
                run_step7_fitting(cfg=cfg, frame_box_xyz=frame_box_xyz)
            step_times[7.1] = time.perf_counter() - _t0

        if should_run_step(7.2, resume_from_step):
            _t0 = time.perf_counter()
            with progress.step(7.2, PIPELINE_STEPS[7.2]):
                run_step8_layers(cfg)
            step_times[7.2] = time.perf_counter() - _t0

        if should_run_step(7.3, resume_from_step):
            _t0 = time.perf_counter()
            with progress.step(7.3, PIPELINE_STEPS[7.3]):
                run_step9_curvature_cfg(cfg)
            step_times[7.3] = time.perf_counter() - _t0

        run_post_pipeline_cleanup()

        write_step_timings_txt(step_times, RESULTS_DIR)

        progress.notice("🎉 Full pipeline finished successfully.")
    finally:
        progress.close()


if __name__ == "__main__":
    main()
