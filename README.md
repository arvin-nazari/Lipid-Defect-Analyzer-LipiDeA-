# LipiDeA v1.0

```text
 _     _       _______      ___    __   _____ 
| |   (_)     (_)  _  \    / _ \  /  | |  _  |
| |    _ _ __  _| | | |___/ /_\ \ `| | | |/' |
| |   | | '_ \| | | | / _ \  _  |  | | |  /| |
| |___| | |_) | | |/ /  __/ | | | _| |_\ |_/ /
\_____/_| .__/|_|___/ \___\_| |_/ \___(_)___/ 
        | |                                   
        |_|                                    
```

LipiDeA (Lipid Defect Analyzer) is an open-source Python tool for identifying and quantifying lipid packing defects using trajectories from  molecular dynamics simulations. It uses surface-adaptive triangulated meshes to analyze membrane packing in curved, heterogeneous, bilayer, and monolayer systems without assuming a fixed membrane normal. LipiDeA quantifies defect size, coverage, packing defect constants, local curvature, and defect composition, including contributions from phospholipid tails and neutral lipids. The current implementation is designed for coarse-grained molecular dynamics trajectories.

## Pipeline overview

```
LipiDeA requires two files to start an analysis:

	1. Trajectory PDB file: a multi-frame PDB trajectory for analysis.
	2. *.in file: a configuration file that specifies the criteria and parameters for the LipiDeA analysis.

LipiDeA automatically creates two directories during the analysis:

	1. results: contains the analysis results and output files.
	2. pdb_files: contains PDB files generated during the analysis.

      │
      ▼

Step 1  atom csv                → results/atom_info.csv
      ▼
Step 2  tiling and cutting      → pdb_files/
      ▼
Step 3  mesh generation         → results/meshes/
      ▼
Step 4  defect finding          → results/defect/raw/
      ▼
Step 5  deduplication           → results/defect/cluster/
      ▼
Step 6  pruning                 → results/defect/pruned/, results/pruned.csv
      ▼
Step 7.1  fitting / analysis    → results/fitting/
      ▼
Step 7.2  layer classification  → results/classification/
      ▼
Step 7.3  curvature             → results/curvature/
      │
      ▼
results/step_timings.txt
```

## Prerequisites

Prepare your trajectory as **one multi-frame PDB file**.

- **Wrap the trajectory.** All molecules should remain inside the primary  periodic box, with no molecules
  split across periodic box boundries. Unwrapped coordinates can interfere with leaflet reconstruction and 
  defect detection in Steps 3-4.

- **Remove water and ions.** Water and ions are not required for packing-defect analysis. Removing them 
  reduces the number of atoms and improves computational efficiency throughout the workflow.

Example commands used in GROMACS:

```bash
gmx trjconv -f traj.xtc -s topol.tpr -pbc mol -o wrapped.xtc              # wrap molecules
gmx trjconv -f wrapped.xtc -s topol.tpr -n index.ndx -o input/traj.pdb    # strip water/ions via index group
```

## Requirements

One conda (or other Python) environment with:

- `numpy`
- `scipy` (used for KD-trees — Steps 3, 7.2, 7.3)
- `matplotlib` (used for plotting — Steps 7.1, 7.2, 7.3)
- `MDAnalysis` (used to calculate box sizes and leaflet splitting — Steps 3-7)
- `open3d` (used for mesh construction, KD-tree, and PLY output)
- `pymeshlab` (used for mesh refinement and curvature)
- `numba` (used for parallel defect classification — Step 4)

Install, e.g.:

```bash
conda create -n lipidea -c conda-forge python=3.12 numpy scipy matplotlib mdanalysis open3d pymeshlab numba
conda activate lipidea
```

## Running

1. Place your multi-frame PDB trajectory in an accessible location, e.g. `input/traj.pdb`.
2. Copy a `*.in` file and edit to specify the parameters for your system.
3. Run LipiDeA with:

```bash
python main.py prep.in
```

## List of output files

```
<folder containing your input PDB>/
├── traj.pdb                                          your input
├── pdb_files/                                        Step 2  tiled + cut frames
└── results/
    ├── atom_info.csv                                 Step 1
    ├── box_size.csv                                  per-frame box, cached
    ├── run.log                                       verbose: off only
    ├── step_timings.txt                              wall-clock time per step
    ├── meshes/
    │   ├── upper/, lower/                            Step 3  leaflet surface PLYs
    │   └── npz/frame_####_mesh_predefect.npz         Step 3
    ├── defect/
    │   ├── raw/{upper,lower}/                        Step 4  per-triangle defect PLYs
    │   ├── raw/npz/frame_####_radius_defects.npz     Step 4
    │   ├── cluster/npz/                              Step 5
    │   └── pruned/{upper,lower}/, pruned/npz/        Step 6
    ├── pruned.csv                                    Step 6  final cluster table
    ├── fitting/                                      Step 7.1  summaries + PNG/PDF plots
    ├── classification/                               Step 7.2  layers CSV/TXT
    │   └── layers/<bilayer|monolayer>/<upper|lower>/ Step 7.2  PLYs
    └── curvature/
        ├── npz/frame_####_curvature_<side>.npz       Step 7.3
        ├── full_mesh/<side>/                         Step 7.3  heatmap PLYs
        └── defects/
            ├── defects_curvature.csv                 Step 7.3  frame, cluster, area_angstrom2, H
            └── <side>/                               Step 7.3  merged defect-cluster PLYs
```

Cluster labels are written as `side:index` (e.g. `upper:3`) in every CSV, so
the leaflet is encoded in the cluster column rather than a separate one.
Curvature `H` is the raw area-weighted mean curvature in Å⁻¹, not normalized.


## Visualization

Several steps write `.ply` meshes alongside their NPZ data. These open
directly in **VMD**, **ChimeraX**, **MeshLab**, and the **Open3D GUI**. We
recommend VMD or MeshLab for the best results.

- `results/meshes/{upper,lower}/` — full leaflet surfaces (Step 3)
- `results/defect/raw/{upper,lower}/` — colored per-triangle defect classification (Step 4)
- `results/defect/pruned/{upper,lower}/` — colored final cleaned clusters (Step 6)
- `results/classification/layers/…` — colored clusters split by monolayer/bilayer (Step 7.2)
- `results/curvature/full_mesh/<side>/` — colored curvature heatmaps (Step 7.3)
- `results/curvature/defects/<side>/` — defect clusters colored by curvature (Step 7.3)

VMD or ChimeraX is suggested for overlaying a mesh on the original
trajectory — load the matching frame from your input PDB alongside the `.ply`
from the same frame and check that the mesh and defect regions line up with
the actual lipid coordinates. MeshLab or the Open3D GUI is quicker for a
one-off look, and is what you need for anything where the coloring carries the
information: drag and drop the file in.

**VMD does not display the vertex colors.** It draws the whole mesh in a
single flat color (blue by default), so the defect-type, per-cluster,
monolayer/bilayer, and curvature coloring written into these files is
invisible there. Open the file in **MeshLab** or the **Open3D GUI** to see the
colors.

### Recommended usage

To see what a frame actually looks like, load the frame's coordinates and its
`.ply` files together in the same session. Everything listed here sits in the
same folder as your input PDB.

1. **Pick a frame index.** Frame numbering is 0-based and consistent across
   every step — frame 0 is the first frame of your input trajectory.
2. **Load the coordinates.** Use `pdb_files/frame_<N>.pdb`, the tiled-and-cut
   frame Step 2 wrote, not a frame from your original trajectory. That is the
   file every mesh was built from, so the two share a coordinate frame exactly.
3. **Load the meshes for that same frame** as additional molecules in the same
   session:
   - `results/meshes/upper/upper_mesh_frame_<N>.ply` and
     `results/meshes/lower/lower_mesh_frame_<N>.ply` — the two leaflet surfaces.
   - `results/defect/pruned/upper/frame_<NNNN>_pruned_clusters_upper.ply` (and
     the `lower` equivalent) — the final defect clusters sitting on those
     surfaces.


## Tips and troubleshooting

- **Many `UNKNOWN` atom types reported in Step 1.** Check that `neutral` and
  the force-field folders under `lib/lists/ff/<MODEL>/` match your topology.
  You can also resume from Step 2 after the first run and hand-edit
  `results/atom_info.csv`.

- **Errors for step 7.2 and 7.3.** Both read the Step 3 mesh NPZs.
  Step 7.3 also reads the Step 6 pruned NPZs. Please check them.

- **Runtime.** `results/step_timings.txt` gives per-step wall clock for the
  steps that ran in that session. Mesh generation is time-consuming for large systems.
  Raise `cpu_workers` to speed up Steps 2, 5, 6, 7.2, and 7.3; it does not affect
  Step 3 and 4.

## Citation

If you use LipiDeA in published work, please cite:

> *Citation pending.*

**DOI:** *TBD (Zenodo)*

## Authors

Arvin Nazari and Yu-ming M. Huang

