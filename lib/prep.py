"""Step 1 and Step 2: force-field lookup tables and PDB frame preparation.

Two jobs live here:

  Step 1  Build results/atom_info.csv -- one row per unique (residue, bead)
          pair in the trajectory, carrying its force-field type, van der Waals
          radius, and head/tail/neutral flag. Step 4 reads that CSV to decide
          which beads cover a triangle.

  Step 2  Split the input trajectory into per-frame PDBs, tile each frame
          3x3 in x/y, and cut a centered window back out. The tiling gives
          Step 3's surface reconstruction real neighbors across the periodic
          boundary instead of a hard edge.

Also owns results/box_size.csv, the canonical per-frame (Lx, Ly, Lz) source
shared by Steps 2, 5, and 7.
"""

import csv
import glob
import io
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from . import progress
# ATOM and HETATM are both treated as real atoms everywhere in this module.
ATOM_RECORDS = {"ATOM", "HETATM"}

# ==================================================
# Shared helpers
# ==================================================
def normalize_resname(resname, model):
    """Normalize a residue/molecule name so lookups match across sources.
    resname: raw name from a PDB or force-field file.
    model:   "AA" or "CG".
    Returns: the normalized name as str.
    """
    r = (resname or "").strip().upper()
    if model.upper() == "CG":
        return r[:4]
    return r


def parse_neutral_resnames(values, model):
    """Normalize the neutral-lipid names from prep.in's `neutral` key.

    Neutral lipids (TAG, DAG, ...) get flag "n" in atom_info.csv, which
    Step 4 uses to tell a neutral-lipid-dominant defect from a
    phospholipid-tail-dominant one.

    values: list of name strings, or empty.
    model:  "AA" or "CG", passed through to normalize_resname.
    Returns: set of normalized names; empty set when nothing was given.
    """
    neutral = set()

    for item in values or []:
        for part in item.replace(";", ",").split(","):
            name = part.strip()
            if not name:
                continue
            neutral.add(normalize_resname(name, model))

    return neutral

def auto_flag_from_residue_and_atom_type(resname, atom_type, neutral_resnames):
    """Assign the head/tail/neutral flag written into atom_info.csv.

        residue listed in prep.in's `neutral` key -> "n"
        force-field atom_type starts with C       -> "t"
        everything else                           -> "h"

    Neutral overrides atom_type: a TAG bead is "n" even though its type
    starts with C.

    Returns: "n", "t", or "h".
    """
    rname = (resname or "").strip().upper()
    atype = (atom_type or "").strip().upper()

    if rname in neutral_resnames:
        return "n"

    if atype.startswith("C"):
        return "t"

    return "h"



# ==================================================
# Per-frame box size, cached to results/box_size.csv
# ==================================================
# Canonical (Lx, Ly, Lz) source for Steps 2, 5, and 7. Built once and cached
# to disk, so a resume reads the CSV instead of rescanning the trajectory.
#
# Computed regardless of resume_from: Steps 2, 5, and 7 can each be entered
# directly without Step 1 having run in this session, and all three need it.

def _frame_idx_from_split_filename(path, frame_prefix):
    """Recover the frame index from a split frame filename.

    split_multiframe_pdb writes frame_0.pdb, frame_1.pdb, ... so the index is
    whatever follows the prefix.
    """
    stem = Path(path).stem
    prefix = f"{frame_prefix}_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected split frame filename: {path}")
    return int(stem[len(prefix):])


def read_per_frame_xyz_box(input_pdb):
    """Return {frame_idx: (Lx, Ly, Lz)} for every frame in a multi-frame PDB.

    Box size is the atom-coordinate bounding box for that frame (max - min
    per axis)


    Frame boundaries follow the same convention as split_multiframe_pdb --
    a bare END delimits frames, while MODEL/ENDMDL are skipped rather than
    treated as boundaries. That is what keeps frame_idx here identical to
    frame_idx everywhere else in the pipeline.

    input_pdb: path to the multi-frame trajectory PDB.
    Returns:   dict {frame_idx: (Lx, Ly, Lz)} in Angstroms.
    """
    input_pdb = Path(input_pdb)

    boxes = {}
    frame_idx = 0
    xs = []
    ys = []
    zs = []

    def finalize_frame():
        nonlocal frame_idx
        if not xs:
            return
        lx = max(xs) - min(xs)
        ly = max(ys) - min(ys)
        lz = max(zs) - min(zs)
        boxes[frame_idx] = (lx, ly, lz)
        frame_idx += 1

    with input_pdb.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            rec = line[:6].strip()

            if rec == "END":
                finalize_frame()
                xs, ys, zs = [], [], []
                continue

            if rec in {"MODEL", "ENDMDL"}:
                continue

            if rec in ATOM_RECORDS:
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except ValueError:
                    continue
            # Every other record (CRYST1, REMARK, TITLE, ...) is ignored --
            # box size comes purely from atom coordinates.

    # A final frame with no trailing END still counts. split_multiframe_pdb
    # handles this same edge case the same way.
    finalize_frame()

    if not boxes:
        raise ValueError(f"No frames with ATOM/HETATM records found in {input_pdb}")

    print(f"[box_size] Read {len(boxes)} frame box(es) from {input_pdb} (atom bounding box, text scan).")
    return boxes

def get_or_build_frame_box_csv(input_pdb, results_dir):
    """Load results/box_size.csv, or compute and write it if absent.

    First run pays a full scan of the trajectory; every run after that is a
    CSV read. Delete the CSV to force a rebuild -- which is what the error
    messages in Steps 2 and 5 tell the user to do when a cached CSV does not
    cover every frame the run needs (usually a leftover from a different
    input PDB).

    input_pdb:   path to the multi-frame trajectory PDB.
    results_dir: folder holding box_size.csv; created if missing.
    Returns:     dict {frame_idx: (Lx, Ly, Lz)} in Angstroms.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "box_size.csv"

    if csv_path.is_file():
        boxes = {}
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    frame = int(row["frame"])
                    lx = float(row["X(AA)"])
                    ly = float(row["Y(AA)"])
                    lz = float(row["Z(AA)"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Malformed row in {csv_path}: {row!r}") from exc
                boxes[frame] = (lx, ly, lz)
        if not boxes:
            raise ValueError(f"{csv_path} exists but has no frame rows.")
        print(f"[box_size] Loaded {len(boxes)} frame box(es) from cache: {csv_path}")
        return boxes

    print(f"[box_size] {csv_path} not found -- computing per-frame box sizes now.")
    boxes = read_per_frame_xyz_box(input_pdb)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "X(AA)", "Y(AA)", "Z(AA)"])
        for frame in sorted(boxes):
            lx, ly, lz = boxes[frame]
            writer.writerow([frame, f"{lx:.4f}", f"{ly:.4f}", f"{lz:.4f}"])

    print(f"[box_size] Wrote {len(boxes)} frame box(es) to {csv_path}")
    return boxes


# ==================================================
# PDB scanning for atom_info.csv
# ==================================================
def iter_pdb_res_atom(pdb_path):
    """Yield (resname, atom_name) for every ATOM/HETATM record in a PDB.


    Names are yielded raw. normalize_resname is applied by the caller.
    """
    pdb_path = Path(pdb_path)

    with pdb_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            record = line[0:6].strip()
            if record not in ATOM_RECORDS:
                continue

            atom_name = line[12:16].strip()
            resname = line[17:21].strip()

            if atom_name and resname:
                yield (resname, atom_name)


def unique_pairs_from_pdb(pdb_path, model):
    """Return unique (resname, atom_name) pairs in first-seen order.

    One row per pair is what atom_info.csv holds -- the table is a lookup
    keyed by (residue, bead), not a per-atom listing, so a 200k-bead system
    still produces only a few dozen rows.

    Order is first-seen rather than sorted so the CSV reads in the same
    order as the PDB, which makes an unexpected residue easy to spot.

    Returns: list of (normalized_resname, atom_name) tuples.
    """
    seen = set()
    pairs = []

    for resname_raw, atom_name in iter_pdb_res_atom(pdb_path):
        resname = normalize_resname(resname_raw, model)
        pair = (resname, atom_name.strip())
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)

    return pairs


# ==================================================
# Force-field type parsing
# ==================================================
# Builds {resname: {atom_name: atom_type}} from whatever force-field files
# ship with the model. AA reads AMBER-style .lib/.prepc; CG reads Martini
# .itp. build_types_map (below) is the only entry point callers need --
# these are its file-format backends.

# All-Atom (WIP)
# ---------------------------------------------------------------------------
def parse_lib_file(lib_path):
    """Parse one AMBER .lib file into {resname: {atom_name: atom_type}}.

    Reads only the !entry.<RES>.unit.atoms tables; every other section
    (connectivity, coordinates, ...) is skipped.
    """
    lib_path = Path(lib_path)
    res_to_atoms = {}

    current_res = None
    inside_atoms_table = False

    with lib_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("!entry.") and ".unit.atoms" in line:
                parts = line.split(".")
                if len(parts) >= 3:
                    current_res = parts[1].strip().upper()
                    res_to_atoms.setdefault(current_res, {})
                    inside_atoms_table = True
                continue

            if line.startswith("!entry.") and ".unit.atoms" not in line:
                inside_atoms_table = False
                current_res = None
                continue

            if inside_atoms_table and current_res and line.startswith('"'):
                pieces = line.split('"')
                if len(pieces) >= 4:
                    atom_name = pieces[1].strip()
                    atom_type = pieces[3].strip()
                    if atom_name and atom_type:
                        res_to_atoms[current_res][atom_name] = atom_type

    return res_to_atoms

def merge_lib_folder(types, lib_folder, pattern="*.lib", overwrite=False):
    """Merge every matching .lib file in a folder into types, in place.

    types:     {resname: {atom_name: atom_type}} dict to update.
    lib_folder: folder to scan.
    pattern:   glob pattern; files are read in sorted order for determinism.
    overwrite: True lets a later file replace an earlier file's value for
               the same (resname, atom_name); False (default) keeps the
               first value seen.
    Returns:   the same types dict, for chaining.
    """
    lib_folder = Path(lib_folder)

    for lib_file in sorted(lib_folder.glob(pattern)):
        lib_map = parse_lib_file(lib_file)
        for res, atom_map in lib_map.items():
            types.setdefault(res, {})
            for atom, atype in atom_map.items():
                if overwrite:
                    types[res][atom] = atype
                else:
                    types[res].setdefault(atom, atype)

    return types

def parse_prepc_file(prepc_path):
    """Parse one AMBER .prepc file into {resname: {atom_name: atom_type}}.

    A .prepc lists atoms as numbered rows under a "RESNAME XYZ ..." header;
    non-numeric rows and the DUMM placeholder atom are skipped.
    """
    prepc_path = Path(prepc_path)

    resname = None
    res_to_atoms = {}

    with prepc_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue

            toks = stripped.split()

            if len(toks) >= 3 and toks[1] == "XYZ":
                resname = toks[0].strip().upper()
                res_to_atoms.setdefault(resname, {})
                continue

            try:
                int(toks[0])
            except ValueError:
                continue

            if resname is None or len(toks) < 3:
                continue

            atom_name = toks[1].strip()
            atom_type = toks[2].strip()

            if atom_name == "DUMM":
                continue

            res_to_atoms[resname][atom_name] = atom_type

    return res_to_atoms

def merge_prepc_folder(types, prepc_folder, pattern="*.prepc", overwrite=False):
    """Merge every matching .prepc file in a folder into types, in place.

    Same merge semantics as merge_lib_folder -- see its docstring.
    """
    prepc_folder = Path(prepc_folder)

    for prepc_file in sorted(prepc_folder.glob(pattern)):
        prepc_map = parse_prepc_file(prepc_file)
        for res, atom_map in prepc_map.items():
            types.setdefault(res, {})
            for atom, atype in atom_map.items():
                if overwrite:
                    types[res][atom] = atype
                else:
                    types[res].setdefault(atom, atype)

    return types
#-----------------------------------------------------------------------------------------


# Coarse-grain
#-----------------------------------------------------------------------------------------
def parse_itp_file(itp_path):
    """Parse one Martini/GROMACS molecule .itp into {resname: {atom_name: bead_type}}.

    Reads [ moleculetype ] for the residue name and [ atoms ] for the bead
    type and atom name columns. Any other section is ignored.
    """
    itp_path = Path(itp_path)

    res_to_atoms = {}
    current_res = None
    section = None

    with itp_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]").strip().lower()
                continue

            if section == "moleculetype":
                data = line.split(";")[0].strip()
                if not data:
                    continue
                toks = data.split()
                if len(toks) >= 1:
                    current_res = toks[0].strip().upper()
                    if current_res:
                        res_to_atoms.setdefault(current_res, {})
                continue

            if section == "atoms":
                if current_res is None:
                    continue

                data = line.split(";")[0].strip()
                if not data:
                    continue

                toks = data.split()
                # GROMACS [ atoms ] column order:
                # id  type  resnr  residu  atom  cgnr  charge  ...
                if len(toks) < 7:
                    continue

                bead_type = toks[1].strip()
                atom_name = toks[4].strip()

                if atom_name and bead_type:
                    res_to_atoms[current_res][atom_name] = bead_type

    return res_to_atoms

def iter_itp_files(folder):
    """Return every .itp file directly inside folder, case-insensitive, sorted.

    Deliberately does not choose files by name or size -- every .itp/.ITP
    file in the folder is considered for both atom-type and radius parsing.
    Subfolders are not searched.
    """
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".itp"
    )

def merge_itp_folder(types, itp_folder, pattern="*.itp", overwrite=False):
    """Merge every .itp file in a folder into types, in place.


    types:     {resname: {atom_name: atom_type}} dict to update.
    itp_folder: folder to scan.
    pattern:   glob pattern; files are read in sorted order for determinism.
    overwrite: True lets a later file replace an earlier file's value for
               the same (resname, atom_name); False (default) keeps the
               first value seen.
    Returns:   the same types dict, for chaining.

    pattern is accepted for call-site symmetry with merge_lib_folder and
    merge_prepc_folder but is NOT used: files come from iter_itp_files,
    which always scans *.itp/*.ITP case-insensitively regardless of what
    is passed here.
    """
    itp_folder = Path(itp_folder)

    for itp_file in iter_itp_files(itp_folder):
        itp_map = parse_itp_file(itp_file)
        for res, atom_map in itp_map.items():
            types.setdefault(res, {})
            for atom, atype in atom_map.items():
                if overwrite:
                    types[res][atom] = atype
                else:
                    types[res].setdefault(atom, atype)

    return types
# --------------------------------------------------------------------------------

# ==================================================
# Radius parsing
# ==================================================
# Builds {atom_type: vdw_radius_angstrom}. AA reads AMBER .dat RE tables;
# CG derives radii from Martini [ nonbond_params ] self-pair sigma values.
# build_radii_map is the entry point; these are its format backends.


# All-Atom (WIP)
# ---------------------------------------------------------------------------
def parse_radius_dat(dat_path):
    """Parse one AMBER radius .dat file into {atom_type: radius_angstrom}.

    Reads only rows inside the RE (radius) section, identified by a line
    whose last token is "RE". Earlier LJ-parameter sections are skipped.
    """
    dat_path = Path(dat_path)
    radius_map = {}
    in_re_section = False

    with dat_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            toks = line.split()

            if len(toks) >= 2 and toks[-1] == "RE":
                in_re_section = True
                continue

            if not in_re_section:
                continue

            if len(toks) < 2:
                continue

            atom_type = toks[0]
            try:
                radius = float(toks[1])
            except ValueError:
                continue

            radius_map[atom_type] = radius

    return radius_map

def merge_radius_folder(radius_folder, pattern="*.dat", overwrite=False):
    """Merge every matching .dat file in a folder into one radius map.

    overwrite: True lets a later file replace an earlier file's radius for
               the same atom_type; False (default) keeps the first seen.
    Returns:   {atom_type: radius_angstrom}, merged across all files.
    """
    radius_folder = Path(radius_folder)
    merged = {}

    for dat_file in sorted(radius_folder.glob(pattern)):
        one = parse_radius_dat(dat_file)
        for k, v in one.items():
            if overwrite:
                merged[k] = v
            else:
                merged.setdefault(k, v)

    return merged
# ---------------------------------------------------------------------------


#Coarse-grain
# ---------------------------------------------------------------------------
def detect_martini_ff_version(itp_path):
    """Detect Martini 2 vs Martini 3 from an .itp file's first non-empty line.

        ;;;;; Martini 3 Force Field: Particle definition and LJ interactions
            -> "MARTINI3"
        ; MARTINI FORCEFIELD V2.2
            -> "MARTINI2"

    Only the first non-empty line is checked, by design -- this identifies
    which file format the rest of the parser should expect, not which
    version any particular molecule was built for.

    Returns: "MARTINI2", "MARTINI3", or None if unrecognized.
    """
    itp_path = Path(itp_path)

    with itp_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            upper = line.upper()

            if "MARTINI 3 FORCE FIELD" in upper or "MARTINI3 FORCE FIELD" in upper:
                return "MARTINI3"

            if (
                "MARTINI FORCEFIELD V2" in upper
                or "MARTINI FORCE FIELD V2" in upper
                or "MARTINI V2" in upper
            ):
                return "MARTINI2"

            # Only the first non-empty line is ever checked.
            return None

    return None


def martini2_c6_c12_to_sigma_nm(c6, c12):
    """Convert Martini 2 Lennard-Jones C6/C12 to sigma, in nm.

    Martini 2 [ nonbond_params ] stores C6/C12; Martini 3 stores sigma
    directly. This converts the former to the latter so both versions
    reduce to one downstream radius calculation.

        sigma = (C12 / C6) ** (1/6)
    """
    c6 = float(c6)
    c12 = float(c12)

    if c6 <= 0.0 or c12 <= 0.0:
        raise ValueError(f"Invalid Martini 2 C6/C12 values: C6={c6}, C12={c12}")

    return (c12 / c6) ** (1.0 / 6.0)


def parse_martini_self_sigma(itp_path):
    """Parse Martini [ nonbond_params ] self-pairs into {bead_type: sigma_nm}.

    Only self-pairs (i == j) are used, since a bead's own vdW radius comes
    from its self-interaction, not from any cross term. Handles both
    parameter formats transparently:

        Martini 3:  i  j  func  sigma  epsilon
        Martini 2:  i  j  func  C6     C12

    Molecule topology .itp files (no [ nonbond_params ] section) are not
    force-field parameter files and correctly return {} rather than raising.
    """
    itp_path = Path(itp_path)
    ff_version = detect_martini_ff_version(itp_path)

    sigma_self = {}
    section = None
    saw_nonbond_params = False

    with itp_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]").strip().lower()
                if section == "nonbond_params":
                    saw_nonbond_params = True
                continue

            if line.startswith(";"):
                continue

            if section != "nonbond_params":
                continue

            data = line.split(";")[0].strip()
            if not data:
                continue

            toks = data.split()
            if len(toks) < 5:
                continue

            i = toks[0].strip()
            j = toks[1].strip()

            if i != j:
                continue

            if ff_version == "MARTINI3":
                try:
                    sigma_nm = float(toks[3])
                except ValueError:
                    continue

            elif ff_version == "MARTINI2":
                try:
                    c6 = float(toks[3])
                    c12 = float(toks[4])
                    sigma_nm = martini2_c6_c12_to_sigma_nm(c6, c12)
                except ValueError:
                    continue

            else:
                raise RuntimeError(
                    f"Could not detect Martini version for file: {itp_path.resolve()}\n"
                    "The file contains [ nonbond_params ], so the parser needs to know "
                    "whether column 4/5 mean sigma/epsilon or C6/C12.\n"
                    "Expected first non-empty line to look like one of:\n"
                    "  ;;;;; Martini 3 Force Field: Particle definition and LJ interactions\n"
                    "  ; MARTINI FORCEFIELD V2.2"
                )

            sigma_self[i] = sigma_nm

    if not saw_nonbond_params:
        return {}

    return sigma_self



def martini_sigma_to_radius_angstrom(sigma_nm):
    """Convert a Martini LJ sigma (nm) to a vdW radius in Angstroms.

    Uses the PackMem convention:  radius = 0.5 * 2^(1/6) * sigma
    """
    r_nm = 0.5 * (2.0 ** (1.0 / 6.0)) * float(sigma_nm)
    return r_nm * 10.0

def merge_martini_self_sigma_folder(radius_folder, overwrite=False):
    """Parse Martini self-sigma values from every .itp file in a folder.

    Supports Martini 2 (C6/C12) and Martini 3 (sigma/epsilon) [ nonbond_params ]
    transparently -- see parse_martini_self_sigma. Files with no
    [ nonbond_params ] section are silently skipped, not an error, since
    most molecule .itp files are topology-only.

    overwrite: True lets a later file replace an earlier file's sigma for
               the same bead_type; False (default) keeps the first seen.
    Returns:   (sigma_self, files_used) -- the merged {bead_type: sigma_nm}
               map, and the list of .itp files that actually contributed
               a [ nonbond_params ] self-pair.
    """
    radius_folder = Path(radius_folder)
    itp_files = iter_itp_files(radius_folder)

    if not itp_files:
        raise FileNotFoundError(f"No .itp files found in CG radius folder: {radius_folder.resolve()}")

    merged = {}
    files_used = []

    for itp_file in itp_files:
        one = parse_martini_self_sigma(itp_file)
        if not one:
            continue

        files_used.append(itp_file)
        for bead_type, sigma_nm in one.items():
            if overwrite:
                merged[bead_type] = sigma_nm
            else:
                merged.setdefault(bead_type, sigma_nm)

    return merged, files_used

# ---------------------------------------------------------------------------


# AA parts are WIP
def build_types_map(model, type_folder):
    """Build {resname: {atom_name: atom_type}} for the given model.

    AA: merges every *.lib and *.prepc file in type_folder.
    CG: merges every *.itp file in type_folder.
    """
    model = model.upper().strip()
    type_folder = Path(type_folder)

    types = {}

    if model == "AA":
        merge_lib_folder(types, type_folder, pattern="*.lib", overwrite=False)
        merge_prepc_folder(types, type_folder, pattern="*.prepc", overwrite=False)
        return types

    if model == "CG":
        merge_itp_folder(types, type_folder, pattern="*.itp", overwrite=False)
        return types

    raise ValueError(f"Unknown model: {model!r}. Expected 'AA' or 'CG'.")

def build_radii_map(model, radius_folder):
    """Build {atom_type: radius_angstrom} for the given model.

    AA: merges every *.dat file in radius_folder.
    CG: derives radii from Martini self-sigma across every .itp file.

    Returns: (radii, ff_files_used). ff_files_used is [] for AA, and for CG
             lists every .itp file that contributed a self-pair -- useful
             for confirming which force-field files were actually read.
    """
    model = model.upper().strip()
    radius_folder = Path(radius_folder)

    if model == "AA":
        return merge_radius_folder(radius_folder, pattern="*.dat", overwrite=False), []

    if model == "CG":
        sigma_self, ff_files_used = merge_martini_self_sigma_folder(
            radius_folder=radius_folder,
            overwrite=False,
        )
        radii = {
            bead_type: martini_sigma_to_radius_angstrom(sigma)
            for bead_type, sigma in sigma_self.items()
        }
        if not radii:
            raise RuntimeError(
                f"No CG radii parsed from any .itp file in '{radius_folder.resolve()}'. "
                "Make sure at least one .itp contains [ nonbond_params ] with "
                "self-pairs like 'P1 P1 1 sigma eps'."
            )
        return radii, ff_files_used

    raise ValueError(f"Unknown model: {model!r}. Expected 'AA' or 'CG'.")


def write_atom_info_csv(pdb_file, output_csv, model, type_folder, radius_folder, neutral_resnames):
    """Build results/atom_info.csv: one row per unique (resname, atom_name).

    Columns: resname, atom_name, atom_type, vdw_radius, flag.
    atom_type is "UNKNOWN" and vdw_radius is blank when no force-field file
    defines that (resname, atom_name) pair -- Step 4 treats UNKNOWN beads as
    non-covering, so an unexpectedly high UNKNOWN count usually means the
    wrong type_folder or radius_folder was given. User should always check the file.

    Returns: (counts, unknown_rows, n_pairs, ff_files_used)
        counts:        Counter of flag -> row count ("h"/"t"/"n").
        unknown_rows:  list of (resname, atom_name) that hit UNKNOWN.
        n_pairs:       total unique pairs written.
        ff_files_used: from build_radii_map -- see its docstring.
    """
    pairs = unique_pairs_from_pdb(pdb_file, model)
    types = build_types_map(model, type_folder)
    radii, ff_file_used = build_radii_map(model, radius_folder)

    counts = Counter()
    unknown_rows = []

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["resname", "atom_name", "atom_type", "vdw_radius", "flag"])

        for resname, atom_name in pairs:
            atom_type = types.get(resname, {}).get(atom_name, "UNKNOWN")
            vdw_radius = radii.get(atom_type) if atom_type != "UNKNOWN" else None
            flag = auto_flag_from_residue_and_atom_type(
                resname=resname,
                atom_type=atom_type,
                neutral_resnames=neutral_resnames,
            )

            writer.writerow([resname, atom_name, atom_type, vdw_radius, flag])
            counts[flag] += 1

            if atom_type == "UNKNOWN":
                unknown_rows.append((resname, atom_name))

    return counts, unknown_rows, len(pairs), ff_file_used

def find_default_representative_pdb(input_dir, pattern):
    """Return the first frame PDB matching pattern, sorted, for atom_info.csv.

    Step 1 only needs one representative frame -- see unique_pairs_from_pdb's
    docstring for why scanning just the first frame is sufficient.
    """
    infiles = [Path(p) for p in sorted(glob.glob(str(input_dir / pattern)))]
    if not infiles:
        raise FileNotFoundError(f"No files matched {pattern!r} in {input_dir.resolve()}")
    return infiles[0]

# ==================================================
# Multi-frame PDB splitting
# ==================================================
# Step 1's other output: one PDB per frame, ready for Step 2's tiling/cutting.

def split_multiframe_pdb(multiframe_pdb, output_dir, frame_prefix="frame"):
    """Split one multi-frame PDB into frame_0.pdb, frame_1.pdb, ...

    A bare END line delimits frames; MODEL/ENDMDL wrappers are skipped, not
    treated as boundaries. This is the convention every other frame-index
    reader in this module (read_per_frame_xyz_box, _frame_idx_from_split_filename)
    assumes, so frame_idx stays consistent across the whole pipeline.

    Chunks with no ATOM/HETATM records are dropped (header-only fragments).
    A final frame with no trailing END is still written.

    If a frame's chunk carries no CRYST1 line of its own, the most recent
    CRYST1 seen so far is copied into it -- handles trajectories that store
    the box once near the top instead of repeating it every frame.

    Returns: number of frame files written.
    """
    multiframe_pdb = Path(multiframe_pdb)
    output_dir = Path(output_dir)

    if not multiframe_pdb.is_file():
        raise FileNotFoundError(f"Multi-frame PDB not found: {multiframe_pdb.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_lines = []
    last_cryst1 = None
    n_frames = 0

    def has_atom_records(lines):
        return any(line[:6].strip() in ATOM_RECORDS for line in lines)

    def has_cryst1(lines):
        return any(line[:6].strip() == "CRYST1" for line in lines)

    def write_frame(lines):
        nonlocal n_frames

        if not has_atom_records(lines):
            return False

        out_lines = list(lines)

        if last_cryst1 is not None and not has_cryst1(out_lines):
            out_lines.insert(0, last_cryst1)

        out_path = output_dir / f"{frame_prefix}_{n_frames}.pdb"
        with out_path.open("w", encoding="utf-8") as out:
            for line in out_lines:
                out.write(line.rstrip("\n") + "\n")
            out.write("END\n")

        n_frames += 1
        return True

    with multiframe_pdb.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            rec = line[:6].strip()

            if rec == "CRYST1":
                last_cryst1 = line

            if rec == "END":
                write_frame(frame_lines)
                frame_lines = []
                continue

            if rec in {"MODEL", "ENDMDL"}:
                continue

            frame_lines.append(line)

    write_frame(frame_lines)

    if n_frames == 0:
        raise ValueError(f"No frames with ATOM/HETATM records were found in {multiframe_pdb.resolve()}")

    return n_frames


# ==================================================
# Step 2: tile + cut PDB preparation
# ==================================================
# Generic PDB read/write (parse_pdb, write_pdb) plus the tile-and-cut
# geometry that gives Step 3's surface reconstruction real neighbors across
# the periodic boundary instead of a hard box edge.


def parse_pdb(path):
    """Parse a PDB file into (box, atoms, other_lines).

    box:         (Lx, Ly, Lz, alpha, beta, gamma) from CRYST1, or None if
                 the file has no CRYST1 record.
    atoms:       list of dicts, one per ATOM/HETATM record, with fixed-width
                 PDB fields already split out (serial, name, resName, x/y/z,
                 occupancy, tempFactor, element, charge, ...).
    other_lines: every non-ATOM/HETATM line except END, in original order --
                 this is what preserves CRYST1, REMARK, TITLE, etc. when the
                 file is later rewritten by write_pdb.

    Malformed atom lines are skipped with a warning rather than raising,
    since a single corrupt line in one frame of a long trajectory shouldn't
    abort the whole run.
    """
    box = None
    atoms = []
    other_lines = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            rec = line[:6].strip()

            if rec == "CRYST1":
                try:
                    Lx = float(line[6:15])
                    Ly = float(line[15:24])
                    Lz = float(line[24:33])
                    alpha = float(line[33:40])
                    beta = float(line[40:47])
                    gamma = float(line[47:54])
                    box = (Lx, Ly, Lz, alpha, beta, gamma)
                except ValueError:
                    print(f"[warn] Could not parse CRYST1 in {path}: {line!r}")
                other_lines.append(line)

            elif rec in ATOM_RECORDS:
                try:
                    atoms.append({
                        "record": line[0:6],
                        "serial": int(line[6:11]),
                        "name": line[12:16],
                        "altLoc": line[16],
                        "resName": line[17:21],
                        "chainID": line[21],
                        "resSeq": line[22:26],
                        "iCode": line[26],
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                        "occupancy": line[54:60] if len(line) >= 60 else "",
                        "tempFactor": line[60:66] if len(line) >= 66 else "",
                        "element": line[76:78] if len(line) >= 78 else "  ",
                        "charge": line[78:80] if len(line) >= 80 else "  ",
                    })
                except (ValueError, IndexError) as e:
                    print(f"[warn] Skipping malformed atom line in {path}: {e}")

            elif rec == "END":
                pass
            else:
                other_lines.append(line)

    return box, atoms, other_lines



def estimate_box(atoms):
    """Estimate a box from atom coordinates when a PDB has no CRYST1 record.

    Bounding box of the atoms plus a fixed 2 A pad per side. This is a
    fallback only -- CRYST1 is preferred wherever it's available, since an
    estimated box has no physical meaning beyond "big enough to hold the
    atoms without touching the edge."

    Returns: (Lx, Ly, Lz, 90.0, 90.0, 90.0) -- always assumes orthorhombic.
    """
    if not atoms:
        raise ValueError("Cannot estimate box because no atoms were parsed.")

    xs = [a["x"] for a in atoms]
    ys = [a["y"] for a in atoms]
    zs = [a["z"] for a in atoms]
    pad = 2.0
    Lx = max(xs) - min(xs) + 2 * pad
    Ly = max(ys) - min(ys) + 2 * pad
    Lz = max(zs) - min(zs) + 2 * pad
    print(f"[info] No CRYST1 found. Estimated box: Lx={Lx:.3f}, Ly={Ly:.3f}, Lz={Lz:.3f}")
    return (Lx, Ly, Lz, 90.0, 90.0, 90.0)


def format_cryst1(Lx, Ly, Lz, alpha, beta, gamma, sgroup="P 1", z=1):
    """Format one PDB CRYST1 record from box dimensions and angles."""
    return (
        f"CRYST1"
        f"{Lx:9.3f}{Ly:9.3f}{Lz:9.3f}"
        f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} "
        f"{sgroup:<11s}{z:4d}"
    )


def format_atom(a):
    """Format one atom dict (see parse_pdb) back into a fixed-width PDB ATOM line.

    Blank optional fields (occupancy, tempFactor, element, charge) get PDB-
    standard defaults rather than being left empty, since some downstream
    tools (PyMeshLab, MDAnalysis) reject non-standard blank columns.

    Column offsets must match parse_pdb's read offsets exactly -- if you
    change one, change the other, or every round-tripped PDB will be
    silently misaligned.
    """
    occ = a["occupancy"] if str(a["occupancy"]).strip() else "  1.00"
    bfac = a["tempFactor"] if str(a["tempFactor"]).strip() else "  0.00"
    elem = a["element"] if str(a["element"]).strip() else "  "
    charge = a["charge"] if str(a["charge"]).strip() else "  "

    return (
        f"{a['record']:<6}{a['serial']:5d} "
        f"{a['name']:<4}{a['altLoc']}"
        f"{a['resName'][:4]:<4}"
        f"{a['chainID']}"
        f"{a['resSeq']}{a['iCode']}"
        f"   "
        f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}"
        f"{occ:>6}{bfac:>6}"
        f"          {elem:>2}{charge:>2}"
    )


def write_pdb(path, box, atoms, other_lines=None):
    """Write atoms and box back out to a PDB file.

    other_lines (from parse_pdb) is replayed as-is except that any CRYST1
    line in it is replaced with one built from box, so the box passed here
    always wins over whatever was originally in the file. If other_lines
    has no CRYST1 at all, one is appended.
    """
    if other_lines is None:
        other_lines = []

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    Lx, Ly, Lz, alpha, beta, gamma = box
    has_cryst1 = any(line[:6].strip() == "CRYST1" for line in other_lines)

    with path.open("w", encoding="utf-8") as fh:
        for line in other_lines:
            if line[:6].strip() == "CRYST1":
                fh.write(format_cryst1(Lx, Ly, Lz, alpha, beta, gamma) + "\n")
            else:
                fh.write(line + "\n")

        if not has_cryst1:
            fh.write(format_cryst1(Lx, Ly, Lz, alpha, beta, gamma) + "\n")

        for a in atoms:
            fh.write(format_atom(a) + "\n")

        fh.write("END\n")



def xy_bounds_and_center_from_atoms(atoms):
    """Return x/y bounds and center from atom coordinates as stored in the PDB.
    """
    if not atoms:
        raise ValueError("Cannot calculate x/y center because no atoms were parsed.")

    xs = [a["x"] for a in atoms]
    ys = [a["y"] for a in atoms]

    xlo = min(xs)
    xhi = max(xs)
    ylo = min(ys)
    yhi = max(ys)
    cx = 0.5 * (xlo + xhi)
    cy = 0.5 * (ylo + yhi)

    return xlo, xhi, ylo, yhi, cx, cy


def tile_atoms(atoms, Lx, Ly, nx=3, ny=3):
    """Tile atoms nx by ny around the original box center.

    Offsets run -L, 0, +L (for nx=ny=3), so the original copy is the
    middle tile, not the lower-left one. That is what lets the original
    box, the tiled system, and the later cut window all share the same
    physical x/y center.
    """
    if nx < 1 or ny < 1:
        raise ValueError(f"nx and ny must be positive, got nx={nx}, ny={ny}")

    x_offsets = [ix - (nx // 2) for ix in range(nx)]
    y_offsets = [iy - (ny // 2) for iy in range(ny)]

    tiled = []
    serial = 1
    for ix in x_offsets:
        for iy in y_offsets:
            for a in atoms:
                b = dict(a)
                b["x"] = a["x"] + ix * Lx
                b["y"] = a["y"] + iy * Ly
                b["serial"] = serial
                tiled.append(b)
                serial += 1
                if serial > 99999:
                    serial = 1
    return tiled



def cut_center_box_atoms(atoms, center_x, center_y, cut_Lx, cut_Ly):
    """Keep a cut_Lx by cut_Ly window centered on (center_x, center_y).

    Returns: (kept_atoms, (xlo, xhi, ylo, yhi, cx, cy))
    """
    cx = float(center_x)
    cy = float(center_y)

    xlo = cx - cut_Lx / 2.0
    xhi = cx + cut_Lx / 2.0
    ylo = cy - cut_Ly / 2.0
    yhi = cy + cut_Ly / 2.0

    kept = []
    serial = 1

    for a in atoms:
        if xlo <= a["x"] <= xhi and ylo <= a["y"] <= yhi:
            b = dict(a)
            # Preserve the original/tiled coordinate frame -- do not
            # subtract xlo/ylo here.
            b["x"] = a["x"]
            b["y"] = a["y"]
            b["serial"] = serial
            kept.append(b)
            serial += 1
            if serial > 99999:
                serial = 1

    return kept, (xlo, xhi, ylo, yhi, cx, cy)


def process_one_pdb(
    infile,
    outfile,
    cut_scale=1.5,
    nx=3,
    ny=3,
    keep_tiled=False,
    tiled_dir=None,
    frame_box_xyz=None,
):
    """Tile one frame 3x3 and cut a centered cut_scale-times-original window back out.
    
    
    every step
    below uses the coordinate center of the original atoms as the shared
    center -- never the box lengths alone.

    frame_box_xyz, when given, is this frame's (Lx, Ly, Lz) from the
    canonical results/box_size.csv (see get_or_build_frame_box_csv). This is
    the box actually used for tiling

    Returns: True if the frame had atoms and was written, False if skipped.
    """
    infile = Path(infile)
    outfile = Path(outfile)

    print(f"\n=== Processing: {infile.name} ===")

    box, atoms, other_lines = parse_pdb(infile)

    if not atoms:
        print("[warn] No atoms found, skipping.")
        return False

    if frame_box_xyz is not None:
        old_Lx, old_Ly, old_Lz = (float(v) for v in frame_box_xyz)
        alpha, beta, gamma = (box[3], box[4], box[5]) if box is not None else (90.0, 90.0, 90.0)
        print(
            f"[box_size] {infile.name}: using per-frame box from box_size.csv -> "
            f"Lx={old_Lx:.3f}, Ly={old_Ly:.3f}, Lz={old_Lz:.3f}"
        )
    else:
        if box is None:
            box = estimate_box(atoms)
        old_Lx, old_Ly, old_Lz, alpha, beta, gamma = box

    orig_xlo, orig_xhi, orig_ylo, orig_yhi, original_center_x, original_center_y = (
        xy_bounds_and_center_from_atoms(atoms)
    )

    if cut_scale <= 0:
        raise ValueError(f"cut_scale must be positive, got {cut_scale!r}")

    cut_Lx = old_Lx * cut_scale
    cut_Ly = old_Ly * cut_scale

    if cut_Lx > old_Lx * nx or cut_Ly > old_Ly * ny:
        raise ValueError(
            "Requested cut box is larger than the tiled box. "
            f"cut=({cut_Lx:.3f}, {cut_Ly:.3f}), "
            f"tiled=({old_Lx * nx:.3f}, {old_Ly * ny:.3f})"
        )

    # Preserve the original coordinate frame -- wrapping into 0..L here
    # would move a zero-centered system and break the shared center below.
    base_atoms = atoms

    tiled_atoms = tile_atoms(base_atoms, old_Lx, old_Ly, nx=nx, ny=ny)
    tiled_box = (old_Lx * nx, old_Ly * ny, old_Lz, alpha, beta, gamma)

    temp_path = None
    try:
        if keep_tiled:
            if tiled_dir is None:
                tiled_dir = outfile.parent / "tiled"
            tiled_dir = Path(tiled_dir)
            tiled_dir.mkdir(parents=True, exist_ok=True)
            temp_path = tiled_dir / f"{infile.stem}_tiled.pdb"
        else:
            fd, temp_name = tempfile.mkstemp(suffix=".pdb", prefix="tiled_")
            os.close(fd)
            temp_path = Path(temp_name)

        write_pdb(temp_path, tiled_box, tiled_atoms, other_lines)

        tiled_box2, tiled_atoms2, _ = parse_pdb(temp_path)
        kept, info = cut_center_box_atoms(
            atoms=tiled_atoms2,
            center_x=original_center_x,
            center_y=original_center_y,
            cut_Lx=cut_Lx,
            cut_Ly=cut_Ly,
        )

        # The saved frame describes the cut_scale-times-original cut size.
        # Coordinates stay in the original frame so the cut region shares
        # the same center as the original and tiled systems.
        new_box = (cut_Lx, cut_Ly, tiled_box2[2], tiled_box2[3], tiled_box2[4], tiled_box2[5])
        centered_other_lines = [
            line for line in other_lines
            if not line.startswith("REMARK CENTERED_CUT_XY")
        ]
        centered_other_lines.insert(
            0,
            (
                "REMARK ORIGINAL_XY_BOUNDS "
                f"X=[{orig_xlo:.3f},{orig_xhi:.3f}] "
                f"Y=[{orig_ylo:.3f},{orig_yhi:.3f}] "
                f"CENTER=({original_center_x:.3f},{original_center_y:.3f})"
            ),
        )
        centered_other_lines.insert(
            1,
            (
                "REMARK CENTERED_CUT_XY "
                f"X=[{info[0]:.3f},{info[1]:.3f}] "
                f"Y=[{info[2]:.3f},{info[3]:.3f}] "
                f"CENTER=({info[4]:.3f},{info[5]:.3f})"
            ),
        )
        write_pdb(outfile, new_box, kept, centered_other_lines)

        xlo, xhi, ylo, yhi, cx, cy = info
        print(f"input box=({old_Lx:.3f}, {old_Ly:.3f})  cut box=({cut_Lx:.3f}, {cut_Ly:.3f})")
        print(f"original atom bounds X=[{orig_xlo:.3f}, {orig_xhi:.3f}]  Y=[{orig_ylo:.3f}, {orig_yhi:.3f}]")
        print(f"shared coordinate center=({cx:.3f}, {cy:.3f})  cut X=[{xlo:.3f}, {xhi:.3f}]  Y=[{ylo:.3f}, {yhi:.3f}]")
        print(f"input atoms={len(base_atoms)}  tiled atoms={len(tiled_atoms)}  output atoms={len(kept)}")
        print(f"wrote: {outfile}")

        return True

    finally:
        if temp_path and (not keep_tiled) and temp_path.exists():
            temp_path.unlink()


def _resolve_step2_worker_count(workers, n_jobs):
    """Clamp Step 2's requested worker count to a safe, usable range.

    Never below 1, never above n_jobs (no point spawning more workers than
    there are frames to process). Falls back to 1 on a bad/missing value
    rather than raising, since this is a performance knob, not a
    correctness one.
    """
    try:
        worker_count = int(workers or 1)
    except (TypeError, ValueError):
        worker_count = 1

    worker_count = max(1, worker_count)
    return max(1, min(worker_count, max(1, int(n_jobs))))


def _process_one_pdb_capture(infile, outfile, cut_scale, nx, ny, keep_tiled, tiled_dir, frame_box_xyz=None):
    """Run one Step 2 tile/cut job in a worker process, capturing its stdout.

    Capturing output here (instead of letting it print directly) keeps the
    terminal readable when several frames are processed concurrently --
    the parent process prints each worker's captured text as one block
    instead of every worker interleaving line by line.

    Returns: (frame_filename, ok, captured_stdout_text)
    """
    infile = Path(infile)
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = process_one_pdb(
            infile=infile,
            outfile=Path(outfile),
            cut_scale=cut_scale,
            nx=nx,
            ny=ny,
            keep_tiled=keep_tiled,
            tiled_dir=(None if tiled_dir is None else Path(tiled_dir)),
            frame_box_xyz=frame_box_xyz,
        )
    return infile.name, bool(ok), buf.getvalue()



def process_pdb_folder(
    input_dir,
    output_dir,
    cut_scale=1.5,
    nx=3,
    ny=3,
    pattern="frame_*.pdb",
    keep_tiled=False,
    tiled_dir=None,
    workers=1,
    frame_box_xyz=None,
    frame_prefix="frame",
):
    """Tile and cut every split frame PDB in input_dir, in parallel if workers > 1.

    Each frame's box is looked up in frame_box_xyz up front, before any
    worker process is spawned, so a missing frame fails fast with a clear
    message instead of surfacing deep inside a worker.

    Returns: (n_ok, n_total) -- frames successfully written vs frames found.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if keep_tiled and tiled_dir is not None:
        Path(tiled_dir).mkdir(parents=True, exist_ok=True)

    infiles = [Path(p) for p in sorted(glob.glob(str(input_dir / pattern)))]
    if not infiles:
        raise FileNotFoundError(f"No files matched {pattern!r} in {input_dir.resolve()}")

    # Guard against accidentally overwriting the raw split PDBs.
    if input_dir.resolve() == output_dir.resolve():
        raise RuntimeError(
            "output_dir is the same as input_dir. "
            "This would overwrite your original PDBs. Please use a different output folder."
        )

    print(f"Found {len(infiles)} PDB file(s) for tiling/cut.")
    progress.set_total(len(infiles))
    # Resolve every frame's box_size.csv entry before spawning any worker,
    # so a missing frame index fails immediately with a clear message
    # rather than deep inside a worker process.
    frame_boxes = [None] * len(infiles)
    if frame_box_xyz is not None:
        for i, infile in enumerate(infiles):
            frame_idx = _frame_idx_from_split_filename(infile, frame_prefix)
            if frame_idx not in frame_box_xyz:
                raise KeyError(
                    f"Frame {frame_idx} ({infile.name}) has no matching row in "
                    "box_size.csv. Delete results/box_size.csv and rerun to rebuild it."
                )
            frame_boxes[i] = frame_box_xyz[frame_idx]

    worker_count = _resolve_step2_worker_count(workers, len(infiles))
    if worker_count <= 1:
        n_ok = 0
        for infile, fbox in zip(infiles, frame_boxes):
            outfile = output_dir / f"{infile.stem}.pdb"
            ok = process_one_pdb(
                infile=infile,
                outfile=outfile,
                cut_scale=cut_scale,
                nx=nx,
                ny=ny,
                keep_tiled=keep_tiled,
                tiled_dir=tiled_dir,
                frame_box_xyz=fbox,
            )
            if ok:
                n_ok += 1
            progress.tick()
        return n_ok, len(infiles)

    print(
        f"Step 2 tiling/cutting running with {worker_count} worker processes "
        f"over {len(infiles)} frame(s)."
    )

    n_ok = 0
    futures = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for infile, fbox in zip(infiles, frame_boxes):
            outfile = output_dir / f"{infile.stem}.pdb"
            fut = executor.submit(
                _process_one_pdb_capture,
                infile,
                outfile,
                cut_scale,
                nx,
                ny,
                keep_tiled,
                tiled_dir,
                fbox,
            )
            futures[fut] = infile

        for fut in as_completed(futures):
            infile = futures[fut]
            try:
                frame_name, ok, text = fut.result()
            except Exception as exc:
                raise RuntimeError(f"Step 2 tiling/cutting failed for {infile.name}") from exc

            if text:
                print(text, end="" if text.endswith("\n") else "\n")
            print(f"[Step 2 worker done] {frame_name}: {'OK' if ok else 'SKIPPED'}")
            if ok:
                n_ok += 1
            progress.tick()

    return n_ok, len(infiles)



