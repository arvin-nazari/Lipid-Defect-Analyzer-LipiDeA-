"""Step 7 fitting and analysis, plus Step 8/9 packing-defect-constant post-analyses.

Step 7 (helpers.py's run_step7_fitting) has five independently-toggleable
analyses, each writing its own summary .txt (and PNG/PDF plot where
applicable) from results/pruned.csv (Step 6's output):
  fitting                run_simple_fit -- pi fit via P(A) log-linear decay
  cluster_count          run_cluster_count
  coverage               run_coverage
  distribution           run_distribution
  defect_type_coverage    run_defect_type_coverage -- reads pruned NPZs
                          directly, not pruned.csv, since defect_type is
                          per-triangle and pruned.csv is per-cluster

Two more entry points are post-analyses run after Step 8 and Step 9
respectively, both reusing the same P(A) log-linear fit machinery as Step
7's fitting analysis, just applied per-region or per-curvature-segment
instead of pooled:
  run_defect_constant_fit         Step 8 -- pi fit split by monolayer/bilayer
  run_curvature_segment_fit       Step 9 -- pi fit split by curvature (H) bin

finite_positive / histogram_probability / fit_log_probability /
set_plot_style / _style_axis / join_analysis_sections below are the shared
machinery all seven entry points build on.
"""

import csv
import math
import os
from pathlib import Path
import glob
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MultipleLocator

# ==================================================
# Small, production Step-7 fitting helper
# ==================================================

def finite_positive(values):
    """Filter a sequence down to finite, strictly positive values, as a float array.

    Used everywhere an area or H value is loaded from CSV or NPZ -- a
    non-positive or non-finite value (NaN, inf, zero, or a parse artifact)
    is silently dropped rather than raising, since a single malformed row
    in a large CSV shouldn't abort the whole analysis.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]
    return arr


def load_cluster_areas(csv_path):
    """Load every defect/cluster area from Step 6's pruned.csv, as a flat array.

    Supported area column names (first match wins) are intentionally small
    but compatible with the existing workflow and the old fit.py script (should probably remove it later):
        - area_angstrom2       current Step-6 pruned.csv

    Raises RuntimeError if the CSV has rows but none survive
    finite_positive (e.g. every area was zero or unparseable).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Step-7 fitting input CSV not found: {csv_path.resolve()}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {csv_path}")

        columns = [str(c).strip() for c in reader.fieldnames]
        area_col = None
        for candidate in ("area_angstrom2", "patch_area_A2", "area"):
            if candidate in columns:
                area_col = candidate
                break

        if area_col is None:
            raise ValueError(
                "Step-7 fitting CSV must contain one area column: "
                "area_angstrom2, patch_area_A2, or area. "
                f"Found columns: {columns}"
            )

        areas = []
        for row in reader:
            try:
                areas.append(float(row[area_col]))
            except (TypeError, ValueError):
                continue

    areas = finite_positive(areas)
    if areas.size == 0:
        raise RuntimeError(f"No positive defect areas found in {csv_path}")
    return areas


def histogram_probability(values, bin_width=1.0):
    """Build a PackMem/fit.py-style normalized histogram of P(A) vs. area.

    BINNING DETAIL WORTH KNOWING: bin edges are shifted by half a bin width
    (edges = 0.5*bw, 1.5*bw, 2.5*bw, ...), so that the returned bin
    midpoints land exactly on whole multiples of bin_width (bw, 2*bw, 3*bw,
    ...) -- matching the reference PackMem/fit.py convention.

    values:    raw values (areas); finite_positive is applied internally,
               so callers don't need to pre-filter.
    bin_width: bin width in the same units as values (Å^2 for areas).
    Returns: (bin_mids, probabilities, counts, bin_edges). All four are
             empty arrays if values had no positive finite entries.
             probabilities is all-zero (not NaN) if total count is zero.
    """
    v = finite_positive(values)
    if v.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    bw = float(bin_width)
    if bw <= 0.0:
        raise ValueError(f"bin_width must be > 0, got {bin_width!r}")

    vmax = float(np.max(v))
    edges = np.arange(0.0, math.floor(vmax / bw) * bw + bw, bw) + 0.5 * bw
    if edges.size < 2:
        edges = np.array([0.0, vmax + bw], dtype=float)

    counts, bin_edges = np.histogram(v, bins=edges)
    mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total = int(counts.sum())
    probabilities = counts.astype(float) / float(total) if total > 0 else counts.astype(float)
    return mids, probabilities, counts.astype(int), bin_edges


def fit_log_probability(bin_mids, probabilities, counts, min_defect_size, max_defect_size, min_probability=None):
    """Fit ln(P(A)) = intercept + slope * A over a requested area/probability window, and derive pi.

    This is the core packing-defect-constant fit used by every entry point
    in this file (Step 7's pooled fitting, Step 8's per-region fit, Step
    9's per-curvature-segment fit) -- only the input histogram differs
    between them.

    Selects histogram bins where: the bin is finite and has positive
    probability, the probability is at or above min_probability (a noise
    floor -- bins below it are excluded from the fit but still shown in
    plots), and the bin's area falls within [min_defect_size,
    max_defect_size]. Fits a straight line to ln(P) vs. A via ordinary
    least squares over exactly those bins.

    pi (the packing-defect constant) is 1/|slope| -- the area at which
    P(A) would have decayed by a factor of e under the fitted exponential.
    NaN if the fitted slope is exactly zero.

    bin_mids, probabilities, counts: from histogram_probability.
    min_defect_size, max_defect_size: the area window to fit within (Å^2).
    min_probability: probability floor; None means no floor (equivalent to 0).
    Returns: (summary, fit_mask) -- summary is a dict of fit statistics
             (see below), fit_mask is a boolean array aligned with bin_mids
             marking which bins were used in the fit.
    Raises: ValueError if min_defect_size >= max_defect_size, or if
            min_probability is given but not in (0, 1]. RuntimeError if
            fewer than 3 bins survive the window (not enough to fit a line
            with a meaningful residual).


    """
    min_a = float(min_defect_size)
    max_a = float(max_defect_size)
    if min_a >= max_a:
        raise ValueError(
            f"min_defect_size must be smaller than max_defect_size. Got {min_a} and {max_a}."
        )

    if min_probability is None:
        min_p = 0.0
    else:
        min_p = float(min_probability)
        if not (np.isfinite(min_p) and 0.0 < min_p <= 1.0):
            raise ValueError(f"min_probability must satisfy 0 < P <= 1. Got {min_probability!r}.")

    x_all = np.asarray(bin_mids, dtype=float)
    y_all = np.asarray(probabilities, dtype=float)
    counts_all = np.asarray(counts, dtype=int)

    fit_mask = (
        np.isfinite(x_all)
        & np.isfinite(y_all)
        & (y_all > 0.0)
        & (y_all >= min_p)
        & (x_all >= min_a)
        & (x_all <= max_a)
    )

    x = x_all[fit_mask]
    y = y_all[fit_mask]
    fit_counts = counts_all[fit_mask]

    if x.size < 3:
        raise RuntimeError(
            "Not enough histogram bins in the requested Step-7 fit window after applying "
            "the area and minimum-probability cutoffs. "
            f"Found {x.size}; need at least 3. Try widening min/max defect size "
            "or lowering min_probability."
        )

    log_y = np.log(y)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, log_y, rcond=None)[0]
    intercept = float(beta[0])
    slope = float(beta[1])
    fitted = X @ beta
    residual = log_y - fitted

    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0.0 else float("nan")
    dof = max(int(x.size) - 2, 1)
    rmse = float(np.sqrt(sse / dof))

    pi_a2 = float(abs(1.0 / slope)) if slope != 0.0 else float("nan")

    summary = {
        "pi_A2": pi_a2,
        "slope": slope,
        "intercept": intercept,
        "b_prefactor_exp_intercept": float(np.exp(intercept)),
        "r2": r2,
        "rmse_log_probability": rmse,
        "n_fit_points": int(x.size),
        "n_total_histogram_bins_positive": int(np.sum(np.isfinite(y_all) & (y_all > 0.0))),
        "n_total_defects_in_fit_bins": int(np.sum(fit_counts)),
        "requested_min_defect_size_A2": min_a,
        "requested_max_defect_size_A2": max_a,
        "applied_min_probability": float(min_p),
        "x_fit_min": float(np.min(x)),
        "x_fit_max": float(np.max(x)),
        "y_fit_min": float(np.min(y)),
        "y_fit_max": float(np.max(y)),
        "warning_slope_positive": bool(slope > 0.0),
    }
    return summary, fit_mask


def set_plot_style(font_size=10, font_family="Arial"):
    """Apply a shared matplotlib rcParams style used by every plot in this file.

    font_family is applied only if it's actually installed and discoverable
    by matplotlib's font manager -- silently falls back to matplotlib's
    default font otherwise, rather than raising on a machine without Arial.
    """
    rc = {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": max(font_size - 1, 6),
        "ytick.labelsize": max(font_size - 1, 6),
        "axes.linewidth": 1.1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
    if font_family:
        try:
            from matplotlib import font_manager
            available = {f.name for f in font_manager.fontManager.ttflist}
            if font_family in available:
                rc["font.family"] = font_family
        except Exception:
            pass
    plt.rcParams.update(rc)


def _style_axis(ax):
    """Apply shared axis styling (grid, spines, tick marks) used by every plot in this file."""
    ax.grid(True, which="major", axis="both", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(length=4.0, width=1.0)



def save_pdf_fit_plot(bin_mids, probabilities, fit_mask, summary, output_base, dpi=300, min_probability=None):
    """Save Step 7's main P(A) fit plot as PNG and PDF, styled like the original fit.py/PackMem plots.

    The underlying fit is still ln(P(A)) = intercept + slope*A (see
    fit_log_probability), but what's actually plotted is normal probability
    P(A) on a logarithmic y-axis.


    bin_mids, probabilities: the full histogram, from histogram_probability.
    fit_mask:      boolean array aligned with bin_mids, from fit_log_probability.
    summary:       the fit_log_probability summary dict -- pi_A2, r2,
                   n_fit_points, intercept, slope are read from it directly
                   for the fit line and the in-plot stats box.
    output_base:   Path without extension; .png and .pdf are both written.
    min_probability: drawn as a dashed reference line if given and > 0.
    Returns: (png_path, pdf_path).
    Raises: RuntimeError if fit_mask has no True entries after filtering to
            finite, positive, plottable bins -- should not happen in
            practice, since fit_log_probability already requires at least
            3 fit points to succeed.
    """
    valid = np.isfinite(bin_mids) & np.isfinite(probabilities) & (probabilities > 0.0)
    x = np.asarray(bin_mids[valid], dtype=float)
    y = np.asarray(probabilities[valid], dtype=float)
    fm = np.asarray(fit_mask[valid], dtype=bool)

    if not np.any(fm):
        raise RuntimeError("Internal error: fit mask contains no valid plotted bins.")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        x[~fm], y[~fm],
        s=80,
        facecolors="white",
        edgecolors="#9E9E9E",
        linewidths=0.9,
        label="Outside fit window",
        zorder=2,
    )

    ax.scatter(
        x[fm], y[fm],
        s=120,
        color="#1f77b4",
        edgecolors="black",
        linewidths=0.35,
        label="Fit bins",
        zorder=3,
    )

    if np.any(fm):
        xline = np.linspace(np.min(x[fm]), np.max(x[fm]), 300)
        yline = np.exp(summary["intercept"] + summary["slope"] * xline)
        ax.plot(
            xline, yline,
            color="#C0392B",
            linewidth=3,
            label="Exponential fit",
            zorder=4,
        )
        ax.axvspan(
            np.min(x[fm]), np.max(x[fm]),
            color="#1f77b4",
            alpha=0.08,
            zorder=1,
        )

    if min_probability is not None:
        min_probability = float(min_probability)
        if np.isfinite(min_probability) and min_probability > 0.0:
            ax.axhline(
                min_probability,
                color="#555555",
                linestyle="--",
                linewidth=1.8,
                zorder=5,
            )

    ax.set_xlabel(r"Defect area $A$ ($\mathrm{\AA}^2$)")
    ax.set_ylabel(r"Probability $P(A)$")
    ax.set_title("Probability fit", pad=12)

    # Show normal probabilities on a logarithmic probability axis.
    # This avoids compressing the exponential tail while keeping the y-axis labels as P(A).
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))

    # Always show the full histogram on the x-axis, not only the fitted window.
    # This keeps all positive and empty area bins within the plotted x range.
    full_x = np.asarray(bin_mids, dtype=float)
    full_x = full_x[np.isfinite(full_x)]
    if full_x.size > 1:
        dx = np.diff(np.sort(np.unique(full_x)))
        dx = dx[dx > 0.0]
        bin_step = float(np.median(dx)) if dx.size else 1.0
    else:
        bin_step = 1.0

    plot_right = float(np.max(full_x)) + 0.5 * bin_step if full_x.size else float(np.max(x))
    plot_right = max(plot_right, float(np.max(x[fm])) + 0.5 * bin_step)
    ax.set_xlim(left=0.0, right=plot_right)

    # Keep Matplotlib's default x tick labeling.
    # The x-limit above still shows the full histogram/bin range.

    visible = (x >= 0.0) & (x <= plot_right) & (y > 0.0)
    positive_y = y[visible] if np.any(visible) else y[y > 0.0]
    y_candidates = [float(np.min(positive_y)), float(np.min(y[fm]))]
    if min_probability is not None and np.isfinite(float(min_probability)) and float(min_probability) > 0.0:
        y_candidates.append(float(min_probability))
    ymin = max(min(y_candidates) * 0.5, np.finfo(float).tiny)
    ymax = min(max(float(np.max(positive_y)) * 1.8, float(np.max(y[fm])) * 1.8), 1.0)
    if ymin < ymax:
        ax.set_ylim(bottom=ymin, top=ymax)

    stats_text = (
        f"$\\pi$ = {summary['pi_A2']:.2f} $\\mathrm{{\\AA}}^2$\n"
        f"$R^2$ = {summary['r2']:.3f}\n"
        f"n = {summary['n_fit_points']}"
    )

    ax.text(
        0.03, 0.97,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#CCCCCC",
            alpha=0.95,
        ),
    )

    _style_axis(ax)
    ax.grid(True, which="minor", axis="y", color="#EAEAEA", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path



def write_text_summary(summary, summary_path):
    """Write Step 7 fitting's per-run diagnostics as a plain-text summary file.

    Every value here comes from the summary dict run_simple_fit assembles --
    the fit_log_probability output plus the extra keys run_simple_fit adds
    afterward (input_csv, output_prefix, bin_width_A2, n_total_defects,
    biggest/average/smallest_area_A2, png_plot, pdf_plot). This function
    assumes all of those keys are already present; it does no fitting or
    computation of its own.


    Returns summary_path, unchanged, for convenience chaining.
    """
    lines = [
        "=== Pi fit diagnostics ===",
        f"input csv = {summary['input_csv']}",
        f"output prefix = {summary['output_prefix']}",
        f"bin width = {float(summary['bin_width_A2']):.4f} Å^2",
        f"requested min defect size = {float(summary['requested_min_defect_size_A2']):.4f} Å^2",
        f"requested max defect size = {float(summary['requested_max_defect_size_A2']):.4f} Å^2",
        f"minimum probability power = {float(summary['min_probability_power']):g}",
        f"minimum probability = {float(summary['min_probability']):.6g}",
        f"total defects = {int(summary['n_total_defects'])}",
        f"positive histogram bins = {int(summary['n_total_histogram_bins_positive'])}",
        f"pi = {float(summary['pi_A2']):.4f} Å^2",
        f"R^2 = {float(summary['r2']):.4f}",
        f"slope = {float(summary['slope']):.6g}",
        f"intercept = {float(summary['intercept']):.6g}",
        f"fit points = {int(summary['n_fit_points'])}",
        f"defects in fit bins = {int(summary['n_total_defects_in_fit_bins'])}",
        f"fitting range = {float(summary['x_fit_min']):.4f} to {float(summary['x_fit_max']):.4f} Å^2",
        f"biggest area = {float(summary['biggest_area_A2']):.4f} Å^2",
        f"average area = {float(summary['average_area_A2']):.4f} Å^2",
        f"smallest area = {float(summary['smallest_area_A2']):.4f} Å^2",
        f"PNG plot = {summary['png_plot']}",
        f"PDF plot = {summary['pdf_plot']}",
    ]

    if bool(summary.get("warning_slope_positive", False)):
        lines.append(
            "WARNING: fitted slope is positive. "
            "The exponential decay assumption is likely failing for this window."
        )

    summary_path.write_text(join_analysis_sections(["\n".join(lines)]), encoding="utf-8")
    return summary_path


def run_simple_fit(
    cluster_csv,
    output_dir,
    output_prefix,
    min_defect_size,
    max_defect_size,
    min_probability_power,
    bin_width=1.0,
    dpi=300,
):
    """Step 7's "fitting" analysis: load areas, fit P(A) decay, save plot + text summary.

    Flow: load_cluster_areas -> histogram_probability -> fit_log_probability
    -> save_pdf_fit_plot -> write_text_summary.

    output_prefix: used as-is for naming (<prefix>_pdf_fit.png/.pdf,
                   <prefix>_fit_summary.txt) -- only the filename component
                   is kept via Path(...).name, so a prefix containing path
                   separators is silently flattened to its basename rather
                   than creating subdirectories.
    Returns: the summary dict (see write_text_summary's docstring for what
             it contains), with summary_txt added pointing at the written
             text file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(str(output_prefix).strip()).name
    if not prefix:
        raise ValueError("output_name_prefix cannot be empty.")

    set_plot_style(font_size=10, font_family="Arial")

    min_probability_power = float(min_probability_power)
    min_probability = float(10.0 ** min_probability_power)
    if not (np.isfinite(min_probability) and 0.0 < min_probability <= 1.0):
        raise ValueError(
            "min_probability must be given as a base-10 power that produces "
            f"0 < probability <= 1. Got power={min_probability_power!r}, "
            f"probability={min_probability!r}."
        )

    areas = load_cluster_areas(cluster_csv)
    x_all, y_all, counts_all, _ = histogram_probability(areas, bin_width=bin_width)
    summary, fit_mask = fit_log_probability(
        x_all,
        y_all,
        counts_all,
        min_defect_size=min_defect_size,
        max_defect_size=max_defect_size,
        min_probability=min_probability,
    )

    summary.update({
        "min_probability_power": min_probability_power,
        "min_probability": min_probability,
    })

    output_base = output_dir / f"{prefix}_pdf_fit"
    png_path, pdf_path = save_pdf_fit_plot(
        x_all,
        y_all,
        fit_mask,
        summary,
        output_base=output_base,
        dpi=dpi,
        min_probability=min_probability,
    )

    summary.update({
        "input_csv": str(Path(cluster_csv)),
        "output_prefix": prefix,
        "bin_width_A2": float(bin_width),
        "n_total_defects": int(areas.size),
        "biggest_area_A2": float(np.max(areas)),
        "average_area_A2": float(np.mean(areas)),
        "smallest_area_A2": float(np.min(areas)),
        "png_plot": str(png_path),
        "pdf_plot": str(pdf_path),
    })

    summary_path = output_dir / f"{prefix}_fit_summary.txt"
    write_text_summary(summary, summary_path)
    summary["summary_txt"] = str(summary_path)
    return summary



# ==================================================
# Section-based analysis summary support
# ==================================================
# Every analysis writes one section. Sections are joined by this dashed line.
# To add a new analysis: build its section text and append it to the section
# list in run_analysis_summary(), and it will be separated automatically.


ANALYSIS_SECTION_SEPARATOR = "-" * 60


def join_analysis_sections(sections):
    """Join one or more analysis section-text blocks with a dashed separator line.

    Blank or falsy entries in sections are dropped before joining, not
    turned into empty separator-only blocks.
    """
    clean = [s.strip("\n") for s in sections if s and s.strip()]
    return ("\n" + ANALYSIS_SECTION_SEPARATOR + "\n").join(clean) + "\n"


def load_cluster_rows(csv_path):
    """Load (frame, side, area) rows from Step-6 pruned.csv.

    side is recovered from the 'cluster' label prefix (upper:/lower:) that
    write_clusters_csv_from_pruned_npz in cleanup.py writes (e.g.
    "upper:3"). area comes from the first available area column -- same
    three-name search as load_cluster_areas above (area_angstrom2,
    patch_area_A2, area); the two lists are not unified, keep both in sync
    if a new column name is ever added.

    Used by count_clusters_per_frame, sum_areas_per_frame, and
    cluster_stats_per_frame -- the shared row-loading step behind Step 7's
    cluster_count, coverage, and distribution analyses.


    Returns: list of (frame:int, side:str, area:float|None) tuples, one per
             CSV row (excluding rows whose frame itself doesn't parse).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Cluster input CSV not found: {csv_path.resolve()}")

    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = [str(c).strip() for c in (reader.fieldnames or [])]
        if "frame" not in columns:
            raise ValueError(f"Cluster CSV must contain a 'frame' column. Found: {columns}")
        label_col = "cluster" if "cluster" in columns else None
        area_col = None
        for candidate in ("area_angstrom2", "patch_area_A2", "area"):
            if candidate in columns:
                area_col = candidate
                break

        for row in reader:
            try:
                frame = int(float(row["frame"]))
            except (TypeError, ValueError, KeyError):
                continue
            label = str(row.get(label_col, "")).strip().lower() if label_col else ""
            side = "upper" if label.startswith("upper") else "lower" if label.startswith("lower") else "other"
            area = None
            if area_col is not None:
                try:
                    area = float(row[area_col])
                except (TypeError, ValueError, KeyError):
                    area = None
            rows.append((frame, side, area))
    return rows


def count_clusters_per_frame(csv_path):
    """Return a per-frame cluster-count table: {frame: {"upper": n, "lower": n, "total": n}}.

    """
    from collections import defaultdict

    counts = defaultdict(lambda: {"upper": 0, "lower": 0, "total": 0})
    for frame, side, _area in load_cluster_rows(csv_path):
        if side in ("upper", "lower"):
            counts[frame][side] += 1
        counts[frame]["total"] += 1
    return dict(counts)


def build_cluster_count_section(csv_path):
    """Build the 'defect clusters per frame' summary section text."""
    per_frame = count_clusters_per_frame(csv_path)

    lines = ["=== Defect clusters per frame ==="]
    if not per_frame:
        lines.append("No cluster rows found.")
        return "\n".join(lines)

    frames = sorted(per_frame)
    total_upper = sum(per_frame[fr]["upper"] for fr in frames)
    total_lower = sum(per_frame[fr]["lower"] for fr in frames)
    total_all = sum(per_frame[fr]["total"] for fr in frames)
    n_frames = len(frames)

    lines.append(f"frames with clusters = {n_frames}")
    lines.append(f"total clusters = {total_all} (upper = {total_upper}, lower = {total_lower})")
    lines.append(f"mean clusters per frame = {total_all / n_frames:.4f}")
    lines.append("")
    lines.append(f"{'frame':>8}  {'upper':>6}  {'lower':>6}  {'total':>6}")
    for fr in frames:
        c = per_frame[fr]
        lines.append(f"{fr:>8}  {c['upper']:>6}  {c['lower']:>6}  {c['total']:>6}")

    return "\n".join(lines)


def run_cluster_count(cluster_csv, output_dir, output_prefix):
    """Step 7's "cluster_count" analysis. Writes its own separate summary .txt file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(str(output_prefix).strip()).name
    if not prefix:
        raise ValueError("output_name_prefix cannot be empty.")

    section = build_cluster_count_section(cluster_csv)
    summary_path = output_dir / f"{prefix}_cluster_counts.txt"
    summary_path.write_text(join_analysis_sections([section]), encoding="utf-8")

    per_frame = count_clusters_per_frame(cluster_csv)
    return {
        "cluster_count_txt": str(summary_path),
        "n_frames": len(per_frame),
        "n_clusters_total": sum(v["total"] for v in per_frame.values()),
    }


# ==================================================
# Defect-type coverage (neutral lipid vs. phospholipid tail)
# ==================================================

_PRUNED_NPZ_RE = re.compile(r"frame_(\d+)_pruned_clusters_(upper|lower)\.npz$")

def _pruned_npz_files(pruned_npz_dir):
    """Glob every Step-6 pruned NPZ file (both leaflets, all frames) in a directory."""
    pattern = str(Path(pruned_npz_dir) / "frame_*_pruned_clusters_*.npz")
    return sorted(glob.glob(pattern))



def sum_defect_type_areas(pruned_npz_dir):
    """Sum neutral vs. tail triangle area, pooled across every final (pruned) defect triangle.

    defect_type is a per-triangle field written by Step 4 and carried
    through Steps 5/6 unchanged -- it is not in pruned.csv (which is
    per-cluster), so this reads the pruned NPZ files directly instead of
    going through load_cluster_rows. A cluster with a mix of neutral and
    tail triangles contributes to both totals rather than being forced
    into one bucket.


    Returns: dict with neutral_area_A2, tail_area_A2, total_area_A2,
             neutral_pct, tail_pct, n_frames, n_defect_triangles, and
             per_frame ({frame_idx: {"neutral": Å^2, "tail": Å^2}}, pooling
             both leaflets per frame).
    """
    npz_files = _pruned_npz_files(pruned_npz_dir)
    if not npz_files:
        raise FileNotFoundError(f"No pruned NPZ files found in {pruned_npz_dir}")

    neutral_area = 0.0
    tail_area = 0.0
    n_defect_tris = 0
    # {frame: {"neutral": Å^2, "tail": Å^2}}. The entry is created here, above
    # the empty-leaflet skip below, so a frame whose clusters were all pruned
    # away still shows up in the per-frame table as a 0.0/0.0 row instead of
    # silently vanishing from it.
    per_frame = {}

    for npz_path in npz_files:
        m = _PRUNED_NPZ_RE.search(Path(npz_path).name)
        if not m:
            continue
        frame_idx = int(m.group(1))
        per_frame.setdefault(frame_idx, {"neutral": 0.0, "tail": 0.0})

        with np.load(npz_path, allow_pickle=True) as data:
            final_tri_ids = np.asarray(data["final_defect_tri_ids"], dtype=np.int64)
            if final_tri_ids.size == 0:
                continue
            if "defect_type" not in data:
                raise KeyError(
                    f"Missing 'defect_type' in {npz_path}. This pruned NPZ predates "
                    "defect-type classification -- delete results/defect/cluster and "
                    "results/defect/pruned, then rerun Steps 4-6."
                )
            verts = np.asarray(data["verts"], dtype=np.float64)
            tris = np.asarray(data["tris"], dtype=np.int32)
            defect_type = np.asarray(data["defect_type"], dtype=np.int8)

            faces = tris[final_tri_ids]
            a = verts[faces[:, 0]]
            b = verts[faces[:, 1]]
            c = verts[faces[:, 2]]
            areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
            types = defect_type[final_tri_ids]

            frame_neutral = float(areas[types == 1].sum())
            frame_tail = float(areas[types == 0].sum())

            neutral_area += frame_neutral
            tail_area += frame_tail
            n_defect_tris += int(final_tri_ids.size)

            # Upper and lower land on the same frame key, so this accumulates
            # rather than assigns -- the per-frame row is both leaflets pooled,
            # matching the pooled totals above it.
            per_frame[frame_idx]["neutral"] += frame_neutral
            per_frame[frame_idx]["tail"] += frame_tail

    total_area = neutral_area + tail_area
    neutral_pct = 100.0 * neutral_area / total_area if total_area > 0 else 0.0
    tail_pct = 100.0 * tail_area / total_area if total_area > 0 else 0.0

    return {
        "neutral_area_A2": neutral_area,
        "tail_area_A2": tail_area,
        "total_area_A2": total_area,
        "neutral_pct": neutral_pct,
        "tail_pct": tail_pct,
        "n_frames": len(per_frame),
        "n_defect_triangles": n_defect_tris,
        "per_frame": per_frame,
    }


def build_defect_type_section(stats):
    """Build the 'defect type coverage' summary section text from sum_defect_type_areas' output."""
    lines = [
        "=== Defect type coverage (neutral lipid vs. phospholipid tail) ===",
        f"frames = {stats['n_frames']}",
        f"total defect triangles (pooled, both leaflets) = {stats['n_defect_triangles']}",
        f"total defect area (pooled) = {stats['total_area_A2']:.4f} Å^2",
        "",
        f"neutral lipid:      {stats['neutral_area_A2']:.4f} Å^2 ({stats['neutral_pct']:.2f}%)",
        f"phospholipid tail:  {stats['tail_area_A2']:.4f} Å^2 ({stats['tail_pct']:.2f}%)",
    ]

    per_frame = stats["per_frame"]
    if not per_frame:
        return "\n".join(lines)

    lines.append("")
    lines.append(
        f"{'frame':>8}  {'phospholipid tail (Å^2)':>24}  "
        f"{'neutral lipid (Å^2)':>22}  {'total (Å^2)':>16}"
    )
    for fr in sorted(per_frame):
        tail = float(per_frame[fr]["tail"])
        neutral = float(per_frame[fr]["neutral"])
        lines.append(
            f"{fr:>8}  {tail:>24.4f}  {neutral:>22.4f}  {tail + neutral:>16.4f}"
        )

    return "\n".join(lines)



def save_defect_type_plot(stats, output_base, dpi=300):
    """Save a two-bar chart (neutral lipid % vs. phospholipid tail %) from sum_defect_type_areas' output."""
    set_plot_style(font_size=10, font_family="Arial")

    fig, ax = plt.subplots(figsize=(5, 5))
    labels = ["Neutral lipid", "Phospholipid tail"]
    values = [stats["neutral_pct"], stats["tail_pct"]]
    colors = ["#2ecc71", "#e74c3c"]

    ax.bar(labels, values, color=colors, edgecolor="black")
    ax.set_ylabel("Share of total defect area (%)")
    ax.set_ylim(0, 100)
    _style_axis(ax)

    fig.tight_layout()
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def run_defect_type_coverage(pruned_npz_dir, output_dir, output_prefix, dpi=300):
    """Step 7's "defect_type_coverage" analysis: what share of final defect area is
    neutral-lipid-dominant vs. phospholipid-tail-dominant, pooled across every
    frame and leaflet, per Step 4's per-triangle classification.

    Writes its own summary .txt and a two-bar PNG/PDF plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(str(output_prefix).strip()).name
    if not prefix:
        raise ValueError("output_name_prefix cannot be empty.")

    stats = sum_defect_type_areas(pruned_npz_dir)

    section = build_defect_type_section(stats)
    summary_path = output_dir / f"{prefix}_defect_type.txt"
    summary_path.write_text(join_analysis_sections([section]), encoding="utf-8")

    output_base = output_dir / f"{prefix}_defect_type_coverage"
    png_path, pdf_path = save_defect_type_plot(stats, output_base, dpi=dpi)

    return {
        "defect_type_txt": str(summary_path),
        "png_plot": str(png_path),
        "pdf_plot": str(pdf_path),
        "n_frames": stats["n_frames"],
        "neutral_pct": stats["neutral_pct"],
        "tail_pct": stats["tail_pct"],
    }

# ==================================================
# Coverage
# ==================================================
def sum_areas_per_frame(csv_path):
    """Return {frame: {"upper": area_sum, "lower": area_sum}} in Å^2, from Step 6's pruned.csv.

    Only rows with a recognized upper/lower side and a finite positive area
    contribute -- see load_cluster_rows for how side/area are parsed.
    """
    from collections import defaultdict

    sums = defaultdict(lambda: {"upper": 0.0, "lower": 0.0})
    for frame, side, area in load_cluster_rows(csv_path):
        if side in ("upper", "lower") and area is not None and math.isfinite(area) and area > 0.0:
            sums[frame][side] += float(area)
    return dict(sums)



def build_coverage_section(csv_path, box_area_by_frame):
    """Build the 'defect surface coverage per frame' section text.

    Coverage = defect area / that frame's own projected box area, in percent.
      upper = sum(upper areas) / (Lx*Ly)_frame * 100
      lower = sum(lower areas) / (Lx*Ly)_frame * 100
      total = (upper+lower area) / (2*(Lx*Ly)_frame) * 100   (two-leaflet mean)

    box_area_by_frame is keyed by frame index (results/box_size.csv) instead
    of a single constant, since box size can vary frame to frame.

    Raises KeyError if a frame with cluster rows has no matching
    box_size.csv entry -- same "delete and rebuild" pattern used everywhere
    else box_size.csv is consumed in this pipeline.
    """
    if not box_area_by_frame:
        raise ValueError("box_area_by_frame must not be empty for coverage.")

    per_frame = sum_areas_per_frame(csv_path)

    lines = ["=== Defect surface coverage per frame ==="]
    if not per_frame:
        lines.append("No cluster rows found.")
        return "\n".join(lines)

    frames = sorted(per_frame)
    n_frames = len(frames)

    pct = {}
    for fr in frames:
        if fr not in box_area_by_frame:
            raise KeyError(
                f"Frame {fr} appears in {csv_path} but has no matching row in "
                "box_size.csv. Delete results/box_size.csv and rerun to rebuild it."
            )
        box_area = float(box_area_by_frame[fr])
        if not (math.isfinite(box_area) and box_area > 0.0):
            raise ValueError(f"box area for frame {fr} must be positive. Got {box_area!r}.")

        up_area = per_frame[fr]["upper"]
        lo_area = per_frame[fr]["lower"]
        pct[fr] = (
            up_area / box_area * 100.0,
            lo_area / box_area * 100.0,
            (up_area + lo_area) / (2.0 * box_area) * 100.0,
        )

    mean_up = sum(pct[fr][0] for fr in frames) / n_frames
    mean_lo = sum(pct[fr][1] for fr in frames) / n_frames
    mean_tot = sum(pct[fr][2] for fr in frames) / n_frames

    all_areas = [float(box_area_by_frame[fr]) for fr in frames]
    lines.append(f"frames with clusters = {n_frames}")
    lines.append(
        f"box area per leaflet (per-frame) = min {min(all_areas):.4f}, "
        f"mean {sum(all_areas)/len(all_areas):.4f}, max {max(all_areas):.4f} Å^2"
    )
    lines.append(
        f"mean coverage (all frames) = {mean_tot:.4f} % "
        f"(upper = {mean_up:.4f} %, lower = {mean_lo:.4f} %)"
    )
    lines.append("")
    lines.append(f"{'frame':>8}  {'box_area':>10}  {'upper':>9}  {'lower':>9}  {'total':>9}")
    for fr in frames:
        u, l, t = pct[fr]
        lines.append(f"{fr:>8}  {box_area_by_frame[fr]:>10.4f}  {u:>9.4f}  {l:>9.4f}  {t:>9.4f}")

    return "\n".join(lines)


def save_coverage_plot(csv_path, box_area_by_frame, output_base, dpi=300):
    """Coverage-vs-frame line plot (upper/lower/two-leaflet mean %), styled like the other Step-7 plots.

    box_area_by_frame is keyed by frame index, since box size can vary frame
    to frame (see build_coverage_section for the same convention).
    """
    set_plot_style(font_size=10, font_family="Arial")

    if not box_area_by_frame:
        raise ValueError("box_area_by_frame must not be empty for the coverage plot.")

    per_frame = sum_areas_per_frame(csv_path)
    frames = sorted(per_frame)

    fig, ax = plt.subplots(figsize=(8, 6))

    if frames:
        missing = [fr for fr in frames if fr not in box_area_by_frame]
        if missing:
            raise KeyError(
                f"Frame(s) {missing[:5]}{'...' if len(missing) > 5 else ''} appear in "
                f"{csv_path} but have no matching row in box_size.csv."
            )

        upper_pct = [per_frame[fr]["upper"] / float(box_area_by_frame[fr]) * 100.0 for fr in frames]
        lower_pct = [per_frame[fr]["lower"] / float(box_area_by_frame[fr]) * 100.0 for fr in frames]
        total_pct = [
            (per_frame[fr]["upper"] + per_frame[fr]["lower"]) / (2.0 * float(box_area_by_frame[fr])) * 100.0
            for fr in frames
        ]

        ax.plot(frames, upper_pct, color="#1f77b4", linewidth=1.6, marker="o", markersize=3.5, label="Upper")
        ax.plot(frames, lower_pct, color="#C0392B", linewidth=1.6, marker="o", markersize=3.5, label="Lower")
        ax.plot(frames, total_pct, color="#555555", linewidth=1.8, linestyle="--", label="Two-leaflet mean")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Defect coverage (%)")
    ax.set_title("Defect surface coverage vs. frame", pad=12)

    _style_axis(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def run_coverage(cluster_csv, output_dir, output_prefix, box_area_by_frame, dpi=300):
    """Step 7's "coverage" analysis. Writes its own summary .txt plus a coverage-vs-frame plot.

    box_area_by_frame is keyed by frame index (from results/box_size.csv) --
    box size can vary frame to frame, so there is no single constant area.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(str(output_prefix).strip()).name
    if not prefix:
        raise ValueError("output_name_prefix cannot be empty.")

    section = build_coverage_section(cluster_csv, box_area_by_frame)
    summary_path = output_dir / f"{prefix}_coverage.txt"
    summary_path.write_text(join_analysis_sections([section]), encoding="utf-8")

    output_base = output_dir / f"{prefix}_coverage_vs_frame"
    png_path, pdf_path = save_coverage_plot(cluster_csv, box_area_by_frame, output_base, dpi=dpi)

    per_frame = sum_areas_per_frame(cluster_csv)
    areas = [float(v) for v in box_area_by_frame.values()]
    return {
        "coverage_txt": str(summary_path),
        "png_plot": str(png_path),
        "pdf_plot": str(pdf_path),
        "n_frames": len(per_frame),
        "box_area_a2_min": min(areas),
        "box_area_a2_mean": sum(areas) / len(areas),
        "box_area_a2_max": max(areas),
    }


# ==================================================
# Distribution
# ==================================================


def cluster_stats_per_frame(csv_path):
    """Return {frame: {upper_sum, lower_sum, upper_n, lower_n}} in Å^2 / counts, from pruned.csv.

    Only clusters with a finite positive area are included, since an average
    size is undefined for a zero/invalid-area cluster.
    """
    from collections import defaultdict

    stats = defaultdict(lambda: {"upper_sum": 0.0, "lower_sum": 0.0, "upper_n": 0, "lower_n": 0})
    for frame, side, area in load_cluster_rows(csv_path):
        if side not in ("upper", "lower"):
            continue
        if area is None or not math.isfinite(area) or area <= 0.0:
            continue
        stats[frame][f"{side}_sum"] += float(area)
        stats[frame][f"{side}_n"] += 1
    return dict(stats)



def build_distribution_section(csv_path, bin_width, png_path, pdf_path):
    """Build the 'defect size distribution' section text.

    Per-frame columns are average cluster area (Å^2):
      upper = mean area of upper clusters in that frame
      lower = mean area of lower clusters in that frame
      total = pooled mean over all clusters in that frame
              = (upper_sum + lower_sum) / (upper_n + lower_n)
    'total' is the pooled mean, NOT the mean of the two leaflet means; the two
    differ whenever the leaflets hold different cluster counts.
    """
    stats = cluster_stats_per_frame(csv_path)

    lines = ["=== Defect size distribution ==="]
    if not stats:
        lines.append("No cluster rows with positive area found.")
        return "\n".join(lines)

    frames = sorted(stats)

    def _mean(sum_v, n):
        return (sum_v / n) if n > 0 else 0.0

    tot_up_sum = sum(stats[fr]["upper_sum"] for fr in frames)
    tot_lo_sum = sum(stats[fr]["lower_sum"] for fr in frames)
    tot_up_n = sum(stats[fr]["upper_n"] for fr in frames)
    tot_lo_n = sum(stats[fr]["lower_n"] for fr in frames)
    n_all = tot_up_n + tot_lo_n

    grand_up = _mean(tot_up_sum, tot_up_n)
    grand_lo = _mean(tot_lo_sum, tot_lo_n)
    grand_tot = _mean(tot_up_sum + tot_lo_sum, n_all)

    lines.append(f"frames with clusters = {len(frames)}")
    lines.append(f"total clusters = {n_all} (upper = {tot_up_n}, lower = {tot_lo_n})")
    lines.append(
        f"mean cluster size (all clusters) = {grand_tot:.4f} Å^2 "
        f"(upper = {grand_up:.4f}, lower = {grand_lo:.4f})"
    )
    lines.append(f"bin width = {float(bin_width):.4f} Å^2")
    lines.append(f"histogram PNG = {png_path}")
    lines.append(f"histogram PDF = {pdf_path}")
    lines.append("")
    lines.append(f"{'frame':>8}  {'upper':>9}  {'lower':>9}  {'total':>9}")
    for fr in frames:
        s = stats[fr]
        up = _mean(s["upper_sum"], s["upper_n"])
        lo = _mean(s["lower_sum"], s["lower_n"])
        tot = _mean(s["upper_sum"] + s["lower_sum"], s["upper_n"] + s["lower_n"])
        lines.append(f"{fr:>8}  {up:>9.4f}  {lo:>9.4f}  {tot:>9.4f}")

    return "\n".join(lines)


def save_distribution_histogram(areas, output_base, bin_width=1.0, dpi=300):
    """Pooled defect-size histogram (all clusters, both leaflets), raw counts.

    """
    set_plot_style(font_size=10, font_family="Arial")

    v = finite_positive(areas)
    bin_mids, _probs, counts, bin_edges = histogram_probability(v, bin_width=bin_width)

    fig, ax = plt.subplots(figsize=(8, 6))
    if bin_mids.size:
        ax.bar(
            bin_mids, counts,
            width=float(bin_width),
            align="center",
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.set_xlim(left=0.0, right=float(np.max(bin_edges)))

    ax.set_xlabel(r"Defect area $A$ ($\mathrm{\AA}^2$)")
    ax.set_ylabel("Count")
    ax.set_title("Defect size distribution", pad=12)

    _style_axis(ax)
    fig.tight_layout()

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path

def run_distribution(cluster_csv, output_dir, output_prefix, bin_width=1.0, dpi=300):
    """Step 7's "distribution" analysis: pooled size histogram + per-frame average sizes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(str(output_prefix).strip()).name
    if not prefix:
        raise ValueError("output_name_prefix cannot be empty.")

    areas = load_cluster_areas(cluster_csv)
    output_base = output_dir / f"{prefix}_size_distribution"
    png_path, pdf_path = save_distribution_histogram(areas, output_base, bin_width=bin_width, dpi=dpi)

    section = build_distribution_section(cluster_csv, bin_width=bin_width, png_path=png_path, pdf_path=pdf_path)
    summary_path = output_dir / f"{prefix}_distribution.txt"
    summary_path.write_text(join_analysis_sections([section]), encoding="utf-8")

    return {
        "distribution_txt": str(summary_path),
        "png_plot": str(png_path),
        "pdf_plot": str(pdf_path),
        "n_clusters": int(areas.size),
    }

# ======================================================================================
# Layer defect-constant fit (Step 8 post-analysis): monolayer vs bilayer pi
# ======================================================================================
# finite_positive / histogram_probability / fit_log_probability / set_plot_style
# above are reused as-is. Only the CSV loader, the two-panel plot, and the
# run_* wrapper are new.


_DEFECT_CONSTANT_COLORS = {"monolayer": "blue", "bilayer": "red"}


def load_areas_by_region(csv_path):
    """Load finite positive areas from a Step-8 <prefix>_layers.csv, grouped by region_type.


    Returns: {"monolayer": array, "bilayer": array} -- always both keys,
             even if one region had zero rows (empty array, not missing key).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Layers CSV not found: {csv_path.resolve()}")

    areas_by_region = {"monolayer": [], "bilayer": []}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = [c.strip() for c in (reader.fieldnames or [])]
        if "area_angstrom2" not in columns or "region_type" not in columns:
            raise ValueError(
                f"Layers CSV must contain 'area_angstrom2' and 'region_type'. Found: {columns}"
            )

        for row in reader:
            region = str(row.get("region_type", "")).strip().lower()
            if region not in areas_by_region:
                continue
            try:
                areas_by_region[region].append(float(row["area_angstrom2"]))
            except (TypeError, ValueError):
                continue

    return {k: finite_positive(v) for k, v in areas_by_region.items()}



def save_defect_constant_plot(
    areas_by_region,
    min_defect_size,
    max_defect_size,
    min_probability,
    bin_width,
    output_base,
    dpi=300,
    x_tick_step=50.0,
):
    """Two-panel plot: log-scale P(A) decay + fit per region (left), pi bar chart (right).

    Fits monolayer and bilayer independently via fit_log_probability, using
    the same min_defect_size/max_defect_size window for both -- reused
    directly from Step 7's fitting keys.

    A region with zero areas, or too few histogram bins in the fit window,
    is skipped with a warning printed to stdout rather than raising -- a
    system with no bilayer clusters at all (e.g. a pure monolayer test
    system) still produces a valid plot and summary, just missing that
    region's fit.

    max_defect_size, if None, falls back independently per region to that
    region's own maximum area -- so passing None lets monolayer and bilayer
    each use their own natural upper bound instead of a shared one.

    Returns: (png_path, pdf_path, summaries) -- summaries is
             {region: fit_log_probability summary dict}, containing only
             the regions that actually produced a fit.
    """
    set_plot_style(font_size=10, font_family="Arial")

    fig, (ax_decay, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    ax_decay.set_yscale("log")

    summaries = {}

    for region, color in _DEFECT_CONSTANT_COLORS.items():
        areas = areas_by_region.get(region, np.array([]))
        if areas.size == 0:
            print(f"⚠️  [layer_fitting] no '{region}' clusters found -- skipping its fit.")
            continue

        bin_mids, probs, counts, _ = histogram_probability(areas, bin_width=bin_width)
        max_size = max_defect_size if max_defect_size is not None else float(np.max(areas))

        try:
            summary, fit_mask = fit_log_probability(
                bin_mids, probs, counts,
                min_defect_size=min_defect_size,
                max_defect_size=max_size,
                min_probability=min_probability,
            )
        except RuntimeError as exc:
            print(f"⚠️  [layer_fitting] '{region}': {exc} -- skipping its fit.")
            continue

        summaries[region] = summary

        visible = probs > 0.0
        x_vis, y_vis, fm_vis = bin_mids[visible], probs[visible], fit_mask[visible]

        ax_decay.scatter(x_vis, y_vis, s=18, marker="o", facecolors="none",
                          edgecolors=color, linewidths=0.8, alpha=0.75, zorder=2,
                          label=f"{region} data")
        if np.any(fm_vis):
            ax_decay.scatter(x_vis[fm_vis], y_vis[fm_vis], s=24, marker="o", facecolors="none",
                              edgecolors=color, linewidths=1.1, alpha=0.95, zorder=3)

        if np.any(fit_mask):
            x_line = np.linspace(bin_mids[fit_mask].min(), bin_mids[fit_mask].max(), 200)
            y_line = np.exp(summary["intercept"] + summary["slope"] * x_line)
            ax_decay.plot(
                x_line, y_line, color=color, linewidth=1.8,
                label=f"{region} fit ($\\pi$={summary['pi_A2']:.1f} Å², R²={summary['r2']:.2f})",
                zorder=4,
            )

        if summary["warning_slope_positive"]:
            print(f"⚠️  [layer_fitting] '{region}': fitted slope is positive; "
                  "exponential-decay assumption may be failing.")

    ax_decay.axvline(float(min_defect_size), color="0.35", linestyle="--", linewidth=1.2, zorder=1)
    ax_decay.axvline(float(max_defect_size), color="0.35", linestyle="--", linewidth=1.2, zorder=1)
    if min_probability is not None:
        ax_decay.axhline(float(min_probability), color="0.35", linestyle="--", linewidth=1.2, zorder=1)

    ax_decay.set_xlabel(r"Defect area $A$ (Å$^2$)")
    ax_decay.set_ylabel(r"Probability $P(A)$")
    ax_decay.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_decay.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax_decay.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_decay.xaxis.set_major_locator(MultipleLocator(x_tick_step))
    ax_decay.legend(frameon=False, loc="best", handlelength=2.2)
    _style_axis(ax_decay)

    regions_present = [r for r in _DEFECT_CONSTANT_COLORS if r in summaries]
    ax_bar.bar(
        regions_present,
        [summaries[r]["pi_A2"] for r in regions_present],
        color=[_DEFECT_CONSTANT_COLORS[r] for r in regions_present],
        edgecolor="black",
    )
    ax_bar.set_ylabel(r"Packing defect constant $\pi$ (Å$^2$)")
    _style_axis(ax_bar)

    fig.tight_layout()
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path, summaries


def _build_defect_constant_section(prefix, summaries, min_defect_size, max_defect_size, min_probability):
    """Build the 'layer defect-constant fit' summary section text, from save_defect_constant_plot's summaries."""
    min_p_desc = "none" if min_probability is None else f"{float(min_probability):.6g}"
    lines = [
        "=== Layer defect-constant fit (monolayer vs bilayer) ===",
        f"output prefix = {prefix}",
        f"fit window = [{float(min_defect_size):.3f}, {float(max_defect_size):.3f}] Å^2",
        f"min probability floor = {min_p_desc}",
    ]
    if not summaries:
        lines.append("No region had enough data for a fit.")
        return "\n".join(lines)

    for region in ("monolayer", "bilayer"):
        s = summaries.get(region)
        if s is None:
            lines.append(f"{region}: skipped (no data or not enough histogram bins in window)")
            continue
        lines.append(
            f"{region}: pi = {s['pi_A2']:.4f} Å^2, R^2 = {s['r2']:.4f}, "
            f"slope = {s['slope']:.6g}, n_fit_points = {s['n_fit_points']}"
            + (" [WARNING: positive slope]" if s["warning_slope_positive"] else "")
        )
    return "\n".join(lines)


def run_defect_constant_fit(
    layers_csv,
    output_dir,
    output_prefix,
    min_defect_size,
    max_defect_size,
    min_probability,
    bin_width=1.0,
    dpi=300,
):
    """Step 8 post-analysis: fit the packing-defect constant pi separately for
    monolayer and bilayer clusters, from a completed <prefix>_layers.csv.

    Writes (all in output_dir):
      <prefix>_defect_constant.txt
      <prefix>_defect_constant.png / .pdf

    Returns: dict with summary_txt, png_plot, pdf_plot, and pi_by_region
             ({region: pi_A2}, only for regions that produced a fit).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(str(output_prefix).strip()).name
    if not prefix:
        raise ValueError("output_name_prefix cannot be empty.")

    areas_by_region = load_areas_by_region(layers_csv)

    output_base = output_dir / f"{prefix}_defect_constant"
    png_path, pdf_path, summaries = save_defect_constant_plot(
        areas_by_region,
        min_defect_size=min_defect_size,
        max_defect_size=max_defect_size,
        min_probability=min_probability,
        bin_width=bin_width,
        output_base=output_base,
        dpi=dpi,
    )

    section = _build_defect_constant_section(prefix, summaries, min_defect_size, max_defect_size, min_probability)
    summary_path = output_dir / f"{prefix}_defect_constant.txt"
    summary_path.write_text(join_analysis_sections([section]), encoding="utf-8")

    return {
        "summary_txt": str(summary_path),
        "png_plot": str(png_path),
        "pdf_plot": str(pdf_path),
        "pi_by_region": {r: s["pi_A2"] for r, s in summaries.items()},
    }


# ======================================================================================
# Curvature-segment defect-constant fit (Step 9 post-analysis): pi by H bin
# ======================================================================================
# finite_positive / histogram_probability / fit_log_probability / set_plot_style /
# join_analysis_sections above are reused as-is. Only the CSV loader (splits by
# H instead of region_type), the plot, and the run_* wrapper are new.


def load_areas_by_h_segment(csv_path, n_segments):
    """Load (area, H) from a Step-9 defects_curvature.csv and split areas into
    n_segments equal-width bins across the H range actually present in this CSV.

    Bin edges are data-dependent, not fixed -- H isn't guaranteed to span any
    particular range, since defects_curvature.csv only covers whichever
    clusters got written this run. Each segment key is "seg1", "seg2", ...
    in increasing-H order.

    Binning is closed on the right for the last segment only (a value
    exactly at h_max lands in the last segment, not dropped as
    out-of-range); every other segment is half-open [lo, hi).

    Raises RuntimeError if every row's H is identical -- there's no
    meaningful way to split a single value into multiple segments.

    Returns: (areas_by_segment, stats) --
        areas_by_segment: {"seg1": array, "seg2": array, ...}, one entry
                          per segment, finite_positive already applied.
        stats: {"h_min", "h_max", "edges", "segment_keys"} -- edges is the
              (n_segments+1)-length array of bin boundaries; segment_keys
              is the ordered list of keys used above, both needed by
              save_curvature_segment_plot to label the H range per segment.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Defects curvature CSV not found: {csv_path.resolve()}")

    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = [str(c).strip() for c in (reader.fieldnames or [])]
        required = {"area_angstrom2", "H"}
        if not required.issubset(columns):
            raise ValueError(f"Defects curvature CSV must contain {required}. Found: {columns}")

        for row in reader:
            try:
                area = float(row["area_angstrom2"])
                h = float(row["H"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(area) and math.isfinite(h):
                rows.append((area, h))

    if not rows:
        raise RuntimeError(f"No valid (area_angstrom2, H) rows found in {csv_path}")

    h_values = np.array([h for _, h in rows], dtype=float)
    h_max = float(h_values.max())
    h_min = float(h_values.min())
    if h_max == h_min:
        raise RuntimeError(
            f"H is constant ({h_min:.6g}) across every row in {csv_path}; "
            "cannot split into segments."
        )

    n_segments = max(1, int(n_segments))
    edges = np.linspace(h_min, h_max, n_segments + 1)
    segment_keys = [f"seg{i + 1}" for i in range(n_segments)]
    areas_by_segment = {k: [] for k in segment_keys}

    for area, h in rows:
        idx = int(np.searchsorted(edges, h, side="right") - 1)
        idx = min(max(idx, 0), n_segments - 1)  # closed on the right for the last bin
        areas_by_segment[segment_keys[idx]].append(area)

    stats = {"h_min": h_min, "h_max": h_max, "edges": edges, "segment_keys": segment_keys}
    return {k: finite_positive(v) for k, v in areas_by_segment.items()}, stats

def save_curvature_segment_plot(
    areas_by_segment,
    stats,
    min_defect_size,
    max_defect_size,
    min_probability,
    bin_width,
    output_base,
    dpi=300,
    x_tick_step=50.0,
):
    """Two-panel plot: log-scale P(A) decay + fit per H segment (left), pi bar chart (right).

    Same structure as save_defect_constant_plot (Step 8's monolayer/bilayer
    version) 

    stats: from load_areas_by_h_segment -- h_min/h_max/edges/segment_keys
           used here to build each segment's "H in [lo, hi)" legend label.
    Returns: (png_path, pdf_path, summaries) -- summaries is
             {segment_key: fit_log_probability summary dict}, containing
             only segments that actually produced a fit.
    """
    set_plot_style(font_size=10, font_family="Arial")

    edges = stats["edges"]
    segment_keys = stats["segment_keys"]
    n_segments = len(segment_keys)
    segment_colors = plt.cm.coolwarm(np.linspace(0, 1, n_segments))
    segment_labels = {
        key: f"H\u2208[{edges[i]:.2f}, {edges[i+1]:.2f}{']' if i == n_segments - 1 else ')'}"
        for i, key in enumerate(segment_keys)
    }

    fig, (ax_decay, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    ax_decay.set_yscale("log")

    summaries = {}

    for i, key in enumerate(segment_keys):
        color = segment_colors[i]
        areas = areas_by_segment.get(key, np.array([]))
        if areas.size == 0:
            print(f"⚠️  [curvature_fitting] no clusters in segment '{segment_labels[key]}' -- skipping its fit.")
            continue

        bin_mids, probs, counts, _ = histogram_probability(areas, bin_width=bin_width)
        max_size = max_defect_size if max_defect_size is not None else float(np.max(areas))

        try:
            summary, fit_mask = fit_log_probability(
                bin_mids, probs, counts,
                min_defect_size=min_defect_size,
                max_defect_size=max_size,
                min_probability=min_probability,
            )
        except RuntimeError as exc:
            print(f"⚠️  [curvature_fitting] segment '{segment_labels[key]}': {exc} -- skipping its fit.")
            continue

        summaries[key] = summary

        visible = probs > 0.0
        x_vis, y_vis, fm_vis = bin_mids[visible], probs[visible], fit_mask[visible]

        ax_decay.scatter(x_vis, y_vis, s=18, marker="o", facecolors="none",
                          edgecolors=color, linewidths=0.8, alpha=0.75, zorder=2)
        if np.any(fm_vis):
            ax_decay.scatter(x_vis[fm_vis], y_vis[fm_vis], s=24, marker="o", facecolors="none",
                              edgecolors=color, linewidths=1.1, alpha=0.95, zorder=3)

        if np.any(fit_mask):
            x_line = np.linspace(bin_mids[fit_mask].min(), bin_mids[fit_mask].max(), 200)
            y_line = np.exp(summary["intercept"] + summary["slope"] * x_line)
            legend_label = f"{segment_labels[key]} ($\\pi$={summary['pi_A2']:.1f} Å², R²={summary['r2']:.2f})"
            ax_decay.plot(x_line, y_line, color=color, linewidth=1.8, label=legend_label, zorder=4)

        if summary["warning_slope_positive"]:
            print(f"⚠️  [curvature_fitting] segment '{segment_labels[key]}': fitted slope is positive; "
                  "exponential-decay assumption may be failing.")

    ax_decay.axvline(float(min_defect_size), color="0.35", linestyle="--", linewidth=1.2, zorder=1)
    ax_decay.axvline(float(max_defect_size), color="0.35", linestyle="--", linewidth=1.2, zorder=1)
    if min_probability is not None:
        ax_decay.axhline(float(min_probability), color="0.35", linestyle="--", linewidth=1.2, zorder=1)

    ax_decay.set_xlabel(r"Defect area $A$ (Å$^2$)")
    ax_decay.set_ylabel(r"Probability $P(A)$")
    ax_decay.yaxis.set_major_locator(LogLocator(base=10.0))
    ax_decay.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax_decay.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_decay.xaxis.set_major_locator(MultipleLocator(x_tick_step))
    ax_decay.legend(frameon=False, loc="best", handlelength=2.2)
    _style_axis(ax_decay)

    segments_present = [k for k in segment_keys if k in summaries]
    ax_bar.bar(
        [segment_labels[k] for k in segments_present],
        [summaries[k]["pi_A2"] for k in segments_present],
        color=[segment_colors[segment_keys.index(k)] for k in segments_present],
        edgecolor="black",
    )
    ax_bar.set_ylabel(r"Packing defect constant $\pi$ (Å$^2$)")
    ax_bar.tick_params(axis="x", rotation=25)
    _style_axis(ax_bar)

    fig.tight_layout()
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path, summaries


def _build_curvature_segment_section(prefix, stats, summaries, min_defect_size, max_defect_size, min_probability):
    """Build the 'curvature-segment defect-constant fit' summary section text.

    stats: from load_areas_by_h_segment. summaries: from
    save_curvature_segment_plot -- only segments that produced a fit.
    """
    edges = stats["edges"]
    segment_keys = stats["segment_keys"]
    min_p_desc = "none" if min_probability is None else f"{float(min_probability):.6g}"

    lines = [
        "=== Curvature-segment defect-constant fit (pi vs H) ===",
        f"output prefix = {prefix}",
        f"H range = [{stats['h_min']:.4f}, {stats['h_max']:.4f}]",
        f"segments = {len(segment_keys)}",
        f"fit window = [{float(min_defect_size):.3f}, {float(max_defect_size):.3f}] Å^2",
        f"min probability floor = {min_p_desc}",
        "",
    ]
    for i, key in enumerate(segment_keys):
        label = f"H in [{edges[i]:.3f}, {edges[i+1]:.3f}{']' if i == len(segment_keys) - 1 else ')'}"
        s = summaries.get(key)
        if s is None:
            lines.append(f"{label}: skipped (no data or not enough histogram bins in window)")
            continue
        lines.append(
            f"{label}: pi = {s['pi_A2']:.4f} Å^2, R^2 = {s['r2']:.4f}, "
            f"slope = {s['slope']:.6g}, n_fit_points = {s['n_fit_points']}"
            + (" [WARNING: positive slope]" if s["warning_slope_positive"] else "")
        )
    return "\n".join(lines)

def run_curvature_segment_fit(
    defects_csv,
    output_dir,
    output_prefix,
    min_defect_size,
    max_defect_size,
    min_probability,
    n_segments=5,
    bin_width=1.0,
    dpi=300,
):
    """Step 9 post-analysis: fit the packing-defect constant pi separately per
    H segment, from a completed defects_curvature.csv.

    Called from helpers.py's run_step9_curvature_cfg, always -- there is no
    curvature_fitting on/off toggle; this sub-analysis always runs after
    Step 9's own curvature classification writes defects_curvature.csv.
    min_defect_size, max_defect_size, and min_probability are always Step
    7's already-resolved fitting values (see helpers.py's
    _resolve_curvature_fit_bound / _resolve_curvature_fit_min_probability),
    same reuse pattern as Step 8's run_defect_constant_fit. n_segments is
    CURVATURE_FIT_SEGMENTS in main.py -- a code-level constant, not a
    prep.in key.

    Writes (all in output_dir):
      <prefix>_curvature_segments.txt
      <prefix>_curvature_segments.png / .pdf

    Returns: dict with summary_txt, png_plot, pdf_plot, pi_by_segment
             ({segment_key: pi_A2}, only segments that produced a fit),
             and n_segments (the actual count used, from stats).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = Path(str(output_prefix).strip()).name
    if not prefix:
        raise ValueError("output_name_prefix cannot be empty.")

    areas_by_segment, stats = load_areas_by_h_segment(defects_csv, n_segments)

    output_base = output_dir / f"{prefix}_curvature_segments"
    png_path, pdf_path, summaries = save_curvature_segment_plot(
        areas_by_segment, stats,
        min_defect_size=min_defect_size,
        max_defect_size=max_defect_size,
        min_probability=min_probability,
        bin_width=bin_width,
        output_base=output_base,
        dpi=dpi,
    )

    section = _build_curvature_segment_section(
        prefix, stats, summaries, min_defect_size, max_defect_size, min_probability
    )
    summary_path = output_dir / f"{prefix}_curvature_segments.txt"
    summary_path.write_text(join_analysis_sections([section]), encoding="utf-8")

    return {
        "summary_txt": str(summary_path),
        "png_plot": str(png_path),
        "pdf_plot": str(pdf_path),
        "pi_by_segment": {k: s["pi_A2"] for k, s in summaries.items()},
        "n_segments": len(stats["segment_keys"]),
    }


