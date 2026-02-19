"""thermal_pmv_review_plots.py

PMV review plots for the Thermal_Analysis notebook:
- Loads BASE zone files from /content (PMV_zone1.csv / PMV_zone2.csv)
- Loads all run files created by the optimizer from /content/pmv_runs/run_###_Zone_#.csv
- For each file: plots Month x Hour heatmap of mean PMV (occupied only)
- Prints hot/cold hour counts using thresholds (defaults: +/-1.0)
- Produces a summary bar chart across all files

Designed to be imported and run with minimal Colab code.
"""

from __future__ import annotations

import os
import glob
import re
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# -------------------------
# Default paths (Colab)
# -------------------------
DEFAULT_BASE_Z1_PATH = "/content/PMV_zone1.csv"
DEFAULT_BASE_Z2_PATH = "/content/PMV_zone2.csv"
DEFAULT_RUN_DIR = "/content/pmv_runs"


# -------------------------
# Colormaps (match swatches)
# -------------------------
def _build_colormaps():
    cold_cmap = mcolors.LinearSegmentedColormap.from_list(
        "cold_grad_desat",
        ["#e6f7fb", "#cdeff6", "#a8e3ef", "#72c8da", "#3aa6bf"],
    )
    hot_cmap = mcolors.LinearSegmentedColormap.from_list(
        "hot_grad",
        ["#f8d7dd", "#f2a9b3", "#e56b7c", "#d23b56", "#b3223c"],
    )

    pmv_cmap = mcolors.LinearSegmentedColormap.from_list(
        "pmv_custom_div",
        [
            cold_cmap(1.0),
            cold_cmap(0.65),
            cold_cmap(0.35),
            "#f2f2f2",
            hot_cmap(0.35),
            hot_cmap(0.65),
            hot_cmap(1.0),
        ],
        N=256,
    ).copy()
    pmv_cmap.set_bad(color="lightgrey")

    return cold_cmap, hot_cmap, pmv_cmap


COLD_CMAP, HOT_CMAP, PMV_CMAP = _build_colormaps()


# -------------------------
# Helpers
# -------------------------
def _nice_label(path: str, base_z1_path: str, base_z2_path: str) -> str:
    base = os.path.basename(path)

    if base.lower() == os.path.basename(base_z1_path).lower():
        return "BASE (Zone 1)"
    if base.lower() == os.path.basename(base_z2_path).lower():
        return "BASE (Zone 2)"

    m = re.search(r"run_(\d+)_Zone_(\d)\.csv$", base, flags=re.IGNORECASE)
    if m:
        run_n = int(m.group(1))
        z_n = int(m.group(2))
        return f"RUN {run_n} (Zone {z_n})"

    return base


def _order_key(label: str):
    mz = re.search(r"\(Zone (\d)\)", label)
    zone = int(mz.group(1)) if mz else 9

    if label.startswith("BASE"):
        run = -1
    else:
        mr = re.search(r"RUN (\d+)", label)
        run = int(mr.group(1)) if mr else 999999

    return (zone, run)


def _get_pmv_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["PMV_new", "PMV"]:
        if c in df.columns:
            return c
    return None


def _get_clo_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["clo_used", "clo", "CLO", "Clothing", "clothing"]:
        if c in df.columns:
            return c
    clo_like = [c for c in df.columns if "clo" in c.lower()]
    return clo_like[0] if clo_like else None


# -------------------------
# Plotting
# -------------------------
def plot_pmv_heatmap_month_hour(Z: np.ndarray, title: str):
    """Plot Month (x) vs Hour (y) heatmap for PMV."""
    finite = Z[np.isfinite(Z)]
    vmax = 1.0 if finite.size == 0 else float(np.nanmax(np.abs(finite)))
    vmax = max(0.5, min(vmax, 3.0))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(Z, aspect="auto", cmap=PMV_CMAP, norm=norm, origin="upper")

    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(np.arange(0, 24, 3))
    ax.set_yticklabels([str(h) for h in range(0, 24, 3)])
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day")
    ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 24, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PMV")

    plt.tight_layout()
    plt.show()


def pmv_month_hour_heatmap_and_stats(
    df: pd.DataFrame,
    title: str,
    pmv_hot_thr: float = 1.0,
    pmv_cold_thr: float = -1.0,
) -> Optional[Dict[str, Any]]:
    pmv_col = _get_pmv_col(df)
    clo_col = _get_clo_col(df)

    if pmv_col is None or "Month" not in df.columns or "Hour" not in df.columns:
        print(f"{title}: missing PMV/Month/Hour -> skipping")
        return None

    d = df.copy()
    d["Month"] = pd.to_numeric(d["Month"], errors="coerce")
    d["Hour"] = pd.to_numeric(d["Hour"], errors="coerce")

    ####Patch for occupancy
    if "OCC" in d.columns:
        occ_val = pd.to_numeric(d["OCC"], errors="coerce")
        occ = occ_val >= 0.2  # <-- change threshold
    else:
        occ = pd.Series(True, index=d.index)

    pmv = pd.to_numeric(d[pmv_col], errors="coerce")
    valid = occ & np.isfinite(pmv) & d["Month"].between(1, 12) & d["Hour"].between(0, 23)

    # Skip files where PMV exists but has no computable values (often empty PMV column)
    if int(np.sum(valid)) == 0:
        print(f"{title}: PMV column is empty or non-computable for occupied hours -> skipping")
        return None

    g = (
        pd.DataFrame(
            {
                "Month": d.loc[valid, "Month"].astype(int),
                "Hour": d.loc[valid, "Hour"].astype(int),
                "PMV": pmv.loc[valid].astype(float),
            }
        )
        .groupby(["Hour", "Month"])["PMV"]
        .mean()
        .unstack("Month")
    )
    g = g.reindex(index=range(0, 24), columns=range(1, 13))
    Z = g.to_numpy(dtype=float)

    plot_pmv_heatmap_month_hour(Z, title=f"{title} | Annual PMV (occupied only; unoccupied/NaN = grey)")

    pmv_valid = pmv.loc[valid].to_numpy(dtype=float)
    hot_hours = int(np.sum(pmv_valid >= pmv_hot_thr))
    cold_hours = int(np.sum(pmv_valid <= pmv_cold_thr))
    total_used = int(np.sum(valid))

    day_clo = None
    night_clo = None

    if clo_col is not None and "Hour" in d.columns:
        clo = pd.to_numeric(d[clo_col], errors="coerce")
        hrs = pd.to_numeric(d["Hour"], errors="coerce")
        ok = np.isfinite(clo) & hrs.between(0, 23)

        hrs_i = hrs.loc[ok].astype(int)
        night_mask = hrs_i.isin([22, 23, 0, 1, 2, 3, 4, 5]).to_numpy()
        clo_vals = clo.loc[ok].to_numpy(dtype=float)

        day_vals = clo_vals[~night_mask]
        night_vals = clo_vals[night_mask]

        if np.isfinite(day_vals).any():
            day_clo = float(np.nanmean(day_vals))
        if np.isfinite(night_vals).any():
            night_clo = float(np.nanmean(night_vals))

    print(f"{title}")
    print(f"Clothing column used: {clo_col if clo_col is not None else 'N/A'}")
    print(f"Avg DAY clothing (whole year):   {day_clo:.3f}" if day_clo is not None else "Avg DAY clothing (whole year):   N/A")
    print(
        f"Avg NIGHT clothing (whole year): {night_clo:.3f}" if night_clo is not None else "Avg NIGHT clothing (whole year): N/A (no night samples)"
    )
    print(f"Hot hours  (PMV >= {pmv_hot_thr:+.1f}): {hot_hours}")
    print(f"Cold hours (PMV <= {pmv_cold_thr:+.1f}): {cold_hours}")
    print(f"Computable occupied rows used for hot/cold counts: {total_used}")
    print("-" * 80)

    return {"label": title, "hot_hours": hot_hours, "cold_hours": cold_hours}


def plot_summary_bars(summary_rows: List[Dict[str, Any]], pmv_hot_thr: float, pmv_cold_thr: float):
    if not summary_rows:
        print("No summaries to plot.")
        return

    summary_rows = sorted(summary_rows, key=lambda r: _order_key(r["label"]))
    labels = [r["label"] for r in summary_rows]
    cold = [r["cold_hours"] for r in summary_rows]
    hot = [r["hot_hours"] for r in summary_rows]

    x = np.arange(len(labels))
    w = 0.38

    cold_color = COLD_CMAP(0.92)
    hot_color = HOT_CMAP(0.85)

    fig, ax = plt.subplots(figsize=(12, 4.1))
    ax.bar(x - w / 2, cold, width=w, color=cold_color, label="Cold hours")
    ax.bar(x + w / 2, hot, width=w, color=hot_color, label="Hot hours")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", rotation_mode="anchor")

    ax.set_xlabel("Run (by zone)")
    ax.set_ylabel("Hours (occupied, PMV computable)")
    ax.set_title(f"Hot vs Cold hours (PMV≥{pmv_hot_thr:+.1f}, PMV≤{pmv_cold_thr:+.1f})")

    for i in range(len(labels) - 1):
        ax.axvline(i + 0.5, color="#d0d0d0", linewidth=0.8, zorder=0)

    zone_flags = np.array([("Zone 2" in lab) for lab in labels], dtype=bool)
    if zone_flags.any() and (~zone_flags).any():
        first_zone2_idx = int(np.argmax(zone_flags))
        if first_zone2_idx > 0:
            ax.axvline(first_zone2_idx - 0.5, color="black", linewidth=1.4, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24)
    plt.show()


# -------------------------
# Public entrypoint
# -------------------------
def plot_all_base_and_runs(
    base_z1_path: str = DEFAULT_BASE_Z1_PATH,
    base_z2_path: str = DEFAULT_BASE_Z2_PATH,
    run_dir: str = DEFAULT_RUN_DIR,
    pmv_hot_thr: float = 1.0,
    pmv_cold_thr: float = -1.0,
):
    """Plot PMV heatmaps for base + all run files and show summary bars."""

    paths: List[str] = []

    if os.path.exists(base_z1_path):
        paths.append(base_z1_path)
    if os.path.exists(base_z2_path):
        paths.append(base_z2_path)

    run_paths = sorted(glob.glob(os.path.join(run_dir, "run_*_Zone_*.csv")))
    paths.extend(run_paths)

    if not paths:
        print(
            "No files found.\n"
            f"Expected base at:\n  - {base_z1_path}\n  - {base_z2_path}\n"
            f"And runs at:\n  - {run_dir}/run_###_Zone_#.csv"
        )
        return

    summaries: List[Dict[str, Any]] = []
    for p in paths:
        label = _nice_label(p, base_z1_path, base_z2_path)
        dfp = pd.read_csv(p)
        s = pmv_month_hour_heatmap_and_stats(dfp, label, pmv_hot_thr=pmv_hot_thr, pmv_cold_thr=pmv_cold_thr)
        if s is not None:
            summaries.append(s)

    plot_summary_bars(summaries, pmv_hot_thr=pmv_hot_thr, pmv_cold_thr=pmv_cold_thr)
