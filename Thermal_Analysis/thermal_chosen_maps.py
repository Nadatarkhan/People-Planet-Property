"""
thermal/thermal_chosen_maps.py

Chosen-scenario visualizations:
1) Annual occupancy recommendation map (Zone 1 vs Zone 2), based on lower mean |PMV| by (Month, Hour)
2) Build a combined CSV (run_###_combined.csv) in /content/Chosen_scenario
3) Plot clothing (clo) heatmap FROM THE COMBINED CSV using combined_OCC as occupancy mask

Designed for Google Colab paths:
- input:  /content/Chosen_scenario/run_###_Zone_1.csv and run_###_Zone_2.csv
- output: /content/Chosen_scenario/run_###_combined.csv
"""

from __future__ import annotations

import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =========================
# Config
# =========================
CHOSEN_DIR = "/content/Chosen_scenario"

# Colors
ZONE1_COLOR = "#ffffff"   # Zone 1 = white
ZONE2_COLOR = "#b3223c"   # Zone 2 = deep red

# Clothing map colormap (blue gradient)
CLO_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "clo_blue_grad",
    ["#e6f7fb", "#cdeff6", "#a8e3ef", "#72c8da", "#3aa6bf"],
)
CLO_CMAP.set_bad(color="lightgrey")  # unoccupied/NaN


# =========================
# Helpers
# =========================
def _find_chosen_pair(chosen_dir: str = CHOSEN_DIR) -> tuple[int, str, str]:
    """Find matching run_###_Zone_1.csv and run_###_Zone_2.csv in chosen_dir."""
    z1 = glob.glob(os.path.join(chosen_dir, "run_*_Zone_1.csv"))
    z2 = glob.glob(os.path.join(chosen_dir, "run_*_Zone_2.csv"))
    if not z1 or not z2:
        raise FileNotFoundError(
            f"Could not find both Zone_1 and Zone_2 files in {chosen_dir}. "
            f"Expected run_###_Zone_1.csv and run_###_Zone_2.csv"
        )

    def rid(p: str):
        m = re.search(r"run_(\d+)_Zone_[12]\.csv$", os.path.basename(p))
        return int(m.group(1)) if m else None

    z1_map = {rid(p): p for p in z1 if rid(p) is not None}
    z2_map = {rid(p): p for p in z2 if rid(p) is not None}

    common = sorted(set(z1_map) & set(z2_map))
    if not common:
        raise FileNotFoundError(
            f"Found Zone_1 and Zone_2 files in {chosen_dir}, but no matching run numbers."
        )

    run_id = common[0]  # keep as-is (earliest match)
    return run_id, z1_map[run_id], z2_map[run_id]


def _get_pmv_col(df: pd.DataFrame) -> str | None:
    for c in ["PMV", "PMV_new"]:
        if c in df.columns:
            return c
    return None


def _get_clo_col(df: pd.DataFrame) -> str | None:
    # Prefer combined first
    for c in ["combined_clo_used", "combined_clo", "combined_CLO"]:
        if c in df.columns:
            return c
    # Then regular
    for c in ["clo_used", "clo", "CLO", "Clothing", "clothing"]:
        if c in df.columns:
            return c
    clo_like = [c for c in df.columns if "clo" in c.lower()]
    return clo_like[0] if clo_like else None


def _get_occ_col(df: pd.DataFrame, prefer_combined: bool = False) -> str | None:
    """
    For combined CSVs, prefer combined_OCC.
    For zone CSVs, prefer OCC.
    """
    if prefer_combined:
        for c in ["combined_OCC", "OCC"]:
            if c in df.columns:
                return c
    else:
        for c in ["OCC", "combined_OCC"]:
            if c in df.columns:
                return c
    # fallback: anything that looks like occupancy
    cand = [c for c in df.columns if c.lower() in ("occ", "occupancy", "occupied")]
    return cand[0] if cand else None


def _month_hour_score(df: pd.DataFrame, pmv_col: str) -> np.ndarray:
    """
    Returns a 24x12 array score[hour, month-1] = mean(|PMV|) on OCC==1 and finite PMV.
    NaN where no data.
    """
    d = df.copy()
    for c in ["Month", "Hour", "OCC", pmv_col]:
        if c not in d.columns:
            raise ValueError(f"Missing required column '{c}'")

    d["Month"] = pd.to_numeric(d["Month"], errors="coerce")
    d["Hour"] = pd.to_numeric(d["Hour"], errors="coerce")
    occ = pd.to_numeric(d["OCC"], errors="coerce") == 1
    pmv = pd.to_numeric(d[pmv_col], errors="coerce")

    valid = occ & np.isfinite(pmv) & d["Month"].between(1, 12) & d["Hour"].between(0, 23)
    if valid.sum() == 0:
        return np.full((24, 12), np.nan, dtype=float)

    score = (
        pd.DataFrame(
            {
                "Month": d.loc[valid, "Month"].astype(int),
                "Hour": d.loc[valid, "Hour"].astype(int),
                "val": np.abs(pmv.loc[valid].astype(float).to_numpy()),
            }
        )
        .groupby(["Hour", "Month"])["val"]
        .mean()
        .unstack("Month")
    )
    score = score.reindex(index=range(0, 24), columns=range(1, 13))
    return score.to_numpy(dtype=float)


def _month_hour_clo(df: pd.DataFrame, clo_col: str, occ_col: str) -> np.ndarray:
    """
    Returns a 24x12 array clo_mean[hour, month-1] = mean(clo) on occ_col==1 and finite clo.
    NaN where no data.
    """
    d = df.copy()
    for c in ["Month", "Hour", occ_col, clo_col]:
        if c not in d.columns:
            raise ValueError(f"Missing required column '{c}'")

    d["Month"] = pd.to_numeric(d["Month"], errors="coerce")
    d["Hour"] = pd.to_numeric(d["Hour"], errors="coerce")

    occ = pd.to_numeric(d[occ_col], errors="coerce") == 1
    clo = pd.to_numeric(d[clo_col], errors="coerce")

    valid = occ & np.isfinite(clo) & d["Month"].between(1, 12) & d["Hour"].between(0, 23)
    if valid.sum() == 0:
        return np.full((24, 12), np.nan, dtype=float)

    g = (
        pd.DataFrame(
            {
                "Month": d.loc[valid, "Month"].astype(int),
                "Hour": d.loc[valid, "Hour"].astype(int),
                "clo": clo.loc[valid].astype(float),
            }
        )
        .groupby(["Hour", "Month"])["clo"]
        .mean()
        .unstack("Month")
    )
    g = g.reindex(index=range(0, 24), columns=range(1, 13))
    return g.to_numpy(dtype=float)


# =========================
# Plot helpers
# =========================
def _apply_month_hour_axes(ax, title: str) -> None:
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(np.arange(0, 24, 3))
    ax.set_yticklabels([str(h) for h in range(0, 24, 3)])
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day")
    ax.set_title(title)

    ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 24, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_annual_occupancy_map(zone_choice: np.ndarray, title: str) -> None:
    """
    zone_choice: 24x12 array with:
      1 => choose Zone 1
      2 => choose Zone 2
      NaN => no data (grey)
    """
    cmap = mcolors.ListedColormap([ZONE1_COLOR, ZONE2_COLOR])
    cmap.set_bad(color="lightgrey")

    Z = zone_choice.copy()
    Z01 = np.full_like(Z, np.nan, dtype=float)
    Z01[np.isfinite(Z) & (Z == 1)] = 0.0
    Z01[np.isfinite(Z) & (Z == 2)] = 1.0

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.imshow(Z01, aspect="auto", cmap=cmap, origin="upper", vmin=0, vmax=1)

    _apply_month_hour_axes(ax, title)

    import matplotlib.patches as mpatches
    leg = [
        mpatches.Patch(facecolor=ZONE1_COLOR, edgecolor="#cccccc", label="Choose Zone 1"),
        mpatches.Patch(facecolor=ZONE2_COLOR, edgecolor="none", label="Choose Zone 2"),
        mpatches.Patch(facecolor="lightgrey", edgecolor="none", label="No data / unoccupied"),
    ]
    ax.legend(handles=leg, loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()


def plot_clothing_map(clo_Z: np.ndarray, title: str) -> None:
    """
    clo_Z: 24x12 array (mean clo by hour, month), NaNs allowed.
    """
    finite = clo_Z[np.isfinite(clo_Z)]
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if np.isfinite(vmin) and np.isfinite(vmax) and (vmax - vmin) < 1e-9:
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(clo_Z, aspect="auto", origin="upper", cmap=CLO_CMAP, vmin=vmin, vmax=vmax)

    _apply_month_hour_axes(ax, title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("clo (mean, occupied)")

    plt.tight_layout()
    plt.show()


# =========================
# Combined CSV builder
# =========================
def build_and_save_combined_csv(
    run_id: int,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    zone_choice: np.ndarray,
    chosen_dir: str = CHOSEN_DIR,
) -> str:
    """
    Creates a time-aligned combined CSV by choosing Zone 1 vs Zone 2 per row based on (Month, Hour)
    from `zone_choice`. Writes chosen-zone values into "combined_*" columns.
    Saves into chosen_dir with: run_###_combined.csv
    """
    for k in ["Month", "Day", "Hour"]:
        if k not in df1.columns or k not in df2.columns:
            raise ValueError(f"Both zone CSVs must contain {k} to build a combined file.")

    d1 = df1.copy()
    d2 = df2.copy()

    for d in (d1, d2):
        d["Month"] = pd.to_numeric(d["Month"], errors="coerce").astype("Int64")
        d["Day"] = pd.to_numeric(d["Day"], errors="coerce").astype("Int64")
        d["Hour"] = pd.to_numeric(d["Hour"], errors="coerce").astype("Int64")

    def _choose_zone_for_row(m, h):
        if pd.isna(m) or pd.isna(h):
            return np.nan
        m = int(m)
        h = int(h)
        if not (1 <= m <= 12 and 0 <= h <= 23):
            return np.nan
        return zone_choice[h, m - 1]

    d1["_zone_choice"] = [_choose_zone_for_row(m, h) for m, h in zip(d1["Month"].tolist(), d1["Hour"].tolist())]

    merged = d1.merge(
        d2,
        on=["Month", "Day", "Hour"],
        how="inner",
        suffixes=("_z1", "_z2"),
    )

    if merged.empty:
        raise ValueError("Zone 1 and Zone 2 files did not align on (Month, Day, Hour); combined CSV would be empty.")

    zc = (
        merged["_zone_choice"].to_numpy(dtype=float)
        if "_zone_choice" in merged.columns
        else np.array([_choose_zone_for_row(m, h) for m, h in zip(merged["Month"], merged["Hour"])], dtype=float)
    )

    merged["chosen_zone"] = np.where(zc == 1, "Zone_1", np.where(zc == 2, "Zone_2", "NoData"))

    candidates = ["TDryBulb", "MRT", "RelHum", "rh_pct", "OCC", "clo_used", "PMV", "PPD", "met_used", "vr_used"]
    for col in candidates:
        c1 = f"{col}_z1"
        c2 = f"{col}_z2"
        if c1 in merged.columns and c2 in merged.columns:
            merged[f"combined_{col}"] = np.where(zc == 1, merged[c1], np.where(zc == 2, merged[c2], np.nan))

    keep = ["Month", "Day", "Hour", "chosen_zone"] + [c for c in merged.columns if c.startswith("combined_")]
    combined = merged[keep].copy()
    combined.insert(0, "run_id", int(run_id))

    os.makedirs(chosen_dir, exist_ok=True)
    out_path = os.path.join(chosen_dir, f"run_{run_id:03d}_combined.csv")
    combined.to_csv(out_path, index=False)
    print(f"✅ Saved combined CSV: {out_path}")
    return out_path


# =========================
# Main
# =========================
def plot_chosen_annual_zone_occupancy_map_and_plot_combined_clothing(chosen_dir: str = CHOSEN_DIR) -> str:
    run_id, z1_path, z2_path = _find_chosen_pair(chosen_dir)

    df1 = pd.read_csv(z1_path)
    df2 = pd.read_csv(z2_path)

    c1 = _get_pmv_col(df1)
    c2 = _get_pmv_col(df2)
    if c1 is None or c2 is None:
        raise ValueError("Could not find a PMV column in one of the chosen files. Expected 'PMV' or 'PMV_new'.")

    s1 = _month_hour_score(df1, c1)
    s2 = _month_hour_score(df2, c2)

    zone_choice = np.full((24, 12), np.nan, dtype=float)
    both = np.isfinite(s1) & np.isfinite(s2)
    only1 = np.isfinite(s1) & ~np.isfinite(s2)
    only2 = ~np.isfinite(s1) & np.isfinite(s2)

    zone_choice[only1] = 1
    zone_choice[only2] = 2
    zone_choice[both] = np.where(s1[both] <= s2[both], 1, 2)

    # 1) Occupancy recommendation map
    plot_annual_occupancy_map(
        zone_choice,
        title=f"Chosen scenario run_{run_id:03d} | Annual occupancy recommendation (lower mean |PMV|)",
    )

    # 2) Build combined CSV
    combined_path = build_and_save_combined_csv(run_id, df1, df2, zone_choice, chosen_dir=chosen_dir)

    # 3) Clothing map FROM COMBINED CSV, using combined_OCC
    dfc = pd.read_csv(combined_path)
    clo_col = _get_clo_col(dfc)
    occ_col = _get_occ_col(dfc, prefer_combined=True)

    if clo_col is None:
        print("⚠️ Combined file: could not find a clothing column (expected combined_clo_used). Skipping clothing map.")
        return combined_path
    if occ_col is None:
        print("⚠️ Combined file: could not find an occupancy column (expected combined_OCC). Skipping clothing map.")
        return combined_path

    cloZ = _month_hour_clo(dfc, clo_col=clo_col, occ_col=occ_col)
    plot_clothing_map(
        cloZ,
        title=f"Chosen scenario run_{run_id:03d} | Combined clothing (mean {clo_col}, occupied={occ_col})",
    )

    return combined_path


if __name__ == "__main__":
    plot_chosen_annual_zone_occupancy_map_and_plot_combined_clothing()
