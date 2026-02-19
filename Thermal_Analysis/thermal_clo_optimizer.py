"""thermal_clo_optimizer.py

Hourly clothing (clo) optimization for TWO zones using pythermalcomfort.

What it does
- Reads two CSVs from /content by default:
    /content/PMV_zone1.csv
    /content/PMV_zone2.csv
- For each occupied hour (OCC==1), grid-searches clo within user-specified day/night bounds
  to minimize |PMV|.
- Writes two outputs per run into /content/pmv_runs:
    run_###_Zone_1.csv
    run_###_Zone_2.csv
  and adds: zone_label, run_label, met_used, vr_used, rh_pct, is_night, clo_used, PMV, PPD.
- Comfort reporting uses PMV in [-1, +1].

Designed for Google Colab + ipywidgets.
"""

from __future__ import annotations

import os
import glob
import re
import warnings
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

from pythermalcomfort.models import pmv_ppd_ashrae


# =========================
# Silence warnings EARLY (first-run safe)
# =========================
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numba")
logging.getLogger("jupyter_client").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)


# Colab widget manager (needed for ipywidgets sliders)
try:
    from google.colab import output  # type: ignore

    output.enable_custom_widget_manager()
except Exception:
    pass


# =========================
# Defaults
# =========================
DEFAULT_Z1_PATH = "/content/PMV_zone1.csv"
DEFAULT_Z2_PATH = "/content/PMV_zone2.csv"
DEFAULT_RUN_DIR = "/content/pmv_runs"

STEP = 0.1  # fixed step for hourly clothing optimization (grid search)

PMV_COMFORT_LO = -1.0
PMV_COMFORT_HI = 1.0

night_hours = {22, 23, 0, 1, 2, 3, 4, 5}


# =========================
# IO helpers
# =========================
def _prepare_df(path: str, zone_label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Tip: run `!ls -lh /content` to confirm the filenames."
        )

    d = pd.read_csv(path).copy()
    d["zone_label"] = zone_label

    # required columns
    for c in ["RelHum", "Hour", "OCC", "TDryBulb", "MRT"]:
        if c not in d.columns:
            raise ValueError(f"{zone_label}: CSV must contain '{c}' column.")

    # RH handling: if RelHum looks like 0–1, convert to percent; if already 0–100, keep as-is.
    rel = pd.to_numeric(d["RelHum"], errors="coerce")
    # guard against all-NaN
    rel_max = np.nanmax(rel.to_numpy()) if np.isfinite(rel.to_numpy()).any() else np.nan
    d["rh_pct"] = rel * 100.0 if (np.isfinite(rel_max) and rel_max <= 1.5) else rel.astype(float)

    # Night flag
    d["Hour"] = pd.to_numeric(d["Hour"], errors="coerce")
    d["is_night"] = d["Hour"].isin(night_hours)

    return d


def next_run_number(run_dir: str) -> int:
    files = glob.glob(os.path.join(run_dir, "run_*_Zone_*.csv"))
    nums = []
    for f in files:
        m = re.search(r"run_(\d+)_Zone_", os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


# =========================
# PMV helpers
# =========================
def pmv_for_candidate_clo(
    tdb: float,
    tr: float,
    rh: float,
    met: float,
    vr: float,
    clo_values: np.ndarray,
) -> np.ndarray:
    """Compute PMV for a vector of candidate clo values for ONE hour."""
    n = int(len(clo_values))
    res = pmv_ppd_ashrae(
        tdb=np.full(n, tdb, dtype=float),
        tr=np.full(n, tr, dtype=float),
        vr=np.full(n, vr, dtype=float),
        rh=np.full(n, rh, dtype=float),
        met=np.full(n, met, dtype=float),
        clo=np.asarray(clo_values, dtype=float),
        wme=np.zeros(n, dtype=float),
    )
    return np.asarray(res.pmv, dtype=float)


def optimize_clo_per_row(
    tdb: float,
    tr: float,
    rh: float,
    met: float,
    vr: float,
    clo_lo: float,
    clo_hi: float,
    step: float = STEP,
) -> Tuple[float, float]:
    """For one row, find clo in [clo_lo, clo_hi] (step) that minimizes |PMV|."""
    if not (np.isfinite(clo_lo) and np.isfinite(clo_hi) and clo_hi >= clo_lo):
        return (np.nan, np.nan)

    clo_vals = np.round(np.arange(clo_lo, clo_hi + 1e-12, step), 3)
    if clo_vals.size == 0:
        return (np.nan, np.nan)

    pmv_vals = pmv_for_candidate_clo(tdb, tr, rh, met, vr, clo_vals)
    finite = np.isfinite(pmv_vals)
    if finite.sum() == 0:
        return (np.nan, np.nan)

    idx = int(np.argmin(np.abs(pmv_vals[finite])))
    clo_best = float(clo_vals[finite][idx])
    pmv_best = float(pmv_vals[finite][idx])
    return (clo_best, pmv_best)


# =========================
# Core runner
# =========================
def _run_one_zone(
    dfin: pd.DataFrame,
    zone_label: str,
    met: float,
    vr: float,
    day_lo: float,
    day_hi: float,
    night_lo: float,
    night_hi: float,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    df_out = dfin.copy()

    # overwrite any existing PMV/PPD columns (start fresh)
    for col in ["PMV", "PPD", "PMV_new", "PPD_new"]:
        if col in df_out.columns:
            df_out.drop(columns=[col], inplace=True)

    df_out["met_used"] = float(met)
    df_out["vr_used"] = float(vr)

    df_out["clo_used"] = np.nan
    df_out["PMV"] = np.nan
    df_out["PPD"] = np.nan

    #####Ocupancy patch#####
    occ_mask = (pd.to_numeric(df_out["OCC"], errors="coerce") == 0.2)
    occ_n = int(occ_mask.sum())
    if occ_n == 0:
        return df_out, occ_mask.to_numpy(dtype=bool), {
            "occ_n": 0,
            "ok_inputs_n": 0,
            "optimized_count": 0,
            "pmv_computed_n": 0,
            "pct_ok": 0.0,
            "note": "No occupied hours (OCC==1).",
        }

    occ_idx = df_out.index[occ_mask].to_numpy()
    occ_rows = df_out.loc[occ_idx].copy()

    tdb = pd.to_numeric(occ_rows["TDryBulb"], errors="coerce").to_numpy(dtype=float)
    tr = pd.to_numeric(occ_rows["MRT"], errors="coerce").to_numpy(dtype=float)
    rh = pd.to_numeric(occ_rows["rh_pct"], errors="coerce").to_numpy(dtype=float)
    is_night = occ_rows["is_night"].to_numpy(dtype=bool)

    ok_inputs = np.isfinite(tdb) & np.isfinite(tr) & np.isfinite(rh)
    ok_inputs_n = int(ok_inputs.sum())
    if ok_inputs_n == 0:
        return df_out, occ_mask.to_numpy(dtype=bool), {
            "occ_n": occ_n,
            "ok_inputs_n": 0,
            "optimized_count": 0,
            "pmv_computed_n": 0,
            "pct_ok": 0.0,
            "note": "Occupied rows exist, but TDryBulb/MRT/RH are non-numeric or NaN.",
        }

    clo_best_list = np.full(len(occ_idx), np.nan, dtype=float)
    optimized_count = 0

    for i in range(len(occ_idx)):
        if not ok_inputs[i]:
            continue

        clo_lo = float(night_lo if is_night[i] else day_lo)
        clo_hi = float(night_hi if is_night[i] else day_hi)

        clo_best, _ = optimize_clo_per_row(
            tdb=float(tdb[i]),
            tr=float(tr[i]),
            rh=float(rh[i]),
            met=float(met),
            vr=float(vr),
            clo_lo=clo_lo,
            clo_hi=clo_hi,
            step=STEP,
        )

        clo_best_list[i] = clo_best
        if np.isfinite(clo_best):
            optimized_count += 1

    df_out.loc[occ_idx, "clo_used"] = clo_best_list

    ok_final = ok_inputs & np.isfinite(clo_best_list)
    idx_ok = occ_idx[ok_final]
    if len(idx_ok) == 0:
        return df_out, occ_mask.to_numpy(dtype=bool), {
            "occ_n": occ_n,
            "ok_inputs_n": ok_inputs_n,
            "optimized_count": optimized_count,
            "pmv_computed_n": 0,
            "pct_ok": 0.0,
            "note": "No occupied rows produced a finite PMV within selected clothing bounds. Widen bounds.",
        }

    res = pmv_ppd_ashrae(
        tdb=tdb[ok_final],
        tr=tr[ok_final],
        vr=np.full(int(ok_final.sum()), float(vr)),
        rh=rh[ok_final],
        met=np.full(int(ok_final.sum()), float(met)),
        clo=clo_best_list[ok_final],
        wme=np.zeros(int(ok_final.sum())),
    )

    df_out.loc[idx_ok, "PMV"] = np.asarray(res.pmv, dtype=float)
    df_out.loc[idx_ok, "PPD"] = np.asarray(res.ppd, dtype=float)

    occ_pmv = df_out.loc[occ_mask, "PMV"].to_numpy(dtype=float)
    finite = np.isfinite(occ_pmv)
    pmv_computed_n = int(finite.sum())

    pct_ok = (
        100.0
        * float(np.mean((occ_pmv[finite] >= PMV_COMFORT_LO) & (occ_pmv[finite] <= PMV_COMFORT_HI)))
        if pmv_computed_n
        else 0.0
    )

    return df_out, occ_mask.to_numpy(dtype=bool), {
        "occ_n": occ_n,
        "ok_inputs_n": ok_inputs_n,
        "optimized_count": optimized_count,
        "pmv_computed_n": pmv_computed_n,
        "pct_ok": pct_ok,
        "note": "",
    }


# =========================
# Public UI launcher
# =========================
def launch_clothing_optimizer_ui(
    z1_path: str = DEFAULT_Z1_PATH,
    z2_path: str = DEFAULT_Z2_PATH,
    run_dir: str = DEFAULT_RUN_DIR,
) -> None:
    """Launch the ipywidgets UI in Colab."""

    os.makedirs(run_dir, exist_ok=True)

    df_z1 = _prepare_df(z1_path, "Zone_1")
    df_z2 = _prepare_df(z2_path, "Zone_2")
    zone_dfs = {"Zone_1": df_z1, "Zone_2": df_z2}

    # --- UI widgets ---
    day_clo_range = widgets.FloatRangeSlider(
        value=[0.7, 1.1],
        min=0.0,
        max=1.5,
        step=0.1,
        description="Day clo",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="620px"),
    )

    night_clo_range = widgets.FloatRangeSlider(
        value=[1.0, 1.2],
        min=0.0,
        max=1.5,
        step=0.1,
        description="Night clo",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="620px"),
    )

    met_w = widgets.FloatSlider(
        value=1.1,
        min=0.7,
        max=2.0,
        step=0.1,
        description="met",
        continuous_update=False,
        readout_format=".1f",
        layout=widgets.Layout(width="620px"),
    )

    vr_w = widgets.FloatSlider(
        value=0.1,
        min=0.05,
        max=0.6,
        step=0.05,
        description="vr (m/s)",
        continuous_update=False,
        readout_format=".2f",
        layout=widgets.Layout(width="620px"),
    )

    run_btn = widgets.Button(
        description="Optimize hourly clo + Save (Zone_1 + Zone_2)",
        button_style="primary",
    )
    out = widgets.Output()

    def _on_run(_=None):
        with out:
            clear_output(wait=True)

            met = float(met_w.value)
            vr = float(vr_w.value)
            day_lo, day_hi = map(float, day_clo_range.value)
            night_lo, night_hi = map(float, night_clo_range.value)

            run_id = next_run_number(run_dir)

            results = {}
            for zone_label, dfin in zone_dfs.items():
                df_out, occ_mask, stats = _run_one_zone(
                    dfin,
                    zone_label,
                    met,
                    vr,
                    day_lo,
                    day_hi,
                    night_lo,
                    night_hi,
                )

                run_label = f"run_{run_id:03d}_{zone_label}"
                df_out["run_label"] = run_label
                df_out["zone_label"] = zone_label

                out_path = os.path.join(run_dir, f"run_{run_id:03d}_{zone_label}.csv")
                df_out.to_csv(out_path, index=False)

                results[zone_label] = {
                    "df_out": df_out,
                    "occ_mask": occ_mask,
                    "stats": stats,
                    "out_path": out_path,
                }

            print(f"Saved TWO files for run {run_id:03d}:")
            for zone_label in ["Zone_1", "Zone_2"]:
                r = results[zone_label]
                st = r["stats"]
                print(f"  - {zone_label}: {r['out_path']}")
                if st.get("note"):
                    print(f"    ⚠️ {st['note']}")
                print(
                    f"    Occupied rows: {st['occ_n']} | finite inputs (T/MRT/RH): {st['ok_inputs_n']}"
                )
                print(
                    f"    Optimized clo: {st['optimized_count']}/{st['occ_n']} | PMV computed: {st['pmv_computed_n']}/{st['occ_n']}"
                )
                print(
                    f"    Comfort on computable OCC rows (PMV in [{PMV_COMFORT_LO:+.1f}, {PMV_COMFORT_HI:+.1f}]): {st['pct_ok']:.1f}%"
                )

            print(f"\nSettings: step={STEP} | met={met:.2f} | vr={vr:.2f}")
            print(f"Day bounds:   [{day_lo:.1f}, {day_hi:.1f}]")
            print(f"Night bounds: [{night_lo:.1f}, {night_hi:.1f}]")
            print("-" * 80)

            show_cols = [
                "ZoneName",
                "Month",
                "Day",
                "Hour",
                "TDryBulb",
                "MRT",
                "rh_pct",
                "OCC",
                "is_night",
                "clo_used",
                "PMV",
                "PPD",
                "met_used",
                "vr_used",
                "zone_label",
                "run_label",
            ]

            for zone_label in ["Zone_1", "Zone_2"]:
                r = results[zone_label]
                df_out = r["df_out"]
                occ_mask = r["occ_mask"]
                cols = [c for c in show_cols if c in df_out.columns]

                display(widgets.HTML(f"<b>{zone_label} (first 20 occupied rows)</b>"))
                display(df_out.loc[occ_mask, cols].head(20))

    run_btn.on_click(_on_run)

    display(
        widgets.VBox(
            [
                widgets.HTML(
                    "<b>Hourly clothing optimization (2 zones)</b><br>"
                    "Loads <code>/content/PMV_zone1.csv</code> and <code>/content/PMV_zone2.csv</code>, then for each zone: "
                    "each occupied hour gets its own optimized <code>clo_used</code> within day/night bounds (step=0.1). "
                    "<code>PMV</code> and <code>PPD</code> are overwritten with newly computed values. "
                    f"Comfort reporting uses PMV in [{PMV_COMFORT_LO:+.1f}, {PMV_COMFORT_HI:+.1f}]. "
                    "Outputs are saved as <code>run_###_Zone_1.csv</code> and <code>run_###_Zone_2.csv</code> and include "
                    "<code>zone_label</code> + <code>run_label</code> columns."
                ),
                day_clo_range,
                night_clo_range,
                met_w,
                vr_w,
                run_btn,
                out,
            ]
        )
    )
