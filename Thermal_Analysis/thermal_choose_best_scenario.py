"""
thermal_choose_best_scenario.py

Pick the best *paired* PMV run (Zone 1 + Zone 2 share the same run_id) by minimizing
total discomfort hours, then copy the chosen pair into /content/Chosen_scenario.

Assumptions:
- Run files are created by the optimizer as:
    /content/pmv_runs/run_###_Zone_1.csv
    /content/pmv_runs/run_###_Zone_2.csv
- Each file includes:
    - OCC (1 = occupied)
    - PMV (or PMV_new)
- "Discomfort" is defined as:
    OCC==1 & finite PMV & (PMV > PMV_HOT_THR OR PMV < PMV_COLD_THR)

Typical usage in Colab:
    from thermal_choose_best_scenario import pick_best_paired_run_min_discomfort
    pick_best_paired_run_min_discomfort()
"""

from __future__ import annotations

import os
import glob
import re
import shutil
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Defaults (Colab-friendly)
# -------------------------
RUN_DIR_DEFAULT = "/content/pmv_runs"
CHOSEN_DIR_DEFAULT = "/content/Chosen_scenario"

# PMV thresholds (updated comfort band)
PMV_HOT_THR_DEFAULT = 1.0
PMV_COLD_THR_DEFAULT = -1.0

# Validation check: anti-empty gate
MIN_COMPUTABLE_OCC_ROWS_PER_ZONE_DEFAULT = 24


# -------------------------
# Internal helpers
# -------------------------
def _pmv_col(d: pd.DataFrame) -> Optional[str]:
    if "PMV" in d.columns:
        return "PMV"
    if "PMV_new" in d.columns:
        return "PMV_new"
    return None


def _computable_occ_mask(d: pd.DataFrame, pmv_col: str) -> Optional[pd.Series]:
    if "OCC" not in d.columns:
        return None
    occ = pd.to_numeric(d["OCC"], errors="coerce") == 1
    pmv = pd.to_numeric(d[pmv_col], errors="coerce")
    return occ & np.isfinite(pmv)


def _discomfort_hours(d: pd.DataFrame, pmv_col: str, pmv_hot_thr: float, pmv_cold_thr: float) -> Optional[int]:
    computable = _computable_occ_mask(d, pmv_col)
    if computable is None:
        return None
    pmv = pd.to_numeric(d[pmv_col], errors="coerce")
    discomfort = computable & ((pmv > pmv_hot_thr) | (pmv < pmv_cold_thr))
    return int(discomfort.sum())


def _count_computable_occ(d: pd.DataFrame, pmv_col: str) -> Optional[int]:
    computable = _computable_occ_mask(d, pmv_col)
    if computable is None:
        return None
    return int(computable.sum())


def _run_id_from_path(path: str) -> Optional[int]:
    m = re.search(r"run_(\d+)_Zone_[12]\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else None


# -------------------------
# Public API
# -------------------------
def pick_best_paired_run_min_discomfort(
    run_dir: str = RUN_DIR_DEFAULT,
    chosen_dir: str = CHOSEN_DIR_DEFAULT,
    pmv_hot_thr: float = PMV_HOT_THR_DEFAULT,
    pmv_cold_thr: float = PMV_COLD_THR_DEFAULT,
    min_computable_occ_rows_per_zone: int = MIN_COMPUTABLE_OCC_ROWS_PER_ZONE_DEFAULT,
    clear_existing_chosen: bool = True,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Picks ONE run number k that exists for BOTH zones:
      - run_kkk_Zone_1.csv
      - run_kkk_Zone_2.csv

    Objective:
      - minimize total discomfort hours across both zones combined

    Robustness gate (anti-empty):
      - skip any run where either zone has too few computable occupied rows
        (prevents selecting an "empty" scenario).

    Returns:
      dict with chosen run_id + copied paths, or None if no valid run.
    """
    os.makedirs(chosen_dir, exist_ok=True)

    z1_files = glob.glob(os.path.join(run_dir, "run_*_Zone_1.csv"))
    z2_files = glob.glob(os.path.join(run_dir, "run_*_Zone_2.csv"))

    z1 = {_run_id_from_path(p): p for p in z1_files if _run_id_from_path(p) is not None}
    z2 = {_run_id_from_path(p): p for p in z2_files if _run_id_from_path(p) is not None}

    common_ids = sorted(set(z1.keys()) & set(z2.keys()))
    if not common_ids:
        if verbose:
            print(f"No paired Zone_1/Zone_2 run files found in {run_dir}")
            print("Expected filenames like: run_001_Zone_1.csv and run_001_Zone_2.csv")
        return None

    best: Optional[Tuple[int, int, int, int, int, int]] = None
    # best = (total_discomfort, rid, dh1, dh2, comp1, comp2)

    rejected: List[Tuple[int, str]] = []

    for rid in common_ids:
        p1, p2 = z1[rid], z2[rid]
        try:
            d1 = pd.read_csv(p1)
            d2 = pd.read_csv(p2)
        except Exception as e:
            rejected.append((rid, f"read_error: {e}"))
            continue

        c1 = _pmv_col(d1)
        c2 = _pmv_col(d2)
        if (c1 is None) or (c2 is None):
            rejected.append((rid, "missing PMV column"))
            continue

        comp1 = _count_computable_occ(d1, c1)
        comp2 = _count_computable_occ(d2, c2)
        if (comp1 is None) or (comp2 is None):
            rejected.append((rid, "missing OCC or non-computable PMV"))
            continue

        # Robustness gate: prevent empty/near-empty selections
        if (comp1 < min_computable_occ_rows_per_zone) or (comp2 < min_computable_occ_rows_per_zone):
            rejected.append((rid, f"too_few_computable_occ (Z1={comp1}, Z2={comp2})"))
            continue

        dh1 = _discomfort_hours(d1, c1, pmv_hot_thr=pmv_hot_thr, pmv_cold_thr=pmv_cold_thr)
        dh2 = _discomfort_hours(d2, c2, pmv_hot_thr=pmv_hot_thr, pmv_cold_thr=pmv_cold_thr)
        if (dh1 is None) or (dh2 is None):
            rejected.append((rid, "could not compute discomfort"))
            continue

        total = int(dh1 + dh2)

        if (best is None) or (total < best[0]):
            best = (total, rid, dh1, dh2, int(comp1), int(comp2))

    if best is None:
        if verbose:
            print("No valid paired runs found after filtering empties.")
            print(f"Filter: each zone must have >= {min_computable_occ_rows_per_zone} computable occupied rows.")
            if rejected:
                print("\nRejected examples (up to 10):")
                for rid, reason in rejected[:10]:
                    print(f"  - run_{rid:03d}: {reason}")
        return None

    total, rid, dh1, dh2, comp1, comp2 = best

    # Clear existing chosen scenario (optional but usually desired)
    if clear_existing_chosen:
        for old in glob.glob(os.path.join(chosen_dir, "run_*_Zone_*.csv")):
            try:
                os.remove(old)
            except Exception:
                pass

    # Copy BOTH files into Chosen_scenario
    out_paths: List[str] = []
    for src in (z1[rid], z2[rid]):
        dst = os.path.join(chosen_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        out_paths.append(dst)

    if verbose:
        print(f"✅ Chosen paired scenario (same run number for both zones): run_{rid:03d}")
        print(f"Validity gate (anti-empty): computable OCC rows per zone >= {min_computable_occ_rows_per_zone}")
        print(f"Computable occupied rows: Zone 1 = {comp1}, Zone 2 = {comp2}")
        print(
            f"Discomfort hours definition: OCC==1 & finite PMV & "
            f"(PMV>{pmv_hot_thr:+.1f} OR PMV<{pmv_cold_thr:+.1f})"
        )
        print(f"Zone 1 discomfort hours: {dh1}")
        print(f"Zone 2 discomfort hours: {dh2}")
        print(f"TOTAL discomfort hours (Zone1+Zone2): {total}")
        print("Saved to:")
        for p in out_paths:
            print("  -", p)

    return {
        "run_id": rid,
        "zone1_path": out_paths[0],
        "zone2_path": out_paths[1],
        "total_discomfort_hours": total,
        "zone1_computable_occ_rows": comp1,
        "zone2_computable_occ_rows": comp2,
        "pmv_hot_thr": float(pmv_hot_thr),
        "pmv_cold_thr": float(pmv_cold_thr),
        "min_computable_occ_rows_per_zone": int(min_computable_occ_rows_per_zone),
    }


if __name__ == "__main__":
    pick_best_paired_run_min_discomfort()
