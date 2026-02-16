"""
thermal/threshold_events_ui.py

Chosen_scenario peak heat/cold parsing with sliders (Colab-friendly).
- Auto-loads the newest *combined* CSV from /content/Chosen_scenario:
      run_###_combined.csv   (preferred)
  Falls back to newest CSV in the folder if needed.
- Prefers combined temperature column(s):
      combined_TDryBulb (preferred)
  then falls back to other dry-bulb candidates.
- Hot threshold slider: 25..45 °C
- Cold threshold slider: -20..15 °C
- Duration sliders (5-hour increments):
    * Minimum HOT duration (hours):  5..100 step 5
    * Minimum COLD duration (hours): 5..100 step 5
- Highlights hot/cold consecutive events on the DBT plot
- Prints cumulative exceedance hours + event-hours

Usage in Colab (after placing this file under /content/thermal):
    from thermal.threshold_events_ui import launch_threshold_ui
    launch_threshold_ui()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


# Try to enable the Colab widget manager (safe no-op elsewhere)
try:
    from google.colab import output  # type: ignore

    output.enable_custom_widget_manager()
except Exception:
    pass


# ----------------------------
# Config
# ----------------------------
DEFAULT_CHOSEN_DIR = "/content/Chosen_scenario"


@dataclass
class LoadedSeries:
    csv_path: str
    dbt_col: str
    df: pd.DataFrame
    dbt: np.ndarray


def autopick_combined_csv(chosen_dir: str = DEFAULT_CHOSEN_DIR) -> str:
    """Pick newest combined CSV if present, else newest CSV in folder."""
    if not os.path.isdir(chosen_dir):
        raise FileNotFoundError(f"Folder not found: {chosen_dir}")

    def _list_csvs(filter_fn):
        return sorted(
            [
                os.path.join(chosen_dir, f)
                for f in os.listdir(chosen_dir)
                if f.lower().endswith(".csv") and filter_fn(f.lower())
            ],
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )

    combined = _list_csvs(lambda s: "combined" in s)
    if combined:
        return combined[0]

    files = _list_csvs(lambda s: True)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {chosen_dir}")
    return files[0]


def get_dbt_col(df: pd.DataFrame) -> Optional[str]:
    """Prefer combined dry-bulb candidates; fall back to generic DBT candidates."""
    preferred = [
        "combined_TDryBulb",
        "combined_DBT",
        "combined_DryBulb",
        "combined_OutdoorDBT",
        "combined_OutDryBulb",
        "TDryBulb",
        "DBT",
        "DryBulb",
        "OutdoorDBT",
        "OutDryBulb",
        "drybulb",
        "tdb",
    ]
    for c in preferred:
        if c in df.columns:
            return c

    candidates = [
        c
        for c in df.columns
        if "combined" in c.lower()
        and (("dbt" in c.lower()) or ("drybulb" in c.lower()) or ("tdrybulb" in c.lower()))
    ]
    if candidates:
        return candidates[0]

    candidates = [c for c in df.columns if ("dbt" in c.lower()) or ("drybulb" in c.lower())]
    return candidates[0] if candidates else None


def load_combined_temperature_series(
    chosen_dir: str = DEFAULT_CHOSEN_DIR, csv_path: Optional[str] = None
) -> LoadedSeries:
    """Load CSV + return sorted DBT series."""
    csv_path = csv_path or autopick_combined_csv(chosen_dir)
    df = pd.read_csv(csv_path).copy()

    dbt_col = get_dbt_col(df)
    if dbt_col is None:
        raise ValueError(
            "Could not find a dry-bulb column. Expected something like combined_TDryBulb (preferred) "
            "or TDryBulb/DBT/DryBulb/OutdoorDBT."
        )

    # numeric time cols if present
    for c in ["Month", "Day", "Hour"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort by time if possible (keeps hour-of-year order for typical E+ exports)
    if all(c in df.columns for c in ["Month", "Day", "Hour"]):
        df = df.sort_values(["Month", "Day", "Hour"], kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    dbt = pd.to_numeric(df[dbt_col], errors="coerce").to_numpy(dtype=float)
    return LoadedSeries(csv_path=csv_path, dbt_col=dbt_col, df=df, dbt=dbt)


# ----------------------------
# Event detection helpers
# ----------------------------
def segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive (start, end) segments where mask is True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0])

    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [mask.size - 1]

    return list(zip(starts, ends))


def filter_by_min_len(segments: List[Tuple[int, int]], min_len: int) -> List[Tuple[int, int]]:
    min_len = int(min_len)
    return [(s, e) for (s, e) in segments if (e - s + 1) >= min_len]


def count_hours_in_segments(segments: List[Tuple[int, int]]) -> int:
    return int(sum((e - s + 1) for s, e in segments))


# ----------------------------
# Plotting
# ----------------------------
def plot_threshold_events(
    dbt: np.ndarray,
    dbt_col: str,
    csv_path: str,
    hot_thr: float,
    cold_thr: float,
    min_hot_len: int,
    min_cold_len: int,
    title: str = "",
) -> None:
    x = np.arange(len(dbt))

    hot_mask = np.isfinite(dbt) & (dbt >= hot_thr)
    cold_mask = np.isfinite(dbt) & (dbt <= cold_thr)

    hot_segs_all = segments_from_mask(hot_mask)
    cold_segs_all = segments_from_mask(cold_mask)

    hot_segs = filter_by_min_len(hot_segs_all, min_hot_len)
    cold_segs = filter_by_min_len(cold_segs_all, min_cold_len)

    hot_hours_total = int(hot_mask.sum())
    cold_hours_total = int(cold_mask.sum())

    hot_hours_events = count_hours_in_segments(hot_segs)
    cold_hours_events = count_hours_in_segments(cold_segs)

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(x, dbt, linewidth=1.0, alpha=0.75, color="#555555")

    for s, e in cold_segs:
        ax.axvspan(s, e, facecolor="#2b8cbe", alpha=0.25, edgecolor="#2b8cbe", linewidth=0.6)
    for s, e in hot_segs:
        ax.axvspan(s, e, facecolor="#d7301f", alpha=0.22, edgecolor="#d7301f", linewidth=0.6)

    ax.grid(True, color="#d0d0d0", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylabel(f"{dbt_col} (°C)")
    ax.set_xlabel("Hour of year")
    ax.set_title(title)

    ax.axhline(hot_thr, linestyle="--", linewidth=1.0, color="#d7301f", alpha=0.85)
    ax.axhline(cold_thr, linestyle="--", linewidth=1.0, color="#2b8cbe", alpha=0.85)

    plt.tight_layout()
    plt.show()

    print(f"File: {os.path.basename(csv_path)}")
    print(f"Using temperature column: {dbt_col}")
    print(f"Hot threshold:  ≥ {hot_thr:.1f} °C | Minimum HOT duration (hours):  {int(min_hot_len)}")
    print(f"Cold threshold: ≤ {cold_thr:.1f} °C | Minimum COLD duration (hours): {int(min_cold_len)}")
    print(f"Cumulative hot exceedance hours (any hours):   {hot_hours_total}")
    print(f"Cumulative cold exceedance hours (any hours):  {cold_hours_total}")
    print(f"Hot event hours  (consecutive ≥{int(min_hot_len)}):  {hot_hours_events}  | events: {len(hot_segs)}")
    print(f"Cold event hours (consecutive ≥{int(min_cold_len)}): {cold_hours_events} | events: {len(cold_segs)}")


# ----------------------------
# UI launcher
# ----------------------------
def launch_threshold_ui(
    chosen_dir: str = DEFAULT_CHOSEN_DIR,
    csv_path: Optional[str] = None,
    title: str = "Chosen scenario (combined) | Indoor Temperature Exceedances",
) -> LoadedSeries:
    """
    Launch interactive threshold/event parsing UI in a notebook.

    Returns the LoadedSeries (csv_path, df, dbt_col, dbt) for convenience.
    """
    loaded = load_combined_temperature_series(chosen_dir=chosen_dir, csv_path=csv_path)

    style_wide_label = {"description_width": "260px"}
    layout_wide = widgets.Layout(width="820px")

    hot_thr_w = widgets.FloatSlider(
        value=35.0, min=25.0, max=45.0, step=0.5,
        description="Hot threshold (°C)",
        continuous_update=False,
        layout=layout_wide, style=style_wide_label
    )
    min_hot_w = widgets.IntSlider(
        value=10, min=5, max=100, step=5,
        description="Minimum HOT duration (hours)",
        continuous_update=False,
        layout=layout_wide, style=style_wide_label
    )
    cold_thr_w = widgets.FloatSlider(
        value=0.0, min=-20.0, max=15.0, step=0.5,
        description="Cold threshold (°C)",
        continuous_update=False,
        layout=layout_wide, style=style_wide_label
    )
    min_cold_w = widgets.IntSlider(
        value=10, min=5, max=100, step=5,
        description="Minimum COLD duration (hours)",
        continuous_update=False,
        layout=layout_wide, style=style_wide_label
    )

    run_btn = widgets.Button(description="Update plot", button_style="primary")
    out = widgets.Output()

    def _update(_=None):
        with out:
            clear_output(wait=True)
            plot_threshold_events(
                dbt=loaded.dbt,
                dbt_col=loaded.dbt_col,
                csv_path=loaded.csv_path,
                hot_thr=float(hot_thr_w.value),
                cold_thr=float(cold_thr_w.value),
                min_hot_len=int(min_hot_w.value),
                min_cold_len=int(min_cold_w.value),
                title=title,
            )

    run_btn.on_click(_update)

    display(
        widgets.VBox(
            [
                widgets.HTML(
                    "<b>Chosen_scenario (combined): peak heat/cold parsing</b><br>"
                    "Red spans = hot events (DBT ≥ hot threshold) meeting the minimum HOT duration (hours).<br>"
                    "Blue spans = cold events (DBT ≤ cold threshold) meeting the minimum COLD duration (hours)."
                ),
                hot_thr_w,
                min_hot_w,
                cold_thr_w,
                min_cold_w,
                run_btn,
                out,
            ]
        )
    )

    _update()  # initial render
    return loaded
