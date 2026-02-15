"""
thermal_upload_loader.py

Google Colab helper to upload TWO PMV zone CSVs from your local machine,
save them under /content, then load + preview them.

Recommended filenames:
  - PMV_zone1.csv
  - PMV_zone2.csv

Usage in Colab:
  from thermal_upload_loader import upload_and_load_zone_csvs
  df_zone1, df_zone2 = upload_and_load_zone_csvs()
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import pandas as pd


def upload_and_load_zone_csvs(
    target_dir: str = "/content",
    rename_map: Optional[Dict[str, str]] = None,
    expected_names: Tuple[str, str] = ("PMV_zone1.csv", "PMV_zone2.csv"),
    preview_rows: int = 5,
):
    """
    Upload two zone CSVs in Google Colab and save them to `target_dir`.

    Parameters
    ----------
    target_dir:
        Where to save the uploaded files (default: /content).
    rename_map:
        Optional mapping to rename uploaded filenames to the expected names.
        Example: {"zone1.csv":"PMV_zone1.csv", "zone2.csv":"PMV_zone2.csv"}
    expected_names:
        The two filenames expected after upload/rename.
    preview_rows:
        How many head() rows to print/return for quick sanity check.

    Returns
    -------
    (df_zone1, df_zone2)
    """
    # Import Colab upload tool
    try:
        from google.colab import files  # type: ignore
    except Exception as e:
        raise RuntimeError("This helper is intended to run inside Google Colab.") from e

    os.makedirs(target_dir, exist_ok=True)

    print(f"Please upload your two files: {expected_names[0]} and {expected_names[1]}")
    uploaded = files.upload()

    # Save uploaded bytes
    for name, data in uploaded.items():
        out_path = os.path.join(target_dir, name)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Saved: {out_path}")

    # Optional rename
    rename_map = rename_map or {}
    for src, dst in rename_map.items():
        src_path = os.path.join(target_dir, src)
        dst_path = os.path.join(target_dir, dst)
        if os.path.exists(src_path):
            os.replace(src_path, dst_path)
            print(f"Renamed: {src} -> {dst}")

    z1_path = os.path.join(target_dir, expected_names[0])
    z2_path = os.path.join(target_dir, expected_names[1])

    missing = [p for p in (z1_path, z2_path) if not os.path.exists(p)]
    if missing:
        print("Missing expected file(s):")
        for p in missing:
            print(" -", p)
        print("\nTip: confirm uploaded names with:")
        print(f"!ls -lh {target_dir}")
        raise FileNotFoundError("Please upload the missing file(s) (or set rename_map).")

    df_zone1 = pd.read_csv(z1_path)
    df_zone2 = pd.read_csv(z2_path)

    print("Zone 1 shape:", df_zone1.shape, "| path:", z1_path)
    print("Zone 2 shape:", df_zone2.shape, "| path:", z2_path)

    # Preview (works in Colab/Jupyter)
    try:
        from IPython.display import display  # type: ignore
        display(df_zone1.head(preview_rows))
        display(df_zone2.head(preview_rows))
    except Exception:
        # Fallback for non-notebook environments
        print(df_zone1.head(preview_rows))
        print(df_zone2.head(preview_rows))

    return df_zone1, df_zone2


if __name__ == "__main__":
    upload_and_load_zone_csvs()
