"""
thermal_data_loader.py

Utilities to fetch the PMV zone CSVs used in the People-Planet-Property Thermal Analysis
notebooks.

Default behavior:
- Downloads PMV_zone1.csv and PMV_zone2.csv from GitHub (raw) if they do not already
  exist locally.
- Returns the loaded pandas DataFrames.

Designed to be imported from Colab with minimal boilerplate.

Example (Colab):
    from thermal_data_loader import load_zone_csvs
    df1, df2 = load_zone_csvs()
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import pandas as pd


@dataclass(frozen=True)
class GitHubCSVSpec:
    """Specification for loading CSVs from a GitHub repo."""
    repo_owner: str = "Nadatarkhan"
    repo_name: str = "People-Planet-Property"
    branches: Tuple[str, ...] = ("main", "master")
    # Repo-relative paths
    files: Optional[Dict[str, str]] = None  # populated in __post_init__

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "files",
            self.files
            or {
                "PMV_zone1.csv": "Thermal_Analysis/data/PMV_zone1.csv",
                "PMV_zone2.csv": "Thermal_Analysis/data/PMV_zone2.csv",
            },
        )

    def raw_candidates(self) -> Dict[str, List[str]]:
        """Return candidate raw.githubusercontent URLs for each CSV."""
        out: Dict[str, List[str]] = {}
        assert self.files is not None
        for name, repo_path in self.files.items():
            out[name] = [
                f"https://raw.githubusercontent.com/{self.repo_owner}/{self.repo_name}/{branch}/{repo_path}"
                for branch in self.branches
            ]
        return out


def _download_if_missing(local_path: str, urls: Iterable[str]) -> str:
    """Download a file into `local_path` if missing, trying each URL."""
    if os.path.exists(local_path):
        return local_path

    last_err: Optional[Exception] = None
    for url in urls:
        try:
            print(f"Downloading: {url}")
            urllib.request.urlretrieve(url, local_path)
            print(f"Saved to: {local_path}")
            return local_path
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Could not download '{os.path.basename(local_path)}' from GitHub. "
        "Check the file path and branch name."
    ) from last_err


def load_zone_csvs(
    dest_dir: str = "/content",
    spec: Optional[GitHubCSVSpec] = None,
    force_redownload: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Zone 1 and Zone 2 PMV CSVs.

    Parameters
    ----------
    dest_dir:
        Where to save the CSVs locally (defaults to /content in Colab).
    spec:
        GitHubCSVSpec defining the repo owner/name/branches and file paths.
    force_redownload:
        If True, deletes local copies before downloading.

    Returns
    -------
    (df_zone1, df_zone2)
    """
    os.makedirs(dest_dir, exist_ok=True)
    spec = spec or GitHubCSVSpec()
    candidates = spec.raw_candidates()

    z1_local = os.path.join(dest_dir, "PMV_zone1.csv")
    z2_local = os.path.join(dest_dir, "PMV_zone2.csv")

    if force_redownload:
        for p in (z1_local, z2_local):
            if os.path.exists(p):
                os.remove(p)

    z1_path = _download_if_missing(z1_local, candidates["PMV_zone1.csv"])
    z2_path = _download_if_missing(z2_local, candidates["PMV_zone2.csv"])

    df_z1 = pd.read_csv(z1_path)
    df_z2 = pd.read_csv(z2_path)
    return df_z1, df_z2


if __name__ == "__main__":
    df1, df2 = load_zone_csvs()
    print("Zone 1 shape:", df1.shape)
    print("Zone 2 shape:", df2.shape)
