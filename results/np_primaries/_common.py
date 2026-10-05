"""Shared I/O for the legacy n/p diagnostics.

Set ``NP_PRIMARY_DIR`` to the directory containing ``*_prim.csv`` files.
The production analysis lives in ``scripts/np_ratio_smash_check.py``; these
small programs are retained as focused diagnostics.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def sample_dir() -> Path:
    configured = os.environ.get("NP_PRIMARY_DIR")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[2] / "data" / "np_primaries"
    if not root.is_dir():
        raise FileNotFoundError(
            f"Primary CSV directory not found: {root}. "
            "Set NP_PRIMARY_DIR=/path/to/npsample."
        )
    return root


def primary_files(dataset: str) -> list[Path]:
    files = sorted(sample_dir().glob(f"{dataset}_*_prim.csv"))
    if not files:
        raise FileNotFoundError(
            f"No {dataset}_*_prim.csv files under {sample_dir()}"
        )
    return files


def read_primary(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    # The pandas C engine is sufficient for CSV and avoids an undeclared
    # run-time dependency on pyarrow in these diagnostics.
    return pd.read_csv(path, usecols=usecols)
