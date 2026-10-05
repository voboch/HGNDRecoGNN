#!/usr/bin/env python3
"""Regression tests for the reproducible n/p analysis."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from np_ratio_smash_check import (  # noqa: E402
    event_clustered_ratio,
    load_dataset,
    validate_inputs,
)
from np_ratio_smash_check_stream import analyze_dataset  # noqa: E402


def _write(path: Path, rows: list[dict[str, float | int]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(row: int, pdg: int, ekin: float = 1.0) -> dict[str, float | int]:
    return {
        "Row": row,
        "PDG": pdg,
        "Ekin": ekin,
        "Pt": 0.2,
        "Pz": 1.0,
        "Rapid": 1.0,
        "B": -1.0,
        "NPrim": -1.0,
    }


def test_event_key_includes_source_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / "zeroSpot_1_prim.csv", [_row(0, 2112), _row(0, 2212)])
        _write(root / "zeroSpot_2_prim.csv", [_row(0, 2112), _row(0, 2212)])
        loaded = load_dataset(root, "zeroSpot", 0.9863)
        assert loaded.event_count == 2
        assert loaded.frame["event_key"].nunique() == 2


def test_event_clustered_ratio_and_invalid_centrality() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [_row(0, 2112), _row(0, 2112), _row(0, 2212), _row(1, 2112), _row(1, 2212)]
        loaded = {}
        for dataset in ("zeroSpot", "defaultSpot", "bigSpot"):
            _write(root / f"{dataset}_1_prim.csv", rows)
            loaded[dataset] = load_dataset(root, dataset, 0.9863)
        frame = loaded["zeroSpot"].frame
        result = event_clustered_ratio(
            frame,
            np.ones(len(frame), dtype=bool),
            frame["Ekin"].to_numpy(),
            np.array([0.0, 2.0]),
            min_count=1,
        )
        assert np.isclose(result.loc[0, "ratio"], 1.5)
        assert result.loc[0, "stat_error"] > 0
        validation = validate_inputs(loaded)
        assert validation["centrality_B_valid"] is False
        assert validation["participant_count_valid"] is False
        assert validation["decision"] == "DECLINE"


def test_streaming_statistics_match_in_memory_result() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nested = root / "smash_xecs_2.87gev_hardSkyrme_zeroSpot" / "run"
        nested.mkdir(parents=True)
        rows = [
            _row(0, 2112, 1.0),
            _row(0, 2112, 1.0),
            _row(0, 2212, 1.0),
            _row(1, 2112, 1.0),
            _row(1, 2212, 1.0),
        ]
        _write(nested / "0001_prim.csv", rows)
        streamed, info = analyze_dataset(root, "zeroSpot", 0.9863, 1, None)
        ekin = streamed[
            (streamed["figure"] == "ekin")
            & (streamed["panel"] == "abs_y_lt_0p5")
            & (streamed["bin_lo"] == 1.0)
        ].iloc[0]
        assert info["files"] == 1
        assert info["events"] == 2
        assert np.isclose(ekin["ratio"], 1.5)
        assert ekin["stat_error"] > 0


if __name__ == "__main__":
    test_event_key_includes_source_file()
    test_event_clustered_ratio_and_invalid_centrality()
    test_streaming_statistics_match_in_memory_result()
    print("all n/p tests passed")
