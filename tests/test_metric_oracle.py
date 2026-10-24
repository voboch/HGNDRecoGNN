"""A small rank-pair and residual oracle independent of sklearn metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import pytest
from HGNDRecoGNN.scripts.diagnose_student_pipeline import measure


def test_count_threshold_auc_and_energy_populations_against_hand_oracle():
    # Positive scores .5,.9 against negatives .1,.5: pair wins 1,.5,1,1.
    # Equality to threshold .5 is excluded. Negative truth energy is deliberately
    # nonzero and must never enter signal-only regression metrics.
    f = pd.DataFrame(
        {
            "cl_label": [1, 1, 0, 0],
            "cl_score": [0.5, 0.9, 0.1, 0.5],
            "e_true": [1.0, 2.0, 999.0, 999.0],
            "e_pred": [0.5, 3.5, -999.0, -999.0],
        }
    )
    m = measure(f, 0.5)
    assert m["n"] == 4 and m["signal"] == 2
    assert m["selected"] == m["selected_signal"] == 1
    assert m["auc"] == pytest.approx(3.5 / 4)
    assert m["purity"] == 1 and m["efficiency"] == 0.5
    assert m["bias_GeV"] == 0.5 and m["mae_GeV"] == 1
    assert m["selected_bias_GeV"] == m["selected_mae_GeV"] == 1.5


def test_empty_selection_and_single_class_are_explicit():
    f = pd.DataFrame(
        {
            "cl_label": [0, 0],
            "cl_score": [0.1, 0.2],
            "e_true": [0.0, 0.0],
            "e_pred": [4.0, 8.0],
        }
    )
    m = measure(f, 0.3)
    for key in (
        "auc",
        "ap",
        "purity",
        "efficiency",
        "bias_GeV",
        "mae_GeV",
        "selected_bias_GeV",
        "selected_mae_GeV",
    ):
        assert m[key] is None
