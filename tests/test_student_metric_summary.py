"""Check truth alignment, signal-only energy errors and seed SD semantics."""

import pandas as pd
import pytest

from HGNDRecoGNN.scripts.summarize_student_metrics import (
    METRICS, MODELS, SEEDS, candidate_metrics, summarise,
)


def fixture_frames():
    truth = pd.DataFrame({"row": [10, 10, 20], "cluster": [0, 1, 0],
                          "true_y": [0, 1, 1], "true_E": [0., 1., 2.]})
    predicted = pd.DataFrame({"row": [10, 10, 20], "cluster": [0, 1, 0],
                              "true_y": [0, 1, 1], "score": [.1, .9, .8],
                              "pred_E": [100., 1.2, 1.8]})
    return predicted, truth


def test_energy_errors_join_ids_and_ignore_background():
    predicted, truth = fixture_frames()
    actual = candidate_metrics(predicted.iloc[::-1], truth)
    assert actual["roc_auc"] == actual["pr_auc"] == 1.
    assert actual["regression_candidates"] == 2
    assert actual["energy_mae_GeV"] == pytest.approx(.2, abs=1e-7)
    assert actual["energy_rmse_GeV"] == pytest.approx(.2, abs=1e-7)
    assert actual["energy_bias_GeV"] == pytest.approx(0., abs=1e-7)


def test_different_candidate_or_label_population_is_rejected():
    predicted, truth = fixture_frames()
    changed = predicted.copy()
    changed.loc[0, "row"] = 999
    with pytest.raises(ValueError, match="population or labels"):
        candidate_metrics(changed, truth)
    changed = predicted.copy()
    changed.loc[0, "true_y"] = 1
    with pytest.raises(ValueError, match="population or labels"):
        candidate_metrics(changed, truth)


def test_summary_is_sample_seed_sd_and_requires_all_three_seeds():
    rows = [{"model": model, "seed": seed, **dict.fromkeys(METRICS, value)}
            for model in MODELS for seed, value in zip(SEEDS, (.1, .2, .3))]
    summary = summarise(rows)
    for model in MODELS:
        for metric in METRICS:
            assert summary[model][metric]["mean"] == pytest.approx(.2)
            assert summary[model][metric]["seed_sd"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="Incomplete or repeated"):
        summarise(rows[:-1])
