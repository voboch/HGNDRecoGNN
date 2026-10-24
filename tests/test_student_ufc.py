"""Mechanism checks for the separate UFC inference/evaluation branch."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from HGNDRecoGNN.data.student_ufc import (
    UFCRecipe, attach_targets, build_ufc_event, prepare_observables, truth_inventory,
)
from HGNDRecoGNN.models.student_ufc import (
    MODEL_CLASSES, build_student_model, load_student_checkpoint,
)
from HGNDRecoGNN.training.student_ufc import (
    evaluate_events, match_unique_objects, predict_candidates, seed_metrics,
    validate_predictions,
)


@pytest.fixture
def hits():
    return pd.DataFrame({
        "Row": [10, 10, 10, 10, 20, 30],
        "LayerId": [1, 2, 1, 2, 1, 1], "RowId": [1, 1, 10, 10, 1, 1],
        "ColumnId": [1] * 6, "fELoss": [.01] * 5 + [.001],
        "fX": [100.] * 6, "fY": [0.] * 6, "fZ": [500.] * 6,
        "fTime_nonoise": [30., 30.1, 30., 30.1, 30., 30.],
        "fTime": [30.15, 30.25, 30.15, 30.25, 30.15, 30.15],
        "PDG": [2112] * 6, "Side": [5] * 6,
        "Id": [101, 101, 102, 102, 103, 104],
        "Ekin": [1.] * 6, "eToF_up_2sig": [2.] * 6, "eToF_dn_2sig": [0.] * 6,
    })


def graphs_from(frame, spectral=False):
    recipe = UFCRecipe(time_column="fTime_nonoise")
    obs, rows = prepare_observables(frame, recipe)
    graphs = [g for _, event in obs.groupby("Row")
              for g in build_ufc_event(event, recipe, "fixture", spectral=spectral)]
    return graphs, obs, rows


def test_observables_survive_absent_or_invalid_truth(hits):
    expected, _, _ = graphs_from(hits)
    # Missing MC fields and invalid truth must have no effect on inference hits.
    invalid = hits.copy()
    invalid[["PDG", "Side", "Id", "Ekin"]] = np.nan
    absent = hits.drop(columns=["PDG", "Side", "Id", "Ekin", "eToF_up_2sig", "eToF_dn_2sig"])
    for frame in (invalid, absent):
        actual, _, rows = graphs_from(frame)
        assert list(rows) == [10, 20, 30]
        assert len(actual) == 2
        for a, b in zip(actual, expected):
            assert torch.equal(a.x, b.x)
            assert torch.equal(a.edge_index, b.edge_index)
            assert "y_cls" not in a
    with pytest.raises(ValueError, match="Invalid MC"):
        attach_targets(expected, invalid)


def test_targets_include_objects_without_candidates_and_omit_cut_hits(hits):
    graphs, obs, _ = graphs_from(hits)
    attach_targets(graphs, hits)
    assert [g.signal_ids.tolist() for g in graphs] == [[101], [102]]
    assert all(g.y_cls.item() == 1 for g in graphs)
    truth = truth_inventory(obs, hits, "fixture", energy_min=.1)
    assert truth == {("fixture", 10): [101, 102], ("fixture", 20): [103]}


def test_timing_choice_is_explicit_and_changes_recipe_and_tof(hits):
    first = UFCRecipe(time_column="fTime_nonoise")
    second = UFCRecipe(time_column="fTime")
    assert first.fingerprint() != second.fingerprint()
    a, _ = prepare_observables(hits, first)
    b, _ = prepare_observables(hits, second)
    ga = build_ufc_event(a[a.Row == 10], first, "fixture")[0]
    gb = build_ufc_event(b[b.Row == 10], second, "fixture")[0]
    assert not torch.equal(ga.x[:, 5], gb.x[:, 5])


class FixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, batch):
        return 2. - 4. * batch.cluster.float(), torch.ones(batch.num_graphs, device=batch.x.device)


def test_event_results_do_not_depend_on_batch_boundaries_and_include_empty_events(hits):
    graphs, obs, rows = graphs_from(hits)
    attach_targets(graphs, hits)
    keys = [("fixture", int(row)) for row in rows]
    truth = truth_inventory(obs, hits, "fixture", energy_min=.1)
    results = []
    for size in (1, 2, 256):
        predictions = predict_candidates(FixedModel(), iter(graphs), batch_size=size)
        events, result = evaluate_events(predictions, keys, truth_objects=truth,
                                         threshold=.5, reco_energy_min=.1, topk=3)
        assert events.row.tolist() == [10, 20, 30]
        assert events.n_candidates.tolist() == [2, 0, 0]
        assert result["events"] == 3
        assert result["event_efficiency"] == .5
        assert result["eligible_object_efficiency"] == pytest.approx(1 / 3)
        results.append(result)
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize("memberships,truth,expected", [
    ([[1, 2], [1]], [1, 2], 2),  # Greedy assignment can incorrectly return one.
    ([[1], [1]], [1], 1),        # Duplicated candidates cannot count one neutron twice.
    ([[1, 2]], [1, 2], 1),      # A merged candidate cannot reconstruct two objects.
    ([[]], [1], 0),
])
def test_unique_matching(memberships, truth, expected):
    selected = pd.DataFrame({"signal_ids": [json.dumps(v) for v in memberships]})
    assert match_unique_objects(selected, truth) == expected


def prediction_frame(seeds=(42, 7, 21)):
    return pd.DataFrame([{"model_key": "baseline_graphsage", "seed": seed,
                          "source_id": "fixture", "row": i, "cluster": 0,
                          "score": score, "pred_E": 1., "true_y": label}
                         for seed in seeds for i, (label, score) in enumerate(((1, .9), (0, .1)))])


def test_csv_auc_is_per_seed_and_incomplete_seed_means_are_unavailable():
    complete = seed_metrics(prediction_frame())
    assert len(complete["per_seed"]) == 3
    assert complete["summary"][0]["auc_mean"] == 1.
    incomplete = seed_metrics(prediction_frame(seeds=(42, 7)))
    assert incomplete["summary"][0]["auc_mean"] is None
    assert incomplete["summary"][0]["auc_seed_sd"] is None


def test_duplicate_predictions_and_changed_populations_are_rejected():
    frame = prediction_frame()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_predictions(pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(ValueError, match="Different candidate"):
        seed_metrics(frame.drop(index=0))
    changed = frame.copy()
    changed.loc[0, "true_y"] = 0
    with pytest.raises(ValueError, match="Different candidate"):
        seed_metrics(changed)


@pytest.mark.parametrize("model_key", sorted(MODEL_CLASSES))
def test_original_checkpoint_schema_loads_strictly_for_all_architectures(tmp_path, hits, model_key):
    torch.set_num_threads(1)
    model = build_student_model(model_key).eval()
    config = {"in_dim": 6, "hidden_dim": 128, "num_layers": 4, "dropout": .1,
              "pe_dim": 4, "pe_hidden": 64, "heads": 4, "dynamic_knn": 8}
    path = tmp_path / "fixture.pt"
    torch.save({**config, "state_dict": model.state_dict()}, path)
    recovered, _ = load_student_checkpoint(path, model_key)
    graphs, _, _ = graphs_from(hits, spectral=model_key.startswith("spectral_"))
    first = predict_candidates(model, graphs, batch_size=2)
    second = predict_candidates(recovered, graphs, batch_size=1)
    np.testing.assert_allclose(first.score, second.score, atol=1e-6)
    np.testing.assert_allclose(first.pred_E, second.pred_E, atol=1e-6)
