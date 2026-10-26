"""Synthetic fixtures for the retrospective R_n ladder; no real transfer data is read."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts import reco_rn_retrospective as rr  # noqa: E402


def frame(score, label, e_true, e_pred, rows=None):
    n = len(score)
    return pd.DataFrame({"Row": np.arange(n) if rows is None else rows, "cl_score": score,
                         "e_pred": e_pred, "cl_label": label, "e_true": e_true})


def test_band_edges_are_half_open_and_skip_the_middle():
    e = np.array([0.5, 0.99, 1.0, 1.5, 1.99, 2.0, 3.0])
    assert rr.band_counts(e, np.ones(len(e))).tolist() == [2.0, 2.0]


def test_selection_is_strict_and_rungs_use_their_energy():
    f = frame(score=[0.5, 0.6, 0.6, 0.2], label=[1, 1, 0, 1],
              e_true=[2.5, 0.5, 0.0, 3.0], e_pred=[0.8, 2.2, 2.4, 2.1])
    c = rr.cluster_counts(f, cut=0.5)
    assert c[0].tolist() == [2, 1]   # C1: all positives on e_true
    assert c[1].tolist() == [0, 1]   # C2: score 0.5 is not above the cut
    assert c[2].tolist() == [1, 0]   # C3: the same cluster on e_pred
    assert c[3].tolist() == [2, 0]   # C4: background enters on e_pred


def test_injection_raises_c1_hardness_by_exactly_the_weight():
    rng = np.random.default_rng(1)
    n = 500
    f = frame(score=rng.random(n), label=rng.integers(0, 2, n),
              e_true=rng.uniform(0.3, 4, n), e_pred=rng.uniform(0, 4, n))
    base = rr.ratio(rr.cluster_counts(f, 0.4))
    shifted = rr.ratio(rr.cluster_counts(f, 0.4, rr.injection_weight(f)))
    assert shifted[0] / base[0] == pytest.approx(rr.INJECT, rel=1e-12)


def test_truth_rungs_cut_the_processed_range_and_candidate_events():
    events = pd.DataFrame({"acc_n_hi": [1, 2, 3, 4], "acc_n_lo": [5, 6, 7, 8]},
                          index=pd.Index([0, 1, 2, 3], name="Row"))
    t = rr.truth_counts(events, span=3, candidate_rows=np.array([1, 3]))
    assert t.tolist() == [[10, 26], [6, 18], [6, 14]]


def test_event_counts_group_by_event_and_drop_empty_events():
    f = frame(score=[0.9, 0.9, 0.9, 0.1], label=[1, 0, 1, 1], e_true=[0, 0, 0, 0],
              e_pred=[2.5, 0.5, 1.5, 2.5], rows=[7, 7, 8, 9])
    assert rr.event_counts(f, 0.5).tolist() == [[1, 1]]


def test_empty_denominator_is_refused():
    with pytest.raises(ValueError):
        rr.ratio(np.array([3.0, 0.0]))


@pytest.mark.parametrize("diff,diff_sd,truth,state", [
    (0.2, 0.2, 4.0, "pass"), (0.2, 0.6, 4.0, "not resolvable"),
    (3.0, 1.0, 4.0, "fail"), (1.6, 0.25, 4.0, "fail"),
])
def test_closure_gate_has_three_states(diff, diff_sd, truth, state):
    assert rr.gate_g2(diff, diff_sd, truth) == state


def test_null_gate_judges_the_largest_z():
    assert rr.null_gate([0.1, -2.4]) == "pass"
    assert rr.null_gate([2.6]) == "marginal"
    assert rr.null_gate([-3.0]) == "stop"


@pytest.mark.parametrize("diff,diff_sd,state", [
    (0.1, 0.3, "pass"), (0.1, 0.8, "not resolvable"), (2.0, 0.5, "fail"), (0.0, 0.0, "pass"),
])
def test_energy_migration_gate_is_unresolvable_above_the_m4_size(diff, diff_sd, state):
    assert rr.gate_g3(diff, diff_sd) == state


def test_a_gate_passes_only_for_every_seed():
    assert rr.combine_seeds(["pass"] * 3) == "pass"
    assert rr.combine_seeds(["pass", "not resolvable", "pass"]) == "not resolvable"
    assert rr.combine_seeds(["pass", "fail", "not resolvable"]) == "fail"


def test_files_needed_reproduces_the_review_estimate():
    # A paired SD of 1.4 points at T1 = 4.4 % with 195 and 51 files gives about
    # 520 files per potential, the review v9 S4 estimate.
    n = rr.files_needed(1.4, 4.4, 195, 51)
    assert 480 < n < 560
    # At the needed count the 2-SD interval of a null difference equals the threshold.
    sd_at_n = 1.4 * np.sqrt((2 / n) / (1 / 195 + 1 / 51))
    assert 2 * sd_at_n == pytest.approx(0.25 * 4.4)


def synthetic_counts(rng, shift=0.0):
    out = {}
    for tag, n in zip(rr.TAGS, (40, 38, 12)):
        hi = rng.poisson(300 * (1 + (shift if tag == "bigSpot" else 0)), size=n)
        lo = rng.poisson(450, size=n)
        truth = np.stack([np.stack([hi, lo], 1)] * 3, axis=1).astype(float)
        clusters = np.stack([np.stack([truth[:, 0]] * 4, axis=1)] * len(rr.SEEDS), axis=1)
        out[tag] = {"truth": truth, "clusters": clusters,
                    "injected": clusters * np.array([1.05, 1.0]) if tag == "zeroSpot" else None,
                    "per_event": {k: rng.poisson(1.0, size=(200, 2)).astype(float) + [0, 1]
                                  for k in ("T1",) + rr.SEEDS},
                    "coverage": []}
    return out


def test_identical_rungs_give_exactly_zero_paired_steps():
    result, arrays = rr.analyse(synthetic_counts(np.random.default_rng(3), shift=0.05))
    for key, step in result["steps"].items():
        assert step["central_percent"] == pytest.approx(0.0, abs=1e-9), key
        assert step["bootstrap_sd_percent"] == pytest.approx(0.0, abs=1e-9), key
    for key, diff in result["seed_differences_C4"].items():
        assert diff["bootstrap_sd_percent"] == pytest.approx(0.0, abs=1e-9), key
    endpoint = result["endpoint"]["C4/seed_42"]["central_percent"]
    assert endpoint == pytest.approx(result["endpoint"]["T1/seed_42"]["central_percent"])
    k = result["injection"]["seed_42"]["recovery_k"]
    assert k["central"] == pytest.approx(1.0) and k["bootstrap_sd"] == pytest.approx(0.0, abs=1e-9)
    assert arrays["endpoint"].shape == (rr.N_BOOT, len(rr.SEEDS), len(rr.RUNGS))


def test_bootstrap_is_reproducible():
    a, _ = rr.analyse(synthetic_counts(np.random.default_rng(4)))
    b, _ = rr.analyse(synthetic_counts(np.random.default_rng(4)))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_gates_predictions_and_figure_run_on_a_synthetic_result(tmp_path):
    result, _ = rr.analyse(synthetic_counts(np.random.default_rng(5), shift=0.05))
    g = rr.gates(result, {"zeroSpot": 40, "defaultSpot": 38, "bigSpot": 12})
    assert set(g["all_seeds"]) == {"G1_same_sign", "G2_closure", "G3_energy_migration"}
    assert g["decision"]["seed_42"]["largest_step"] in rr.COMPONENT_STEPS
    checks = rr.prediction_check(result, g)
    assert len(checks) == len(rr.PREDICTED) * len(rr.SEEDS) + len(rr.PREDICTED_GLOBAL)
    rr.plot(result, tmp_path)
    assert (tmp_path / "figures" / "ladder.pdf").stat().st_size > 0


def test_changed_pinned_receipt_stops_the_run(tmp_path):
    base = tmp_path / rr.TRANSFER
    base.mkdir(parents=True)
    (base / "full_transfer_summary_bundle.json").write_text("{}")
    (base / "bound_inventory.json").write_text("{}")
    pins = {str(rr.TRANSFER / "full_transfer_summary_bundle.json"): "0" * 64,
            str(rr.TRANSFER / "bound_inventory.json"): "0" * 64}
    (base / "full_metrics_verified.json").write_text(json.dumps({"input_sha256": pins}))
    with pytest.raises(ValueError, match="Pinned transfer receipt changed"):
        rr.load_inputs(tmp_path)
