"""Verify the locked validation selection and paired predictions on acceptance."""

import argparse
import importlib
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
old = importlib.import_module("HGNDRecoGNN.reports.energy_head_dombay_20261111.accept")
PREDICTION_ARMS = ("unit_mse", "capped_raw", "capped_calibrated")


def verify(folder, summary, plan):
    path = folder / "validation_choice.json"
    old.verify_file(path, summary["validation_choice_sha256"])
    choice = old.read(path)
    assert choice == summary["validation_choice"]
    control = plan["controls"][summary["task"]]
    assert choice["control_sha256"] == control["branch_sha256"]
    selection = plan["selection"]
    assert [r["cap"] for r in choice["candidates"]] == selection["caps"]
    counts = np.asarray(choice["training_counts"])
    assert len(counts) == 10 and np.all(counts >= 0)
    for r in choice["candidates"]:
        cap = r["cap"]
        old.verify_file(folder / f"cap_{cap}.pt", r["checkpoint_sha256"])
        weights = np.asarray(r["bin_weights"])
        assert (weights[counts > 0] > 0).all() and np.all(weights <= cap)
        assert np.isclose(weights @ counts / counts.sum(), 1, atol=1e-12, rtol=0)
        ref = choice["reference"]
        m = r["metrics"]
        assert m["count"] == ref["count"] and m["tail_count"] == ref["tail_count"] >= 25
        mae_ratio = m["mae_MeV"] / (ref["mae_MeV"] * selection["mae_ratio_limit"])
        width_ratio = m["tail_half_width_percent"] / (
            ref["tail_half_width_percent"] * selection["width_ratio_limit"]
        )
        assert r["guard_pass"] == (mae_ratio <= 1 and width_ratio <= 1)
        assert np.isclose(r["guard_overrun"], max(0, mae_ratio - 1, width_ratio - 1),
                          atol=1e-15, rtol=0)
    chosen = min(choice["candidates"], key=lambda r: (
        r["guard_overrun"], abs(r["metrics"]["tail_bias_percent"]), r["cap"]
    ))
    assert (choice["selected_cap"], choice["guard_pass"]) == (chosen["cap"], chosen["guard_pass"])
    c = choice["calibration"]
    x, y = np.asarray(c["x_e_pred"]), np.asarray(c["y_e_true"])
    assert len(x) == len(y) >= 2 and x[0] > 0 and y[0] > 0
    assert (np.diff(x) > 0).all() and (np.diff(y) > 0).all()
    previous = HERE.parent / "energy_head_dombay_20261111/exports" / f"task_{summary['task']}"
    maximum = 0.0
    for cid in summary["test_files"]:
        member = f"predictions/{cid}.npz"
        with np.load(folder / member) as new, np.load(previous / member) as old_pred:
            assert set(new.files) == {"keys", *PREDICTION_ARMS}
            assert np.array_equal(new["keys"], old_pred["keys"])
            np.testing.assert_allclose(new["unit_mse"], old_pred["unit_mse"],
                                       atol=2e-5, rtol=2e-5, err_msg=cid)
            maximum = max(maximum, float(np.max(np.abs(new["unit_mse"] - old_pred["unit_mse"]))))
            energy = new["capped_raw"].astype(float)
            expected = np.interp(energy, x, y)
            expected[energy < x[0]] = energy[energy < x[0]] * y[0] / x[0]
            expected[energy > x[-1]] = energy[energy > x[-1]] * y[-1] / x[-1]
            assert np.array_equal(expected, new["capped_calibrated"])
    return {"unit_replay_max_abs_GeV": maximum, "calibration_replay": "exact"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "target", "evidence"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("task", "job"):
        parser.add_argument("--" + name, required=True, type=int)
    args = parser.parse_args()
    caps = old.read(HERE / "plan.json")["selection"]["caps"]
    old.accept(args.bundle, args.target, args.evidence, args.task, args.job, HERE,
               PREDICTION_ARMS, tuple(f"cap_{c}" for c in caps), verify)
