"""Select a bounded energy weighting using the corresponding validation fold."""

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
previous = importlib.import_module("HGNDRecoGNN.reports.energy_head_dombay_20261111.run")
from HGNDRecoGNN.scripts.smash_train import fit_calibration  # noqa: E402


def bounded_weights(counts, cap):
    """Square-root inverse frequency with mean one and a hard per-object cap."""
    n = counts.sum()
    active = counts > 0
    frequency = counts[active] / n
    relative = 1 / np.sqrt(frequency)
    low, high = 0.0, cap / relative.min()
    for _ in range(52):  # Bisection to float64 precision.
        scale = (low + high) / 2
        if np.dot(frequency, np.minimum(scale * relative, cap)) < 1:
            low = scale
        else:
            high = scale
    weights = np.zeros(len(counts))
    weights[active] = np.minimum(high * relative, cap)
    if not np.isclose(np.dot(counts, weights) / n, 1, rtol=0, atol=1e-12):
        raise ValueError("Capped training weights are not mean one")
    return weights


def metrics(target, energy):
    selected = (target >= 4) & (target < 5)
    if selected.sum() < 25:  # Inherited response minimum.
        raise ValueError("Too few validation positives in the registered 4–5 GeV bin")
    error = energy - target
    q16, median, q84 = np.quantile(error[selected] / target[selected], [.16, .5, .84])
    return {
        "count": len(target), "tail_count": int(selected.sum()),
        "mae_MeV": float(1000 * np.abs(error).mean(dtype=np.float64)),
        "tail_bias_percent": float(100 * median), "tail_half_width_percent": float(50 * (q84 - q16)),
    }


def choose(rows, reference, selection):
    mae_limit = reference["mae_MeV"] * selection["mae_ratio_limit"]
    width_limit = reference["tail_half_width_percent"] * selection["width_ratio_limit"]
    for row in rows:
        m = row["metrics"]
        row["guard_pass"] = m["mae_MeV"] <= mae_limit and m["tail_half_width_percent"] <= width_limit
        row["guard_overrun"] = max(0, m["mae_MeV"] / mae_limit - 1,
                                    m["tail_half_width_percent"] / width_limit - 1)
    return min(rows, key=lambda r: (r["guard_overrun"], abs(r["metrics"]["tail_bias_percent"]),
                                    r["cap"]))


def fit_policy(args, original, train, validation, seed, main):
    plan = json.loads(args.plan.read_text())
    selection = plan["selection"]
    control = plan["controls"][args.task]
    path = previous.extract_members(control["bundle_dir"], control["manifest_sha256"],
                                    ["unit_mse.pt"], args.work / "control")["unit_mse.pt"]
    if previous.sha(path) != control["branch_sha256"]:
        raise ValueError("Unweighted control branch changed")
    unit = deepcopy(original)
    unit.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    unit.eval()
    vx, vy, _ = validation
    target = vy.numpy()
    reference = metrics(target, previous.score(unit, vx, main))
    values = train[1].numpy()
    counts, edges = np.histogram(values, bins=previous.BINS)
    index = np.digitize(values, edges[1:-1])
    if not np.array_equal(counts, np.bincount(index, minlength=len(counts))):
        raise ValueError("Training weights and histogram populations differ")
    candidates, branches, val_predictions = [], {}, {}
    for cap in selection["caps"]:
        bin_weights = bounded_weights(counts, cap)
        weights = bin_weights[index]
        name = f"cap_{cap}"
        branch, history = previous.fit_branch(original, train, validation, weights, seed, main, name)
        torch.save(branch.state_dict(), args.output / f"{name}.pt")
        previous.write_json(args.output / f"{name}_history.json", history)
        energies = previous.score(branch, vx, main)
        candidates.append({
            "cap": cap, "metrics": metrics(target, energies), "bin_weights": bin_weights.tolist(),
            "max_weight": float(weights.max()),
            "effective_positives": float(weights.sum() ** 2 / np.square(weights).sum()),
            "checkpoint_sha256": previous.sha(args.output / f"{name}.pt"),
        })
        branches[cap], val_predictions[cap] = branch, energies

    chosen = choose(candidates, reference, selection)
    cap = chosen["cap"]
    calibration = fit_calibration(target, val_predictions[cap])
    choice = {
        "reference": reference, "control_sha256": control["branch_sha256"],
        "candidates": candidates, "selected_cap": cap, "guard_pass": chosen["guard_pass"],
        "training_counts": counts.tolist(), "training_edges_GeV": edges.tolist(),
        "calibration": calibration,
    }
    choice_path = args.output / "validation_choice.json"
    previous.write_json(choice_path, choice)
    print(f"Validation locked cap={cap}, guard_pass={chosen['guard_pass']}", flush=True)
    return (
        {"unit_mse": unit, "capped_raw": branches[cap]},
        {"capped_calibrated": ("capped_raw", calibration)},
        {"validation_choice": choice, "validation_choice_sha256": previous.sha(choice_path)},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plan", "split", "work", "output", "source-sums"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--task", type=int, choices=range(len(previous.TASKS)), required=True)
    previous.run(parser.parse_args(), fit_policy)
