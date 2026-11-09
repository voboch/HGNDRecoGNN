"""Summarize the accepted per-file counts with the registered shared draws."""

from hashlib import sha256
import json

import numpy as np

from analyze import ARMS, HERE, SEEDS

TAGS = ("zeroSpot", "defaultSpot", "bigSpot")
PAIRS = {"18/0": (1, 0), "90/0": (2, 0), "90/18": (2, 1)}
METRICS = ("bias", "mae")


def divide(num, den):
    if not (den > 0).all():
        raise ValueError("Undefined reported ratio")
    return num / den


def with_sd(central, replicas):
    return {"value": central.tolist(), "file_sd": replicas.std(axis=0, ddof=1).tolist()}


def main():
    execution = json.loads((HERE / "execution.json").read_bytes())
    raw = (HERE / "counts.npz").read_bytes()
    assert sha256(raw).hexdigest() == execution["counts_sha256"]
    results = {
        "date": execution["date"],
        "seeds": SEEDS,
        "arms": ARMS,
        "potentials_MeV": [0, 18, 90],
        "bands": ["[0.3,1)", "[2,infinity)"],
        "states": ["negative", "[0,0.3)", "[0.3,1)", "[1,2)", "[2,infinity)"],
        "coverage_order": [
            "geometric T1",
            "native linked",
            "split only",
            "merge only",
            "split and merge",
        ],
        "flow_order": [
            "isolated true",
            "selected true",
            "selected raw",
            "selected calibrated",
        ],
        "error_selection_order": ["own score cut", "intersection of both cuts"],
        "error_energy_order": ["raw", "calibrated"],
        "uncertainty": "SD of 400 paired whole-file replicas stratified by potential and even part",
        "counts_sha256": execution["counts_sha256"],
        "summary_code_sha256": sha256((HERE / "summarize.py").read_bytes()).hexdigest(),
    }
    with np.load(HERE / "counts.npz") as data:
        ids, weights = data["content"], data["weights"]
        arrays = {
            key: data[key]
            for key in ("coverage", "survival", "flow", "migration", "errors")
        }
    groups = [np.char.startswith(ids, tag + "/") for tag in TAGS]

    def totals(key):
        array = arrays[key]
        central = np.stack([array[mask].sum(axis=0) for mask in groups])
        draws = np.stack(
            [np.tensordot(weights[:, mask], array[mask], axes=1) for mask in groups],
            axis=1,
        )
        return central, draws

    coverage, coverage_draw = totals("coverage")
    survival, survival_draw = totals("survival")
    bands = [2, 4]
    rate = 100 * divide(survival[..., bands], coverage[:, None, None, 1, bands])
    rate_draw = 100 * divide(
        survival_draw[..., bands], coverage_draw[:, :, None, None, 1, bands]
    )
    results["coverage_counts"] = coverage.tolist()
    results["survival_counts"] = survival.tolist()
    results["survival_percent"] = with_sd(rate, rate_draw)
    results["survival_B_minus_A_pp"] = with_sd(
        rate[:, :, 1] - rate[:, :, 0], rate_draw[:, :, :, 1] - rate_draw[:, :, :, 0]
    )
    # Equal seed weighting in the figure, with the same file draw applied to all seeds.
    results["survival_seed_mean_percent"] = with_sd(
        rate.mean(axis=1), rate_draw.mean(axis=2)
    )

    flow, flow_draw = totals("flow")
    results["flow_counts"] = flow.tolist()
    results["migration_counts"] = totals("migration")[0].tolist()
    ratio = divide(flow[..., 4], flow[..., 2])
    ratio_draw = divide(flow_draw[..., 4], flow_draw[..., 2])
    results["isolated_Rn"] = with_sd(ratio, ratio_draw)
    results["isolated_contrasts_percent"] = {}
    for pair, (a, b) in PAIRS.items():
        contrast = 100 * (divide(ratio[a], ratio[b]) - 1)
        draws = 100 * (divide(ratio_draw[:, a], ratio_draw[:, b]) - 1)
        results["isolated_contrasts_percent"][pair] = {
            "levels": with_sd(contrast, draws),
            "score_minus_isolated_truth_pp": with_sd(
                contrast[..., 1] - contrast[..., 0], draws[..., 1] - draws[..., 0]
            ),
            "calibrated_minus_selected_truth_pp": with_sd(
                contrast[..., 3] - contrast[..., 1], draws[..., 3] - draws[..., 1]
            ),
            "calibrated_minus_isolated_truth_pp": with_sd(
                contrast[..., 3] - contrast[..., 0], draws[..., 3] - draws[..., 0]
            ),
        }
    err, err_draw = totals("errors")
    metrics = 1000 * divide(err[..., 1:], err[..., :1])
    metrics_draw = 1000 * divide(err_draw[..., 1:], err_draw[..., :1])
    results["energy_error_count"] = err[..., 0].astype(int).tolist()
    results["energy_metric_order"] = METRICS
    results["energy_error_MeV"] = with_sd(metrics, metrics_draw)
    common = metrics[:, :, :, 1]
    common_draw = metrics_draw[:, :, :, :, 1]
    results["common_energy_B_minus_A_MeV"] = with_sd(
        common[:, :, 1] - common[:, :, 0],
        common_draw[:, :, :, 1] - common_draw[:, :, :, 0],
    )
    results["common_energy_calibrated_minus_raw_MeV"] = with_sd(
        common[..., 1, :] - common[..., 0, :],
        common_draw[..., 1, :] - common_draw[..., 0, :],
    )
    results["common_energy_seed_mean_MeV"] = with_sd(
        common.mean(axis=1), common_draw.mean(axis=2)
    )
    (HERE / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
