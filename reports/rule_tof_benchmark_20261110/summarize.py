"""Summarize matched candidate accounting with shared whole-file replicas."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BAND_LABELS = ("below_0p3", "0p3_to_1", "1_to_2", "2_to_4", "at_least_4")
ARMS = ("rules", "gnn")


def estimate(values):
    center = float(values[0])
    q16, q84 = np.quantile(values[1:], (0.16, 0.84))
    return dict(value=center, file_sd=float(np.std(values[1:], ddof=1)),
                q16=float(q16), q84=float(q84))


def pair(values):
    means = values.mean(axis=1)
    return {
        "rules": estimate(means[:, 0]),
        "gnn": estimate(means[:, 1]),
        "gnn_minus_rules": estimate(means[:, 1] - means[:, 0]),
        "by_seed": dict(zip(("42", "123", "456"), values[0].tolist(), strict=True)),
    }


def scope(weights, counts, bands, multiplicity, error, keys):
    def fold(values):
        return np.tensordot(weights, values, axes=1)
    totals = fold(counts)
    c = dict(zip(keys, np.moveaxis(totals, -1, 0), strict=True))
    t, k, ev = c["truth"], c["candidates"], c["events"]
    rates = {
        "direct_coverage_percent": 100 * c["truth_with_edge"] / t,
        "direct_miss_percent": 100 * c["truth_missed"] / t,
        "split_percent_of_truth": 100 * c["truth_split"] / t,
        "merge_percent_of_selected": 100 * c["candidate_merge"] / k,
        "mixed_merge_percent_of_selected": 100 * c["candidate_mixed_merge"] / k,
        "in_window_direct_percent_of_selected": 100 * c["candidate_in"] / k,
        "out_only_percent_of_selected": 100 * c["candidate_out_only"] / k,
        "no_edge_percent_of_selected": 100 * c["candidate_no_edge"] / k,
        "exclusive_truth_deficit_percent": 100 * c["exclusive_truth_deficit"] / t,
        "exclusive_candidate_excess_percent": 100 * c["exclusive_candidate_excess"] / k,
        "zero_candidate_events_percent": 100 * c["zero_candidates"] / ev,
        "candidates_per_event": k / ev,
    }
    by_truth = fold(bands)
    rates["direct_coverage_true_0p3_to_4_percent"] = (
        100 * by_truth[..., 1, 1:4].sum(axis=-1) / by_truth[..., 0, 1:4].sum(axis=-1)
    )
    by_mult = fold(multiplicity)
    e = fold(error)
    if not np.array_equal(e[..., 0, 0], e[..., 1, 0]):
        raise ValueError("Energy comparison is not paired by primary")
    return {
        "counts": {key: pair(values) for key, values in c.items()},
        "rates": {key: pair(values) for key, values in rates.items()},
        "truth_band_coverage": {
            label: pair(100 * by_truth[..., 1, i] / by_truth[..., 0, i])
            for i, label in enumerate(BAND_LABELS)
        },
        "event_multiplicity": {
            "order": ["0", "1", "2", "3_or_more"],
            "central_truth_rows_candidate_columns_by_seed_arm": by_mult[0].tolist(),
            "no_geometric_truth_percent_by_candidate_count": [
                pair(100 * by_mult[..., 0, n] / by_mult[..., 0, :].sum(axis=-1))
                for n in range(4)
            ],
        },
        "common_energy": {
            "n": pair(e[..., 0]), "bias_MeV": pair(1000 * e[..., 1] / e[..., 0]),
            "mae_MeV": pair(1000 * e[..., 2] / e[..., 0]),
        },
    }


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(counts_path, export, output):
    with np.load(counts_path) as a:
        tag = np.asarray([value.split("/")[0] for value in a["content"]])
        weights = np.vstack([np.ones(len(tag), np.int64), a["weights"]])
        scopes = {"pooled": np.ones(len(tag), bool),
                  "0": tag == "zeroSpot", "18": tag == "defaultSpot", "90": tag == "bigSpot"}
        result = {
            "scope": "same-event direct associations on fixed SMASH simulation before spectrum corrections",
            "file_count": len(tag), "bootstrap_draws": len(weights) - 1, "arms": ARMS,
            "counts_sha256": sha(counts_path),
            "submission_sha256": sha(HERE / "reach_submission.json"),
            "receipt_sha256": sha(export / "receipt.json"),
            "manifest_sha256": sha(export / "manifest.json"),
            "literal_rule_counts": dict(zip(("selected", "soft"), a["literal_counts"].sum(axis=0).tolist(), strict=True)),
            "gnn_window_rejections_by_seed": {
                seed: {"at_or_below_0p3": below, "at_or_above_4": above}
                for seed, (below, above) in zip(
                    ("42", "123", "456"), a["gnn_window_loss"].sum(axis=0).tolist(), strict=True
                )
            },
            "rule_rejections": dict(zip(a["rule_rejection_keys"].tolist(),
                                        a["rule_rejections"].sum(axis=0).tolist(), strict=True)),
            "scopes": {name: scope(weights[:, use], a["counts"][use], a["truth_bands"][use],
                                   a["multiplicity"][use], a["error"][use], a["count_keys"].tolist())
                       for name, use in scopes.items()},
        }
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("counts", "export", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    main(args.counts, args.export, args.output)
