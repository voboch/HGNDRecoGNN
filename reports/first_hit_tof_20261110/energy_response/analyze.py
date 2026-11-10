"""Reduce matched energies in the established truth-energy bins."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
sys.path.insert(0, str(SOURCE))
from compare import METHODS, SEEDS, load, reduce_matched  # noqa: E402

# The existing HGND performance edges resolve the soft band finely and
# widen the sparse high-energy tail. Omit the bin below the 0.3 GeV label floor.
EDGES = np.array([0.3, 0.5, 0.7, 1, 1.4, 1.8, 2.2, 2.6, 3, 3.5, 4, 5, 6.5])
MIN_BIN = 25  # Existing minimum for central quantile estimates.
METRICS = ("median_reco_GeV", "linearity_percent", "resolution_percent")


def quantiles(values, file_index, weights, probabilities):
    """Linear percentiles after repeating each observation by its file weight."""
    order = values.argsort()
    sorted_values = values[order]
    out = []
    for draw in weights:
        cumulative = np.cumsum(draw[file_index[order]], dtype=np.int64)
        rank = (cumulative[-1] - 1) * np.asarray(probabilities)
        lo = np.floor(rank).astype(int)
        lower = sorted_values[np.searchsorted(cumulative, lo, side="right")]
        upper = sorted_values[np.searchsorted(cumulative, np.ceil(rank), side="right")]
        out.append(lower + (rank - lo) * (upper - lower))
    return np.stack(out)


def main():
    pieces = [[] for _ in SEEDS]

    def collect(file_index, repeat, truth, estimates):
        files = np.full(len(truth), file_index, dtype=np.int32)
        pieces[repeat].append((files, truth, estimates))

    result, ledger = reduce_matched(collect)
    # Require exact agreement with the accepted comparison before binning.
    assert json.loads(json.dumps(result)) == load(SOURCE / "results.json")
    with np.load(SOURCE / "counts.npz") as accepted:
        for name, array in ledger.items():
            assert np.array_equal(array, accepted[name]), name
    weights = np.vstack([np.ones(ledger["content"].size, int), ledger["weights"]])
    samples = []
    for rows in pieces:
        files, truth, estimates = zip(*rows)
        samples.append(
            (np.concatenate(files), np.concatenate(truth), np.hstack(estimates))
        )

    rows = []
    csv_rows = []
    for low, high in zip(EDGES[:-1], EDGES[1:]):
        selected = [(t >= low) & (t < high) for _, t, _ in samples]
        counts = [int(mask.sum()) for mask in selected]
        bin_row = {"low_GeV": low, "high_GeV": high, "counts": counts, "methods": {}}
        rows.append(bin_row)
        file_ids = [f[mask] for (f, _, _), mask in zip(samples, selected)]
        bin_row["file_counts"] = [int(np.unique(f).size) for f in file_ids]
        if min(counts) < MIN_BIN or any(
            not (weights[:, np.unique(f)].sum(1) > 0).all() for f in file_ids
        ):
            continue
        truth_values = [t[mask] for (_, t, _), mask in zip(samples, selected)]
        bin_row["median_truth_GeV"] = float(
            np.mean([np.median(t) for t in truth_values])
        )
        draws_by_method = {}
        for m, method in enumerate(METHODS):
            repeats = []
            for (f, t, e), mask in zip(samples, selected):
                energy = e[m, mask]
                med = quantiles(energy, f[mask], weights, [0.5])[:, 0]
                q16, q50, q84 = quantiles(
                    energy / t[mask] - 1, f[mask], weights, [0.16, 0.5, 0.84]
                ).T
                repeats.append(np.column_stack((med, 100 * q50, 50 * (q84 - q16))))
            repeats = np.stack(repeats, axis=1)
            draws_by_method[method] = repeats.mean(axis=1)
            value = draws_by_method[method][0]
            sd = draws_by_method[method][1:].std(axis=0, ddof=1)
            bin_row["methods"][method] = {
                metric: {
                    "value": float(v),
                    "file_sd": float(s),
                    "by_repeat": per_repeat.tolist(),
                }
                for metric, v, s, per_repeat in zip(METRICS, value, sd, repeats[0].T)
            }
            for metric, v, s, per_repeat in zip(METRICS, value, sd, repeats[0].T):
                csv_rows.append(
                    [
                        low,
                        high,
                        *counts,
                        *bin_row["file_counts"],
                        method,
                        metric,
                        v,
                        s,
                        *per_repeat,
                    ]
                )
        # Compare on the same files, timing validity mask and training repeats.
        paired = draws_by_method[METHODS[4]] - draws_by_method[METHODS[0]]
        bin_row["smash_raw_minus_original_tof"] = {
            metric: {"value": float(v), "file_sd": float(s)}
            for metric, v, s in zip(METRICS, paired[0], paired[1:].std(axis=0, ddof=1))
        }

    result = {
        "source_job": result["source_job"],
        "repeats": SEEDS,
        "bootstrap_draws": len(weights) - 1,
        "selected_per_repeat": [len(t) for _, t, _ in samples],
        "truth_minmax_GeV_by_repeat": [
            [float(t.min()), float(t.max())] for _, t, _ in samples
        ],
        "below_first_edge": [int((t < EDGES[0]).sum()) for _, t, _ in samples],
        "at_or_above_last_edge": [int((t >= EDGES[-1]).sum()) for _, t, _ in samples],
        "bins": rows,
    }
    assert all(
        sum(b["counts"][i] for b in rows)
        + result["below_first_edge"][i]
        + result["at_or_above_last_edge"][i]
        == total
        for i, total in enumerate(result["selected_per_repeat"])
    )
    (HERE / "bins.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (HERE / "bins.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "low_GeV",
                "high_GeV",
                "n_1",
                "n_2",
                "n_3",
                "files_1",
                "files_2",
                "files_3",
                "method",
                "metric",
                "value",
                "file_sd",
                "repeat_1",
                "repeat_2",
                "repeat_3",
            ]
        )
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
