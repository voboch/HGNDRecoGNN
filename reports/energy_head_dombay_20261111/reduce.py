"""Compare energy response and hardness with archived paired file draws."""

import argparse
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIRST_HIT = ROOT / "reports/first_hit_tof_20261110"
FIXED = ROOT / "reports/rn_score_cut_20261110"
ODD = ROOT / "reports/rule_tof_odd_20261110"
BASE = ROOT / "results/smash_train_20261028"
SEEDS = (42, 123, 456)
KEYS = ["Row", "ClusterID", "cl_istop"]
ARMS = ("baseline_raw", "unit_mse", "energy_weighted_mse")
ALL_ARMS = ("mc_truth", "baseline_calibrated", *ARMS)
PAIRS = {"18/0": (1, 0), "90/0": (2, 0), "90/18": (2, 1)}
ODD_REFERENCE = "3c994767445146594af810359d5842628ea5d65bbd7d78fa890ba83f3973733b"
sys.path.insert(0, str(FIRST_HIT))
from compare import reduce_matched  # noqa: E402


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return sha256_file(stream)


def sha256_file(stream):
    digest = sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Input digest changed: {path}")
    return path


def compare_inputs(export, plan, source_sums):
    runs = {}
    for path in sorted(export.glob("task_*/summary.json")):
        info = load(path)
        key = info["fold"], info["seed"]
        run = next(x for x in plan["folds"][key[0]]["runs"] if x["task"] == info["task"])
        assert info["state"] == "complete" and info["epochs"] == 15 and key not in runs
        assert info["plan_sha256"] == digest(HERE / "plan.json")
        assert info["source_sums_sha256"] == source_sums
        assert info["baseline_complete_sha256"] == run["complete_sha256"]
        assert info["split_sha256"] == plan["split_sha256"]
        assert info["prepare_manifest_sha256"] == plan["prepare"]["manifest_sha256"]
        runs[key] = (path.parent, info)
    assert set(runs) == {(fold, seed) for fold in range(3) for seed in SEEDS}
    registry = {}
    for job, fold_plan in zip((4466339, 4467086, 4467091), plan["folds"], strict=True):
        path = BASE / f"evaluate_{job}"
        entry = fold_plan["evaluation"]["summary_files"]["evaluation/summary.json"]
        info = load(checked(path / "evaluation_summary.json", entry))
        for cid in info["test_files"]:
            assert cid not in registry
            registry[cid] = (path / "extracted/evaluation", info)
        for seed in SEEDS:
            run = runs[info["fold"], seed][1]
            assert run["test_files"] == info["test_files"]
            assert run["initial_epoch"] == info["training"][str(seed)]["selection"]["e_en"]
    assert len(registry) == 568
    return runs, registry


def predictions(cid, seed, registry, runs):
    root, info = registry[cid]
    raw_ref = info["predictions"]["B0"][str(seed)][cid]
    raw = pd.read_csv(checked(root / raw_ref["path"], raw_ref["sha256"]),
                      usecols=[*KEYS, "cl_score", "e_pred"])
    with np.load(runs[info["fold"], seed][0] / "predictions" / (cid + ".npz")) as new:
        assert np.array_equal(new["keys"], raw[KEYS].to_numpy())
        energy = np.stack([raw.e_pred, *(new[a] for a in ARMS[1:])])
        assert np.isfinite(energy).all()
    return raw, energy


def summary(draws):
    value = float(draws[0])
    low, high = np.quantile(draws[1:], [.16, .84])
    return {"value": value, "q16": float(low), "q84": float(high),
            "half_width": float((high - low) / 2), "file_sd": float(draws[1:].std(ddof=1))}


def differences(values):
    return {
        "weighted_minus_original": values[..., 2] - values[..., 0],
        "weighted_minus_unit": values[..., 2] - values[..., 1],
        "unit_minus_original": values[..., 1] - values[..., 0],
    }


def relative_bins(runs, registry):
    spec = spec_from_file_location("accepted_bins", FIRST_HIT / "energy_response/analyze.py")
    analyzer = module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    matched = [[] for _ in SEEDS]

    def visit(cid, repeat, indices, truth, reference):
        _, e = predictions(cid, SEEDS[repeat], registry, runs)
        assert np.array_equal(e[0, indices], reference[4])
        matched[repeat].append((cid, truth, e[:, indices]))

    comparison, ledger = reduce_matched(visit_keys=visit)
    assert json.loads(json.dumps(comparison)) == load(FIRST_HIT / "results.json")
    with np.load(FIRST_HIT / "counts.npz") as accepted:
        assert all(np.array_equal(value, accepted[key]) for key, value in ledger.items())
    ids = ledger["content"].tolist()
    index = {cid: i for i, cid in enumerate(ids)}
    weights = np.vstack([np.ones(len(ids), dtype=int), ledger["weights"]])
    samples = []
    for items in matched:
        files, truth, energy = zip(*items)
        f = np.concatenate([np.full(len(t), index[c], dtype=int) for c, t in zip(files, truth)])
        samples.append((f, np.concatenate(truth), np.hstack(energy)))

    def bin_metrics(samples):
        by_seed = []
        for files, truth, energies in samples:
            metrics = []
            for e in energies:
                q16, q50, q84 = analyzer.quantiles(e / truth - 1, files, weights, [.16, .5, .84]).T
                metrics.append(np.stack([100 * q50, 50 * (q84 - q16)], axis=-1))
            by_seed.append(np.stack(metrics, axis=1))
        return np.stack(by_seed, axis=1)

    outputs = []
    accepted_bins = load(FIRST_HIT / "energy_response/bins.json")
    accepted_bins = accepted_bins["bins"] if isinstance(accepted_bins, dict) else accepted_bins
    for old, lo, hi in zip(accepted_bins, analyzer.EDGES[:-1], analyzer.EDGES[1:], strict=True):
        assert (lo, hi) == (old["low_GeV"], old["high_GeV"])
        selected = []
        for files, truth, energy in samples:
            keep = (truth >= lo) & (truth < hi)
            selected.append((files[keep], truth[keep], energy[:, keep]))
        counts = [len(t) for _, t, _ in selected]
        assert counts == old["counts"]
        row = {"range_GeV": [lo, hi], "counts": counts}
        outputs.append(row)
        if not old["methods"]:
            continue
        values = bin_metrics(selected)
        mean = values.mean(axis=1)
        row["median_truth_GeV"] = old["median_truth_GeV"]
        for m, key in enumerate(("linearity_percent", "resolution_percent")):
            assert np.isclose(mean[0, 0, m], old["methods"]["SMASH-trained raw"][key]["value"],
                              atol=1e-12, rtol=0)
            draws = mean[:, :, m]
            row[key] = {
                **{arm: {**summary(draws[:, a]), "by_seed": values[0, :, a, m].tolist()}
                   for a, arm in enumerate(ARMS)},
                **{name: summary(d) for name, d in differences(draws).items()},
            }
    absolute = []
    for files, truth, energy in samples:
        multiplicities = weights[:, files]
        absolute.append(1000 * (multiplicities @ np.abs(energy - truth).T) /
                        multiplicities.sum(axis=1)[:, None])
    maes = np.stack(absolute, axis=1)
    means = maes.mean(axis=1)
    return {
        "matched_counts": [len(t) for _, t, _ in samples], "bins": outputs,
        "mae_MeV": {
            **{arm: {**summary(means[:, a]), "by_seed": maes[0, :, a].tolist()}
               for a, arm in enumerate(ARMS)},
            **{name: summary(d) for name, d in differences(means).items()},
        },
    }


def band_counts(energy):
    return [
        np.sum((energy > .3) & (energy < 1), axis=-1),
        np.sum((energy >= 2) & (energy < 4), axis=-1),
        np.sum(energy >= 4, axis=-1),
    ]


def odd_contrast(runs, registry, output):
    reference = load(checked(ODD / "results.json", ODD_REFERENCE))
    with np.load(ODD / "counts.npz") as prior:
        ids, draws, tags = prior["content"], prior["draws"], prior["tags"]
        prior_names = prior["arms"].tolist()
        assert len(ids) == 282 and draws.shape == (400, 282)
        truth_index, gnn_index = prior_names.index("mc"), prior_names.index("gnn")
        # The accepted ledger stores 2 <= E < 4 and E >= 4 separately.
        old = np.stack([prior["soft"], prior["hard"], prior["overflow"]], axis=-1)
        counts = np.zeros((len(ids), len(SEEDS), len(ALL_ARMS), 3), dtype=np.int64)
        counts[:, :, 0] = old[:, :, truth_index]
        counts[:, :, 1] = old[:, :, gnn_index]
    with np.load(FIXED / "fixed_results/files.npz") as fixed, np.load(FIXED / "fixed_results/draw_weights.npz") as saved:
        keep = np.isin(fixed["parts"], (1, 3, 5))
        assert np.array_equal(ids, fixed["ids"][keep]) and np.array_equal(ids, saved["ids"][keep])
        assert np.array_equal(draws, saved["weights"][:, keep])
        assert fixed["spans"][keep].sum() == reference["event_count"] == 1165216
        assert np.array_equal(counts[:, 0, 0, 1:].sum(axis=-1), fixed["truth"][keep, 1, 0])
        assert np.array_equal(counts[:, 0, 0, 0], fixed["truth"][keep, 1, 1])
        expected = fixed["objects"][keep, :, 1, 2]
        hist = fixed["spectra"][keep, :, 1, 4]
        edges = fixed["edges"]
        accepted_raw = np.stack([
            hist[..., (edges[:-1] >= .3) & (edges[1:] <= 1)].sum(axis=-1),
            hist[..., (edges[:-1] >= 2) & (edges[1:] <= 4)].sum(axis=-1),
            hist[..., edges[:-1] >= 4].sum(axis=-1),
        ], axis=-1)
    cuts = load(FIXED / "choice.json")["per_fold_seed_score_cuts"]
    for i, cid in enumerate(ids):
        _, info = registry[cid]
        for s, seed in enumerate(SEEDS):
            raw, energy = predictions(cid, seed, registry, runs)
            select = raw.cl_score.to_numpy() > cuts[f"{info['fold']}/{seed}"]
            assert select.sum() == expected[i, s]
            counts[i, s, 2:] = np.asarray(band_counts(energy[:, select])).T
    assert np.array_equal(counts[:, :, 2], accepted_raw)
    np.savez_compressed(output / "odd_counts.npz", content=ids, tags=tags, draws=draws,
                        counts=counts, arms=ALL_ARMS, seeds=SEEDS)
    tag_order = ("zeroSpot", "defaultSpot", "bigSpot")
    draws = np.vstack([np.ones(len(ids), dtype=int), draws])
    sums = np.stack([np.einsum("df,fsac->dsac", draws[:, tags == t], counts[tags == t])
                     for t in tag_order], axis=1)
    assert (sums[..., 0] > 0).all()
    out = {}
    for band, hard in (("open_2_GeV", sums[..., 1:].sum(axis=-1)), ("2_to_4_GeV", sums[..., 1])):
        rn = hard / sums[..., 0]
        assert (rn > 0).all()
        old_contrasts = reference["open_upper" if band == "open_2_GeV" else "within_reach"]["contrasts"]
        result = {}
        for pair, (a, b) in PAIRS.items():
            by_seed = 100 * (rn[:, a] / rn[:, b] - 1)
            c = by_seed.mean(axis=1)
            bias = c - c[:, :1]
            for method, values in (("mc", c[:, 0]), ("gnn", c[:, 1]), ("gnn_minus_mc", bias[:, 1])):
                for field, value in summary(values).items():
                    name = "half_width_68" if field == "half_width" else field
                    assert np.isclose(value, old_contrasts[pair][method][name], rtol=0, atol=1e-12), (
                        band, pair, method, name
                    )
            result[pair] = {
                "contrast_percent": {
                    arm: {**summary(c[:, j]), "by_seed": by_seed[0, :, j].tolist()}
                    for j, arm in enumerate(ALL_ARMS)
                },
                "paired_bias_vs_mc_pp": {arm: summary(bias[:, j])
                                         for j, arm in enumerate(ALL_ARMS) if j},
                "paired_change_pp": {name: summary(d) for name, d in differences(c[:, 2:]).items()},
                "change_in_abs_bias_pp": {
                    name: summary(d) for name, d in differences(np.abs(bias[:, 2:])).items()
                },
                "change_in_half_width_pp": {
                    name: float(d) for name, d in differences(
                        np.array([summary(c[:, j])["half_width"] for j in range(2, len(ALL_ARMS))])
                    ).items()
                },
            }
        out[band] = {"Rn_by_setting_and_arm": {
            tag: {arm: rn[0, t, :, j].tolist() for j, arm in enumerate(ALL_ARMS)}
            for t, tag in enumerate(tag_order)
        }, "pairs": result}
    return out


def main(args):
    plan = load(HERE / "plan.json")
    for path, expected in plan["reduction_pins"].items():
        checked(ROOT / path, expected)
    runs, registry = compare_inputs(args.exports, plan, args.source_sums)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "matched_even_energy": relative_bins(runs, registry),
        "odd_hardness": odd_contrast(runs, registry, args.output),
        "source_sums_sha256": args.source_sums, "plan_sha256": digest(HERE / "plan.json"),
        "run_summaries": [info for _, info in runs.values()],
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--source-sums", required=True)
    main(p.parse_args())
