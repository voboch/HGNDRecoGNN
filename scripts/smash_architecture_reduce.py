"""Reduce the registered raw-energy SMASH architecture comparison.

Inputs are accepted, extracted prediction bundles. The pinned odd release
selects raw70 alone or raw70 with a frozen even policy. The historical
Dombay replay follows a separate input contract.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BENCHMARK = REPORTS / "rule_tof_benchmark_20261110"
sys.path.insert(0, str(BENCHMARK))
from reduce import COUNT_KEYS, accepted, isolation, metric, require  # noqa: E402
from contrast import PAIRS, TAGS  # noqa: E402

BASE = ROOT / "results/smash_train_20261028"
PRIMARY = REPORTS / "primary_association_20261106"
ODD = REPORTS / "rule_tof_odd_20261110"
SEEDS = ("42", "123", "456")
FOLDS = (0, 1, 2)
MODELS = ("student_sage", "student_dynedge", "smash_hetero_sage")
# Scoring and fit receipts use model-major, then fold-major task indices.
CHALLENGER_TASKS = tuple((model, fold, seed) for model in MODELS[1:] for fold in FOLDS for seed in SEEDS)
ODD_RELEASE_SCHEMA = "smash-architecture-odd-release-v1"
RAW70_ONLY = "raw70_only"
WITH_FROZEN_POLICY = "raw70_with_frozen_even_policy"
SPLIT_SHA = "848ebf492b254b75e09297b95f89d7c2773f7877afe5e629520d520904a59886"
PREPARE_SHA = "c0156ec47c8ef9d7e5b4b714660b53a4e8e5fc3a10ad9a0c6421ecc7b1130501"
PREPARE_SUMMARY_SHA = "2b58f9ccce0efccb4c66afa96e93c4bbd124c1c61a22593baa05c26544de7133"
ARCHIVES = {
    "even": (BENCHMARK / "counts.npz", "d1be578faaf364a182f68550da67262c3cc1ed085b4bb6e01a05e12fcfd68e2c"),
    "odd": (ODD / "counts.npz", "034c4b4e0667ecbf9a7dd1fe07b605ca04a4b05700e94bbd6c9862dccecda312"),
}
BASELINE = {
    4466339: "30e456e22b83d78b1d947247f338da771353b7918bbf3e5f3c3ebb7eecdcd6a7",
    4467086: "6306bc293bdce9011ac5800929e4fd373ee63a86de3089bfbcde57e7254f984a",
    4467091: "2b4fcbb19376874674a0b9fb7ef675a14780266f9755f1e734ce131e09b7154c",
}
REFERENCE_CEILING = REPORTS / "smash_architecture_20261112/reference_ceiling.json"
REFERENCE_CEILING_SHA = "2eb8bdfa344b5fc447ec255af3043971c0dee47e6f143b81366046fa43fe42c7"
# Inherited comparison bands, threshold grid and file resampling in plan §§3–5.
PARTS = {"even": (0, 2, 4), "odd": (1, 3, 5)}
SPAN_TOTAL = {"even": 1_180_595, "odd": 1_165_216}
TARGETS = tuple(str(p) for p in range(60, 100, 5))
THRESHOLD_GRID = np.linspace(.001, .999, 401)
STAGES = ("all", "score", "score_energy")
TRUTH_RANGES = {
    "all": (0, np.inf), "0p3_to_4": (.3, 4), "below_0p3": (0, .3),
    "0p3_to_1": (.3, 1), "1_to_2": (1, 2), "2_to_4": (2, 4),
    "at_least_4": (4, np.inf),
}
# Finer soft bins and wider sparse tail from first_hit_tof_20261110/energy_response.
ENERGY_EDGES = (.3, .5, .7, 1, 1.4, 1.8, 2.2, 2.6, 3, 3.5, 4, 5, 6.5)
KEYS = ["Row", "ClusterID", "cl_istop"]
COLUMNS = KEYS + ["cl_label", "e_true", "cluster_size", "cl_score", "e_pred_raw", "e_pred"]


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_json(path, expected=None):
    raw = path.read_bytes()
    if expected is not None:
        require(sha256(raw).hexdigest() == expected, f"Digest differs: {path}")
    return json.loads(raw)


@dataclass(frozen=True)
class PredictionSet:
    root: Path
    entries: dict
    cuts: dict
    empty: frozenset
    selection: dict
    provenance: dict

    def frame(self, content, span):
        if content in self.empty:
            return pd.DataFrame(columns=COLUMNS, dtype=np.float64)
        entry = self.entries[content]
        path = (self.root / entry["path"]).resolve()
        require(path.is_relative_to(self.root.resolve()), f"Prediction leaves bundle: {path}")
        raw = accepted.checked(path, entry["sha256"])
        frame = pd.read_csv(BytesIO(raw), usecols=COLUMNS)[COLUMNS]
        keys = frame[KEYS].to_numpy()
        require(not frame.empty and not frame.duplicated(KEYS).any()
                and np.isfinite(frame).all(axis=None)
                and np.equal(keys, np.floor(keys)).all()
                and frame.Row.between(0, span - 1).all()
                and frame.ClusterID.ge(0).all() and frame.cl_istop.isin([0, 1]).all()
                and frame.cl_label.isin([0, 1]).all() and frame.cl_score.between(0, 1).all()
                and frame.cluster_size.gt(0).all()
                and np.equal(frame.cluster_size, np.floor(frame.cluster_size)).all()
                and frame.e_true.ge(0).all() and frame.e_pred_raw.ge(0).all()
                and np.array_equal(frame.e_pred, frame.e_pred_raw),
                f"Invalid raw prediction rows: {path}")
        return frame


def baseline_cuts(info, fold, seed, roles, scan_rn):
    root = BASE / f"train_4466016/extracted/task_{info['task']}/train"
    run = read_json(root / "complete.json", info["complete_sha256"])
    require(run["state"] == "complete" and run["evaluable"] and run["epochs"] == 30
            and run["model"] == MODELS[0] and (run["fold"], str(run["seed"])) == (fold, seed)
            and run["identity"]["split_sha256"] == SPLIT_SHA
            and run["identity"]["prepare_manifest_sha256"] == PREPARE_SHA
            and all(run["identity"][f"{role}_files"] == ids for role, ids in roles.items())
            and all(run["selection"][key] == info["selection"][key] for key in ("e_cls", "e_en")),
            f"Accepted GraphSAGE fit differs: {root}")
    name = "validation/threshold_scan_B.csv"
    scan = pd.read_csv(BytesIO(accepted.checked(root / name, run["files"][name])))
    require(np.allclose(scan.threshold, THRESHOLD_GRID, rtol=0, atol=1e-9),
            f"GraphSAGE threshold grid differs: {root}")
    total = scan.n_selected.to_numpy()
    signal = scan.n_selected_signal.to_numpy()
    cuts = {}
    for target in TARGETS if scan_rn else ("70",):
        valid = np.flatnonzero((total > 0) & (100 * signal >= int(target) * total))
        cuts[target] = float(THRESHOLD_GRID[valid[0]]) if valid.size else None
    require(cuts["70"] == run["cuts"]["B0"] == info["cuts"]["B0"],
            f"GraphSAGE 70% validation cut differs: {root}")
    return cuts


def inputs(scope, bundle_roots, scan_rn, odd_release, odd_release_digest):
    archive, digest = ARCHIVES[scope]
    require(sha(archive) == digest, f"Accepted file draws differ: {archive}")
    split = read_json(BASE / "split/split_manifest.json", SPLIT_SHA)
    groups = {f"{g['tag']}/{g['kept']}": g for g in split["groups"] if g["truth_bound"]}
    group_contents = {g["id"]: f"{g['tag']}/{g['kept']}" for g in split["groups"]}
    roles = {
        int(fold): {role: sorted(group_contents[g] for g in entry["partitions"][short])
                    for role, short in (("train", "train"), ("validation", "val"))}
        for fold, entry in split["folds"].items()
    }
    require(len(groups) == 568, "Registered truth-bound split")
    wanted = {cid for cid, g in groups.items() if g["part"] in PARTS[scope]}
    if scope == "even":
        records_path = PRIMARY / "result/summary.json"
        expected = "50563de5950c168a452c9445368460eb9b472b8841e634326fc7334f5947e1b3"
    else:
        records_path = ODD / "plan.json"
        expected = "ca5261eca36567c661f2ccea945da3f3fbd6bf89af6d9883aa3a5810269140b3"
    records = {row["content"]: row for row in read_json(records_path, expected)["files"]}
    with np.load(archive, allow_pickle=False) as saved:
        content = saved["content"]
        tags = np.asarray([groups[cid]["tag"] for cid in content])
        draws = saved["weights" if scope == "even" else "draws"]
        if scope == "even":
            # In the accepted benchmark the geometric counts repeat across methods and seeds.
            all_truth = saved["truth_bands"][..., 0, :]
            require((all_truth == all_truth[:, :1, :1]).all()
                    and all_truth[:, 0, 0, :].sum() == 897_563, "Even geometric truth")
            truth = all_truth[:, 0, 0, :][:, [1, 3, 4]]
        else:
            require(saved["arms"][0] == "mc" and np.array_equal(tags, saved["tags"]),
                    "Odd MC arm or potential order")
            truth = np.column_stack([saved[key][:, 0, 0] for key in ("soft", "hard", "overflow")])
            require(all((saved[key][..., 0] == saved[key][:, :1, 0]).all()
                        for key in ("soft", "hard", "overflow")), "Odd MC seed invariance")
        require(saved["seeds"].tolist() == list(SEEDS), "File archive seed order")
    parts = np.asarray([groups[cid]["part"] for cid in content])
    require(set(records) == set(content) == wanted and len(content) == len(wanted)
            and draws.shape == (400, len(content)) and np.issubdtype(draws.dtype, np.integer)
            and (draws >= 0).all(), "Registered content and file draws")
    for tag in TAGS.values():
        for part in PARTS[scope]:
            mask = (tags == tag) & (parts == part)
            require(mask.any() and (draws[:, mask].sum(axis=1) == mask.sum()).all(),
                    f"Bootstrap changes stratum size: {tag}/{part}")
    spans = np.array([records[cid]["processed_span"] for cid in content])
    require(spans.sum() == SPAN_TOTAL[scope], "Accepted event-span total differs")
    if scope == "odd":
        certified = np.array([records[cid]["certified_events"] for cid in content])
        require((certified >= spans).all() and np.count_nonzero(certified > spans) == 63
                and (certified - spans).sum() == 91, "Odd shortened spans")
        require(np.array_equal(np.column_stack([truth[:, 1] + truth[:, 2], truth[:, 0]]),
                               [records[cid]["open_truth_hi_lo"] for cid in content]),
                "Odd certified geometric counts")
    weights = np.vstack([np.ones(len(content), dtype=np.int64), draws])

    by_fold = {}
    predictions = {}
    for job, pin in BASELINE.items():
        root = BASE / f"evaluate_{job}"
        meta = read_json(root / "evaluation_summary.json", pin)
        fold = meta["fold"]
        test = {cid for cid, g in groups.items() if g["part"] in split["folds"][str(fold)]["test_parts"]}
        require(meta["state"] == "complete" and not meta["test_threshold_tuning"]
                and set(meta["test_files"]) == test
                and meta["identity"]["split_sha256"] == SPLIT_SHA
                and meta["identity"]["prepare_manifest_sha256"] == PREPARE_SHA,
                f"Accepted evaluation differs: {root}")
        require(fold not in by_fold, f"Duplicate baseline fold {fold}")
        by_fold[fold] = test & wanted
        for seed in SEEDS:
            info = meta["training"][seed]
            cuts = baseline_cuts(info, fold, seed, roles[fold], scan_rn)
            predictions[(MODELS[0], fold, seed)] = PredictionSet(
                root / "extracted/evaluation", meta["predictions"]["B0"][seed],
                cuts, frozenset(meta["empty_after_cuts"]), info["selection"],
                {"evaluation_sha256": pin, "fit_sha256": info["complete_sha256"]},
            )
    fold_for = {cid: fold for fold, ids in by_fold.items() for cid in ids}
    require(set(by_fold) == set(FOLDS) and set(fold_for) == wanted
            and sum(map(len, by_fold.values())) == len(wanted), "Cross-fit coverage")
    if scope == "odd":
        require(all(records[cid]["fold"] == fold_for[cid] for cid in wanted), "Odd TEST fold")
    source_plans = set()
    tasks = {key: str(task) for task, key in enumerate(CHALLENGER_TASKS)}
    for root in bundle_roots:
        raw = (root / "summary.json").read_bytes()
        meta = json.loads(raw)
        model, fold, seed = meta["model"], int(meta["fold"]), str(meta["seed"])
        key = model, fold, seed
        ids, empty = meta["test_files"], frozenset(meta["empty_after_cuts"])
        entries, selection, cuts = meta["predictions"], meta["selection"], meta["cuts"]
        require(meta["schema"] == "smash-architecture-predictions-v1" and meta["state"] == "complete"
                and model in MODELS[1:] and fold in by_fold and seed in SEEDS
                and meta["test_scope"] == scope and meta["split_sha256"] == SPLIT_SHA
                and meta["prepare_manifest_sha256"] == PREPARE_SHA
                and meta["prepare_summary_sha256"] == PREPARE_SUMMARY_SHA and meta["epochs"] == 30
                and all(meta[f"{role}_files"] == expected for role, expected in roles[fold].items())
                and len(ids) == len(set(ids)) and set(ids) == by_fold[fold]
                and set(entries) == set(ids) - empty and empty <= set(ids)
                and all(isinstance(selection[h], int) and 0 <= selection[h] < 30
                        for h in ("e_cls", "e_en"))
                and set(cuts) >= (set(TARGETS) if scan_rn else {"70"}),
                f"Invalid architecture prediction receipt: {root}")
        require(all(cut is None or np.any(np.isclose(cut, THRESHOLD_GRID, rtol=0, atol=1e-12))
                    for cut in cuts.values()), f"Off-grid score cut: {root}")
        require(key not in predictions, f"Duplicate evaluation: {root}")
        fit_digest = meta["fit_receipt_sha256"]
        require(isinstance(fit_digest, str) and len(fit_digest) == 64,
                f"Missing complete fit receipt hash: {root}")
        source_plans.add((meta["source_sums_sha256"], meta["plan_sha256"]))
        provenance = {"summary_sha256": sha256(raw).hexdigest(), "fit_sha256": fit_digest,
                      "source_sums_sha256": meta["source_sums_sha256"], "plan_sha256": meta["plan_sha256"]}
        if scope == "odd":
            require(meta.get("odd_release_sha256") == odd_release_digest,
                    f"Odd prediction release digest differs: {root}")
            require(meta["plan_sha256"] == odd_release["plan_sha256"]
                    and fit_digest == odd_release["fit_receipt_sha256"][tasks[key]],
                    f"Odd release does not cover this plan and fit: {root}")
            provenance["odd_release_sha256"] = odd_release_digest
        predictions[key] = PredictionSet(root, entries, cuts, empty, selection, provenance)
    require(set(predictions) == {(model, fold, seed) for model in MODELS
                                for fold in by_fold for seed in SEEDS},
            "Expected one complete export per architecture, TEST fold and seed")
    require(len(source_plans) == 1, "Prediction exports span different source or plan pins")
    return content, tags, weights, records, fold_for, truth, predictions


def align(frame, keys, target, context):
    index = pd.MultiIndex.from_frame(frame[KEYS])
    order = index.get_indexer(keys)
    require(len(frame) == len(keys) and (order >= 0).all(), f"Candidate identity: {context}")
    ordered = frame.iloc[order]
    if target is not None:
        require(np.array_equal(ordered[["cl_label", "e_true"]].to_numpy(np.float32), target[0])
                and np.array_equal(ordered.cluster_size, target[1]), f"Native targets or size: {context}")
    return ordered


def geometry(record, reference, manifest):
    cid, span = record["content"], record["processed_span"]
    root = PRIMARY / "result" / cid
    raw = {}
    for name in ("events.csv.gz", "neutrons.npz"):
        pin = record["files"][name]
        require(pin == manifest[f"{cid}/{name}"], f"Primary archive pin: {cid}/{name}")
        raw[name] = accepted.checked(root / name, pin)
    events = pd.read_csv(BytesIO(raw["events.csv.gz"]), compression="gzip")
    events = events.loc[events.Row < span]
    require(np.array_equal(events.Row, np.arange(span)), f"{cid}: complete event inventory")
    with np.load(BytesIO(raw["neutrons.npz"]), allow_pickle=False) as z:
        row, energy = z["Row"], z["Ekin"]
        inside = z["in_rectangle"] & (row < span)
        objects = pd.MultiIndex.from_arrays([row, z["Id"]])
        key_array = np.column_stack([z["cluster_row"], z["cluster_local_id"], z["cluster_side"]])
        primary, candidate = accepted.edge_indices(z["observed_edges"], objects, key_array)
    require(objects.is_unique and np.isfinite(energy).all() and (energy >= 0).all()
            and np.array_equal(np.bincount(row[inside], minlength=span), events.n_all)
            and accepted.digest(key_array, "<i8") == reference["keys_sha256"]
            and len(key_array) == record["clusters"], f"{cid}: primary/candidate binding")
    keys = pd.MultiIndex.from_arrays(key_array.T)
    require(keys.is_unique, f"{cid}: duplicate candidate keys")
    unique = isolation(primary, candidate, len(objects), len(keys))
    return keys, key_array[:, 0], row, energy, inside, primary, candidate, unique


def case_predictions(content, record, fold, predictions, keys=None, native=None):
    labels, target, scores, energies = None, None, [], []
    for seed in SEEDS:
        score_by_model, energy_by_model = [], []
        for model in MODELS:
            source = predictions[(model, fold, seed)]
            frame = source.frame(content, record["processed_span"])
            if keys is None:
                keys = pd.MultiIndex.from_frame(frame[KEYS])
            frame = align(frame, keys, target, f"{content}/{model}/{seed}")
            if target is None:
                target = (frame[["cl_label", "e_true"]].to_numpy(np.float32),
                          frame.cluster_size.to_numpy(np.int64))
                labels = frame.cl_label.to_numpy(bool)
                if native is not None:
                    require(accepted.digest(target[0], "<f4") == native["targets_sha256"]
                            and accepted.digest(target[1], "<i8") == native["sizes_sha256"],
                            f"Accepted native targets differ: {content}")
            score_by_model.append(frame.cl_score.to_numpy(np.float64))
            energy_by_model.append(frame.e_pred_raw.to_numpy(np.float64))
        scores.append(score_by_model)
        energies.append(energy_by_model)
    require((target[0][labels, 1] > 0).all(), f"Nonpositive native target: {content}")
    return labels, target[0][:, 1], np.asarray(scores), np.asarray(energies)


def cuts_for(fold, predictions, targets):
    return np.asarray([[predictions[(model, fold, seed)].cuts.get(targets[model], np.nan)
                        for model in MODELS] for seed in SEEDS], dtype=np.float64)


def candidate_bands(energy, keep, labels):
    """Candidate counts for strict floor, bounded hard band and the open tail."""
    bands = ((energy > .3) & (energy < 1), (energy >= 2) & (energy < 4), energy >= 4,
             np.ones_like(keep), labels, energy == .3)
    return np.stack([np.count_nonzero(keep & mask, axis=-1) for mask in bands], axis=-1)


def count_ratio(numerator, denominator, scale=1):
    return np.divide(scale * numerator, denominator, where=denominator > 0,
                     out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan))


def interval(values):
    valid = np.isfinite(values)
    result = {"value": float(values[0]) if valid[0] else None,
              "invalid_draws": int(np.count_nonzero(~valid[1:]))}
    if not valid.all():
        return result | dict(file_sd=None, q16=None, q84=None, half_width_68=None)
    q16, q84 = np.quantile(values[1:], (.16, .84))
    return result | {"file_sd": float(values[1:].std(ddof=1)), "q16": float(q16),
                     "q84": float(q84), "half_width_68": float((q84 - q16) / 2)}


def clean_values(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def paired(values):
    """Average repeated models inside each draw and retain training-seed centers."""
    means = values.mean(axis=1)
    return {
        "by_model": {model: interval(means[:, i]) | {"by_seed": clean_values(values[0, :, i])}
                     for i, model in enumerate(MODELS)},
        "minus_graphsage": {model: interval(means[:, i] - means[:, 0])
                            | {"by_seed": clean_values(values[0, :, i] - values[0, :, 0])}
                            for i, model in enumerate(MODELS[1:], 1)},
    }


def band_sums(weights, tags, counts):
    return {
        setting: np.tensordot(weights[:, tags == tag], counts[tags == tag], axes=1)
        for setting, tag in TAGS.items()
    }


def seed_test(differences):
    n = len(SEEDS)
    result = {"by_seed": clean_values(differences), "mean": None, "seed_se": None,
              "paired_t": None, "df": n - 1, "two_sided_p": None, "holm_p": None,
              "unadjusted_critical_5pct": float(student_t.ppf(.975, n - 1))}
    if np.isfinite(differences).all():
        mean, se = differences.mean(), differences.std(ddof=1) / np.sqrt(n)
        result.update(mean=float(mean), seed_se=float(se))
        if se > 0:
            t = mean / se
            result.update(paired_t=float(t), two_sided_p=float(2 * student_t.sf(abs(t), n - 1)))
    return result


def adjust_holm(tests):
    if all(result["two_sided_p"] is not None for result in tests.values()):
        ordered = sorted(tests.values(), key=lambda result: result["two_sided_p"])
        largest = 0
        for remaining, result in zip(range(len(ordered), 0, -1), ordered, strict=True):
            largest = max(largest, min(1, remaining * result["two_sided_p"]))
            result["holm_p"] = largest


def hardness(weights, tags, truth, counts):
    # MC is deterministic across seeds and uses the same file draws.
    sums = band_sums(weights, tags, counts)
    mc_sums = band_sums(weights, tags, truth)
    output = {}
    for window, include_tail in (("open_upper", True), ("within_reach", False)):
        ratios, mc = {}, {}
        settings = {}
        for setting in TAGS:
            x, ref = sums[setting], mc_sums[setting]
            hard = x[..., 1] + (x[..., 2] if include_tail else 0)
            mc_hard = ref[:, 1] + (ref[:, 2] if include_tail else 0)
            ratios[setting] = count_ratio(hard, x[..., 0])
            mc[setting] = count_ratio(mc_hard, ref[:, 0])
            settings[setting] = {
                "counts_by_seed_model": x[0].tolist(), "mc_counts": ref[0].tolist(),
                "reconstructed_Rn": paired(ratios[setting]), "geometric_mc_Rn": interval(mc[setting]),
                "draws_with_zero_soft_by_seed_model": (x[1:, ..., 0] == 0).sum(axis=0).tolist(),
            }
        contrasts = {}
        for a, b in PAIRS:
            values = count_ratio(ratios[a], ratios[b], 100) - 100
            target = count_ratio(mc[a], mc[b], 100) - 100
            means = values.mean(axis=1)
            delta = means - target[:, None]
            seed_delta = values[0] - target[0]
            contrasts[f"{a}/{b}"] = {
                "geometric_mc_contrast_percent": interval(target),
                "reconstructed_contrast_percent": paired(values),
                "paired_shift_from_mc_pp": {
                    model: interval(delta[:, i]) | {"by_seed": clean_values(seed_delta[:, i])}
                    for i, model in enumerate(MODELS)
                },
                "width_change_from_graphsage_pp": {
                    model: width_delta(means[:, i], means[:, 0])
                    for i, model in enumerate(MODELS[1:], 1)
                },
                "paired_change_in_absolute_mc_shift_pp": {
                    model: interval(np.abs(delta[:, i]) - np.abs(delta[:, 0]))
                    for i, model in enumerate(MODELS[1:], 1)
                },
            }
            # Absolute differences after averaging seeds are not the t-test estimand.
            if include_tail and (a, b) == ("90", "0"):
                tests = {model: seed_test(np.abs(seed_delta[:, i]) - np.abs(seed_delta[:, 0]))
                         for i, model in enumerate(MODELS[1:], 1)}
                adjust_holm(tests)
                contrasts[f"{a}/{b}"]["paired_seed_tests_for_absolute_mc_shift_pp"] = tests
        output[window] = {"settings": settings, "contrasts": contrasts}
    return output


def width_delta(a, b):
    left, right = interval(a), interval(b)
    return {key: None if left[key] is None or right[key] is None else left[key] - right[key]
            for key in ("file_sd", "half_width_68")}


def classification(tags, rows, weights, complete_operating_point):
    result = {}
    for name, mask in {"pooled": np.ones(len(tags), bool),
                       **{key: tags == tag for key, tag in TAGS.items()}}.items():
        scope_rows = [row for row, include in zip(rows, mask, strict=True) if include]
        labels = np.concatenate([r[0] for r in scope_rows])
        score = np.concatenate([r[1] for r in scope_rows], axis=-1)
        require(labels.any() and (~labels).any(), f"Classifier labels in {name}")
        auc, ap = np.empty((len(SEEDS), len(MODELS))), np.empty((len(SEEDS), len(MODELS)))
        for si in range(len(SEEDS)):
            for mi in range(len(MODELS)):
                auc[si, mi] = roc_auc_score(labels, score[si, mi])
                ap[si, mi] = average_precision_score(labels, score[si, mi])
        def seed_metrics(values):
            return {model: {"value": float(values[:, i].mean()),
                            "seed_sd": float(values[:, i].std(ddof=1)),
                            "by_seed": values[:, i].tolist()}
                    for i, model in enumerate(MODELS)}
        result[name] = {
            "native_candidates": int(len(labels)), "native_positives": int(labels.sum()),
            "ROC_AUC": seed_metrics(auc), "AP": seed_metrics(ap),
        }
        if complete_operating_point:
            passed = np.stack([np.stack([r[2].sum(-1), (r[2] & r[0]).sum(-1)], axis=-1)
                               for r in scope_rows])
            total_positive = np.asarray([r[0].sum() for r in scope_rows])
            totals = np.tensordot(weights[:, mask], passed, axes=1)
            result[name].update(
                purity=paired(count_ratio(totals[..., 1], totals[..., 0])),
                recall=paired(count_ratio(totals[..., 1], (weights[:, mask] @ total_positive)[:, None, None])),
            )
    return result


def matching_rates(counts):
    c = dict(zip(COUNT_KEYS, np.moveaxis(counts, -1, 0), strict=True))
    metrics = {
        "direct_coverage_percent": ("truth_with_edge", "truth"),
        "direct_miss_percent": ("truth_missed", "truth"),
        "split_percent_of_truth": ("truth_split", "truth"),
        "merge_percent_of_candidates": ("candidate_merge", "candidates"),
        "mixed_merge_percent_of_candidates": ("candidate_mixed_merge", "candidates"),
        "no_direct_link_percent_of_candidates": ("candidate_no_edge", "candidates"),
        "exclusive_truth_deficit_percent": ("exclusive_truth_deficit", "truth"),
        "exclusive_candidate_excess_percent": ("exclusive_candidate_excess", "candidates"),
        "zero_candidate_events_percent": ("zero_candidates", "events"),
    }
    return {"counts": {key: value[0].tolist() for key, value in c.items()},
            "rates": {name: paired(count_ratio(c[n], c[d], 100))
                      for name, (n, d) in metrics.items()}}


def matching_file(geom, span, energies, score_keep, range_keep):
    _, rows, truth_row, truth_energy, inside, primary, candidate, unique = geom
    dims = energies.shape[:2]
    counts = np.zeros((*dims, len(STAGES), len(TRUTH_RANGES), len(COUNT_KEYS)), dtype=np.int64)
    multiplicity = np.zeros((*dims, len(STAGES), 4, 4), dtype=np.int64)
    for stage in range(len(STAGES)):
        for si, mi in np.ndindex(dims) if stage else [(0, 0)]:
            e = energies[si, mi]
            keep = (np.ones(len(e), dtype=bool), score_keep[si, mi], range_keep[si, mi])[stage]
            for j, (low, high) in enumerate(TRUTH_RANGES.values()):
                truth_mask = inside & (truth_energy >= low) & (truth_energy < high)
                values, mult, _ = metric(rows, truth_row, truth_mask, primary, candidate, keep, e, span)
                counts[si, mi, stage, j] = values
                if j == 0:
                    multiplicity[si, mi, stage] = mult
        if stage == 0:
            # Baseline selection has identical graph/matching, but raw energy counts vary.
            counts[:, :, 0] = counts[0, 0, 0]
            multiplicity[:, :, 0] = multiplicity[0, 0, 0]
            for key, lo, hi in (("soft_candidates", .3, 1), ("hard_candidates", 2, 4)):
                k = COUNT_KEYS.index(key)
                counts[:, :, 0, :, k] = ((energies >= lo) & (energies < hi)).sum(axis=-1)[..., None]
    require(np.array_equal(counts[:, :, 0, :, COUNT_KEYS.index("truth")],
                           counts[:, :, 2, :, COUNT_KEYS.index("truth")]), "Truth span by stage")
    valid = inside & (unique >= 0)
    return counts, multiplicity, (truth_energy[valid], energies[..., unique[valid]], range_keep[..., unique[valid]])


def matching_summary(tags, counts, multiplicity, weights):
    out = {}
    for setting, mask in {"pooled": np.ones(len(tags), bool),
                          **{key: tags == tag for key, tag in TAGS.items()}}.items():
        c = np.tensordot(weights[:, mask], counts[mask], axes=1)
        out[setting] = {
            "by_stage_and_truth_range": {
                stage: {band: matching_rates(c[..., i, j, :])
                        for j, band in enumerate(TRUTH_RANGES)}
                for i, stage in enumerate(STAGES)
            },
            "multiplicity_by_seed_model_stage": multiplicity[mask].sum(axis=0).tolist(),
        }
    return out


def weighted_quantiles(values, file_index, weights):
    """Quantiles with linear interpolation in each whole-file weighted sample."""
    out = np.full((len(weights), 3), np.nan)
    if not len(values):
        return out
    order = values.argsort()
    sorted_values = values[order]
    sorted_files = file_index[order]
    for i, draw in enumerate(weights):
        cumulative = np.cumsum(draw[sorted_files], dtype=np.int64)
        if cumulative[-1]:
            rank = (cumulative[-1] - 1) * np.asarray([.16, .50, .84])
            lower = sorted_values[np.searchsorted(cumulative, np.floor(rank), side="right")]
            upper = sorted_values[np.searchsorted(cumulative, np.ceil(rank), side="right")]
            out[i] = lower + (rank - np.floor(rank)) * (upper - lower)
    return out


def matched_energy(weights, samples):
    file_index = np.repeat(np.arange(len(samples)), [len(s[0]) for s in samples])
    truth, energy, keep = (np.concatenate(items, axis=-1) for items in zip(*samples, strict=True))
    out = {}
    for cohort in ("preselection", "common_selected", "own_selected"):
        out[cohort] = {}
        bands = {"primary": (.3, 4), **{
            f"{lo:g}_to_{hi:g}": (lo, hi) for lo, hi in zip(ENERGY_EDGES[:-1], ENERGY_EDGES[1:], strict=True)
        }, "at_least_6p5": (6.5, np.inf)}
        for band, (lo, hi) in bands.items():
            base = (truth >= lo) & (truth < hi)
            shape = (len(weights), len(SEEDS), len(MODELS))
            n = np.zeros(shape, dtype=np.int64)
            mae, median, width = (np.full(shape, np.nan) for _ in range(3))
            for si, mi in np.ndindex(shape[1:]):
                mask = base
                if cohort == "common_selected":
                    mask = mask & keep[si].all(axis=0)
                elif cohort == "own_selected":
                    mask = mask & keep[si, mi]
                errors = energy[si, mi, mask] - truth[mask]
                files = file_index[mask]
                n[:, si, mi] = weights @ np.bincount(files, minlength=weights.shape[1])
                absolute_sum = np.bincount(files, weights=np.abs(errors), minlength=weights.shape[1])
                mae[:, si, mi] = count_ratio(weights @ absolute_sum, n[:, si, mi], 1000)
                # Inherited first-hit response minimum for a central quantile width.
                if len(errors) >= 25:
                    q16, q50, q84 = weighted_quantiles(errors / truth[mask], files, weights).T
                    median[:, si, mi] = 100 * q50
                    width[:, si, mi] = 50 * (q84 - q16)
            out[cohort][band] = {"matched_counts_by_seed_model": n[0].tolist()}
            for name, values in (("MAE_MeV", mae), ("median_relative_error_percent", median),
                                 ("relative_error_half_width_68_percent", width)):
                summary = paired(values)
                if cohort == "own_selected":
                    # Own-sample rates describe different primary populations.
                    del summary["minus_graphsage"]
                out[cohort][band][name] = summary
    return out


def fit_bindings(predictions):
    return {f"{model}/{fold}/{seed}": {
                "fit_sha256": source.provenance["fit_sha256"], "selection": source.selection,
                "primary_cut": source.cuts["70"],
            }
            for (model, fold, seed), source in sorted(predictions.items())}


def freeze_policy(weights, tags, policy_counts, predictions):
    # One ceiling for the new arms. The calibrated historical 2.46 pp was
    # registered for a different energy treatment and is not copied.
    counts = band_sums(weights, tags, policy_counts)
    valid = (policy_counts[..., 0] >= 0).all(axis=(0, 1))
    ratios = {}
    for setting, x in counts.items():
        lo, hi = x[..., 0], x[..., 1] + x[..., 2]
        valid &= ((lo > 0) & (hi > 0)).all(axis=(0, 1))
        ratios[setting] = count_ratio(hi, lo)
    contrast = (count_ratio(ratios["90"], ratios["0"], 100) - 100).mean(axis=1)
    ti_ref = TARGETS.index("70")
    ref = interval(contrast[:, 0, ti_ref])
    ceiling = read_json(REFERENCE_CEILING, REFERENCE_CEILING_SHA)
    require(ceiling["reference_graphsage_raw70"] == ref
            and ceiling["counts_by_setting_seed"] == {s: x[0, :, 0, ti_ref, :3].tolist()
                                                       for s, x in counts.items()}
            and ceiling["ceiling_file_sd_pp"] == 2 * ref["file_sd"],
            "GraphSAGE raw70 replay differs from the registered width ceiling")
    cap = ceiling["ceiling_file_sd_pp"]
    targets, scan = {}, {}
    for mi, model in enumerate(MODELS):
        rows = []
        for ti, target in enumerate(TARGETS):
            stats = interval(contrast[:, mi, ti]) if valid[mi, ti] else None
            eligible = stats is not None and cap is not None and stats["file_sd"] <= cap
            rows.append({"target": target, "valid_positive_bands": bool(valid[mi, ti]),
                         "eligible": eligible, "contrast_percent": stats,
                         "counts_by_setting_seed": {s: x[0, :, mi, ti].tolist() for s, x in counts.items()}})
        selected = max((r for r in rows if r["eligible"]),
                       key=lambda r: (r["contrast_percent"]["value"], -int(r["target"])), default=None)
        targets[model] = selected["target"] if selected else None
        scan[model] = rows
    return {
        "schema": "smash-architecture-rn-policy-v1", "scope": "even only", "energy": "raw",
        "split_sha256": SPLIT_SHA, "prepare_manifest_sha256": PREPARE_SHA,
        "even_counts_sha256": ARCHIVES["even"][1],
        "ceiling_file_sd_pp": cap, "reference_graphsage_raw70": ref,
        "reference_ceiling_sha256": REFERENCE_CEILING_SHA,
        "selected_targets": targets, "scan": scan, "fits": fit_bindings(predictions),
        "cuts": {f"{m}/{f}/{s}": p.cuts.get(targets[m]) for (m, f, s), p in predictions.items()},
    }


def frozen_targets(frozen, predictions):
    require(frozen["schema"] == "smash-architecture-rn-policy-v1" and frozen["scope"] == "even only"
            and frozen["energy"] == "raw" and frozen["split_sha256"] == SPLIT_SHA
            and frozen["prepare_manifest_sha256"] == PREPARE_SHA
            and frozen["reference_ceiling_sha256"] == REFERENCE_CEILING_SHA
            and frozen["even_counts_sha256"] == ARCHIVES["even"][1]
            and frozen["fits"] == fit_bindings(predictions), "Frozen even policy or fit identity differs")
    targets = frozen["selected_targets"]
    require(set(targets) == set(MODELS) and all(t is None or t in TARGETS for t in targets.values()),
            "Frozen target grid differs")
    if any(t is None for t in targets.values()):
        return None
    for (m, f, s), source in predictions.items():
        cut = source.cuts.get(targets[m])
        require(cut is not None and cut == frozen["cuts"][f"{m}/{f}/{s}"],
                f"Frozen cut for {m}/{f}/{s} differs")
    return targets


def read_odd_release(args):
    if args.scope == "even":
        require(args.odd_release is None and args.odd_release_sha256 is None and args.frozen_policy is None,
                "Odd release and frozen policy apply to odd reduction")
        return None, None

    require(args.odd_release is not None and args.odd_release_sha256,
            "Odd reduction requires --odd-release and --odd-release-sha256")
    release = read_json(args.odd_release, args.odd_release_sha256)
    require(isinstance(release, dict) and release.get("schema") == ODD_RELEASE_SCHEMA
            and "frozen_policy_sha256" in release
            and isinstance(release.get("plan_sha256"), str)
            and isinstance(release.get("fit_receipt_sha256"), dict)
            and set(release["fit_receipt_sha256"]) == {str(task) for task in range(len(CHALLENGER_TASKS))},
            "Invalid odd scoring release")
    policy_digest = release["frozen_policy_sha256"]
    mode = release.get("policy_mode")
    if mode == RAW70_ONLY:
        require(policy_digest is None and args.frozen_policy is None,
                "Raw70-only release requires a null policy digest and no --frozen-policy")
        return release, None

    require(mode == WITH_FROZEN_POLICY and isinstance(policy_digest, str) and args.frozen_policy is not None,
            "Frozen-even-policy release requires a policy digest and --frozen-policy")
    policy = read_json(args.frozen_policy, policy_digest)
    require(isinstance(policy, dict), "Invalid frozen even policy")
    return release, policy


def run(args):
    require(not args.output.exists(), "Output already exists")
    require(not args.scan_rn or args.scope == "even", "Policy selection applies to even reduction")
    release, frozen_policy = read_odd_release(args)
    content, tags, weights, records, fold_for, truth, predictions = inputs(
        args.scope, args.prediction_bundle, args.scan_rn or frozen_policy is not None,
        release, args.odd_release_sha256,
    )
    frozen = frozen_targets(frozen_policy, predictions) if frozen_policy is not None else None
    identities = {
        f"{m}/{f}/{s}": source.provenance | {"selection": source.selection, "cuts": source.cuts}
        for (m, f, s), source in sorted(predictions.items())
    }
    result = {
        "schema": "smash-architecture-reduction-v1", "scope": args.scope, "energy": "raw",
        "files": len(content), "events": SPAN_TOTAL[args.scope],
        "models": MODELS, "seeds": SEEDS, "bootstrap_draws": len(weights) - 1,
        "bootstrap": "whole files, stratified by potential and part, shared across models and seeds",
        "retrospective": True,
        "split_sha256": SPLIT_SHA, "prepare_manifest_sha256": PREPARE_SHA,
        "prepare_summary_sha256": PREPARE_SUMMARY_SHA,
        "accepted_counts_sha256": ARCHIVES[args.scope][1], "reducer_sha256": sha(Path(__file__)),
        "identities": identities,
    }
    missing = [name for name, identity in identities.items() if identity["cuts"]["70"] is None]

    even = args.scope == "even"
    if even:
        native = {r["content"]: r for r in read_json(
            REPORTS / "native_features_20261106/plan.json",
            "f0c33dab8b2e90ce4563323de19af75ac79650ac998c761a752047e4e38a639a",
        )["files"]}
        manifest = {r["member"]: r["sha256"] for r in read_json(
            PRIMARY / "export_4491592/manifest.json",
            "e32d7ea8f4629354864d8a63e76a674c8dfaa4d63d2f941889399c885063ccba",
        )["files"]}
        require(set(native) == set(content), "Even native candidate inventory")
    if release is not None:
        result.update(odd_release_sha256=args.odd_release_sha256, policy_mode=release["policy_mode"],
                      frozen_policy_sha256=release["frozen_policy_sha256"])
        if frozen_policy is not None and frozen is None:
            result["hardness_frozen_even_policy"] = {"state": "operating_point_unavailable"}
    primary_targets = {model: "70" for model in MODELS}
    rows, match, mult, samples, counts, alternate, native_error = [], [], [], [], [], [], []
    for i, cid in enumerate(content):
        record = records[cid]
        fold = fold_for[cid]
        geom = geometry(record, native[cid], manifest) if even else None
        labels, true_target, score, energy = case_predictions(
            cid, record, fold, predictions, keys=geom[0] if even else None,
            native=native[cid] if even else None,
        )
        if missing:
            rows.append((labels, score, None))
            continue
        keep = score > cuts_for(fold, predictions, primary_targets)[..., None]
        counts.append(candidate_bands(energy, keep, labels))
        rows.append((labels, score, keep))
        if even:
            range_keep = keep & (energy > .3) & (energy < 4)
            m, x, sample = matching_file(geom, record["processed_span"], energy, keep, range_keep)
            match.append(m)
            mult.append(x)
            samples.append(sample)
            native_error.append(np.abs(energy[..., labels] - true_target[labels]).sum(axis=-1))
            # Keep the accepted [0.3,1), [2,4), and >=4 geometric MC bands exact.
            mc_energy, inside = geom[3:5]
            require(np.array_equal(candidate_bands(mc_energy, inside, inside)[1:3], truth[i, 1:])
                    and ((mc_energy >= .3) & (mc_energy < 1) & inside).sum() == truth[i, 0],
                    f"{cid}: geometric count reconciliation")
        if args.scan_rn:
            scan = []
            for target in TARGETS:
                cuts = cuts_for(fold, predictions, dict.fromkeys(MODELS, target))
                band = candidate_bands(energy, score > cuts[..., None], labels)
                scan.append(np.where(np.isfinite(cuts)[..., None], band, -1))
            alternate.append(np.stack(scan, axis=-2))
        elif frozen:
            alternate.append(candidate_bands(energy, score > cuts_for(fold, predictions, frozen)[..., None], labels))
        if (i + 1) % 25 == 0:
            print(f"Reduced {i + 1}/{len(content)} {args.scope} files", flush=True)

    result["classification"] = classification(tags, rows, weights, not missing)
    if missing:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "summary.json", result | {
            "state": "operating_point_unavailable", "unreachable_70_percent_purity": missing,
        })
        return

    counts = np.stack(counts)
    arrays = {"content": content, "tags": tags, "draws": weights[1:], "truth": truth, "candidate_counts": counts,
              "candidate_count_keys": ("soft", "hard", "overflow", "total", "native_positive", "at_soft_floor"),
              "seeds": SEEDS, "models": MODELS}
    result["state"] = "complete"
    if even:
        match, mult = np.stack(match), np.stack(mult)
        require(match[:, 0, 0, 0, 0, COUNT_KEYS.index("truth")].sum() == 897_563,
                "Even geometric primary population")
        arrays.update(matching=match, multiplicity=mult, matching_keys=COUNT_KEYS,
                      truth_ranges=tuple(TRUTH_RANGES), stages=STAGES)
        # Retain the already matched sample for exact derived-data energy replay.
        # File indices follow content; no event or primary-track identifier is needed.
        arrays.update(
            isolated_file_index=np.repeat(np.arange(len(samples)), [len(s[0]) for s in samples]),
            isolated_truth_energy=np.concatenate([s[0] for s in samples]),
            isolated_raw_energy=np.concatenate([s[1] for s in samples], axis=-1),
            isolated_range_keep=np.concatenate([s[2] for s in samples], axis=-1),
        )
        result["energy_replay_sample"] = {
            "file_axis": "indices into content, in its saved order",
            "prediction_axes": "seed, model, isolated neutron",
            "population": "all common isolated direct primary links before classifier or reconstructed-energy selection",
            "isolated_range_keep": "raw70 classifier cut AND 0.3 < raw reconstructed energy < 4 GeV; stored mask, not a preselection of the saved energy arrays",
            "truth": "initial MC kinetic energy of the common isolated direct primary neutron",
            "preserved_dtypes_and_order": True,
        }
        result["direct_association"] = matching_summary(tags, match, mult, weights)
        result["matched_energy"] = matched_energy(weights, samples)
        all_positive = np.asarray([r[0].sum() for r in rows])
        result["unselected_native_positive_MAE_MeV"] = paired(count_ratio(
            np.tensordot(weights, native_error, axes=1), (weights @ all_positive)[:, None, None], 1000,
        ))
    else:
        result["hardness_raw70"] = hardness(weights, tags, truth, counts)
    if alternate:
        arrays["alternative_candidate_counts"] = np.stack(alternate)
        if args.scan_rn:
            policy = freeze_policy(weights, tags, arrays["alternative_candidate_counts"], predictions)
        else:
            result["hardness_frozen_even_policy"] = hardness(
                weights, tags, truth, arrays["alternative_candidate_counts"],
            )
            # Registered seed tests apply to raw70 only.
            del result["hardness_frozen_even_policy"]["open_upper"]["contrasts"]["90/0"][
                "paired_seed_tests_for_absolute_mc_shift_pp"
            ]
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "file_counts.npz", **arrays)
    result["file_counts_sha256"] = sha(args.output / "file_counts.npz")
    if args.scan_rn:
        write_json(args.output / "frozen_policy.json", policy)
        result["frozen_policy_sha256"] = sha(args.output / "frozen_policy.json")
    write_json(args.output / "summary.json", result)


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=PARTS, required=True)
    parser.add_argument("--prediction-bundle", type=Path, action="append", required=True,
                        help="Extracted accepted prediction bundle root (one per new model/fold/seed)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-rn", action="store_true", help="Freeze secondary policy on even files")
    parser.add_argument("--odd-release", type=Path, help="Pinned release receipt for odd predictions")
    parser.add_argument("--odd-release-sha256", help="Published digest of the odd release receipt")
    parser.add_argument("--frozen-policy", type=Path, help="Even-only policy bound by the odd release")
    run(parser.parse_args())
