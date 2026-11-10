"""Choose an R_n score-cut policy on even parts, then evaluate it once on odd parts."""

from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "results/smash_train_20261028"
SEEDS = ["42", "123", "456"]
TAGS = ["zeroSpot", "defaultSpot", "bigSpot"]
TARGETS = np.arange(60, 100, 5)  # Inherited 60–95% policy grid.
REF = int(np.flatnonzero(TARGETS == 70)[0])
DRAWS = 400  # Accepted whole-file bootstrap size.
SPLIT = "848ebf492b254b75e09297b95f89d7c2773f7877afe5e629520d520904a59886"


def read(path, expected=None):
    raw = path.read_bytes()
    if expected is not None:
        assert hashlib.sha256(raw).hexdigest() == expected, path
    return raw


def load(path, expected=None):
    return json.loads(read(path, expected))


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inputs():
    split = load(BASE / "split/split_manifest.json", SPLIT)
    groups = {
        g["tag"] + "/" + g["kept"]: g for g in split["groups"] if g["truth_bound"]
    }
    registry, cuts = {}, {}
    for job in (4466339, 4467086, 4467091):
        base = BASE / f"evaluate_{job}"
        meta = load(base / "evaluation_summary.json")
        fold = meta["fold"]
        assert meta["state"] == "complete" and not meta["test_threshold_tuning"]
        assert meta["identity"]["split_sha256"] == SPLIT
        parts = split["folds"][str(fold)]["test_parts"]
        assert set(meta["test_files"]) == {
            cid for cid, g in groups.items() if g["part"] in parts
        }
        for cid in set(meta["test_files"]) & groups.keys():
            assert cid not in registry
            registry[cid] = (base / "extracted/evaluation", meta)
        for seed in SEEDS:
            info = meta["training"][seed]
            root = BASE / f"train_4466016/extracted/task_{info['task']}/train"
            run = load(root / "complete.json", info["complete_sha256"])
            assert (run["fold"], str(run["seed"])) == (fold, seed)
            name = "validation/threshold_scan_B.csv"
            scan = pd.read_csv(BytesIO(read(root / name, run["files"][name])))
            grid = np.linspace(0.001, 0.999, 401)
            assert np.allclose(scan.threshold, grid, rtol=0, atol=1e-9)
            valid = scan.n_selected.to_numpy() > 0
            signal = scan.n_selected_signal.to_numpy()
            total = scan.n_selected.to_numpy()
            cut = np.array(
                [
                    grid[np.flatnonzero(valid & (100 * signal >= p * total))[0]]
                    for p in TARGETS
                ]
            )
            assert cut[REF] == run["cuts"]["B1"] == info["cuts"]["B1"]
            cuts[(fold, seed)] = cut
    assert set(registry) == set(groups)
    return groups, registry, cuts


def reduce(parts, policies, groups, registry, cuts):
    ledger = load(BASE / "heldout_rn/per_file_counts.json")["primary"]["B1"]
    reference = {r["content"]: r for records in ledger.values() for r in records}
    ids = sorted(k for k, g in groups.items() if g["part"] in parts)
    assert len(ids) == (286 if parts == (0, 2, 4) else 282)
    assert REF in policies
    data = np.zeros((len(ids), len(SEEDS), len(policies), 4), dtype=int)
    truth = []
    for i, cid in enumerate(ids):
        base, meta = registry[cid]
        fold = meta["fold"]
        ref = reference[cid]
        assert ref["fold"] == fold
        truth.append(ref["truth_hi_lo"][1])
        for s, seed in enumerate(SEEDS):
            entry = meta["predictions"]["B1"][seed][cid]
            frame = pd.read_csv(
                BytesIO(read(base / entry["path"], entry["sha256"])),
                usecols=["cl_score", "e_pred", "cl_label"],
            )
            score, energy, label = (
                frame[k].to_numpy() for k in ["cl_score", "e_pred", "cl_label"]
            )
            assert np.isfinite(score).all() and np.isfinite(energy).all()
            assert np.isin(label, [0, 1]).all() and ((0 <= score) & (score <= 1)).all()
            cut = cuts[(fold, seed)][policies]
            # Right insertion excludes exact score ties, as in score > cut.
            for j, mask in enumerate(
                [
                    energy >= 2,
                    (energy >= 0.3) & (energy < 1),
                    np.ones(len(score), bool),
                    label == 1,
                ]
            ):
                v = np.sort(score[mask])
                data[i, s, :, j] = len(v) - np.searchsorted(v, cut, side="right")
            assert np.array_equal(
                data[i, s, policies.index(REF), :2], ref["clusters_hi_lo"][s][3]
            ), (cid, seed)
    return ids, data, np.array(truth)


def summarize(ids, data, truth, groups):
    rng = np.random.default_rng(20261106)  # Inherited response-diagnostics seed.
    weights = np.zeros((DRAWS, len(ids)), int)
    tags = np.array([groups[k]["tag"] for k in ids])
    parts = np.array([groups[k]["part"] for k in ids])
    for tag in TAGS:
        for part in np.unique(parts):
            ix = np.flatnonzero((tags == tag) & (parts == part))
            assert len(ix)
            weights[:, ix] = rng.multinomial(
                len(ix), np.full(len(ix), 1 / len(ix)), DRAWS
            )
    weights = np.vstack([np.ones(len(ids), int), weights])
    totals = np.stack(
        [
            np.einsum("df,fspk->dspk", weights[:, tags == t], data[tags == t])
            for t in TAGS
        ],
        axis=1,
    )
    truth_totals = np.stack(
        [weights[:, tags == t] @ truth[tags == t] for t in TAGS], axis=1
    )
    assert (truth_totals > 0).all()
    truth_rn = truth_totals[..., 0] / truth_totals[..., 1]
    truth_delta = 100 * (truth_rn[:, 2] / truth_rn[:, 0] - 1)
    eligible = (totals[..., :2] > 0).all(axis=(0, 1, 2, 4))
    delta = np.full((DRAWS + 1, len(SEEDS), data.shape[2]), np.nan)
    stable = totals[..., eligible, :2]
    rn = stable[..., 0] / stable[..., 1]
    delta[..., eligible] = 100 * (rn[:, 2] / rn[:, 0] - 1)
    return delta, totals[0], truth_delta


def stat(x):
    return {"value": float(x[0]), "file_sd": float(x[1:].std(ddof=1))}


def metrics(delta, totals, truth, policies):
    center = delta.mean(axis=1)
    base = center[:, policies.index(REF)]
    rows = []
    for i, j in enumerate(policies):
        row = {
            "purity_target_percent": int(TARGETS[j]),
            "counts_potential_repeat_hi_lo_total_signal": totals[:, :, i, :].tolist(),
        }
        if np.isfinite(center[:, i]).all():
            row.update(
                contrast=stat(center[:, i]),
                by_repeat=delta[0, :, i].tolist(),
                change=stat(center[:, i] - base),
                gap_to_truth=stat(center[:, i] - truth),
            )
        rows.append(row)
    return rows


def main():
    assert not (HERE / "choice.json").exists()
    groups, registry, cuts = inputs()
    policies = list(range(len(TARGETS)))
    ids, data, truth = reduce((0, 2, 4), policies, groups, registry, cuts)
    delta, totals, truth_delta = summarize(ids, data, truth, groups)
    rows = metrics(delta, totals, truth_delta, policies)
    ceiling = 2 * rows[REF]["contrast"]["file_sd"]  # Inherited precision guard.
    for row in rows:
        row["eligible"] = "contrast" in row and row["contrast"]["file_sd"] <= ceiling
    best = max(
        (j for j, row in enumerate(rows) if row["eligible"]),
        key=lambda j: rows[j]["contrast"]["value"],
    )
    chosen = {
        "truth": stat(truth_delta),
        "scan": rows,
        "maximum_file_sd": ceiling,
        "chosen_target_percent": int(TARGETS[best]),
        "per_fold_seed_score_cuts": {
            f"{f}/{s}": float(c[best]) for (f, s), c in cuts.items()
        },
    }
    write("choice.json", chosen)
    # The freeze record exists before odd prediction files are opened.
    policies = sorted({REF, best})
    ids_f, data_f, truth_f = reduce((1, 3, 5), policies, groups, registry, cuts)
    assert not set(ids) & set(ids_f)
    delta, totals, truth_delta = summarize(ids_f, data_f, truth_f, groups)
    write(
        "fixed_check.json",
        {
            "truth": stat(truth_delta),
            "points": metrics(delta, totals, truth_delta, policies),
        },
    )
    np.savez_compressed(
        HERE / "counts.npz",
        selection_ids=ids,
        selection=data,
        check_ids=ids_f,
        check=data_f,
    )


if __name__ == "__main__":
    main()
