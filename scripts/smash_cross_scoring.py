"""Cross-scoring: two networks of one arm and seed, trained on different files, on the same test files.

Note: reports/smash_cross_scoring_20261029.md (predictions committed before the run). Split part
q in {0, 2, 4} is validated by fold (q/2 - 1) mod 3 and tested by fold q/2, so each of its
truth-bound files has two scorings per arm and seed: the validating fold's (training-export
validation predictions, calibrated with that run's calibration) and the testing fold's (accepted
evaluation CSV), each at its own frozen cut. Per part the ladder steps and band selection
efficiencies of both are computed on identical clusters, and their paired difference is
bootstrapped over whole files with draws shared by both scorings. No training or inference.
Labels and true energies are MC truth; selection and reconstructed energies are reconstructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.scripts import reco_rn_heldout as held  # noqa: E402
from HGNDRecoGNN.scripts import reco_rn_retrospective as retro  # noqa: E402

TAGS, SEEDS = retro.TAGS, retro.SEEDS
ARMS = ("A1", "B1")
BAND = held.BANDS["primary"]
BASE = ROOT / "results/smash_train_20261028"
EVALUATIONS = [BASE / f"evaluate_{j}/extracted/evaluation" for j in (4466339, 4467086, 4467091)]
TRAININGS = BASE / "train_4466016"
TASK = {(f, s): 3 * f + i for f in range(3) for i, s in enumerate(SEEDS)}
NPZ = {"A1": "predictions_A.npz", "B1": "predictions_B.npz"}
MODELS = ("validating_fold", "testing_fold")
STEPS = {"selection": (1, 0), "energy": (2, 1), "background": (3, 2)}  # rows C1..C4 of cluster_counts
SEED = 20261029


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parts(split_path):
    manifest = json.loads(Path(split_path).read_text())
    return {f"{g['tag']}/{c}": g["part"] for g in manifest["groups"] for c in g["contents"]}


def validating_fold(part):
    return {(2 * k + 2) % 6: k for k in range(3)}[part]


def validation_frames(fold, seed, arm, members):
    """{content: frame} of a run's validation predictions with its calibration, and its frozen cut."""
    from HGNDRecoGNN.scripts.smash_train import apply_calibration
    t = TASK[(fold, seed)]
    complete = json.loads((TRAININGS / f"task_{t}/complete.json").read_text())
    d = np.load(TRAININGS / f"extracted/task_{t}/train/validation/{NPZ[arm]}", allow_pickle=False)
    keep = np.isin(d["content"], list(members))
    frame = pd.DataFrame({"content": d["content"][keep], "cl_score": d["cl_score"][keep].astype(float),
                          "e_pred": apply_calibration(complete["calibrations"][arm], d["e_pred"][keep]),
                          "e_true": d["e_true"][keep], "cl_label": d["cl_label"][keep],
                          "cluster_size": d["cluster_size"][keep]})
    return {c: g.reset_index(drop=True) for c, g in frame.groupby("content", sort=False)}, complete["cuts"][arm]


def part_counts(files, members, part, arm):
    """counts[tag] (files, seeds, 2 scorings, 4 rungs, 2 bands) on the part's files, sorted by tag."""
    fold_v, fold_t = validating_fold(part), part // 2
    out = {t: [] for t in TAGS}
    names = {t: [] for t in TAGS}
    per_seed = {}
    for s in SEEDS:
        per_seed[s] = validation_frames(fold_v, s, arm, members)
    for c in sorted(members):
        if files[c]["fold"] != fold_t:
            raise ValueError(f"Test fold of {c} is {files[c]['fold']}, expected {fold_t}")
        rows = []
        for s in SEEDS:
            test = pd.read_csv(files[c]["csv"][arm][s])
            val_frames, val_cut = per_seed[s]
            v = val_frames[c]
            if (len(v) != len(test) or not np.array_equal(v.cl_label.to_numpy(), test.cl_label.to_numpy())
                    or not np.array_equal(v.e_true.to_numpy(np.float32), test.e_true.to_numpy(np.float32))
                    or not np.array_equal(v.cluster_size.to_numpy(), test.cluster_size.to_numpy())):
                raise ValueError(f"Validation rows do not align with the test CSV: {c}")
            a = held.cluster_counts(v, val_cut, BAND)
            b = held.cluster_counts(test, files[c]["cuts"][arm][s], BAND)
            if not np.array_equal(a[0], b[0]):
                raise ValueError(f"Truth-label rung C1 differs between scorings: {c}")
            rows.append([a, b])
        tag = c.split("/", 1)[0]
        out[tag].append(rows)
        names[tag].append(c)
    return {t: np.array(v, float) for t, v in out.items()}, names


def contrasts(sums):
    """K per rung and the band efficiencies from summed counts (..., 4 rungs, 2 bands) per tag."""
    r = {t: sums[t][..., 0] / sums[t][..., 1] for t in TAGS}  # (..., 4)
    k = 100 * (r["bigSpot"] / r["zeroSpot"] - 1)
    eff = {t: sums[t][..., 1, :] / sums[t][..., 0, :] for t in TAGS}  # C2/C1 per band (..., 2)
    e = 100 * (eff["bigSpot"] / eff["zeroSpot"] - 1)
    out = {name: k[..., i] - k[..., j] for name, (i, j) in STEPS.items()}
    out |= {"C4": k[..., 3], "eff_high": e[..., 0], "eff_low": e[..., 1]}
    return out


def analyse(counts, rng):
    """Central values and paired differences (validating - testing scoring) with shared whole-file draws."""
    draws = {t: retro.file_draws(len(counts[t]), rng) for t in TAGS}
    central = contrasts({t: counts[t].sum(axis=0) for t in TAGS})            # (seeds, 2)
    reps = contrasts({t: np.tensordot(draws[t], counts[t], axes=(1, 0)) for t in TAGS})  # (B, seeds, 2)
    out = {}
    for name in central:
        c, r = central[name], reps[name]
        out[name] = {"seed": list(SEEDS),
                     **{m: {"central": c[:, i].tolist(), "sd": r[:, :, i].std(axis=0, ddof=1).tolist()}
                        for i, m in enumerate(MODELS)},
                     "difference": {"central": (c[:, 0] - c[:, 1]).tolist(),
                                    "sd": (r[:, :, 0] - r[:, :, 1]).std(axis=0, ddof=1).tolist()}}
    return out


def run(split_path, output):
    from HGNDRecoGNN.scripts.smash_train_prepare import load_split
    split = load_split(split_path)
    split["sha256"] = sha256(split_path)
    files = held.load_evaluations(EVALUATIONS, split)
    split_parts = parts(split_path)
    rng = np.random.default_rng(SEED)
    result = {"schema": "smash-cross-scoring-v1", "note": "reports/smash_cross_scoring_20261029.md",
              "split_sha256": split["sha256"], "code_sha256": sha256(Path(__file__)), "boot_seed": SEED,
              "n_boot": retro.N_BOOT, "band": "primary", "scorings": list(MODELS), "parts": {}}
    pooled = {arm: {t: [] for t in TAGS} for arm in ARMS}
    for q in (0, 2, 4):
        members = [c for c in files if split_parts[c] == q]
        entry = {"validating_fold": validating_fold(q), "testing_fold": q // 2, "n_files": len(members), "arms": {}}
        for arm in ARMS:
            counts, _ = part_counts(files, members, q, arm)
            entry["arms"][arm] = analyse(counts, rng)
            for t in TAGS:
                pooled[arm][t].append(counts[t])
        result["parts"][str(q)] = entry
    result["pooled"] = {arm: analyse({t: np.concatenate(v) for t, v in pooled[arm].items()}, rng) for arm in ARMS}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, default=BASE / "split/split_manifest.json")
    p.add_argument("--output", type=Path, default=Path("results/smash_cross_scoring_20261029"))
    a = p.parse_args()
    run(a.split, ROOT / a.output)
    print(json.dumps({"status": "complete", "output": str(a.output)}))
