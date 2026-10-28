"""Held-out truth-to-reconstruction R_n ladder for the SMASH-training arms.

Protocol: reports/smash_train_protocol_20261028.md. Every test file was scored
only by models that never saw it (three-fold cross-fitting), with cuts and
calibrations frozen on fold validation. The pure helpers, gates and bootstrap
come unchanged from scripts/reco_rn_retrospective.py; this script adds band
edges as parameters, the five arms, the per-fold file subsets and the
exact-window truth counters. The files were already scored by the frozen
transfer, so this is cross-fitted, not a blind holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.analysis.truth_provenance import load_certified  # noqa: E402
from HGNDRecoGNN.scripts import reco_rn_retrospective as retro  # noqa: E402

TAGS, SEEDS, RUNGS, STEPS = retro.TAGS, retro.SEEDS, retro.RUNGS, retro.STEPS
ARMS = ("A0", "A1", "B0", "B1", "Bcomp")
# B1-A1 is the primary contrast; B0-A0 without calibration; Bcomp-A1 holds the
# composite selector fixed; B1-Bcomp isolates per-head selection.
PAIRS = (("B1", "A1"), ("B0", "A0"), ("Bcomp", "A1"), ("B1", "Bcomp"))
TRUTH = Path("results/truth_reach_20261026/aggregate_4453896_verified/result")
TRUTH_SUMMARY_SHA = "c6613d0b4b33d6a86b971c64c9cd0ca081491c0bcd3035ee8ff176edbdb166aa"
# Band edges are parameters. The primary denominator 0.3 <= E < 1 GeV
# (acc_n_lo_reach) is the v10 adoption: the lower ToF reach of the exact-window
# counters, below which the native label has no positives. E < 1 GeV
# (acc_n_lo) reproduces the retrospective convention as a variant. The high
# band E >= 2 GeV is inherited and unexamined here.
BANDS = {
    "primary": {"low": (0.3, 1.0), "high": 2.0, "truth_low": "acc_n_lo_reach"},
    "below1": {"low": (-np.inf, 1.0), "high": 2.0, "truth_low": "acc_n_lo"},
}
COLUMNS = ["Row", "ClusterID", "cl_istop", "cl_score", "e_pred", "cl_label", "e_true"]
# Predictions committed in the plan before training (percent, or ratios for retention).
PREDICTED = {"fold_0 T1 endpoint": (2.5, 7.0), "C4/T1 retention A1": (0.5, 0.9),
             "C4/T1 retention B1": (0.7, 1.1), "C4(B1)-C4(A1)": (0.0, 2.5)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def band_counts(energy, weight, band):
    """(high, low) weighted counts; energies between the bands enter neither."""
    lo, hi = band["low"]
    return np.array([weight[energy >= band["high"]].sum(), weight[(energy >= lo) & (energy < hi)].sum()], float)


def cluster_counts(frame, cut, band, weight=None):
    """Per-file (high, low) for C1..C4, selection score > cut as in the retrospective."""
    pos = frame.cl_label.to_numpy() == 1
    sel = frame.cl_score.to_numpy() > cut
    e_true, e_pred = frame.e_true.to_numpy(), frame.e_pred.to_numpy()
    w = np.ones(len(frame)) if weight is None else np.asarray(weight, float)
    return np.stack([band_counts(e_true[pos], w[pos], band), band_counts(e_true[pos & sel], w[pos & sel], band),
                     band_counts(e_pred[pos & sel], w[pos & sel], band), band_counts(e_pred[sel], w[sel], band)])


def truth_counts(events, span, candidate_rows, band):
    """(high, low) primary-neutron counts for T0, T1 (Row < span) and T2 (rows with candidates)."""
    rows = events.index.to_numpy()
    hi = events.acc_n_hi.to_numpy(float)
    lo = events[band["truth_low"]].to_numpy(float)
    keep = [np.ones(len(rows), bool), rows < span, np.isin(rows, candidate_rows)]
    return np.array([[hi[k].sum(), lo[k].sum()] for k in keep])


def event_counts(frame, cut, band):
    """Per-event (high, low) of C4, events with nonzero counts only."""
    sel = frame.cl_score.to_numpy() > cut
    e = frame.e_pred.to_numpy()[sel]
    lo, hi = band["low"]
    df = pd.DataFrame({"Row": frame.Row.to_numpy()[sel], "hi": e >= band["high"], "lo": (e >= lo) & (e < hi)})
    g = df.groupby("Row")[["hi", "lo"]].sum()
    return g[(g.hi > 0) | (g.lo > 0)].to_numpy(float)


def load_truth(root, truth_dir):
    summary_path = root / truth_dir / "summary.json"
    if sha256(summary_path) != TRUTH_SUMMARY_SHA:
        raise ValueError("Exact-window truth aggregate changed")
    summary = json.loads(summary_path.read_text())
    events = {}
    for tag in TAGS:
        cert = root / truth_dir / tag
        if sha256(cert / "proof.json") != summary["samples"][tag]["proof_sha256"]:
            raise ValueError(f"Truth proof changed: {tag}")
        ev, _, proof = load_certified(cert, "acceptance")
        if proof["counter_mismatches"] != 0 or proof["coverage"] != "full":
            raise ValueError("Truth certificate is not a complete exact replay")
        events[tag] = ev
    return events


def load_evaluations(paths, split):
    """{content: {fold, cuts[arm][seed], csv[arm][seed]}} from verified evaluation outputs."""
    files = {}
    for directory in paths:
        directory = Path(directory)
        summary = json.loads((directory / "summary.json").read_text())
        if summary["state"] != "complete" or summary["identity"]["split_sha256"] != split["sha256"]:
            raise ValueError(f"Evaluation incomplete or from another split: {directory}")
        fold = summary["fold"]
        if sorted(summary["test_files"]) != sorted(split["folds"][fold]["test"]):
            raise ValueError(f"Evaluation files differ from the split test partition: fold {fold}")
        for cid in summary["test_files"]:
            if cid in summary["empty_after_cuts"]:
                continue
            if cid in files:
                raise ValueError(f"Test file scored in two folds: {cid}")
            item = {"fold": fold, "cuts": {}, "csv": {}}
            for arm in ARMS:
                for seed in SEEDS:
                    record = summary["predictions"][arm][seed][cid]
                    path = directory / record["path"]
                    if sha256(path) != record["sha256"]:
                        raise ValueError(f"Prediction file changed: {path}")
                    item["csv"].setdefault(arm, {})[seed] = path
                    item["cuts"].setdefault(arm, {})[seed] = summary["training"][seed]["cuts"][arm]
            files[cid] = item
    return files


def collect(files, contents, events, band):
    """One pass over every file: per-arm counts in the retrospective's array layout."""
    counts = {arm: {} for arm in ARMS}
    for tag in TAGS:
        members = sorted(c for c in files if c.startswith(tag + "/"))
        acc = {arm: {"truth": [], "clusters": [], "injected": [], "per_event": {k: [] for k in ("T1",) + SEEDS},
                     "coverage": []} for arm in ARMS}
        for cid in members:
            item = files[cid]
            frames = {arm: {s: pd.read_csv(item["csv"][arm][s], usecols=COLUMNS) for s in SEEDS} for arm in ARMS}
            reference = frames["A0"][SEEDS[0]][["Row", "ClusterID", "cl_istop", "cl_label", "e_true"]]
            for arm in ARMS:
                for s in SEEDS:
                    f = frames[arm][s]
                    if not f[reference.columns].equals(reference):
                        raise ValueError(f"Arms or seeds scored different clusters: {cid}")
                    if not np.isfinite(f[["cl_score", "e_pred"]].to_numpy()).all():
                        raise ValueError(f"Nonfinite prediction: {cid}")
            primary_member = contents[cid]["primary_member"] + "_prim.csv"
            primary = events[tag].xs(primary_member, level="member")
            rows = np.unique(reference.Row.to_numpy())
            if not np.isin(rows, primary.index.to_numpy()).all():
                raise ValueError(f"Detector Row absent from the bound primary file: {cid}")
            span = int(rows.max()) + 1
            truth = truth_counts(primary, span, rows, band)
            keep = primary.index.to_numpy() < span
            t1 = primary.loc[keep, ["acc_n_hi", band["truth_low"]]].to_numpy(float)
            weight = retro.injection_weight(frames["A0"][SEEDS[0]]) if tag == "zeroSpot" else None
            for arm in ARMS:
                a = acc[arm]
                cuts = item["cuts"][arm]
                a["coverage"].append({"content": cid, "fold": item["fold"], "primary_member": primary_member,
                                      "events_with_candidates": int(len(rows)), "processed_span": span,
                                      "primary_events": int(len(primary))})
                a["truth"].append(truth)
                a["clusters"].append([cluster_counts(frames[arm][s], cuts[s], band) for s in SEEDS])
                if weight is not None:
                    a["injected"].append([cluster_counts(frames[arm][s], cuts[s], band, weight) for s in SEEDS])
                for s in SEEDS:
                    a["per_event"][s].append(event_counts(frames[arm][s], cuts[s], band))
                a["per_event"]["T1"].append(t1[(t1 > 0).any(axis=1)])
        for arm in ARMS:
            a = acc[arm]
            counts[arm][tag] = {"truth": np.array(a["truth"]), "clusters": np.array(a["clusters"]),
                                "injected": np.array(a["injected"]) if a["injected"] else None,
                                "per_event": {k: np.concatenate(v) for k, v in a["per_event"].items()},
                                "coverage": a["coverage"]}
    return counts


def contrast(arrays, result, kind, first, second):
    """Paired arm difference per rung and step under the shared whole-file draws."""
    out = {}
    for i, seed in enumerate(SEEDS):
        for j, rung in enumerate(RUNGS):
            c = result[first][kind][f"{rung}/seed_{seed}"]["central_percent"] - \
                result[second][kind][f"{rung}/seed_{seed}"]["central_percent"]
            r = arrays[first][kind][:, i, j] - arrays[second][kind][:, i, j]
            out[f"{rung}/seed_{seed}"] = retro.summarize_contrast(c, r)
        for a, b in STEPS:
            ja, jb = RUNGS.index(a), RUNGS.index(b)
            c = ((result[first][kind][f"{a}/seed_{seed}"]["central_percent"]
                  - result[first][kind][f"{b}/seed_{seed}"]["central_percent"])
                 - (result[second][kind][f"{a}/seed_{seed}"]["central_percent"]
                    - result[second][kind][f"{b}/seed_{seed}"]["central_percent"]))
            r = ((arrays[first][kind][:, i, ja] - arrays[first][kind][:, i, jb])
                 - (arrays[second][kind][:, i, ja] - arrays[second][kind][:, i, jb]))
            out[f"{a}-{b}/seed_{seed}"] = retro.summarize_contrast(c, r)
    return out


def ladder(counts):
    """Retrospective analysis per arm; identical file lists give identical draws, so arms pair."""
    n_files = {tag: len(counts[ARMS[0]][tag]["coverage"]) for tag in TAGS}
    for arm in ARMS:
        if {tag: len(counts[arm][tag]["coverage"]) for tag in TAGS} != n_files:
            raise ValueError("Arms analysed on different file lists")
    rng = np.random.default_rng(retro.BOOT_SEED)
    draws = {tag: retro.file_draws(n_files[tag], rng) for tag in TAGS}
    results, arrays, gate = {}, {}, {}
    for arm in ARMS:
        results[arm], arrays[arm] = retro.analyse(counts[arm])
        gate[arm] = retro.gates(results[arm], n_files)
    pairs = {f"{a}-{b}": {kind: contrast(arrays, results, kind, a, b) for kind in ("endpoint", "midpoint")}
             for a, b in PAIRS}
    shared = {tag: hashlib.sha256(draws[tag].astype("<i8").tobytes()).hexdigest() for tag in TAGS}
    return {"n_files": n_files, "arms": results, "gates": gate, "pairs": pairs,
            "shared_draws_sha256": shared}, arrays


def prediction_check(scopes):
    checks = {}
    fold0 = scopes.get("fold_0")
    if fold0 is not None:
        for seed in SEEDS:
            v = fold0["arms"]["A1"]["endpoint"][f"T1/seed_{seed}"]["central_percent"]
            lo, hi = PREDICTED["fold_0 T1 endpoint"]
            checks[f"fold_0 T1 endpoint/seed_{seed}"] = {"value": v, "predicted": [lo, hi], "inside": lo <= v <= hi}
    pooled = scopes["pooled"]
    for seed in SEEDS:
        t1 = pooled["arms"]["A1"]["endpoint"][f"T1/seed_{seed}"]["central_percent"]
        for arm in ("A1", "B1"):
            v = pooled["arms"][arm]["endpoint"][f"C4/seed_{seed}"]["central_percent"] / t1
            lo, hi = PREDICTED[f"C4/T1 retention {arm}"]
            checks[f"C4/T1 retention {arm}/seed_{seed}"] = {"value": v, "predicted": [lo, hi], "inside": lo <= v <= hi}
        v = pooled["pairs"]["B1-A1"]["endpoint"][f"C4/seed_{seed}"]["central_percent"]
        lo, hi = PREDICTED["C4(B1)-C4(A1)"]
        checks[f"C4(B1)-C4(A1)/seed_{seed}"] = {"value": v, "predicted": [lo, hi], "inside": lo <= v <= hi}
    checks["G2 not resolvable in every arm"] = {
        "value": {arm: pooled["gates"][arm]["all_seeds"]["G2_closure"] for arm in ARMS},
        "inside": all(pooled["gates"][arm]["all_seeds"]["G2_closure"] == "not resolvable" for arm in ARMS)}
    return checks


def plot(scopes, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from HGNDRecoGNN.scripts.plot_b_full import apply_style, mark_provenance
    apply_style()
    pooled = scopes["pooled"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.84, 3.0))
    x = np.arange(len(RUNGS))
    colors = {"A1": "#1f4e79", "B1": "#b5651d"}
    for k, arm in enumerate(("A1", "B1")):
        for i, seed in enumerate(SEEDS):
            y = [pooled["arms"][arm]["endpoint"][f"{r}/seed_{seed}"]["central_percent"] for r in RUNGS]
            e = [pooled["arms"][arm]["endpoint"][f"{r}/seed_{seed}"]["bootstrap_sd_percent"] for r in RUNGS]
            left.errorbar(x + (k - .5) * .3 + (i - 1) * .08, y, yerr=e, fmt="o", ms=2.5, capsize=1.5,
                          color=colors[arm], label=arm if i == 0 else None)
    d = pooled["pairs"]["B1-A1"]["endpoint"]
    for i, seed in enumerate(SEEDS):
        right.errorbar(x + (i - 1) * .1, [d[f"{r}/seed_{seed}"]["central_percent"] for r in RUNGS],
                       yerr=[d[f"{r}/seed_{seed}"]["bootstrap_sd_percent"] for r in RUNGS], fmt="o", ms=2.5,
                       capsize=1.5, color=".3")
    for ax in (left, right):
        ax.set_xticks(x)
        ax.set_xticklabels(RUNGS)
        ax.axhline(0, color=".7", lw=.5)
        mark_provenance(ax, "mixed")
    left.set_ylabel(r"$100[R_n(90)/R_n(0)-1]$ (%)")
    right.set_ylabel("B1 − A1, paired (points)")
    left.legend(loc="lower left")
    fig.suptitle("Cross-fitted SMASH test files; T rungs MC truth, C4 reconstructed; "
                 f"bars: SD of {retro.N_BOOT} whole-file resamples", fontsize=6.5)
    (output / "figures").mkdir()
    for ext in ("pdf", "png"):
        fig.savefig(output / "figures" / f"heldout_ladder.{ext}", dpi=160,
                    metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
    plt.close(fig)


def run(evaluations, split_path, output, root=ROOT, truth_dir=TRUTH, make_plot=True):
    started = time.monotonic()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise ValueError(f"Output already exists: {output}")
    from HGNDRecoGNN.scripts.smash_train_prepare import load_split
    split = load_split(split_path)
    split["sha256"] = sha256(split_path)
    files = load_evaluations(evaluations, split)
    events = load_truth(root, truth_dir)
    folds = sorted({f["fold"] for f in files.values()})
    summary = {"schema": "reco-rn-heldout-v1", "split_sha256": split["sha256"], "bands": {
        k: {"low_GeV": [float(x) if np.isfinite(x) else None for x in v["low"]], "high_GeV": v["high"],
            "truth_low": v["truth_low"]} for k, v in BANDS.items()},
        "settings": {"n_boot": retro.N_BOOT, "boot_seed": retro.BOOT_SEED, "inject_weight": retro.INJECT,
                     "selection": "cl_score > cut, cut per file from its fold", "arms": list(ARMS),
                     "pairs": [f"{a}-{b}" for a, b in PAIRS]},
        "code": {"retrospective_sha256": sha256(ROOT / "scripts/reco_rn_retrospective.py"),
                 "heldout_sha256": sha256(Path(__file__))},
        "scope": "cross-fitted SMASH test files; MC truth T0-T2, C1-C3 truth label (C3 reco energy), "
                 "C4 reconstructed; not a blind holdout", "variants": {}}
    replicas, per_file = {}, {}
    for variant, band in BANDS.items():
        scopes = {}
        subsets = {"pooled": files, **{f"fold_{k}": {c: v for c, v in files.items() if v["fold"] == k} for k in folds}}
        for scope, subset in subsets.items():
            counts = collect(subset, split["contents"], events, band)
            scopes[scope], arrays = ladder(counts)
            for arm in ARMS:
                replicas[f"{variant}/{scope}/{arm}/endpoint"] = arrays[arm]["endpoint"]
                replicas[f"{variant}/{scope}/{arm}/midpoint"] = arrays[arm]["midpoint"]
            if scope == "pooled":
                per_file[variant] = {arm: {tag: [{**c, "truth_hi_lo": counts[arm][tag]["truth"][i].tolist(),
                                                  "clusters_hi_lo": counts[arm][tag]["clusters"][i].tolist()}
                                                 for i, c in enumerate(counts[arm][tag]["coverage"])]
                                           for tag in TAGS} for arm in ARMS}
        summary["variants"][variant] = scopes
    summary["predictions"] = prediction_check(summary["variants"]["primary"])
    partial.mkdir(parents=True)
    try:
        if make_plot:
            plot(summary["variants"]["primary"], partial)
        np.savez_compressed(partial / "replicas.npz", **{k.replace("/", "__"): v for k, v in replicas.items()})
        (partial / "per_file_counts.json").write_text(json.dumps(per_file, indent=1) + "\n")
        summary["seconds"] = time.monotonic() - started
        (partial / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    except Exception:
        shutil.rmtree(partial)
        raise
    partial.rename(output)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluations", type=Path, nargs="+", required=True,
                   help="Verified, extracted evaluation/ directories, one per fold")
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/smash_train_20261028/heldout_rn"))
    a = p.parse_args()
    run(a.evaluations, a.split, ROOT / a.output)
    print(json.dumps({"status": "complete", "output": str(a.output)}))
