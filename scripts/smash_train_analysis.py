"""Classification, energy response and the U_sym-dependent energy bias of the SMASH-training arms.

Protocol: reports/smash_train_protocol_20261028.md. Reads verified, extracted
evaluation and training outputs; fits nothing, chooses no cut. Every number is
per arm and per training seed; seeds are never pooled into one prediction set.
Out-of-fold "pooled" rows mix three models per seed label, so per-fold rows are
reported beside them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.scripts.diagnose_student_pipeline import SIZE_EDGES, measure  # noqa: E402
from HGNDRecoGNN.scripts.reco_performance import E_BINS  # noqa: E402

TAGS = ("zeroSpot", "defaultSpot", "bigSpot")
U_MEV = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
SEEDS = ("42", "123", "456")
ARMS = ("A0", "A1", "B0", "B1", "Bcomp")
# The critic's window for the differential bias (review v9 M4), not optimised here.
WINDOW = (1.75, 3.0)
# Inherited from the retrospective (400 whole-file replicates; Monte-Carlo error of
# an SD about 3.5 %); the seed is arbitrary and fixed before the run.
N_BOOT, BOOT_SEED = 400, 20261028
MIN_BIN = 25  # inherited minimum per E_true bin of reco_performance.energy_performance


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def load_predictions(evaluations):
    """{arm: {seed: DataFrame}} over every evaluated test file, with fold, tag and content."""
    frames, cuts = {arm: {s: [] for s in SEEDS} for arm in ARMS}, {}
    for directory in evaluations:
        directory = Path(directory)
        summary = json.loads((directory / "summary.json").read_text())
        if summary["state"] != "complete":
            raise ValueError(f"Incomplete evaluation: {directory}")
        fold = summary["fold"]
        for arm in ARMS:
            for seed in SEEDS:
                cuts.setdefault(arm, {}).setdefault(seed, {})[fold] = summary["training"][seed]["cuts"][arm]
                for cid, record in summary["predictions"][arm][seed].items():
                    path = directory / record["path"]
                    if sha256(path) != record["sha256"]:
                        raise ValueError(f"Prediction file changed: {path}")
                    f = pd.read_csv(path)
                    f["content"], f["tag"], f["fold"] = cid, cid.split("/", 1)[0], fold
                    frames[arm][seed].append(f)
    return {arm: {s: pd.concat(v, ignore_index=True) for s, v in by.items()} for arm, by in frames.items()}, cuts


def selected(frame, cuts):
    """Score > cut, with the cut of each row's fold."""
    cut = frame.fold.map(cuts).to_numpy(float)
    return frame.cl_score.to_numpy() > cut


def classification(frame, cuts):
    """Per potential and pooled, by cluster size; measure() applies one cut, so per fold."""
    out = {}
    for scope, part in [("pooled", frame)] + [(t, frame.loc[frame.tag == t]) for t in TAGS]:
        rows = {}
        for fold, sub in part.groupby("fold"):
            rows[f"fold_{fold}"] = {"all": measure(sub, cuts[fold]), "size": [
                {"lo": lo, "hi": hi, **measure(sub.loc[(sub.cluster_size > lo) & (sub.cluster_size <= hi)], cuts[fold])}
                for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:])]}
        sel = selected(part, cuts)
        y = part.cl_label.to_numpy() == 1
        rows["out_of_fold"] = {"n": len(part), "signal": int(y.sum()), "selected": int(sel.sum()),
                               "purity": float(y[sel].mean()) if sel.any() else None,
                               "efficiency": float(sel[y].mean()) if y.any() else None}
        out[scope] = rows
    return out


def energy_bins(e_true, e_pred, bins=E_BINS):
    """Bias, MAE, median response and half 16-84 width of the relative residual per E_true bin."""
    rows = []
    idx = np.digitize(e_true, bins) - 1
    for k in range(len(bins) - 1):
        m = idx == k
        if m.sum() < MIN_BIN:
            continue
        res, rel = e_pred[m] - e_true[m], e_pred[m] / e_true[m]
        lo, hi = np.percentile(rel - 1, [16, 84])
        rows.append({"e_lo": float(bins[k]), "e_hi": float(bins[k + 1]), "n": int(m.sum()),
                     "mean_e_true": float(e_true[m].mean()), "bias_GeV": float(res.mean()),
                     "mae_GeV": float(np.abs(res).mean()), "median_response": float(np.median(rel)),
                     "half_width_16_84": float(.5 * (hi - lo))})
    return rows


def energy(frame, cuts):
    out = {}
    pos = (frame.cl_label.to_numpy() == 1) & (frame.e_true.to_numpy() > 0)
    sel = selected(frame, cuts)
    for population, mask in (("mc_positive", pos), ("selected_positive", pos & sel)):
        part = frame.loc[mask]
        out[population] = {scope: energy_bins(sub.e_true.to_numpy(), sub.e_pred.to_numpy())
                           for scope, sub in [("pooled", part)] + [(t, part.loc[part.tag == t]) for t in TAGS]}
    return out


def file_sums(frame, edges):
    """Per content and bin: (n, sum of residuals, sum of E_true)."""
    contents = sorted(frame.content.unique())
    index = {c: i for i, c in enumerate(contents)}
    sums = np.zeros((len(contents), len(edges), 3))
    e_true, e_pred = frame.e_true.to_numpy(), frame.e_pred.to_numpy()
    rows = frame.content.map(index).to_numpy(np.int64)
    for j, (lo, hi) in enumerate(edges):
        m = (e_true >= lo) & (e_true < hi)
        for col, value in enumerate((np.ones(m.sum()), e_pred[m] - e_true[m], e_true[m])):
            np.add.at(sums[:, j, col], rows[m], value)
    return sums


def delta_bias(frame, cuts, rng_seed=BOOT_SEED):
    """Delta bias (90-0) and (18-0) per E_BINS bin and in WINDOW, with a whole-file bootstrap.

    The within-bin mean E_true difference is the control: a bias difference that
    tracks it is a spectrum-shape effect inside the bin, not a response change.
    """
    edges = list(zip(E_BINS[:-1], E_BINS[1:])) + [WINDOW]
    pos = (frame.cl_label.to_numpy() == 1) & (frame.e_true.to_numpy() > 0)
    out = {}
    for population, mask in (("mc_positive", pos), ("selected_positive", pos & selected(frame, cuts))):
        part = frame.loc[mask]
        sums = {t: file_sums(part.loc[part.tag == t], edges) for t in TAGS}
        rng = np.random.default_rng(rng_seed)
        draws = {t: rng.multinomial(len(sums[t]), np.full(len(sums[t]), 1 / len(sums[t])), size=N_BOOT)
                 if len(sums[t]) else np.zeros((N_BOOT, 0), np.int64) for t in TAGS}
        stats = {}
        for t in TAGS:
            total = sums[t].sum(axis=0)
            rep = np.tensordot(draws[t], sums[t], axes=(1, 0))
            with np.errstate(invalid="ignore", divide="ignore"):
                stats[t] = {"n": total[:, 0], "bias": total[:, 1] / total[:, 0], "e": total[:, 2] / total[:, 0],
                            "bias_rep": rep[..., 1] / rep[..., 0], "e_rep": rep[..., 2] / rep[..., 0]}
        rows = {}
        for high in ("bigSpot", "defaultSpot"):
            key = f"{U_MEV[high]}-0"
            rows[key] = []
            for j, (lo, hi) in enumerate(edges):
                ok = min(stats[high]["n"][j], stats["zeroSpot"]["n"][j]) >= MIN_BIN
                d = stats[high]["bias"][j] - stats["zeroSpot"]["bias"][j]
                de = stats[high]["e"][j] - stats["zeroSpot"]["e"][j]
                rows[key].append({"e_lo": float(lo), "e_hi": float(hi), "window": (lo, hi) == WINDOW,
                                  "n_high": int(stats[high]["n"][j]), "n_zero": int(stats["zeroSpot"]["n"][j]),
                                  "delta_bias_MeV": 1e3 * d if ok else None,
                                  "delta_bias_sd_MeV": 1e3 * float(np.nanstd(
                                      stats[high]["bias_rep"][:, j] - stats["zeroSpot"]["bias_rep"][:, j], ddof=1)) if ok else None,
                                  "delta_mean_e_true_MeV": 1e3 * de if ok else None,
                                  "delta_mean_e_true_sd_MeV": 1e3 * float(np.nanstd(
                                      stats[high]["e_rep"][:, j] - stats["zeroSpot"]["e_rep"][:, j], ddof=1)) if ok else None})
        out[population] = rows
    return out


def summarize_seeds(values):
    values = [v for v in values if v is not None]
    if len(values) != len(SEEDS):
        return {"seed_values": values, "mean": None, "sample_sd": None}
    return {"seed_values": values, "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}


def load_trainings(trainings):
    runs = {}
    for directory in trainings:
        directory = Path(directory)
        complete = json.loads((directory / "complete.json").read_text())
        for name, digest in complete["files"].items():
            if name.startswith("validation/") or name == "history.json":
                if sha256(directory / name) != digest:
                    raise ValueError(f"Training artefact changed: {directory / name}")
        complete["history"] = json.loads((directory / "history.json").read_text())
        complete["directory"] = directory
        runs[(complete["fold"], str(complete["seed"]))] = complete
    return runs


def figures(runs, delta, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from HGNDRecoGNN.scripts.plot_b_full import apply_style, mark_provenance
    from HGNDRecoGNN.scripts.smash_train import apply_calibration
    apply_style()
    target = output / "figures"
    target.mkdir()
    colors = {"42": "#1f4e79", "123": "#b5651d", "456": "#4f7f3a"}
    folds = sorted({f for f, _ in runs})
    # Per-head validation histories with the selected epochs.
    fig, axes = plt.subplots(len(folds), 2, figsize=(6.84, 2.2 * len(folds)), squeeze=False)
    for r, fold in enumerate(folds):
        for c, (head, key, label) in enumerate((("e_cls", "validation_cluster_cls", "validation BCE"),
                                                ("e_en", "validation_cluster_energy", "validation SmoothL1, positives (GeV)"))):
            ax = axes[r, c]
            for seed in SEEDS:
                run = runs.get((fold, seed))
                if run is None:
                    continue
                y = run["selection"][key]
                ax.plot(range(len(y)), y, color=colors[seed], label=f"seed {seed}")
                ax.axvline(run["selection"][head], color=colors[seed], ls=":", lw=.8)
            ax.set(xlabel="epoch", ylabel=label, title=f"fold {fold}: {head} (dotted: selected)")
            mark_provenance(ax, "mixed")
        axes[r, 0].legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    fig.savefig(target / "head_histories.pdf")
    fig.savefig(target / "head_histories.png", dpi=200)
    plt.close(fig)
    # Validation threshold scans with the frozen cuts.
    fig, axes = plt.subplots(len(folds), 3, figsize=(6.84, 2.2 * len(folds)), squeeze=False)
    for r, fold in enumerate(folds):
        for c, arm in enumerate(("A", "B", "Bcomp")):
            ax = axes[r, c]
            cut_name = {"A": "A1", "B": "B1", "Bcomp": "Bcomp"}[arm]
            for seed in SEEDS:
                run = runs.get((fold, seed))
                if run is None:
                    continue
                scan = pd.read_csv(run["directory"] / f"validation/threshold_scan_{arm}.csv")
                ax.plot(scan.threshold, scan.purity_count, color=colors[seed])
                ax.plot(scan.threshold, scan.efficiency_count, color=colors[seed], ls="--")
                if run["cuts"][cut_name] is not None:
                    ax.axvline(run["cuts"][cut_name], color=colors[seed], ls=":", lw=.8)
            ax.axhline(.70, color=".4", ls=":", lw=.6)
            ax.set(xlabel="score cut", ylabel="purity (solid) / efficiency (dashed)", ylim=(0, 1.03),
                   title=f"fold {fold}: {cut_name} validation")
            mark_provenance(ax, "mixed", loc="lower left")
    fig.tight_layout()
    fig.savefig(target / "threshold_scans.pdf")
    fig.savefig(target / "threshold_scans.png", dpi=200)
    plt.close(fig)
    # Calibration response on validation, raw and calibrated.
    fig, axes = plt.subplots(len(folds), 3, figsize=(6.84, 2.2 * len(folds)), squeeze=False)
    for r, fold in enumerate(folds):
        for c, (arm, cal_name) in enumerate((("A", "A1"), ("B", "B1"), ("Bcomp", "Bcomp"))):
            ax = axes[r, c]
            for seed in SEEDS:
                run = runs.get((fold, seed))
                if run is None:
                    continue
                data = np.load(run["directory"] / f"validation/predictions_{arm}.npz")
                m = (data["cl_label"] == 1) & (data["e_true"] > 0)
                e_true, e_pred = data["e_true"][m].astype(float), data["e_pred"][m].astype(float)
                raw = energy_bins(e_true, e_pred)
                ax.plot([b["mean_e_true"] for b in raw], [b["median_response"] for b in raw], color=colors[seed], ls="--")
                cal = run["calibrations"][cal_name]
                if cal is not None:
                    fixed = energy_bins(e_true, apply_calibration(cal, e_pred))
                    ax.plot([b["mean_e_true"] for b in fixed], [b["median_response"] for b in fixed], color=colors[seed])
            ax.axhline(1, color=".4", lw=.6)
            ax.set(xlabel=r"$E_{\rm true}$ (GeV)", ylabel="median E_reco/E_true", title=f"fold {fold}: {arm} (dashed raw)")
            mark_provenance(ax, "mixed")
    fig.tight_layout()
    fig.savefig(target / "calibration_response.pdf")
    fig.savefig(target / "calibration_response.png", dpi=200)
    plt.close(fig)
    # Delta bias (90-0) per bin, MC-positive clusters, arms side by side, seeds separate.
    fig, ax = plt.subplots(figsize=(6.84, 3.0))
    for k, arm in enumerate(ARMS):
        for i, seed in enumerate(SEEDS):
            rows = [x for x in delta[arm][seed]["mc_positive"]["90-0"] if not x["window"] and x["delta_bias_MeV"] is not None]
            x = np.array([.5 * (r["e_lo"] + r["e_hi"]) for r in rows]) * (1 + .01 * (k - 2)) + .01 * (i - 1)
            ax.errorbar(x, [r["delta_bias_MeV"] for r in rows], yerr=[r["delta_bias_sd_MeV"] for r in rows],
                        fmt="o", ms=2, capsize=1, color=f"C{k}", label=arm if i == 0 else None)
    ax.axhline(0, color=".6", lw=.5)
    ax.axvspan(*WINDOW, color=".9", zorder=0)
    ax.set(xlabel=r"$E_{\rm true}$ (GeV)", ylabel=r"$\Delta$bias(90−0) (MeV)")
    ax.legend(ncol=5, fontsize=6, loc="lower left")
    mark_provenance(ax, "mixed")
    fig.tight_layout()
    fig.savefig(target / "delta_bias.pdf")
    fig.savefig(target / "delta_bias.png", dpi=200)
    plt.close(fig)


def a0_parity(frames, transfer_root):
    """A0 on test files against the stored frozen-transfer CSVs (local check)."""
    import torch
    result = {"files": 0, "max_abs_diff": {"cl_score": 0., "e_pred": 0.}}
    for seed in SEEDS:
        f = frames["A0"][seed]
        for cid, part in f.groupby("content", sort=True):
            tag, member = cid.split("/", 1)
            path = Path(transfer_root) / tag / f"seed_{seed}" / (member + ".csv")
            if not path.exists():
                continue
            ref = pd.read_csv(path)
            for col in ("Row", "ClusterID", "cl_istop", "cl_label", "cluster_size"):
                if not np.array_equal(ref[col].to_numpy(), part[col].to_numpy()):
                    raise ValueError(f"A0 keys differ from the transfer: {cid} {col}")
            for col in ("cl_score", "e_pred"):
                a = torch.tensor(part[col].to_numpy(np.float32))
                b = torch.tensor(ref[col].to_numpy(np.float32))
                torch.testing.assert_close(a, b)
                result["max_abs_diff"][col] = max(result["max_abs_diff"][col], float((a - b).abs().max()))
            result["files"] += 1
    return result


def run(evaluations, trainings, output, transfer_root=None, make_figures=True):
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise ValueError(f"Output already exists: {output}")
    frames, cuts = load_predictions(evaluations)
    runs = load_trainings(trainings)
    summary = {"schema": "smash-train-analysis-v1",
               "scope": "cross-fitted SMASH test files; MC-truth labels and energies, reconstructed scores "
                        "and energies; not a blind holdout",
               "settings": {"E_BINS": E_BINS.tolist(), "window_GeV": list(WINDOW), "size_edges": SIZE_EDGES,
                            "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "min_bin": MIN_BIN},
               "code_sha256": sha256(Path(__file__)), "arms": {}, "seed_summary": {}, "selection": {}}
    delta = {}
    for arm in ARMS:
        summary["arms"][arm], delta[arm] = {}, {}
        for seed in SEEDS:
            f = frames[arm][seed]
            c = cuts[arm][seed]
            delta[arm][seed] = delta_bias(f, c)
            summary["arms"][arm][seed] = {"cuts": c, "classification": classification(f, c), "energy": energy(f, c),
                                          "delta_bias": delta[arm][seed]}
        for key in ("auc", "ap", "purity", "efficiency", "bias_GeV", "mae_GeV"):
            for scope in ("pooled",) + TAGS:
                for fold in sorted({k for s in SEEDS for k in summary["arms"][arm][s]["classification"][scope]
                                    if k.startswith("fold_")}):
                    summary["seed_summary"].setdefault(arm, {}).setdefault(scope, {}).setdefault(fold, {})[key] = \
                        summarize_seeds([summary["arms"][arm][s]["classification"][scope][fold]["all"][key] for s in SEEDS])
    for (fold, seed), run_ in sorted(runs.items()):
        summary["selection"][f"fold_{fold}/seed_{seed}"] = {
            k: run_["selection"][k] for k in ("e_cls", "e_en", "e_comp", "budget_limited", "e_cls_at_epoch_0")}
    flags = [k for k, v in summary["selection"].items() if v["e_cls_at_epoch_0"]]
    summary["stop_rules"] = {"e_cls_at_epoch_0_runs": flags, "e_cls_epoch_0_in_two_seeds_of_a_fold": any(
        sum(f"fold_{fold}/" in k for k in flags) >= 2 for fold in {f for f, _ in runs}),
        "budget_limited_runs": [k for k, v in summary["selection"].items() if any(v["budget_limited"].values())]}
    if transfer_root is not None:
        summary["a0_parity_with_transfer"] = a0_parity(frames, transfer_root)
    partial.mkdir(parents=True)
    try:
        if make_figures:
            figures(runs, delta, partial)
        (partial / "summary.json").write_text(json.dumps(clean(summary), indent=1, allow_nan=False) + "\n")
    except Exception:
        shutil.rmtree(partial)
        raise
    partial.rename(output)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluations", type=Path, nargs="+", required=True)
    p.add_argument("--trainings", type=Path, nargs="+", required=True, help="Extracted train/ directories")
    p.add_argument("--output", type=Path, default=Path("results/smash_train_20261028/analysis"))
    p.add_argument("--transfer-root", type=Path,
                   default=ROOT / "results/smash_full_transfer_20261023/full_transfer_inputs/transfer")
    a = p.parse_args()
    run(a.evaluations, a.trainings, ROOT / a.output, a.transfer_root)
    print(json.dumps({"status": "complete", "output": str(a.output)}))
