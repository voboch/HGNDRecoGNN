"""Export all three fixed potential contrasts from accepted SMASH results.

Uses the saved joint whole-file replicas. No models, raw events, fitting,
threshold choice or new random draws. The historical [0.3,1) / [2,infinity)
bands are retained to keep the report comparable to the accepted ladder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = ("42", "123", "456")
TAGS = {0: "zeroSpot", 18: "defaultSpot", 90: "bigSpot"}
PAIRS = ((18, 0), (90, 0), (90, 18))  # Every unordered pair, oriented high/low.
ARMS = ("A1", "B1")
RUNGS = ("T0", "T1", "T2", "C1", "C2", "C3", "C4")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(central, replicas):
    assert np.isfinite(central) and np.isfinite(replicas).all()
    return {"percent": float(central), "whole_file_bootstrap_sd_percent": float(np.std(replicas, ddof=1))}


def export():
    pins = json.loads((HERE / "input_pins.json").read_text())
    for name, digest in pins["files"].items():
        assert sha(ROOT / name) == digest, name
    base = ROOT / "results/smash_train_20261028"
    summary = json.loads((base / "heldout_rn/summary.json").read_text())
    counts = json.loads((base / "heldout_rn/per_file_counts.json").read_text())["primary"]
    split = json.loads((base / "split/split_manifest.json").read_text())
    arrays = np.load(base / "heldout_rn/replicas.npz")
    pooled = summary["variants"]["primary"]["pooled"]
    assert summary["settings"]["n_boot"] == 400
    assert summary["bands"]["primary"]["low_GeV"] == [0.3, 1.0]
    assert summary["bands"]["primary"]["high_GeV"] == 2.0
    out = {"source_sha256": pins["files"], "scope": pins["scope"],
           "formula": "100*(R(U_high)/R(U_low)-1)",
           "hardness": "R=N(E>=2 GeV)/N(0.3<=E<1 GeV); no upper high-energy cap",
           "bootstrap": "SD of 400 saved joint whole-file resamples; fixed trained models",
           "sample": pooled["n_files"], "pairs": {}, "folds": {}}
    for high, low in PAIRS:
        pair = {"U_high_MeV": high, "U_low_MeV": low, "reconstructed": {}}
        truth_reference = None
        for arm in ARMS:
            result = pooled["arms"][arm]
            c = {u: np.array([result["R"][f"{r}/seed_{s}"][tag]["central"]
                             for s in SEEDS for r in RUNGS]).reshape(3, 7)
                 for u, tag in TAGS.items()}
            # Replay central R from accepted integer per-file counts; no CSVs opened.
            for u, tag in TAGS.items():
                t = np.sum([x["truth_hi_lo"] for x in counts[arm][tag]], axis=0)
                g = np.sum([x["clusters_hi_lo"] for x in counts[arm][tag]], axis=0)
                direct = np.concatenate([np.repeat((t[:, 0] / t[:, 1])[None], 3, axis=0),
                                         g[:, :, 0] / g[:, :, 1]], axis=1)
                assert np.allclose(c[u], direct, rtol=0, atol=1e-14)
            endpoint = arrays[f"primary__pooled__{arm}__endpoint"]
            midpoint = arrays[f"primary__pooled__{arm}__midpoint"]
            relative = {0: np.ones_like(endpoint), 18: 1 + midpoint / 100, 90: 1 + endpoint / 100}
            central = 100 * (c[high] / c[low] - 1)
            replica = 100 * (relative[high] / relative[low] - 1)
            assert replica.shape == (400, 3, 7)
            # Existing endpoint/midpoint numbers must replay before adding 90/18.
            for kind, h in (("endpoint", 90), ("midpoint", 18)):
                for i, seed in enumerate(SEEDS):
                    for j, rung in enumerate(RUNGS):
                        old = result[kind][f"{rung}/seed_{seed}"]
                        assert abs(100 * (c[h][i, j] / c[0][i, j] - 1) - old["central_percent"]) < 1e-11
                        assert abs(np.std(arrays[f"primary__pooled__{arm}__{kind}"][:, i, j], ddof=1)
                                   - old["bootstrap_sd_percent"]) < 1e-11
            for i, seed in enumerate(SEEDS):
                truth = stat(central[i, 1], replica[:, i, 1])
                if truth_reference is None:
                    truth_reference = truth
                    pair["MC_truth_T1"] = truth
                assert np.allclose(list(truth.values()), list(truth_reference.values()), rtol=0, atol=1e-11)
                pair["reconstructed"].setdefault(arm, {})[seed] = {
                    "C4": stat(central[i, 6], replica[:, i, 6]),
                    "C4_minus_T1_points": stat(central[i, 6] - central[i, 1], replica[:, i, 6] - replica[:, i, 1])}
        out["pairs"][f"{high}/{low}"] = pair
    groups = {g["id"]: g for g in split["groups"]}
    for fold, item in split["folds"].items():
        out["folds"][fold] = {role: {"total": len(ids),
            "by_U_MeV": {str(u): sum(groups[c]["tag"] == tag for c in ids) for u, tag in TAGS.items()},
            "without_primary_binding": sum(not groups[c]["truth_bound"] for c in ids)}
            for role, ids in item["partitions"].items()}
    (HERE / "all_pair_contrasts.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    return out


def tables(data):
    lines = ["| U_high / U_low (МэВ) | T1, MC truth: контраст Rₙ (%) |", "| --- | --- |"]
    for key, p in data["pairs"].items():
        s = p["MC_truth_T1"]
        lines.append(f"| {key} | {s['percent']:+.2f} ± {s['whole_file_bootstrap_sd_percent']:.2f} |")
    lines += ["", "| U_high / U_low (МэВ) | Seed | A1 C4: контраст (%) | B1 C4: контраст (%) | B1 C4 − T1 (п.п.; →0) |",
              "| --- | --- | --- | --- | --- |"]
    def fmt(v):
        return f"{v['percent']:+.2f} ± {v['whole_file_bootstrap_sd_percent']:.2f}"
    for key, p in data["pairs"].items():
        for seed in SEEDS:
            a, b = (p["reconstructed"][arm][seed] for arm in ARMS)
            lines.append(f"| {key} | {seed} | {fmt(a['C4'])} | {fmt(b['C4'])} | {fmt(b['C4_minus_T1_points'])} |")
    lines += ["", "| Fold | Train (0 / 18 / 90 МэВ) | Validation (0 / 18 / 90 МэВ) | Test (0 / 18 / 90 МэВ) | Из test-частей исключено unbound |",
              "| --- | --- | --- | --- | --- |"]
    for k, f in data["folds"].items():
        fields = [" / ".join(str(f[role]["by_U_MeV"][str(u)]) for u in TAGS) for role in ("train", "val", "test")]
        lines.append(f"| {k} | {' | '.join(fields)} | {f['unused']['total']} |")
    (HERE / "tables.md").write_text("\n".join(lines) + "\n")


def plot(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.linewidth": 0.8, "axes.grid": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), sharex="col",
                             gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    bounds = []
    for col, (key, p) in enumerate(data["pairs"].items()):
        ax, gap = axes[:, col]
        t = p["MC_truth_T1"]
        ax.axhspan(t["percent"] - t["whole_file_bootstrap_sd_percent"],
                   t["percent"] + t["whole_file_bootstrap_sd_percent"], color="#2b2b2b", alpha=.16, label="T1 MC truth ± file SD")
        ax.axhline(t["percent"], color="#2b2b2b", lw=1)
        ax.axhline(0, color="0.5", ls=":", lw=.8)
        gap.axhline(0, color="#2b2b2b", ls="--", lw=1)
        top_values = [t["percent"] - t["whole_file_bootstrap_sd_percent"], t["percent"] + t["whole_file_bootstrap_sd_percent"], 0]
        bottom_values = [0]
        for arm, offset, color, marker in (("A1", -.1, "#1f77b4", "s"), ("B1", .1, "#b22222", "^")):
            for i, seed in enumerate(SEEDS):
                r = p["reconstructed"][arm][seed]
                for a, field, values in ((ax, "C4", top_values), (gap, "C4_minus_T1_points", bottom_values)):
                    s = r[field]
                    a.errorbar(i + offset, s["percent"], yerr=s["whole_file_bootstrap_sd_percent"], fmt=marker,
                               color=color, capsize=3, ms=6, label=f"C4 {arm}" if i == 0 and a is ax else None)
                    values.extend([s["percent"] - s["whole_file_bootstrap_sd_percent"], s["percent"] + s["whole_file_bootstrap_sd_percent"]])
        for a, values in ((ax, top_values), (gap, bottom_values)):
            lo, hi = min(values), max(values)
            pad = .16 * (hi - lo)  # Space for uncertainty caps and readable labels.
            a.set_ylim(lo - pad, hi + pad)
            assert a.get_ylim()[0] < lo and hi < a.get_ylim()[1]
            a.tick_params(direction="in", top=True, right=True)
            a.set_xlim(-.45, 2.45)
            bounds.append({"pair": key, "panel": "contrast" if a is ax else "residual", "data_min": lo, "data_max": hi, "ylim": list(a.get_ylim())})
        ax.set_title(f"U_sym: {key} MeV\nMC truth and reconstructed", fontsize=11)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        gap.set_xticks(range(3), SEEDS)
        gap.set_xlabel("Training seed; fixed models")
        if col == 0:
            ax.set_ylabel("100 [R(high)/R(low) − 1] (%)")
            gap.set_ylabel("C4 − T1 (points)")
    fig.suptitle("BM@N HGND simulation · corrected SMASH Xe+Cs, 2.87 GeV/nucleon\n"
                 "568 cross-fit files; no B selection (B unavailable); R = N(E≥2)/N(0.3≤E<1)\n"
                 "Bars: SD of 400 saved joint whole-file resamples; reused data, no fresh holdout", fontsize=11)
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"all_pair_contrasts.{suffix}", dpi=170)
    plt.close(fig)
    (HERE / "figure_bounds.json").write_text(json.dumps(bounds, indent=2) + "\n")


if __name__ == "__main__":
    data = export()
    tables(data)
    plot(data)
    print((HERE / "tables.md").read_text())
