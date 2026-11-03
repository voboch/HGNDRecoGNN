"""Summarise accepted S286 ledgers; no raw data, models or new selections.

The six floors, 0.1-GeV soft bins, 400 draws and RNG seed come from the
registered input config. Shared potential/part draws reproduce the accepted
prevalence SDs before describing origin composition.
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
TAGS = (("zeroSpot", 0), ("defaultSpot", 18), ("bigSpot", 90))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    audit = json.loads((HERE / "audit_summary.json").read_text())
    campaign = json.loads((HERE / "campaign_summary.json").read_text())
    receipt = json.loads((HERE / "publication_receipt.json").read_text())
    assert campaign["state"] == "complete" and campaign["files"] == 286
    assert campaign["all_native_replays_exact"] and campaign["direct_primary_proof_bound"]
    assert sha(HERE / "audit_summary.json") == receipt["summary_files"]["audit/summary.json"]
    assert sha(HERE / "campaign_summary.json") == receipt["summary_files"]["summary.json"]
    assert sha(HERE / "inputs.json") == audit["input_manifest_sha256"]
    config = audit["config"]
    split_path = ROOT / config["split"]
    assert sha(split_path) == config["split_sha256"]
    split = json.loads(split_path.read_text())
    groups = {g["tag"] + "/" + g["kept"]: g for g in split["groups"]
              if g["truth_bound"] and g["part"] in (0, 2, 4)}
    assert set(groups) == set(audit["files"])
    assert len(groups) == 286 and all(f["native_replay_exact"] for f in audit["files"].values())
    n_boot = config["bootstrap_proposal"]["resamples"]
    rng = np.random.default_rng(config["bootstrap_proposal"]["seed"])
    result = {"state": "complete", "scope": audit["scope"], "files": 286,
              "source_sha256": {p: sha(HERE / p) for p in ("audit_summary.json", "campaign_summary.json", "inputs.json", "publication_receipt.json")},
              "draws": {"n": n_boot, "seed": config["bootstrap_proposal"]["seed"], "strata": "potential x even part"},
              "by_potential": {}, "total": {}}
    pooled_hist, pooled_hist_r, pooled_n, pooled_nr = 0, 0, 0, 0
    for tag, potential in TAGS:
        ids = sorted(k for k in groups if groups[k]["tag"] == tag)
        fs = [audit["files"][k] for k in ids]
        n = np.array([f["clusters"] for f in fs])
        positive = np.array([[r["signal_clusters"] for r in f["ledger"]] for f in fs])
        draws = np.zeros((n_boot, len(ids)))
        for part in config["selection_parts"]:
            ix = [i for i, cid in enumerate(ids) if groups[cid]["part"] == part]
            draws[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=n_boot)
        fractions = 100 * positive.sum(axis=0) / n.sum()
        replicas = 100 * (draws @ positive) / (draws @ n)[:, None]
        assert np.allclose(fractions, audit["bootstrap"][tag]["positive_fraction_percent"], atol=1e-12, rtol=0)
        assert np.allclose(replicas.std(axis=0, ddof=1), audit["bootstrap"][tag]["bootstrap_sd_percent"], atol=1e-12, rtol=0)
        added = np.array([[f["ledger"][0]["added_vs_native"],
                           f["ledger"][0]["added_origin_clusters"]["direct_primary_only"],
                           f["ledger"][0]["added_origin_clusters"]["nonprimary_track_only"]] for f in fs])
        assert np.array_equal(added[:, 0], added[:, 1:].sum(axis=1))
        fraction = 100 * added.sum(axis=0) / n.sum()
        added_r = 100 * (draws @ added) / (draws @ n)[:, None]
        native = int(positive[:, 3].sum())
        hist = np.array([np.array(f["ledger"][0]["target_histogram"]) - np.array(f["ledger"][3]["target_histogram"]) for f in fs])
        assert np.all(hist >= 0) and not hist[:, 3:].any()
        assert np.array_equal(hist.sum(axis=1), added[:, 0])
        assert all(f["ledger"][0]["target_overflow"] == f["ledger"][3]["target_overflow"] for f in fs)
        rec = {"groups": len(fs), "clusters": int(n.sum()), "observed_hits": sum(f["hits"] for f in fs),
               "native_positive": native, "no_floor_positive": int(positive[:, 0].sum()),
               "added_positive": int(added[:, 0].sum()), "added_direct_only": int(added[:, 1].sum()),
               "added_nonprimary_only": int(added[:, 2].sum()),
               "added_singletons": sum(f["ledger"][0]["added_singletons"] for f in fs),
               "added_vs_native_percent": float(100 * added[:, 0].sum() / native),
               "delta_positive_fraction_points": fraction.tolist(),
               "delta_positive_fraction_bootstrap_sd_points": added_r.std(axis=0, ddof=1).tolist(),
               "new_target_histogram": hist.sum(axis=0).tolist(),
               "observed_hits_t_ge_35ns": sum(f["observed_hits_after_jitter_t_ge_35ns"] for f in fs),
               "floor_scan": audit["bootstrap"][tag]}
        result["by_potential"][str(potential)] = rec
        pooled_hist += hist.sum(axis=0)
        pooled_hist_r += draws @ hist
        pooled_n += n.sum()
        pooled_nr += draws @ n
    totals = result["total"]
    for k in ("clusters", "observed_hits", "native_positive", "no_floor_positive", "added_positive",
              "added_direct_only", "added_nonprimary_only", "added_singletons", "observed_hits_t_ge_35ns"):
        totals[k] = sum(f[k] for f in result["by_potential"].values())
    totals["added_vs_native_percent"] = 100 * totals["added_positive"] / totals["native_positive"]
    totals["new_target_histogram"] = pooled_hist.tolist()
    totals["new_target_fraction_points"] = (100 * pooled_hist / pooled_n).tolist()
    totals["new_target_fraction_bootstrap_sd_points"] = (100 * pooled_hist_r / pooled_nr[:, None]).std(axis=0, ddof=1).tolist()
    totals["observed_time_min_max_ns"] = [min(f["observed_time_min_max_ns"][0] for f in audit["files"].values()),
                                            max(f["observed_time_min_max_ns"][1] for f in audit["files"].values())]
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["| U_sym (МэВ) | Группы S | Все кластеры | Positive при 0,3 | Добавлено без floor | Из них direct-only / nonprimary-only | Прирост доли positives (п.п.) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for u, r in result["by_potential"].items():
        v, sd = r["delta_positive_fraction_points"][0], r["delta_positive_fraction_bootstrap_sd_points"][0]
        lines.append(f"| {u} | {r['groups']} | {r['clusters']} | {r['native_positive']} | {r['added_positive']} | {r['added_direct_only']} / {r['added_nonprimary_only']} | +{v:.4f} ± {sd:.4f} |")
    (HERE / "table.md").write_text("\n".join(lines) + "\n")
    plot(result)
    print(json.dumps(totals))


def plot(result):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), layout="constrained")
    x = np.arange(3)
    for j, label, color, marker, offset in ((0, "All added", "#2b2b2b", "o", -.12),
                                           (1, "Direct-primary qualifying hits only", "#1f77b4", "^", 0),
                                           (2, "Nonprimary qualifying hits only", "#b22222", "s", .12)):
        recs = list(result["by_potential"].values())
        axes[0].errorbar(x + offset, [r["delta_positive_fraction_points"][j] for r in recs],
                         yerr=[r["delta_positive_fraction_bootstrap_sd_points"][j] for r in recs],
                         fmt=marker, color=color, ms=6, capsize=3, label=label)
    axes[0].set_xticks(x, ["0", "18", "90"])
    axes[0].set_xlabel("U_sym (MeV)")
    axes[0].set_ylabel("Added positives / all clusters (percentage points)")
    upper = max(r["delta_positive_fraction_points"][0] + r["delta_positive_fraction_bootstrap_sd_points"][0]
                for r in result["by_potential"].values())
    axes[0].set_ylim(0, 1.3 * upper)  # Room above the data for the in-frame legend.
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    t = result["total"]
    vals = np.array(t["new_target_fraction_points"][:3])
    err = np.array(t["new_target_fraction_bootstrap_sd_points"][:3])
    # Registered 0.1-GeV bins describe where the removed-label population sits.
    axes[1].stairs(vals, [0, .1, .2, .3], color="#2b2b2b")
    axes[1].errorbar([.05, .15, .25], vals, yerr=err, fmt="none", color="#2b2b2b", capsize=3)
    axes[1].axvline(.3, color="#b22222", ls="--", lw=1)
    axes[1].set_xlim(-.01, .34)
    axes[1].set_ylim(0, float(max(vals + err) * 1.3))
    axes[1].set_xlabel("New maximum qualifying MCTrack energy (GeV)")
    axes[1].set_ylabel("New targets in bin / all clusters (percentage points)")
    axes[1].text(.04, .95, "All added targets < 0.3 GeV\nHigher bins unchanged", transform=axes[1].transAxes, va="top", fontsize=9)
    assert np.all(vals - err >= 0) and max(vals + err) < axes[1].get_ylim()[1]
    assert upper < axes[0].get_ylim()[1]
    (HERE / "figure_bounds.json").write_text(json.dumps({"added_fraction_upper_with_sd": upper,
        "left_ylim": list(axes[0].get_ylim()), "target_fraction_upper_with_sd": float(max(vals + err)),
        "right_ylim": list(axes[1].get_ylim())}, indent=2) + "\n")
    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(False)
    fig.suptitle("MC truth label diagnostics · corrected SMASH Xe+Cs, 2.87 GeV/nucleon\n"
                 "S286, unchanged graphs; no B selection; bars: SD of 400 shared group resamples", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"soft_origins.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    build()
