#!/usr/bin/env python3
"""Neutron efficiency against true kinetic energy, with its two stages separated.

The note leans on "efficiency rises from 0.02 below 0.5 GeV to about 0.80 near
2-3 GeV" to argue that the conventional hardness denominator is unrecoverable,
and then quotes no plot for it. This is that plot.

Two curves, because they answer different questions and are often conflated:

  detection   P(selected cluster | neutron reaches the front face). Denominator
              from the vacs export, so this folds in cluster formation and the
              classifier together. It is the curve that decides whether an
              energy region is measurable at all.
  selection   P(selected | a truth-matched cluster already exists). Denominator
              from the clusters themselves, so it isolates the classifier from
              cluster formation. It is the curve an architecture comparison
              should move.

Everything here is MC truth: the abscissa is the true neutron kinetic energy
and both denominators are counted from truth information, which is available in
simulation only.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_b_full import apply_style, assert_contained, save, FULL, DATASETS
from make_spectra_figures import SCAN_LO, SCAN_HI, mark_scan_range

E_BINS = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.2, 5.2])


def wilson(k, n, z=1.0):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    d = 1 + z * z / np.maximum(n, 1)
    c = (p + z * z / (2 * np.maximum(n, 1))) / d
    h = z * np.sqrt(np.divide(p * (1 - p), np.maximum(n, 1)) +
                    z * z / (4 * np.maximum(n, 1) ** 2)) / d
    return np.where(n > 0, c - h, np.nan), np.where(n > 0, c + h, np.nan)


def selection_efficiency(pred_dir, purity):
    """P(selected | truth-matched cluster exists), per sample, against e_true."""
    from reco_experiment import purity_locked_threshold
    out = {}
    for tag in DATASETS:
        p = Path(pred_dir) / tag / "pred_clusters_smash.pkl"
        if not p.exists():
            continue
        c = pd.read_pickle(p)
        t, _, _ = purity_locked_threshold(c, purity)
        sig = c[(c.cl_label == 1) & (c.e_true > 0)]
        et = sig.e_true.to_numpy()
        passed = (sig.cl_score > t).to_numpy()
        idx = np.digitize(et, E_BINS) - 1
        n = np.array([(idx == i).sum() for i in range(len(E_BINS) - 1)], float)
        k = np.array([(passed & (idx == i)).sum() for i in range(len(E_BINS) - 1)], float)
        lo, hi = wilson(k, n)
        out[tag] = {"eff": np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0),
                    "lo": lo, "hi": hi, "n": n, "threshold": float(t)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--detection-json", required=True,
                   help="efficiency_correction.json, which carries the "
                        "arriving-neutron denominator")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--purity", type=float, default=0.7)
    a = p.parse_args()
    apply_style()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    sel = selection_efficiency(a.pred_dir, a.purity)
    det = json.load(open(a.detection_json))
    db = np.array(det["bins"]); de = np.array(det["eps"])
    # the detection binning runs to 6.5 GeV; clip it to the displayed range so
    # the step does not leave the panel
    XMAX = 5.2
    keep = db[:-1] < XMAX
    db = np.concatenate([db[:-1][keep], [min(db[1:][keep][-1], XMAX)]])
    de = de[keep]

    fig, ax = plt.subplots(figsize=(FULL / 1.5, FULL / 2.3))
    mark_scan_range(ax, label=True, y=0.985, va="top")

    ax.step(db, np.concatenate([de[:1], de]), where="pre",
            color="#2b2b2b", lw=1.4,
            label=r"detection: selected $\mid$ reaches front face")
    ctr = 0.5 * (E_BINS[:-1] + E_BINS[1:])
    for tag, st in DATASETS.items():
        if tag not in sel:
            continue
        s = sel[tag]
        ok = np.isfinite(s["eff"]) & (s["n"] >= 25)
        ax.errorbar(ctr[ok], s["eff"][ok],
                    yerr=[s["eff"][ok] - s["lo"][ok], s["hi"][ok] - s["eff"][ok]],
                    color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.4,
                    mfc="none", mew=0.8, elinewidth=0.7, capsize=1.5,
                    label=f"selection, {st['label']}")
    ax.axhline(1.0, color="#C8CED6", lw=0.7, ls=":")
    ax.set_xlabel(r"$E_{\mathrm{kin}}$ (MC truth)  [GeV]")
    ax.set_ylabel("efficiency")
    ax.set_title("Neutron efficiency against true energy (MC truth)", fontsize=7.5)
    ax.legend(loc="lower right", fontsize=6.2)
    ax.set_xlim(0.0, XMAX)
    lo_v = min(0.0, np.nanmin(de))
    ax.set_ylim(lo_v - 0.04, 1.06)
    assert_contained(ax, "eff")
    save(fig, out, "fig_efficiency_vs_ekin")

    json.dump({"e_bins": E_BINS.tolist(),
               "selection": {t: {k: (v.tolist() if hasattr(v, "tolist") else v)
                                 for k, v in sel[t].items()} for t in sel},
               "detection": {"bins": db.tolist(), "eps": de.tolist()}},
              open(out / "efficiency_vs_ekin.json", "w"), indent=1)
    print("efficiency figure complete")


if __name__ == "__main__":
    main()
