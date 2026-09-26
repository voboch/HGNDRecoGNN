#!/usr/bin/env python3
"""Figures for the GNN reconstruction experiment.

Follows docs/plotting_style_protocol.md, including the Appendix A rule that
axis limits are derived from the data and every mark is asserted to lie inside
its panel.
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
from reco_experiment import (fit_energy_calibration, apply_calibration,
                             purity_locked_threshold, ELO, EHI)

RUNGS = [
    ("R_n_truth_primary",  "primary $n$, HGND angular band"),
    ("R_n_truth_reaching", "$n$ reaching the detector"),
    ("R_n_sig_trueE",      "selected clusters, true $E$"),
    ("R_n_reco_calib",     "selected clusters, reconstructed $E$"),
    ("R_n_reco",           "selected clusters, uncalibrated $E$"),
]


def fig_ladder(cmp_, out_dir):
    """Where the measurement is lost, rung by rung."""
    rows = [(lab, cmp_[k]) for k, lab in RUNGS if k in cmp_]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(FULL, FULL / 2.5))
    rel = np.array([r["rel"] for _, r in rows]) * 100
    err = np.array([r["err"] for _, r in rows]) * 100
    col = ["#2b2b2b" if v > 0 else "#B22222" for v in rel]
    ax.axvline(0.0, color="#2b2b2b", lw=0.9)
    for i, (yy, v, e, c) in enumerate(zip(y, rel, err, col)):
        ax.errorbar(v, yy, xerr=e, color=c, marker="o", ms=4.5, mfc="none",
                    mew=1.1, elinewidth=0.9, capsize=2.2, ls="none")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=7.5)
    ax.set_xlabel(r"$R_n$ at $S_{\mathrm{pot}}=90$ relative to $0$ MeV  [%]")
    ax.set_title("The truth-level signal does not survive the energy estimate",
                 fontsize=7.5)
    lo = float((rel - err).min()); hi = float((rel + err).max())
    m = (hi - lo) * 0.12
    ax.set_xlim(lo - m, hi + m)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    assert_contained(ax, "ladder")
    save(fig, out_dir, "fig1_reco_ladder")


def fig_response(pred_dir, out_dir, calib="defaultSpot", purity=0.7):
    """Energy response vs true energy, per sample -- the mechanism."""
    cal = fit_energy_calibration(
        pd.read_pickle(Path(pred_dir) / calib / "pred_clusters_smash.pkl"))
    edges = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0, 4.0])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    fig, (a, b) = plt.subplots(2, 1, figsize=(FULL, FULL / 1.75), sharex=True,
                               gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.07})
    curves, errs = {}, {}
    for tag, st in DATASETS.items():
        p = Path(pred_dir) / tag / "pred_clusters_smash.pkl"
        if not p.exists():
            continue
        c = pd.read_pickle(p)
        t, _, _ = purity_locked_threshold(c, purity)
        s = c[(c.cl_score > t) & (c.cl_label == 1)]
        et = s.e_true.to_numpy()
        res = apply_calibration(s.e_pred.to_numpy(), cal) - et
        idx = np.digitize(et, edges) - 1
        m = np.full(len(ctr), np.nan); e = np.full(len(ctr), np.nan)
        for k in range(len(ctr)):
            sel = idx == k
            if sel.sum() > 30:
                m[k] = res[sel].mean()
                e[k] = res[sel].std(ddof=1) / np.sqrt(sel.sum())
        curves[tag], errs[tag] = m, e
        a.errorbar(ctr, m, yerr=e, color=st["color"], ls=st["ls"], marker=st["marker"],
                   ms=3.5, mfc="none", mew=0.8, elinewidth=0.7, capsize=1.5,
                   label=st["label"])
    a.axhline(0.0, color="#6B727B", lw=0.7, ls=":")
    a.axvline(EHI, color="#6B727B", lw=0.7, ls="--")
    a.set_ylabel(r"$\langle E_{\mathrm{reco}}-E_{\mathrm{true}}\rangle$  [GeV]")
    a.legend(loc="lower left", fontsize=6.5)
    a.set_title(r"One estimator, three samples: the response is not sample-independent",
                fontsize=7.5)
    allv = np.concatenate([np.concatenate([curves[k] + errs[k], curves[k] - errs[k]])
                           for k in curves])
    allv = allv[np.isfinite(allv)]
    pad = (allv.max() - allv.min()) * 0.12
    a.set_ylim(allv.min() - pad, allv.max() + pad)

    ref = "zeroSpot"
    fin = []
    for tag, st in DATASETS.items():
        if tag not in curves or tag == ref:
            continue
        d = (curves[tag] - curves[ref]) * 1000
        de = np.hypot(errs[tag], errs[ref]) * 1000
        b.errorbar(ctr, d, yerr=de, color=st["color"], ls=st["ls"], marker=st["marker"],
                   ms=3.5, mfc="none", mew=0.8, elinewidth=0.7, capsize=1.5)
        fin += [x for x in np.concatenate([d + de, d - de]) if np.isfinite(x)]
    b.axhline(0.0, color="#2b2b2b", lw=0.9)
    b.axvline(EHI, color="#6B727B", lw=0.7, ls="--")
    b.set_xlabel(r"$E_{\mathrm{true}}$  [GeV]")
    b.set_ylabel(r"difference vs $S_{\mathrm{pot}}=0$" "\n" r"[MeV]", fontsize=6.5)
    if fin:
        mm = max(abs(np.array(fin))) * 1.18
        b.set_ylim(-mm, mm)
    b.set_xlim(edges[0], edges[-1])
    for ax, nm in ((a, "resp_a"), (b, "resp_b")):
        assert_contained(ax, nm)
    save(fig, out_dir, "fig2_energy_response")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    apply_style()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    d = json.load(open(a.experiment))
    fig_ladder(d["comparison_90_vs_0"], out)
    fig_response(a.pred_dir, out)
    print("figures complete")


if __name__ == "__main__":
    main()
