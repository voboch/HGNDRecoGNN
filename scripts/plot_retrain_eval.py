#!/usr/bin/env python3
"""Figures for the retrained-reconstruction evaluation.

Follows docs/plotting_style_protocol.md: printed size, 8 pt base type, vector
PDF plus a 200 dpi PNG, colour never the only channel, and axis limits derived
from the data with panel containment asserted.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_b_full import apply_style, assert_contained, save, FULL

OLD = {"color": "#B22222", "marker": "s", "ls": "--", "label": "previous checkpoint"}
NEW = {"color": "#1f77b4", "marker": "^", "ls": "-", "label": "retrained (4 epochs)"}
REFC = "#2b2b2b"


def fig_energy(old, new, out_dir):
    """Linearity and resolution against true energy, with the 10 % targets."""
    fig, (a, b) = plt.subplots(2, 1, figsize=(FULL, FULL / 1.55), sharex=True,
                               gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.08})
    allv = []
    for d, st in ((old, OLD), (new, NEW)):
        e = np.array([r["e_mid"] for r in d["energy_performance"]])
        lin = np.array([r["linearity"] for r in d["energy_performance"]]) * 100
        res = np.array([r["resolution"] for r in d["energy_performance"]]) * 100
        a.plot(e, lin, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5,
               mfc="none", mew=0.9, label=st["label"])
        b.plot(e, res, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5,
               mfc="none", mew=0.9)
        allv += list(lin)
    a.axhspan(-10, 10, color=REFC, alpha=0.10, lw=0)
    a.axhline(0, color=REFC, lw=0.8)
    a.set_ylabel(r"$(E_{\mathrm{reco}}-E_{\mathrm{true}})/E_{\mathrm{true}}$  [%]")
    a.legend(loc="upper right", fontsize=6.5)
    a.set_title("Energy linearity and resolution against the reference targets "
                "(shaded: 10 %)", fontsize=7)
    m = (max(allv) - min(allv)) * 0.10
    a.set_ylim(min(allv) - m, max(allv) + m)
    b.axhspan(0, 10, color=REFC, alpha=0.10, lw=0)
    b.set_ylabel("resolution  [%]")
    b.set_xlabel(r"$E_{\mathrm{true}}$  [GeV]")
    rr = [r["resolution"] * 100 for d in (old, new) for r in d["energy_performance"]]
    b.set_ylim(0, max(rr) * 1.15)
    b.set_xlim(0.4, 5.2)
    for ax, nm in ((a, "lin"), (b, "res")):
        assert_contained(ax, nm)
    save(fig, out_dir, "fig1_energy_performance")


def fig_working_point(old, new, out_dir):
    """Purity against efficiency, energy weighted, with the reference point."""
    fig, ax = plt.subplots(figsize=(FULL / 1.7, FULL / 2.2))
    pts = []
    for d, st in ((old, OLD), (new, NEW)):
        s = d["threshold_scan"]
        eff = np.array([r["efficiency_energy"] for r in s])
        pur = np.array([r["purity_energy"] for r in s])
        ok = np.isfinite(eff) & np.isfinite(pur)
        ax.plot(eff[ok], pur[ok], color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=3, mfc="none", mew=0.8, label=f"{st['label']} (AUC {d['roc_auc']:.3f})")
        pts += list(zip(eff[ok], pur[ok]))
    ax.plot([0.80], [0.87], marker="*", ms=11, color=REFC, ls="none",
            label="reference working point")
    ax.set_xlabel("efficiency  $E_{\\mathrm{true}}/E_{\\mathrm{all\\,signal}}$")
    ax.set_ylabel("purity  $1-E_{\\mathrm{fake}}/E_{\\mathrm{pred}}$")
    # limits derived from the plotted points plus the reference marker
    pts.append((0.80, 0.87))
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    mx = (xs.max() - xs.min()) * 0.08; my = (ys.max() - ys.min()) * 0.10
    ax.set_xlim(xs.min() - mx, xs.max() + mx)
    ax.set_ylim(ys.min() - my, min(ys.max() + my, 1.02))
    ax.legend(loc="lower left", fontsize=6.2)
    assert_contained(ax, "wp")
    save(fig, out_dir, "fig2_working_point")


def fig_sensitivity(scan, out_dir, truth_effect=4.27):
    """Test-set R_n against S_pot, with what the split could have resolved."""
    t = scan.get("test")
    if not t:
        return
    u = np.array([0.0, 18.0, 90.0])
    tags = ["zeroSpot", "defaultSpot", "bigSpot"]
    R = np.array([t["R"][k] for k in tags])
    rel = (R / R[0] - 1) * 100
    err = t["err"] * 100
    fig, ax = plt.subplots(figsize=(FULL / 1.7, FULL / 2.2))
    ax.axhline(0, color=REFC, lw=0.8)
    ax.plot(u, truth_effect * u / 90.0, color=REFC, ls=":", lw=1.1,
            label=f"truth-level response (+{truth_effect:.2f} % at 90 MeV)")
    ax.errorbar(u, rel, yerr=[0, err / np.sqrt(2), err], color=NEW["color"], ls="none",
                marker="^", ms=5, mfc="none", mew=1.1, elinewidth=0.9, capsize=2.5,
                label="reconstructed, test split")
    ax.set_xlabel(r"$S_{\mathrm{pot}}$  [MeV]")
    ax.set_ylabel(r"$R_n$ relative to $S_{\mathrm{pot}}=0$  [%]")
    ax.set_title("Working point fixed on train+val; test split is "
                 f"underpowered ({truth_effect/err:.2f}$\\sigma$ expected)", fontsize=6.8)
    lo = min(rel.min() - err, -2); hi = max(rel.max() + err, truth_effect + 2)
    pad = (hi - lo) * 0.12
    ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-5, 95)
    ax.legend(loc="upper left", fontsize=6.2)
    assert_contained(ax, "sens")
    save(fig, out_dir, "fig3_test_sensitivity")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new", required=True)
    p.add_argument("--old", required=True)
    p.add_argument("--scan", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    apply_style()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    new, old = json.load(open(a.new)), json.load(open(a.old))
    fig_energy(old, new, out)
    fig_working_point(old, new, out)
    fig_sensitivity(json.load(open(a.scan)), out)
    print("figures complete")


if __name__ == "__main__":
    main()
