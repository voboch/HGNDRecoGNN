#!/usr/bin/env python3
"""Where the calibration cut the distributions it was calibrated on.

Parts I and II show the distributions clean, because the search range and the
selected window are *outputs* of Part III: drawing them earlier asks a reader
to accept a conclusion from a figure that has not yet derived it. This figure
puts them back, in the section that owns them, over the same three
distributions:

  left    the isovector signal against energy — why R_thr is searched where it
          is, and where the two species exchange their ordering;
  centre  the reconstructed neutron spectrum — what the two bands actually
          integrate, and how much yield the discarded gap costs;
  right   the detection and selection efficiency — whether both bands sit on
          the plateau, which is the condition the window has to satisfy.

The inputs are the JSON blobs the Part I and II figures already wrote, so the
curves here are the same numbers those figures drew, not a re-derivation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spectra-data", required=True,
                   help="spectra_figure_data.json from make_spectra_figures")
    p.add_argument("--efficiency-data", required=True,
                   help="efficiency_vs_ekin.json from plot_efficiency_vs_ekin")
    p.add_argument("--pred-dir", required=True,
                   help="cluster predictions; the saved spectra JSON records "
                        "only the working point, not the histogram")
    p.add_argument("--threshold", type=float, default=0.60)
    p.add_argument("--scan", required=True,
                   help="the calibrated reco scan, hardness_q060.json")
    p.add_argument("--scan-lo", type=float, default=1.0)
    p.add_argument("--scan-hi", type=float, default=3.0)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    sd = json.load(open(a.spectra_data))
    ed = json.load(open(a.efficiency_data))
    ch = json.load(open(a.scan))["chosen"]
    lo, hi = float(ch["lo_edge"]), float(ch["hi_edge"])
    print(f"selected window: E < {lo:.2f} | E >= {hi:.2f} GeV "
          f"(R_thr {ch['Rthr']:.2f}, delta {ch['delta']:.2f})")

    _plot(sd, ed, lo, hi, a)
    return 0


def _overlay(ax, lo, hi, scan_lo, scan_hi, label_gap=False, label_scan=False):
    """Search range behind, selected window in front."""
    ax.axvspan(scan_lo, scan_hi, color="#2b2b2b", alpha=0.07, lw=0, zorder=0)
    ax.axvspan(lo, hi, color="#8A8A8A", alpha=0.22, lw=0, zorder=0)
    for e in (lo, hi):
        ax.axvline(e, color="#2b2b2b", lw=0.9, ls="--", alpha=0.95)
    if label_scan:
        # left of the discarded gap, along the bottom, so it clears both the
        # spectrum peak and the gap label
        ax.text(scan_lo + 0.22 * (scan_hi - scan_lo), 0.035,
                r"$R_{\rm thr}$ searched", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.0, color="#555555")
    if label_gap:
        ax.text(0.5 * (lo + hi), 0.04, "discarded", rotation=90,
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=5.8, color="#333333")


def _plot(sd, ed, lo, hi, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_b_full import apply_style, hep_label, no_grid, FULL, DATASETS
    apply_style()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(FULL, FULL / 2.8))

    # ── left: the isovector signal, which is what sets where R_thr belongs ──
    rel = sd["np_relation"]
    ctr = np.asarray(rel["e_centres"], float)
    ax1.axhline(1.0, color="#888888", lw=0.8, ls="--")
    for tag, st in DATASETS.items():
        if tag not in rel["np_ratio"] or tag == "zeroSpot":
            continue
        y = np.asarray(rel["np_ratio"][tag], float)
        ax1.plot(ctr, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                 ms=3.0, mfc="none", mew=0.9, lw=1.0, label=st["label"])
    ax1.set_xlabel(r"$E_{\rm kin}$ (MC truth)  [GeV]")
    ax1.set_ylabel(r"$n/p$ relative to $U_{\rm sym}=0$")
    ax1.legend(loc="upper left", fontsize=5.8)

    # ── centre: what the two bands integrate ───────────────────────────────
    import pandas as pd
    edges = np.linspace(0.4, 4.0, 25)
    xc = 0.5 * (edges[:-1] + edges[1:])
    for tag, st in DATASETS.items():
        f = os.path.join(a.pred_dir, tag, "pred_clusters_smash.pkl")
        if not os.path.exists(f):
            continue
        c = pd.read_pickle(f)
        sel = c[c.cl_score > a.threshold]
        n_ev = c.Row.nunique()
        y, _ = np.histogram(sel.e_pred.to_numpy(float), bins=edges)
        dens = y / n_ev / np.diff(edges)
        ax2.plot(xc, dens, color=st["color"], ls=st["ls"], marker=st["marker"],
                 ms=3.0, mfc="none", mew=0.9, lw=1.0, label=st["label"])
    ax2.set_xlabel(r"$E_{\rm reco}$ (reconstructed)  [GeV]")
    ax2.set_ylabel(r"clusters per event  [GeV$^{-1}$]")

    # ── right: whether both bands sit on the efficiency plateau ────────────
    eb = np.asarray(ed["e_bins"], float)
    dbins = np.asarray(ed["detection"]["bins"], float)
    deps = np.asarray(ed["detection"]["eps"], float)
    ax3.step(dbins, np.concatenate([deps[:1], deps]), where="pre",
             color="#2b2b2b", lw=1.3, label="detection")
    sel = ed["selection"]
    ectr = 0.5 * (eb[:-1] + eb[1:])
    for tag, st in DATASETS.items():
        if tag not in sel:
            continue
        q = sel[tag]
        eff = np.asarray(q["eff"], float)
        n = min(len(ectr), len(eff))
        qlo = np.asarray(q["lo"], float)[:n]
        qhi = np.asarray(q["hi"], float)[:n]
        ax3.errorbar(ectr[:n], eff[:n],
                     yerr=[np.clip(eff[:n] - qlo, 0, None),
                           np.clip(qhi - eff[:n], 0, None)],
                     fmt=st["marker"], ms=3.0, mfc="none", mew=0.9,
                     elinewidth=0.7, capsize=1.5, color=st["color"], ls="none")
    ax3.set_xlabel(r"$E_{\rm kin}$ (MC truth)  [GeV]")
    ax3.set_ylabel("efficiency")
    ax3.legend(loc="lower right", fontsize=5.8)
    ax3.set_ylim(0.0, 1.12)

    for k, ax in enumerate((ax1, ax2, ax3)):
        _overlay(ax, lo, hi, a.scan_lo, a.scan_hi,
                 label_gap=(k == 1), label_scan=(k == 1))
    no_grid(ax1, ax2, ax3)
    hep_label(ax3, extra=r"$U_{\rm sym}$ = 0, 18, 90 MeV;  $q>0.60$",
              loc="upper left", fontsize=6.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, f"fig_calibration_context.{ext}"),
                    dpi=200)
    print(f"wrote {a.out_dir}/fig_calibration_context.png")


if __name__ == "__main__":
    sys.exit(main())
