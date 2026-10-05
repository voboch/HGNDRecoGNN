#!/usr/bin/env python3
"""Neutron kinetic-energy spectra on the DCM-QGSM-SMM reference sample.

DCM-QGSM-SMM Xe+CsI at 3.26 AGeV is one sample with one equation of state, so
no U_sym response can be measured on it. What it gives is a generator-independent
check of the reconstruction: the same spectra the U_sym analysis is built from,
at truth and at reconstruction, on data the pipeline was never trained on.

Panels, all marked with their provenance as `AGENTS.md` requires:
  left   MC truth vs reconstructed neutron spectrum
  right  their ratio, which is the detector response the hardness ratio folds

The working point is not free: it is locked to the energy purity given by
`--purity`, the same convention the U_sym analysis uses, and the resulting
score threshold is printed and written into the caption rather than chosen by
eye.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _wilson(k, n, z=1.0):
    """Poisson-ish band on a bin count; symmetric sqrt(k) is wrong at low k."""
    k = np.asarray(k, dtype=float)
    return np.sqrt(np.maximum(k, 1.0))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", required=True, help="pred_clusters_dcm.pkl")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--purity", type=float, default=0.87,
                   help="cluster-count purity the score threshold is locked "
                        "to. This is the same lock the U_sym hardness search "
                        "applies, so the spectra here are drawn at the "
                        "working point the analysis actually uses. Note it is "
                        "count purity, not the energy-weighted purity the "
                        "reference benchmark quotes: at the same 87 %% target "
                        "the two give thresholds 0.72 and 0.65.")
    p.add_argument("--emax", type=float, default=6.0)
    p.add_argument("--bins", type=int, default=30)
    p.add_argument("--label", default="DCM-QGSM-SMM Xe+CsI 3.26 AGeV")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from reco_experiment import purity_locked_threshold

    c = pd.read_pickle(a.clusters)
    # the third return is the fraction of ALL clusters kept, not the recall
    thr, pur, sel_frac = purity_locked_threshold(c, a.purity)
    sel = c.cl_score > thr
    eff = float((sel & (c.cl_label == 1)).sum() / max((c.cl_label == 1).sum(), 1))
    print(f"count-purity-locked threshold = {thr:.3f}  ->  purity {pur:.3f}, "
          f"efficiency (recall) {eff:.3f}, keeps {100 * sel_frac:.1f} % of clusters")

    truth = c.loc[c.cl_label == 1, "e_true"].to_numpy(float)
    reco = c.loc[sel, "e_pred"].to_numpy(float)

    edges = np.linspace(0.0, a.emax, a.bins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    n_t, _ = np.histogram(truth, bins=edges)
    n_r, _ = np.histogram(reco, bins=edges)
    e_t, e_r = _wilson(n_t, None), _wilson(n_r, None)

    ok = n_t > 0
    ratio = np.full_like(ctr, np.nan, dtype=float)
    ratio_err = np.full_like(ctr, np.nan, dtype=float)
    ratio[ok] = n_r[ok] / n_t[ok]
    ratio_err[ok] = ratio[ok] * np.sqrt(
        np.maximum(n_r[ok], 1.0) / n_r[ok] ** 2 + np.maximum(n_t[ok], 1.0) / n_t[ok] ** 2)

    _plot(ctr, n_t, e_t, n_r, e_r, ratio, ratio_err, edges, thr, pur, eff, a)

    out = {
        "clusters": a.clusters, "label": a.label,
        "threshold": float(thr), "count_purity": float(pur),
        "efficiency_recall": float(eff), "selected_fraction": float(sel_frac),
        "n_truth_neutrons": int(n_t.sum()), "n_reco_neutrons": int(n_r.sum()),
        "edges": edges.tolist(), "truth": n_t.tolist(), "reco": n_r.tolist(),
    }
    jp = os.path.join(a.out, "dcm_spectra.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {jp}")
    return 0


def _plot(ctr, n_t, e_t, n_r, e_r, ratio, ratio_err, edges, thr, pur, eff, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_b_full import apply_style, hep_label, no_grid, FULL
    apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, FULL / 2.4))

    ax1.errorbar(ctr, n_t, yerr=e_t, fmt="o", ms=3.5, mfc="none", mew=0.9,
                 elinewidth=0.8, capsize=1.8, color="#2b2b2b",
                 label="MC truth, truth-matched clusters")
    ax1.errorbar(ctr, n_r, yerr=e_r, fmt="s", ms=3.5, mfc="none", mew=0.9,
                 elinewidth=0.8, capsize=1.8, color="#1f77b4",
                 label=rf"reconstructed, $q>{thr:.2f}$")
    ax1.set_xlabel(r"$E_{\rm kin}$  [GeV]")
    ax1.set_ylabel("neutron clusters / bin")
    ax1.set_yscale("log")
    _top = max(float(np.nanmax(n_t)), float(np.nanmax(n_r)))
    _bot = min([v for v in np.concatenate([n_t, n_r]) if v > 0] or [1.0])
    # one extra decade at the top so the in-frame label clears the spectrum
    ax1.set_ylim(_bot * 0.5, _top * 40)
    ax1.legend(loc="lower left", fontsize=6.5)

    ax2.axhline(1.0, color="0.5", lw=0.8, ls="--")
    m = np.isfinite(ratio)
    ax2.errorbar(ctr[m], ratio[m], yerr=ratio_err[m], fmt="o", ms=3.5,
                 mfc="none", mew=0.9, elinewidth=0.8, capsize=1.8,
                 color="#1f77b4")
    ax2.set_xlabel(r"$E_{\rm kin}$  [GeV]")
    ax2.set_ylabel("reconstructed / MC truth")
    ax2.text(0.97, 0.04, "reconstructed / MC truth", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=6.5, style="italic")

    # Appendix A: limits must contain the error-bar ends, not just the points
    lo = np.nanmin((ratio - ratio_err)[m])
    hi = np.nanmax((ratio + ratio_err)[m])
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    ax2.set_ylim(lo - 0.12 * span, hi + 0.12 * span)

    no_grid(ax1, ax2)
    hep_label(ax1, system=r"DCM-QGSM-SMM  Xe+CsI, 3.26$A$ GeV",
              extra=rf"$q>{thr:.2f}$  (purity {pur:.2f}, efficiency {eff:.2f})",
              loc="upper right")
    fig.tight_layout()
    path = os.path.join(a.out, "dcm_neutron_spectra.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
