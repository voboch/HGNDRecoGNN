"""Cluster reconstruction performance: purity, efficiency, linearity, resolution.

Definitions follow the 26.02.2026 Dombay report so the numbers are directly
comparable with its benchmarks:

  energy weighted   purity     = 1 - E_fake / E_predicted
                    efficiency = E_true(selected signal) / E_true(all signal)
  count based       purity     = N_selected_signal / N_selected
                    efficiency = N_selected_signal / N_signal

Reference values on 3.2 AGeV Xe+CsI DCM-QGSM-SMM: ROC AUC about 0.97, a working
point near 80 % efficiency at 87 % purity, linearity within 10 % from 0.7 to
about 5 GeV, and resolution under 10 % at feasible energies.

Linearity is the fractional departure of the median reconstructed energy from
the true energy in bins of true energy; resolution is the half 16-84 percentile
width of (E_reco - E_true)/E_true, which is insensitive to the tails that a
plain standard deviation would let dominate.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

E_BINS = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0, 5.0, 6.5])


def roc_auc(score, label):
    order = np.argsort(score)
    lab = np.asarray(label)[order]
    n_pos, n_neg = lab.sum(), (1 - lab).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = np.empty(len(lab), float)
    s = np.asarray(score)[order]
    i = 0
    while i < len(s):                      # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[lab == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_scan(cl, thresholds):
    s = cl.cl_score.to_numpy()
    y = cl.cl_label.to_numpy().astype(int)
    ep = cl.e_pred.to_numpy()
    et = cl.e_true.to_numpy()
    n_sig, e_sig = int(y.sum()), float(et[y == 1].sum())
    rows = []
    for t in thresholds:
        sel = s > t
        n_sel, n_ss = int(sel.sum()), int((sel & (y == 1)).sum())
        e_pred_sel = float(ep[sel].sum())
        e_fake = float(ep[sel & (y == 0)].sum())
        e_true_sel = float(et[sel & (y == 1)].sum())
        rows.append({
            "threshold": float(t),
            "n_selected": n_sel, "n_selected_signal": n_ss,
            "purity_count": n_ss / n_sel if n_sel else np.nan,
            "efficiency_count": n_ss / n_sig if n_sig else np.nan,
            "purity_energy": 1 - e_fake / e_pred_sel if e_pred_sel > 0 else np.nan,
            "efficiency_energy": e_true_sel / e_sig if e_sig > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def energy_performance(cl, threshold, bins=E_BINS):
    s = cl[(cl.cl_label == 1) & (cl.cl_score > threshold) & (cl.e_true > 0)]
    et, ep = s.e_true.to_numpy(), s.e_pred.to_numpy()
    idx = np.digitize(et, bins) - 1
    rows = []
    for k in range(len(bins) - 1):
        m = idx == k
        if m.sum() < 25:
            continue
        r = (ep[m] - et[m]) / et[m]
        lo, med, hi = np.percentile(r, [16, 50, 84])
        rows.append({
            "e_lo": float(bins[k]), "e_hi": float(bins[k + 1]),
            "e_mid": float(0.5 * (bins[k] + bins[k + 1])),
            "n": int(m.sum()),
            "median_e_reco": float(np.median(ep[m])),
            "linearity": float(med),                     # (reco-true)/true, median
            "resolution": float(0.5 * (hi - lo)),        # half 16-84 width
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", nargs="+", required=True,
                   help="pred_clusters_*.pkl files, pooled for the performance numbers")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=None,
                   help="working point for the energy numbers; default picks the "
                        "threshold closest to the reference 87 %% purity")
    a = p.parse_args()

    cl = pd.concat([pd.read_pickle(f) for f in a.clusters], ignore_index=True)
    print(f"clusters: {len(cl):,}   signal: {int(cl.cl_label.sum()):,} "
          f"({cl.cl_label.mean()*100:.1f} %)")
    auc = roc_auc(cl.cl_score.to_numpy(), cl.cl_label.to_numpy())
    print(f"ROC AUC = {auc:.4f}   (reference ~0.97)")

    scan = threshold_scan(cl, np.round(np.arange(0.05, 0.96, 0.05), 2))
    print("\n  threshold   purity(N)  eff(N)   purity(E)  eff(E)")
    for _, r in scan.iterrows():
        print(f"  {r.threshold:>9.2f}{r.purity_count:>11.3f}{r.efficiency_count:>8.3f}"
              f"{r.purity_energy:>12.3f}{r.efficiency_energy:>8.3f}")

    if a.threshold is None:
        i = (scan.purity_energy - 0.87).abs().idxmin()
        thr = float(scan.loc[i, "threshold"])
        print(f"\n  working point closest to 87 % energy purity: threshold {thr:.2f}"
              f"  -> purity {scan.loc[i,'purity_energy']:.3f}, "
              f"efficiency {scan.loc[i,'efficiency_energy']:.3f}")
    else:
        thr = a.threshold

    ep = energy_performance(cl, thr)
    print(f"\n  energy performance at threshold {thr:.2f}")
    print(f"  {'E_true [GeV]':<14}{'N':>8}{'linearity':>12}{'resolution':>12}")
    for _, r in ep.iterrows():
        flag = "" if abs(r.linearity) <= 0.10 else "  >10%"
        flag += "" if r.resolution <= 0.10 else "  res>10%"
        print(f"  {f'{r.e_lo:.1f}-{r.e_hi:.1f}':<14}{r.n:>8,}"
              f"{r.linearity*100:>+11.1f}%{r.resolution*100:>11.1f}%{flag}")
    ok = ep[(ep.e_mid > 0.7)]
    if len(ok):
        print(f"\n  above 0.7 GeV: max |linearity| = {ok.linearity.abs().max()*100:.1f} %, "
              f"max resolution = {ok.resolution.max()*100:.1f} %"
              f"   (reference: both under 10 %)")
    out = {"n_clusters": int(len(cl)), "roc_auc": auc, "working_threshold": thr,
           "threshold_scan": scan.to_dict("records"),
           "energy_performance": ep.to_dict("records")}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
