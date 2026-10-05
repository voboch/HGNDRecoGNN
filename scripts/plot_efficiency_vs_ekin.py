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
from plot_b_full import hep_label, apply_style, assert_contained, save, FULL, DATASETS
from make_spectra_figures import WINDOW_LO, WINDOW_HI

E_BINS = np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.2, 5.2])


def wilson(k, n, z=1.0):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    d = 1 + z * z / np.maximum(n, 1)
    c = (p + z * z / (2 * np.maximum(n, 1))) / d
    h = z * np.sqrt(np.divide(p * (1 - p), np.maximum(n, 1)) +
                    z * z / (4 * np.maximum(n, 1) ** 2)) / d
    return np.where(n > 0, c - h, np.nan), np.where(n > 0, c + h, np.nan)



def band_assignment_efficiency(pred_dir, threshold, lo_edge, hi_edge, bins):
    """Fraction of selected signal clusters reconstructed into the right band.

    Detection and selection efficiency are both binned in MC-truth energy, so
    neither answers the question the hardness ratio asks: does a neutron of a
    given true energy end up *counted in the band it belongs to*? That depends
    on the reconstructed energy, and it is where the energy resolution enters
    the observable.

    A cluster is correctly assigned when its reconstructed energy falls in the
    same region of the calibrated window as its true energy — below `lo_edge`,
    at or above `hi_edge`, or in the discarded gap between them.

    This measures migration *out* of the correct region only. Clusters migrating
    *in* from the gap contaminate a band without being counted here, and that
    term has to be taken from the response matrix rather than from this curve.
    """
    def _band(e):
        b = np.zeros(len(e), dtype=int)
        b[e < lo_edge] = 1
        b[e >= hi_edge] = 2
        return b

    out = {}
    for tag in DATASETS:
        f = os.path.join(pred_dir, tag, "pred_clusters_smash.pkl")
        if not os.path.exists(f):
            continue
        c = pd.read_pickle(f)
        s = c[(c.cl_score > threshold) & (c.cl_label == 1)]
        et = s.e_true.to_numpy(float)
        ok = _band(et) == _band(s.e_pred.to_numpy(float))
        idx = np.digitize(et, bins) - 1
        eff, lo_e, hi_e, n = [], [], [], []
        for i in range(len(bins) - 1):
            m = idx == i
            k = int(m.sum())
            n.append(k)
            if k < 25:
                eff.append(np.nan); lo_e.append(np.nan); hi_e.append(np.nan)
                continue
            q = float(ok[m].mean())
            # Wilson, as the selection curve uses
            z, den = 1.0, 1.0 + 1.0 / k
            ctr = (q + 1.0 / (2 * k)) / den
            half = z / den * np.sqrt(q * (1 - q) / k + 1.0 / (4 * k * k))
            eff.append(q); lo_e.append(max(q - (ctr - half), 0.0))
            hi_e.append(max((ctr + half) - q, 0.0))
        out[tag] = {"eff": eff, "lo": lo_e, "hi": hi_e, "n": n}
    return out


def selection_efficiency(pred_dir, purity, threshold=None):
    """P(selected | truth-matched cluster exists), per sample, against e_true."""
    from reco_experiment import purity_locked_threshold
    out = {}
    for tag in DATASETS:
        p = Path(pred_dir) / tag / "pred_clusters_smash.pkl"
        if not p.exists():
            continue
        c = pd.read_pickle(p)
        t = float(threshold) if threshold is not None else purity_locked_threshold(c, purity)[0]
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
    p.add_argument("--scan", default=None,
                   help="calibrated reco scan JSON; enables the band-assignment "
                        "and final-efficiency curves")
    p.add_argument("--mark-calibration", action="store_true",
                   help="overlay the Part III window; off by default")
    p.add_argument("--threshold", type=float, default=None,
                   help="Fixed score cut applied to every sample, overriding --purity. "
                        "At q > 0.60 the three samples agree to 0.14 pp in count "
                        "purity and 0.08 pp in recall, so the fixed cut does not "
                        "import a per-sample purity difference. 0.60 is the 87 "
                        "percent energy-purity working point.")
    a = p.parse_args()
    apply_style()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    bae_export = None
    sel = selection_efficiency(a.pred_dir, a.purity, a.threshold)
    thr_used = a.threshold if a.threshold is not None else a.purity
    det = json.load(open(a.detection_json))
    db = np.array(det["bins"]); de = np.array(det["eps"])
    # the detection binning runs to 6.5 GeV; clip it to the displayed range so
    # the step does not leave the panel
    XMAX = 5.2
    keep = db[:-1] < XMAX
    db = np.concatenate([db[:-1][keep], [min(db[1:][keep][-1], XMAX)]])
    de = de[keep]

    fig, ax = plt.subplots(figsize=(FULL / 1.3, FULL / 2.1))
    # the selected window is an output of Part III; this figure belongs to
    # Part II, so it is drawn clean unless explicitly asked otherwise
    if a.mark_calibration:
        ax.axvspan(WINDOW_LO, WINDOW_HI, color="#8A8A8A", alpha=0.16,
                   lw=0, zorder=0)
        for e in (WINDOW_LO, WINDOW_HI):
            ax.axvline(e, color="#2b2b2b", lw=0.8, ls="--", alpha=0.9)
        ax.text(0.5 * (WINDOW_LO + WINDOW_HI), 0.985, "discarded gap",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.0, color="#444444")

    ax.step(db, np.concatenate([de[:1], de]), where="pre",
            color="#2b2b2b", lw=1.4,
            label=r"detection: cluster selected $\mid$ neutron arrives")
    ctr = 0.5 * (E_BINS[:-1] + E_BINS[1:])
    # One selection curve, not three: the samples are indistinguishable here
    # and three copies crowd out the decomposition the figure is for. It is
    # also the least relevant of the curves — it is binned in true energy and
    # so says nothing about which band a cluster lands in.
    for tag, st in list(DATASETS.items())[:1]:
        if tag not in sel:
            continue
        s = sel[tag]
        ok = np.isfinite(s["eff"]) & (s["n"] >= 25)
        ax.errorbar(ctr[ok], s["eff"][ok],
                    yerr=[s["eff"][ok] - s["lo"][ok], s["hi"][ok] - s["eff"][ok]],
                    color="#7F7F7F", ls="-", marker="o", ms=3.0,
                    mfc="none", mew=0.8, elinewidth=0.7, capsize=1.5,
                    label=r"selection: classifier only (all $U_{\rm sym}$ alike)")
    # ── band assignment and the product that actually gates the observable ──
    if a.scan:
        ch = json.load(open(a.scan))["chosen"]
        lo_e, hi_e = float(ch["lo_edge"]), float(ch["hi_edge"])
        bae = band_assignment_efficiency(a.pred_dir, thr_used, lo_e, hi_e,
                                         E_BINS)
        # detection evaluated on the same binning, so the product is defined
        kk = np.clip(np.digitize(ctr, db) - 1, 0, len(de) - 1)
        det_on_bins = de[kk]
        # One curve per U_sym sample, not an average. Detection is common to
        # the three by construction, so any sample dependence of the final
        # efficiency lives entirely in this term: a difference here is a
        # systematic on the response, not a property of the detector.
        for j, (tag, st) in enumerate(DATASETS.items()):
            if tag not in bae:
                continue
            be = np.array(bae[tag]["eff"], float)
            bl = np.array(bae[tag]["lo"], float)
            bh = np.array(bae[tag]["hi"], float)
            good = np.isfinite(be)
            ax.errorbar(ctr[good], be[good], yerr=[bl[good], bh[good]],
                        color=st["color"], ls="none", marker="D", ms=3.2,
                        mfc="none", mew=0.9, elinewidth=0.7, capsize=1.5,
                        label="band assignment, " + st["label"])
            ax.plot(ctr[good], (be * det_on_bins)[good], color=st["color"],
                    ls=st["ls"], marker="v", ms=3.2, mfc="none", mew=0.9,
                    lw=1.0, alpha=0.9,
                    label=("final = detection x band assignment"
                           if j == 0 else "_nolegend_"))
        bae_export = bae
        for e in (lo_e, hi_e):
            ax.axvline(e, color="#2b2b2b", lw=0.8, ls="--", alpha=0.75)
        ax.axvspan(lo_e, hi_e, color="#8A8A8A", alpha=0.13, lw=0, zorder=0)

    ax.axhline(1.0, color="#C8CED6", lw=0.7, ls=":")
    ax.set_xlabel(r"$E_{\mathrm{kin}}$ (MC truth)  [GeV]")
    ax.set_ylabel("efficiency")
    # titles belong in the caption; the frame carries the label instead
    hep_label(ax, extra=r"MC truth;  $U_{\rm sym}$ = 0, 18, 90 MeV",
              loc="upper right")
    ax.legend(loc="lower right", fontsize=5.8)
    ax.set_xlim(0.0, XMAX)
    lo_v = min(0.0, np.nanmin(de))
    ax.set_ylim(lo_v - 0.04, 1.06)
    assert_contained(ax, "eff")
    save(fig, out, "fig_efficiency_vs_ekin_v2")

    json.dump({"e_bins": E_BINS.tolist(),
               "selection": {t: {k: (v.tolist() if hasattr(v, "tolist") else v)
                                 for k, v in sel[t].items()} for t in sel},
               "detection": {"bins": db.tolist(), "eps": de.tolist()},
               "band_assignment": (bae_export if a.scan else None)},
              open(out / "efficiency_vs_ekin.json", "w"), indent=1)
    print("efficiency figure complete")


if __name__ == "__main__":
    main()
