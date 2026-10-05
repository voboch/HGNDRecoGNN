#!/usr/bin/env python3
"""Final U_sym sensitivity from the minimal set of observables.

The observable list had grown to seven — raw yield, neutron hardness, three
shape variants inside 1--2 GeV, an in-band mean energy, and hybrids of each.
Most were introduced to rescue a null and none of them was calibrated the way
the hardness window eventually was, so the table compared tuned quantities
against untuned ones. This reduces it to the two that carry the measurement,
plus the two truth references they are read against:

  n/p yield      the integrated neutron-to-proton yield ratio. This was the
                 baseline observable of the study, so it stays in the main
                 table as the reference the hardness ratios must beat rather
                 than being dropped. It carries no window, so there is nothing
                 to calibrate on it.
  R_n            reconstructed neutron hardness. What the HGND delivers alone.
  R_n / R_p      the same neutron hardness over the truth proton hardness.
                 The HGND does not measure protons, so this is a bound on what
                 it plus an external proton measurement could reach.
  truth R_n      the same observable with no detector, the dilution reference.
  truth R_n/R_p  the ideal double ratio, the upper bound of the whole chain.

All four are calibrated identically: the same (Rthr, delta) grid, the same
admissibility rules, the same development/held-out split, and the window fixed
on development units before the held-out units are opened once.

Why the double ratio is worth the external measurement: at a common window the
proton hardness moves the *opposite* way to the neutron hardness (-5.97 %
against +6.12 % for 90 against 0 MeV), so the isovector response adds in the
ratio instead of cancelling. That is the physics reason the double ratio is
worth roughly twice the single ratio, not a statistical accident.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# (key, label, what it needs, whether the HGND can produce it alone)
# The integrated yields are not in this comparison. R is itself a differential
# yield ratio -- counts above a threshold over counts below it -- so placing an
# integrated yield beside it compares an observable against its own integral
# rather than against a competitor. The yield's insensitivity is established at
# truth in Part I and is the reason R was adopted; it is not a result here.
# Set INCLUDE_BASELINES to restore them.
INCLUDE_BASELINES = False
BASELINES = [
    ("truth_np", r"truth $Y_n/Y_p$", "MC truth"),
    ("reco_yield", r"reco $\widehat Y_n$", "HGND alone"),
    ("hybrid_yield", r"reco $\widehat Y_n$ / truth $Y_p$", "HGND + proton measurement"),
]

ROWS = [
    ("hardness_truth", r"truth $R_n$", "MC truth", False),
    ("truth_dr", r"truth $R_n/R_p$", "MC truth", False),
    ("hardness_q060", r"reco $R_n$", "HGND alone", True),
    ("hybrid_q060", r"reco $R_n$ / truth $R_p$", "HGND + proton measurement", False),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--obs-dir", required=True,
                   help="directory holding the four calibrated scan JSONs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--threshold", type=float, default=0.60,
                   help="the score cut the reconstructed rows were built at")
    p.add_argument("--pred-dir", default=None,
                   help="cluster predictions, for the common-window panel")
    p.add_argument("--hist-dir", default=None,
                   help="truth histograms, for the common-window panel")
    p.add_argument("--common-window", nargs=2, type=float, default=[1.60, 2.40],
                   metavar=("LO", "HI"),
                   help="Each row above sits at its own optimised window, so "
                        "their responses are not comparable with each other: "
                        "the optimiser maximises significance, and a window "
                        "with more statistics can win with a smaller response. "
                        "This second reading puts every quantity on one window "
                        "so the dilution from truth to reconstruction is "
                        "visible. The default is the window the hybrid "
                        "selected.")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    table = []
    bpath = os.path.join(a.obs_dir, "yield_baselines.json")
    if INCLUDE_BASELINES and os.path.exists(bpath):
        bl = json.load(open(bpath))["baselines"]
        for key, label, needs in BASELINES:
            if key not in bl:
                continue
            r = bl[key]
            table.append({
                "key": key, "label": label, "needs": needs,
                "baseline": True, "hgnd_only": needs == "HGND alone",
                "Rthr": None, "delta": None, "lo_edge": None, "hi_edge": None,
                "dev_rel": None, "dev_sigma": None,
                "rel": r["rel"], "err": r["err"], "sigma": r["sigma"],
                "ordered": bool(r["ordered"]),
                "n_admissible": None, "n_ordered": None, "R": r["R"],
                "R_err": r.get("R_err"),
            })
    for key, label, needs, hgnd_only in ROWS:
        path = os.path.join(a.obs_dir, f"{key}.json")
        if not os.path.exists(path):
            print(f"  missing {path} — skipping {label}")
            continue
        d = json.load(open(path))
        c, h = d["chosen"], d["held_out"]
        n_ord = int(sum(1 for r in d["surface"] if r["ordered"]))
        table.append({
            "key": key, "label": label, "needs": needs, "hgnd_only": hgnd_only,
            "baseline": False,
            "Rthr": c["Rthr"], "delta": c["delta"],
            "lo_edge": c["lo_edge"], "hi_edge": c["hi_edge"],
            "dev_rel": c["rel"], "dev_sigma": c["sigma"],
            "slope": h.get("slope"),
            "rel": h["rel"], "err": h["err"], "sigma": h["sigma"],
            "ordered": bool(h["ordered"]),
            "n_admissible": len(d["surface"]), "n_ordered": n_ord,
            "R": h["R"], "R_err": h.get("R_err"),
        })

    _print(table, a.threshold)
    common = (_common_window(a.pred_dir, a.hist_dir, a.common_window, a.threshold)
              if a.pred_dir and a.hist_dir else None)
    out = {"threshold": a.threshold, "rows": table,
           "common_window": common}
    jp = os.path.join(a.out_dir, "final_sensitivity.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {jp}")
    _plot(table, a.out_dir, a.threshold, common)
    _plot_absolute(table, a.out_dir, a.threshold)
    return 0


def _common_window(pred_dir, hist_dir, window, thr):
    """Every quantity at one window, so the dilution ladder is readable.

    Returns the 90/0 response of the truth single ratios, the truth double
    ratio, the reconstructed single ratio and the hybrid, all at `window`.
    """
    import pandas as pd

    lo_e, hi_e = window
    tags = ["zeroSpot", "defaultSpot", "bigSpot"]
    vals = {k: {} for k in ("truth_Rn", "truth_Rp", "truth_dr", "reco_Rn", "hybrid")}
    for t in tags:
        z = np.load(os.path.join(hist_dir, f"{t}_filehist.npz"))
        keys = sorted(k for k in z.files if k.endswith("|acc"))
        per = np.stack([z[k] for k in keys]).astype(float)
        e = z["_e_edges"]
        c = np.sqrt(np.maximum(e[:-1], 1e-12) * e[1:])
        lom, him = c < lo_e, c >= hi_e
        rn = per[:, 0, :][:, him].sum() / per[:, 0, :][:, lom].sum()
        rp = per[:, 1, :][:, him].sum() / per[:, 1, :][:, lom].sum()

        cl = pd.read_pickle(os.path.join(pred_dir, t, "pred_clusters_smash.pkl"))
        sel = cl[cl.cl_score > thr]
        rn_reco = (sel.e_pred >= hi_e).sum() / max((sel.e_pred < lo_e).sum(), 1)

        vals["truth_Rn"][t] = float(rn)
        vals["truth_Rp"][t] = float(rp)
        vals["truth_dr"][t] = float(rn / rp)
        vals["reco_Rn"][t] = float(rn_reco)
        vals["hybrid"][t] = float(rn_reco / rp)

    out = {"window": [lo_e, hi_e], "values": vals, "response": {}}
    for k, v in vals.items():
        out["response"][k] = float(v["bigSpot"] / v["zeroSpot"] - 1.0)
    return out


def _print(table, thr):
    print(f"Final U_sym sensitivity, reconstructed rows at q > {thr:.2f}")
    print(f"{'observable':26s} {'window':>18} {'held-out 90/0-1':>18} "
          f"{'sigma':>7} {'ord':>5} {'ordered/adm':>12}")
    for r in table:
        win = ("no window" if r.get("baseline")
               else f"<{r['lo_edge']:.2f} | >={r['hi_edge']:.2f}")
        frac = ("      —" if r.get("baseline")
                else f"{r['n_ordered']:>5}/{r['n_admissible']:<6}")
        print(f"{r['key']:26s} {win:>18} "
              f"{100*r['rel']:>+9.2f} +- {100*r['err']:<5.2f} "
              f"{r['sigma']:>7.2f} {str(r['ordered']):>5} {frac}")


U_MEV = {"zeroSpot": 0.0, "defaultSpot": 18.0, "bigSpot": 90.0}


def _plot_absolute(table, out_dir, thr):
    """Each observable's value against U_sym, on its own scale.

    The response figure answers "how much does it move"; it cannot show where
    the three samples actually sit, whether the middle one lies between the
    outer two, or how large the move is against the value. The observables span
    0.2 to 1.9 in magnitude, so they cannot share an axis and this is a grid of
    small panels rather than one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from plot_b_full import apply_style, hep_label, no_grid, save, FULL
    apply_style()

    rows = [r for r in table if r.get("R")]
    ncol = 4
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL, FULL / 2.3 * nrow / 2))
    axes = np.atleast_1d(axes).ravel()

    u = np.array([U_MEV[t] for t in ("zeroSpot", "defaultSpot", "bigSpot")])
    for ax, r in zip(axes, rows):
        y = np.array([r["R"][t] for t in ("zeroSpot", "defaultSpot", "bigSpot")],
                     dtype=float)
        e = r.get("R_err") or {}
        ye = np.array([e.get(t, np.nan) for t in
                       ("zeroSpot", "defaultSpot", "bigSpot")], dtype=float)
        base = "#9E9E9E" if r.get("baseline") else (
            "#2b2b2b" if r["needs"] == "MC truth" else "#1f77b4")
        mk = "s" if r.get("baseline") else (
            "o" if r["needs"] == "MC truth" else "^")
        ax.errorbar(u, y, yerr=ye if np.isfinite(ye).all() else None,
                    fmt=mk, ms=4, mfc="none", mew=1.1, elinewidth=0.9,
                    capsize=2.5, color=base, ls="none")
        # a straight line through the outer two, so a reader can see whether
        # the 18 MeV point sits on it
        ax.plot(u[[0, 2]], y[[0, 2]], color=base, ls=":", lw=0.9, alpha=0.8)
        ax.set_title(r["label"], fontsize=6.8)
        ax.set_xlim(-8, 98)
        ax.set_xticks([0, 18, 90])
        lo = float(np.nanmin(y - np.nan_to_num(ye)))
        hi = float(np.nanmax(y + np.nan_to_num(ye)))
        sp = (hi - lo) or abs(hi) * 0.05 or 1.0
        ax.set_ylim(lo - 0.45 * sp, hi + 0.45 * sp)
        ax.tick_params(labelsize=6)
    for ax in axes[len(rows):]:
        ax.set_visible(False)
    for ax in axes[:len(rows)]:
        ax.set_xlabel(r"$U_{\rm sym}$  [MeV]", fontsize=7)
    axes[0].set_ylabel("observable value", fontsize=7)
    no_grid(*axes[:len(rows)])
    # the first panel is small and fully occupied; the unused grid cell is not
    # every panel is small and fully occupied; the label goes above the row
    fig.text(0.005, 0.995,
             "BM@N HGND simulation      "
             r"Xe+CsI, $\sqrt{s_{NN}}$ = 2.87 GeV      "
             rf"held-out units, $q>{thr:.2f}$",
             ha="left", va="top", fontsize=6.4)
    fig.tight_layout()
    save(fig, Path(out_dir), "fig_absolute_vs_spot")
    print("  wrote fig_absolute_vs_spot.pdf / .png")


def _plot(table, out_dir, thr, common=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_b_full import apply_style, save, FULL, hep_label, no_grid
    apply_style()

    ncol = 3 if common else 2
    widths = [1.25, 1, 1.05] if common else [1.25, 1]
    fig, axes = plt.subplots(1, ncol, figsize=(FULL * (1.35 if common else 1.0),
                                               FULL / 2.5),
                             gridspec_kw={"width_ratios": widths})
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if common else None

    y = np.arange(len(table))[::-1]
    rel = np.array([100 * r["rel"] for r in table])
    err = np.array([100 * r["err"] for r in table])
    # truth rows are the references, reconstructed rows are the measurement
    col = ["#9E9E9E" if r.get("baseline") else
           ("#2b2b2b" if r["needs"] == "MC truth" else "#1f77b4") for r in table]
    mk = ["s" if r.get("baseline") else
          ("o" if r["needs"] == "MC truth" else "^") for r in table]

    ax1.axvline(0, color="#888888", lw=0.8, ls="--")
    for yi, ri, ei, ci, mi in zip(y, rel, err, col, mk):
        ax1.errorbar(ri, yi, xerr=ei, color=ci, marker=mi, ms=5, mfc="none",
                     mew=1.2, elinewidth=1.0, capsize=2.5, ls="none")
    ax1.set_yticks(y)
    ax1.set_yticklabels([r["label"] for r in table], fontsize=7)
    ax1.set_xlabel(r"$S_{90}/S_{0}-1$  [%]")
    lo = float((rel - err).min())
    hi = float((rel + err).max())
    sp = hi - lo
    ax1.set_xlim(lo - 0.14 * sp, hi + 0.14 * sp)
    ax1.set_ylim(y.min() - 0.6, y.max() + 0.6)

    sig = np.array([r["sigma"] for r in table])
    ax2.barh(y, sig, color=col, alpha=0.75, height=0.55)
    ax2.axvline(3, color="#B22222", ls="--", lw=1.0)
    ax2.annotate(r"3$\sigma$", xy=(3, y.max() + 0.35), fontsize=6.5,
                 color="#B22222", ha="center")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel(r"separation  [$\sigma$]")
    ax2.set_xlim(0, max(sig.max() * 1.18, 3.4))
    ax2.set_ylim(y.min() - 0.6, y.max() + 0.6)

    if ax3 is not None:
        keys = [("truth_Rn", r"truth $R_n$"), ("truth_Rp", r"truth $R_p$"),
                ("truth_dr", r"truth $R_n/R_p$"), ("reco_Rn", r"reco $R_n$"),
                ("hybrid", r"reco $R_n$/truth $R_p$")]
        yy = np.arange(len(keys))[::-1]
        vv = np.array([100 * common["response"][k] for k, _ in keys])
        cc = ["#2b2b2b"] * 3 + ["#1f77b4"] * 2
        ax3.axvline(0, color="#888888", lw=0.8, ls="--")
        ax3.barh(yy, vv, color=cc, alpha=0.75, height=0.55)
        ax3.set_yticks(yy)
        ax3.set_yticklabels([lab for _, lab in keys], fontsize=6.5)
        ax3.set_xlabel(r"$S_{90}/S_{0}-1$  [%]")
        w = common["window"]
        ax3.text(0.97, 0.04,
                 rf"one window: $E<{w[0]:.2f}$ | $E\geq{w[1]:.2f}$ GeV",
                 transform=ax3.transAxes, ha="right", va="bottom", fontsize=6.5)
        lo3, hi3 = float(vv.min()), float(vv.max())
        sp3 = hi3 - lo3
        ax3.set_xlim(lo3 - 0.16 * sp3, hi3 + 0.16 * sp3)
        ax3.set_ylim(yy.min() - 0.6, yy.max() + 0.6)

    no_grid(*[a for a in axes if a is not None])
    # the baselines sit at zero and the hardness rows to the right of centre,
    # so the upper-right corner of panel 1 is the only clear area
    hep_label(axes[0], extra=rf"$U_{{\rm sym}}$ = 0, 18, 90 MeV;  $q>{thr:.2f}$",
              loc="upper right")
    fig.tight_layout()
    if save is not None:
        from pathlib import Path
        save(fig, Path(out_dir), "fig_final_sensitivity")
        print("  wrote fig_final_sensitivity.pdf / .png")
    else:
        fig.savefig(os.path.join(out_dir, "fig_final_sensitivity.png"), dpi=200)


if __name__ == "__main__":
    sys.exit(main())
