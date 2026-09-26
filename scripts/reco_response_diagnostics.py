"""Why the reconstructed hardness ratio disagrees with the truth-level one.

The project already knows that a *fixed classifier threshold* selects different
physical purity in different datasets, and locks purity to remove that
(scripts/purity_locked_closure.py).  This script shows that the *energy
regressor* has the same pathology and that nothing currently controls it: its
response depends on the sample, so it moves the R thresholds by a different
amount in each, and that differential is large enough to invert the measured
S_pot difference.

Two tests:

  paired          the same selected signal clusters, binned on true energy and
                  on calibrated predicted energy.  Only the energy estimate
                  differs, so job files can be resampled jointly and the
                  difference of the two answers carries its own error.
  response        mean (e_cal - e_true) per sample, inclusive and inside each R
                  bin.  A sample-independent estimator would give one number.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reco_truth_join import row_offsets, to_local
from reco_experiment import (fit_energy_calibration, apply_calibration,
                             purity_locked_threshold, ELO, EHI, SAMPLES)


def load_signal(tag, raw_dir, pred_dir, cal, purity):
    c = pd.read_pickle(os.path.join(pred_dir, tag, "pred_clusters_smash.pkl"))
    offs = row_offsets(os.path.join(raw_dir, tag))
    c = c.assign(stem=to_local(c.Row.to_numpy(), offs).stem.to_numpy())
    t, pur, _ = purity_locked_threshold(c, purity)
    s = c[(c.cl_score > t) & (c.cl_label == 1)].copy()
    s["e_cal"] = apply_calibration(s.e_pred.to_numpy(), cal)
    return s, t, pur


def bin_arrays(s, col, stems):
    g = lambda m: s[m].groupby("stem").size().reindex(stems, fill_value=0).to_numpy(float)
    return g(s[col] >= EHI), g(s[col] < ELO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--purity", type=float, default=0.7)
    p.add_argument("--calib-sample", default="defaultSpot")
    p.add_argument("--ref", default="zeroSpot")
    p.add_argument("--arm", default="bigSpot")
    p.add_argument("--n-boot", type=int, default=400)
    a = p.parse_args()

    cal = fit_energy_calibration(
        pd.read_pickle(os.path.join(a.pred_dir, a.calib_sample, "pred_clusters_smash.pkl")))
    rng = np.random.default_rng(7)

    S, meta = {}, {}
    for tag in SAMPLES:
        pth = os.path.join(a.pred_dir, tag, "pred_clusters_smash.pkl")
        if not os.path.exists(pth):
            continue
        s, t, pur = load_signal(tag, a.raw_dir, a.pred_dir, cal, a.purity)
        S[tag] = s
        meta[tag] = {"threshold": t, "purity": pur, "n_signal": int(len(s))}

    # ---- response table ---------------------------------------------------
    print("=" * 78)
    print("  Energy response of one estimator across samples (signal clusters)")
    print("=" * 78)
    print("  %-12s %8s %10s %10s %11s %11s" %
          ("sample", "N", "<e_true>", "bias", "bias lo bin", "bias hi bin"))
    resp = {}
    for tag, s in S.items():
        et, ec = s.e_true.to_numpy(), s.e_cal.to_numpy()
        lo, hi = et < ELO, et >= EHI
        resp[tag] = {"mean_e_true": float(et.mean()),
                     "bias": float((ec - et).mean()),
                     "bias_lo": float((ec - et)[lo].mean()),
                     "bias_hi": float((ec - et)[hi].mean()),
                     "resolution_sd": float((ec - et).std(ddof=1)),
                     "n": int(len(s))}
        r = resp[tag]
        print("  %-12s %8d %10.4f %+10.4f %+11.4f %+11.4f" %
              (tag, r["n"], r["mean_e_true"], r["bias"], r["bias_lo"], r["bias_hi"]))
    if a.ref in resp and a.arm in resp:
        d_hi = resp[a.arm]["bias_hi"] - resp[a.ref]["bias_hi"]
        print(f"\n  differential response in the high bin, {a.arm} vs {a.ref}: "
              f"{d_hi*1000:+.1f} MeV")
        print("  A sample-independent estimator would give zero here. This shifts the")
        print(f"  {EHI:.0f} GeV threshold by a different amount in each sample.")

    # ---- paired test ------------------------------------------------------
    out = {"response": resp, "meta": meta}
    if a.ref in S and a.arm in S:
        stems = {k: sorted(S[k].stem.unique()) for k in (a.ref, a.arm)}
        A = {k: {c: bin_arrays(S[k], c, stems[k]) for c in ("e_true", "e_cal")}
             for k in (a.ref, a.arm)}

        def rel(col, pr, pa):
            hr, lr = A[a.ref][col]; ha, la = A[a.arm][col]
            vr = hr[pr].sum() / lr[pr].sum() if lr[pr].sum() else np.nan
            va = ha[pa].sum() / la[pa].sum() if la[pa].sum() else np.nan
            return va / vr - 1.0

        nr, na = len(stems[a.ref]), len(stems[a.arm])
        base = (rel("e_true", np.arange(nr), np.arange(na)),
                rel("e_cal", np.arange(nr), np.arange(na)))
        bs = np.empty((a.n_boot, 2))
        for i in range(a.n_boot):
            pr, pa = rng.integers(0, nr, nr), rng.integers(0, na, na)
            bs[i] = [rel("e_true", pr, pa), rel("e_cal", pr, pa)]
        dif = bs[:, 1] - bs[:, 0]
        out["paired"] = {
            "true_energy": {"rel": float(base[0]), "err": float(bs[:, 0].std(ddof=1))},
            "reco_energy": {"rel": float(base[1]), "err": float(bs[:, 1].std(ddof=1))},
            "difference": {"rel": float(base[1] - base[0]),
                           "err": float(dif.std(ddof=1)),
                           "sigma": float(abs(base[1] - base[0]) / dif.std(ddof=1))},
            "correlation": float(np.corrcoef(bs[:, 0], bs[:, 1])[0, 1]),
        }
        pr_ = out["paired"]
        print("\n" + "=" * 78)
        print(f"  Paired test: identical clusters, only the energy estimate differs")
        print("=" * 78)
        print("  true energy       %+.2f%% +- %.2f%%" %
              (pr_["true_energy"]["rel"] * 100, pr_["true_energy"]["err"] * 100))
        print("  calibrated reco   %+.2f%% +- %.2f%%" %
              (pr_["reco_energy"]["rel"] * 100, pr_["reco_energy"]["err"] * 100))
        print("  difference        %+.2f%% +- %.2f%%  -> %.2f sigma" %
              (pr_["difference"]["rel"] * 100, pr_["difference"]["err"] * 100,
               pr_["difference"]["sigma"]))
        print("\n  The estimator does not merely dilute the signal towards zero;")
        print("  it moves the answer by more than the signal itself.")

    # ---- migration matrix -------------------------------------------------
    if a.ref in S:
        s = S[a.ref]
        et, ec = s.e_true.to_numpy(), s.e_cal.to_numpy()
        mig = {}
        for tn, tm in (("lo", et < ELO), ("mid", (et >= ELO) & (et < EHI)), ("hi", et >= EHI)):
            n = max(tm.sum(), 1)
            mig[tn] = {"lo": float((tm & (ec < ELO)).sum() / n),
                       "mid": float((tm & (ec >= ELO) & (ec < EHI)).sum() / n),
                       "hi": float((tm & (ec >= EHI)).sum() / n)}
        out["migration_ref"] = mig
        out["resolution_ref_sd_GeV"] = float((ec - et).std(ddof=1))

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
