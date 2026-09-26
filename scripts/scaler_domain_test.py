"""Is the sample-dependent energy response a network property or a preprocessing one?

`prepare_halves` fitted a StandardScaler per dataset.  `eToF` carries the
neutron energy, so a harder sample gets a wider `eToF` scale and part of the
very difference being measured is divided out before the network sees it --
measured across the scan, 1.9 % in scale and 0.02 sigma in mean.

This runs the identical checkpoint over caches built two ways -- per-sample
scalers, and one shared scaler taken from the 18 MeV midpoint -- and reports the
quantities the progress guide's P1 gates are written against:

  * the differential high-bin energy response between the 90 and 0 MeV samples,
  * the paired displacement between true-energy and reconstructed-energy answers
    on identical clusters,
  * the reconstructed R_n across the three-point scan.

No retraining is involved, so any improvement is attributable to the
normalisation alone.

Usage:
  python3 scripts/scaler_domain_test.py --raw-dir DIR \
      --pred-per-sample DIR --pred-shared DIR --out FILE
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reco_truth_join import row_offsets, to_local
from reco_experiment import (fit_energy_calibration, apply_calibration,
                             purity_locked_threshold, load_vacs, ELO, EHI)

TAGS = [("zeroSpot", 0), ("defaultSpot", 18), ("bigSpot", 90)]


def prep(pred_dir, raw_dir, tag, cal, purity):
    c = pd.read_pickle(os.path.join(pred_dir, tag, "pred_clusters_smash.pkl"))
    offs = row_offsets(os.path.join(raw_dir, tag))
    c = c.assign(stem=to_local(c.Row.to_numpy(), offs).stem.to_numpy())
    t, pur, _ = purity_locked_threshold(c, purity)
    sel = c[c.cl_score > t].copy()
    sel["e_cal"] = apply_calibration(sel.e_pred.to_numpy(), cal)
    return sel, t, pur


def per_file(df, mask, stems):
    return df[mask].groupby("stem").size().reindex(stems, fill_value=0).to_numpy(float)


def arm(raw_dir, pred_dir, purity, rng, n_boot):
    cal = fit_energy_calibration(
        pd.read_pickle(os.path.join(pred_dir, "defaultSpot", "pred_clusters_smash.pkl")))
    out, sig = {}, {}
    for tag, u in TAGS:
        sel, thr, pur = prep(pred_dir, raw_dir, tag, cal, purity)
        stems = sorted(sel.stem.unique())
        v = load_vacs(os.path.join(raw_dir, tag))
        v = v[v.PDG == 2112]
        out[tag] = {
            "U": u, "threshold": thr, "purity": pur, "stems": stems,
            "reco": (per_file(sel, sel.e_cal >= EHI, stems),
                     per_file(sel, sel.e_cal < ELO, stems)),
            "truth": (per_file(v, v.Ekin >= EHI, stems),
                      per_file(v, v.Ekin < ELO, stems)),
        }
        s = sel[(sel.cl_label == 1) & (sel.e_true >= EHI)]
        sig[tag] = {k: float((g.e_cal - g.e_true).mean()) for k, g in s.groupby("stem")}
        out[tag]["sig_stems"] = sorted(sig[tag])
    return out, sig


def ratio(a, p=None):
    if p is None:
        return a[0].sum() / a[1].sum()
    return a[0][p].sum() / a[1][p].sum()


def analyse(out, sig, rng, n_boot):
    res = {}
    # differential high-bin response, 90 vs 0
    z, b = sig["zeroSpot"], sig["bigSpot"]
    sz, sb = out["zeroSpot"]["sig_stems"], out["bigSpot"]["sig_stems"]
    base = np.mean([b[s] for s in sb]) - np.mean([z[s] for s in sz])
    bs = [np.mean([b[s] for s in rng.choice(sb, len(sb))])
          - np.mean([z[s] for s in rng.choice(sz, len(sz))]) for _ in range(n_boot)]
    e = float(np.std(bs, ddof=1))
    res["differential_response_MeV"] = {
        "value": float(base * 1000), "err": e * 1000,
        "sigma_from_zero": float(abs(base) / e) if e else np.nan}

    # paired true-vs-reco displacement on identical clusters
    def rel(key, pz, pb):
        return ratio(out["bigSpot"][key], pb) / ratio(out["zeroSpot"][key], pz) - 1

    nz, nb = len(out["zeroSpot"]["stems"]), len(out["bigSpot"]["stems"])
    dt = rel("truth", np.arange(nz), np.arange(nb))
    dc = rel("reco", np.arange(nz), np.arange(nb))
    arr = np.array([[rel("truth", pz, pb), rel("reco", pz, pb)]
                    for pz, pb in ((rng.integers(0, nz, nz), rng.integers(0, nb, nb))
                                   for _ in range(n_boot))])
    dif = arr[:, 1] - arr[:, 0]
    res["truth_signal"] = {"rel": float(dt), "err": float(arr[:, 0].std(ddof=1))}
    res["reco_signal"] = {"rel": float(dc), "err": float(arr[:, 1].std(ddof=1))}
    res["paired_displacement"] = {
        "rel": float(dc - dt), "err": float(dif.std(ddof=1)),
        "sigma": float(abs(dc - dt) / dif.std(ddof=1)),
        "frac_of_truth": float(abs(dc - dt) / abs(dt)) if dt else np.nan}

    # three-point ordering of reconstructed R_n
    vals = {t: ratio(out[t]["reco"]) for t, _ in TAGS}
    mono = np.mean([
        (lambda v: (v[0] < v[1] < v[2]) or (v[0] > v[1] > v[2]))(
            [ratio(out[t]["reco"], rng.integers(0, len(out[t]["stems"]),
                                                len(out[t]["stems"]))) for t, _ in TAGS])
        for _ in range(n_boot)])
    res["reco_R_n"] = {t: float(v) for t, v in vals.items()}
    res["reco_ordered"] = bool(vals["zeroSpot"] < vals["defaultSpot"] < vals["bigSpot"]
                               or vals["zeroSpot"] > vals["defaultSpot"] > vals["bigSpot"])
    res["reco_monotonic_boot_frac"] = float(mono)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--pred-per-sample", required=True)
    p.add_argument("--pred-shared", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--purity", type=float, default=0.7)
    p.add_argument("--n-boot", type=int, default=400)
    a = p.parse_args()

    report = {}
    for name, pdir in (("per_sample_scalers", a.pred_per_sample),
                       ("shared_scaler", a.pred_shared)):
        rng = np.random.default_rng(23)
        out, sig = arm(a.raw_dir, pdir, a.purity, rng, a.n_boot)
        report[name] = analyse(out, sig, rng, a.n_boot)

    print("=" * 80)
    print("  Sample-dependent energy response: per-sample vs shared normalisation")
    print("  identical checkpoint, no retraining")
    print("=" * 80)
    print(f"  {'quantity':<36}{'per-sample':>20}{'shared':>20}")
    A, B = report["per_sample_scalers"], report["shared_scaler"]
    d = lambda r: r["differential_response_MeV"]
    print(f"  {'differential response [MeV]':<36}"
          f"{d(A)['value']:>+13.1f} ±{d(A)['err']:>5.1f}{d(B)['value']:>+13.1f} ±{d(B)['err']:>5.1f}")
    print(f"  {'  ... sigma from zero':<36}{d(A)['sigma_from_zero']:>20.2f}"
          f"{d(B)['sigma_from_zero']:>20.2f}")
    for k, lab in (("truth_signal", "truth 90/0-1 [%]"),
                   ("reco_signal", "reco  90/0-1 [%]")):
        print(f"  {lab:<36}{A[k]['rel']*100:>+13.2f} ±{A[k]['err']*100:>5.2f}"
              f"{B[k]['rel']*100:>+13.2f} ±{B[k]['err']*100:>5.2f}")
    pa, pb = A["paired_displacement"], B["paired_displacement"]
    print(f"  {'paired displacement [%]':<36}{pa['rel']*100:>+13.2f} ±{pa['err']*100:>5.2f}"
          f"{pb['rel']*100:>+13.2f} ±{pb['err']*100:>5.2f}")
    print(f"  {'  ... sigma':<36}{pa['sigma']:>20.2f}{pb['sigma']:>20.2f}")
    print(f"  {'  ... fraction of truth effect':<36}{pa['frac_of_truth']:>20.2f}"
          f"{pb['frac_of_truth']:>20.2f}")
    print(f"  {'reconstructed R_n ordered 0/18/90':<36}{str(A['reco_ordered']):>20}"
          f"{str(B['reco_ordered']):>20}")
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
