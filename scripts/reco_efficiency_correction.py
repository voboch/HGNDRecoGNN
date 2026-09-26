"""Efficiency- and response-corrected R_n from GNN reconstruction.

Detection and reconstruction efficiency for neutrons is strongly non-linear in
E_kin: a neutron arriving at the front face is far more likely to make a
reconstructable cluster at 3 GeV than at 0.5 GeV.  An uncorrected count ratio
therefore measures the efficiency curve as much as the spectrum, which is why
R_n(reco) came out at 7.9 where the truth value is 0.33.

The correction is a single kernel

    K[i, j] = eps(i) * P(E_reco in bin j | E_true in bin i, selected)

where eps(i) is the probability that a neutron *arriving at the detector* with
true energy in bin i ends up as a selected cluster.  The denominator comes from
the vacs export (particles crossing the detector surface), so eps folds in
cluster formation and the classifier together.  N_reco = K^T N_true is then
solved for N_true and R_n read off the unfolded spectrum.

Two choices matter for bias, and both are deliberate:

  * the kernel is built on ONE sample and applied to all three.  Building it
    per sample would let it absorb the S_pot difference being measured.
  * that sample is the 18 MeV midpoint, which is symmetric between the two
    extremes being compared.  `--kernel-sample` changes it, and
    `--kernel-sample all` pools the three, which is the other defensible
    choice; the two are compared in the report.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from reco_truth_join import row_offsets, to_local
from reco_experiment import purity_locked_threshold, SAMPLES, ELO, EHI, load_vacs

BINS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.5])
LO_BINS = BINS[:-1] < ELO                    # bins entirely below 1 GeV
HI_BINS = BINS[:-1] >= EHI                   # bins entirely at/above 2 GeV


def load_sample(tag, raw_dir, pred_dir, purity):
    c = pd.read_pickle(os.path.join(pred_dir, tag, "pred_clusters_smash.pkl"))
    offs = row_offsets(os.path.join(raw_dir, tag))
    c = c.assign(stem=to_local(c.Row.to_numpy(), offs).stem.to_numpy())
    t, pur, _ = purity_locked_threshold(c, purity)
    v = load_vacs(os.path.join(raw_dir, tag))
    v = v[v.PDG == 2112]
    n_ev = int(c.groupby(["stem", "local_row"]).ngroups) if "local_row" in c else None
    return {"clusters": c, "threshold": t, "purity": pur, "vacs_n": v,
            "n_events": int(offs.span.sum())}


def build_kernel(samples, keys):
    """eps(i) * P(reco j | true i), plus the per-event background spectrum."""
    cl = pd.concat([samples[k]["clusters"].assign(
        _sel=samples[k]["clusters"].cl_score > samples[k]["threshold"]) for k in keys],
        ignore_index=True)
    vn = pd.concat([samples[k]["vacs_n"] for k in keys], ignore_index=True)
    n_ev = sum(samples[k]["n_events"] for k in keys)

    sig = cl[(cl.cl_label == 1) & (cl.e_true > 0)]
    sel = sig[sig._sel]
    nb = len(BINS) - 1
    arriving, _ = np.histogram(vn.Ekin.to_numpy(), bins=BINS)
    made, _ = np.histogram(sel.e_true.to_numpy(), bins=BINS)
    eps = np.divide(made, arriving, out=np.zeros(nb), where=arriving > 0)

    P = np.zeros((nb, nb))
    it = np.digitize(sel.e_true.to_numpy(), BINS) - 1
    ip = np.digitize(sel.e_pred.to_numpy(), BINS) - 1
    for i in range(nb):
        m = it == i
        if m.sum() == 0:
            continue
        cnt = np.bincount(np.clip(ip[m], 0, nb - 1), minlength=nb).astype(float)
        P[i] = cnt / cnt.sum()
    K = eps[:, None] * P

    bkg = cl[(cl.cl_label == 0) & cl._sel]
    bspec, _ = np.histogram(bkg.e_pred.to_numpy(), bins=BINS)
    return {"eps": eps, "P": P, "K": K, "bkg_per_event": bspec / max(n_ev, 1),
            "arriving": arriving, "made": made, "n_events": n_ev}


def unfold(n_reco, K, lam=1e-3):
    A = K.T
    AtA = A.T @ A
    x = np.linalg.solve(AtA + lam * np.trace(AtA) / len(AtA) * np.eye(len(AtA)),
                        A.T @ n_reco)
    return np.clip(x, 0.0, None)


def ratio_from_spectrum(n):
    hi, lo = n[HI_BINS].sum(), n[LO_BINS].sum()
    return hi / lo if lo > 0 else np.nan


def analyse(tag, smp, kern, lam):
    c = smp["clusters"]
    sel = c[c.cl_score > smp["threshold"]]
    n_all, _ = np.histogram(sel.e_pred.to_numpy(), bins=BINS)
    n_sig = np.clip(n_all - kern["bkg_per_event"] * smp["n_events"], 0.0, None)
    n_true = unfold(n_sig, kern["K"], lam)

    v = smp["vacs_n"]
    truth, _ = np.histogram(v.Ekin.to_numpy(), bins=BINS)
    return {
        "R_truth_arriving": float(ratio_from_spectrum(truth.astype(float))),
        "R_reco_raw": float(ratio_from_spectrum(n_all.astype(float))),
        "R_reco_bkgsub": float(ratio_from_spectrum(n_sig)),
        "R_corrected": float(ratio_from_spectrum(n_true)),
        "n_true_spectrum": n_true.tolist(),
        "truth_spectrum": truth.tolist(),
        "n_reco_spectrum": n_all.tolist(),
        "n_events": smp["n_events"], "threshold": smp["threshold"],
    }


def boot_pairs(samples, kern, lam, ref, arm, n_boot, rng):
    """Job-file bootstrap of the corrected and truth ratios, 90 vs 0."""
    out = []
    for _ in range(n_boot):
        vals = {}
        for k in (ref, arm):
            s = samples[k]
            stems = np.array(sorted(s["clusters"].stem.unique()))
            pick = rng.choice(stems, len(stems), replace=True)
            cl = pd.concat([s["clusters"][s["clusters"].stem == x] for x in pick],
                           ignore_index=True)
            vn = pd.concat([s["vacs_n"][s["vacs_n"].stem == x] for x in pick],
                           ignore_index=True)
            sel = cl[cl.cl_score > s["threshold"]]
            n_all, _ = np.histogram(sel.e_pred.to_numpy(), bins=BINS)
            nev = s["n_events"]
            n_sig = np.clip(n_all - kern["bkg_per_event"] * nev, 0.0, None)
            tr, _ = np.histogram(vn.Ekin.to_numpy(), bins=BINS)
            vals[k] = (ratio_from_spectrum(unfold(n_sig, kern["K"], lam)),
                       ratio_from_spectrum(tr.astype(float)))
        out.append((vals[arm][0] / vals[ref][0] - 1 if vals[ref][0] else np.nan,
                    vals[arm][1] / vals[ref][1] - 1 if vals[ref][1] else np.nan))
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--purity", type=float, default=0.7)
    p.add_argument("--kernel-sample", default="defaultSpot")
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--n-boot", type=int, default=200)
    a = p.parse_args()

    rng = np.random.default_rng(11)
    keys = [k for k in SAMPLES
            if os.path.exists(os.path.join(a.pred_dir, k, "pred_clusters_smash.pkl"))]
    samples = {k: load_sample(k, a.raw_dir, a.pred_dir, a.purity) for k in keys}
    kkeys = keys if a.kernel_sample == "all" else [a.kernel_sample]
    kern = build_kernel(samples, kkeys)

    print("=" * 84)
    print(f"  Detection + reconstruction efficiency, kernel from: {', '.join(kkeys)}")
    print("=" * 84)
    print(f"  {'E_true bin [GeV]':<20}{'arriving n':>12}{'-> selected':>13}{'eps':>9}")
    for i in range(len(BINS) - 1):
        print(f"  {f'{BINS[i]:.1f} - {BINS[i+1]:.1f}':<20}"
              f"{kern['arriving'][i]:>12,}{kern['made'][i]:>13,}{kern['eps'][i]:>9.4f}")
    nz = kern["eps"][kern["eps"] > 0]
    print(f"\n  efficiency varies by a factor {nz.max()/max(nz.min(),1e-9):.1f} across the range"
          f" -- this is what makes an uncorrected ratio meaningless")

    res = {k: analyse(k, samples[k], kern, a.lam) for k in keys}
    print("\n" + "=" * 84)
    print("  R_n for neutrons arriving at the detector")
    print("=" * 84)
    print(f"  {'sample':<13}{'truth':>10}{'reco raw':>11}{'bkg-sub':>10}{'corrected':>12}{'corr/truth':>12}")
    for k in keys:
        r = res[k]
        print(f"  {k:<13}{r['R_truth_arriving']:>10.4f}{r['R_reco_raw']:>11.4f}"
              f"{r['R_reco_bkgsub']:>10.4f}{r['R_corrected']:>12.4f}"
              f"{r['R_corrected']/r['R_truth_arriving']:>12.3f}")

    out = {"bins": BINS.tolist(), "eps": kern["eps"].tolist(),
           "kernel_sample": kkeys, "per_sample": res}
    if "zeroSpot" in keys and "bigSpot" in keys:
        bs = boot_pairs(samples, kern, a.lam, "zeroSpot", "bigSpot", a.n_boot, rng)
        rc = res["bigSpot"]["R_corrected"] / res["zeroSpot"]["R_corrected"] - 1
        rt = res["bigSpot"]["R_truth_arriving"] / res["zeroSpot"]["R_truth_arriving"] - 1
        ec, et = np.nanstd(bs[:, 0], ddof=1), np.nanstd(bs[:, 1], ddof=1)
        out["comparison"] = {
            "corrected": {"rel": float(rc), "err": float(ec),
                          "sigma": float(abs(rc) / ec) if ec else np.nan},
            "truth_arriving": {"rel": float(rt), "err": float(et),
                               "sigma": float(abs(rt) / et) if et else np.nan}}
        print("\n" + "=" * 84)
        print("  S_pot = 90 vs 0 MeV, job-file bootstrap")
        print("=" * 84)
        print(f"  truth (arriving)   {rt*100:+7.2f}% +- {et*100:.2f}%   "
              f"{abs(rt)/et if et else float('nan'):.2f} sigma")
        print(f"  corrected reco     {rc*100:+7.2f}% +- {ec*100:.2f}%   "
              f"{abs(rc)/ec if ec else float('nan'):.2f} sigma")
        print("\n  The correction is judged by whether it reproduces the truth row,")
        print("  in sign and size, not by whether it is significant.")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
