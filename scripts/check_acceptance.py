"""Check the A(theta) weighting against the exact per-particle acceptance test.

The weighting is only valid if the laboratory azimuthal distribution is flat.
This tests that directly on primary CSVs, and also reports what the change of
window does to the observables.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hgnd_acceptance import in_acceptance, bin_acceptance, theta_acceptance

COLS = ["Row", "PDG", "Ekin", "Px", "Py", "Pz", "vX", "vY", "vZ"]
ELO, EHI = 1.0, 2.0
TH_LO, TH_HI = 8.9, 13.1          # the old band


def load(paths):
    fr = []
    for p in paths:
        d = pd.read_csv(p, usecols=COLS, engine="c")
        d = d[d.Row != d.Row.iloc[-1]]        # cached extracts are byte-cut
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prim-glob", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    d = load(sorted(glob.glob(a.prim_glob)))
    px, py, pz = (d[c].to_numpy() for c in ("Px", "Py", "Pz"))
    vx, vy, vz = (d[c].to_numpy() for c in ("vX", "vY", "vZ"))
    th = np.degrees(np.arctan2(np.hypot(px, py), pz))
    ek = d.Ekin.to_numpy()
    isn, isp = d.PDG.to_numpy() == 2112, d.PDG.to_numpy() == 2212

    exact = in_acceptance(px, py, pz, vx, vy, vz)
    exact0 = in_acceptance(px, py, pz)          # from the nominal origin
    band = (th >= TH_LO) & (th < TH_HI)

    print(f"=== {a.label or a.prim_glob} ===")
    print(f"  nucleons {len(d):,}   in old theta band {band.sum():,}"
          f"   in rectangle {exact.sum():,}"
          f"   ({exact.sum()/max(band.sum(),1)*100:.1f} % of the band)")
    print(f"  vertex matters: rectangle from true vertex {exact.sum():,} vs "
          f"from origin {exact0.sum():,} "
          f"({(exact0.sum()-exact.sum())/max(exact.sum(),1)*100:+.2f} %)")

    # azimuthal uniformity of the population the weighting averages over
    sel = (th >= 8.0) & (th < 13.5)
    phi = np.degrees(np.arctan2(py, px))[sel] % 360.0
    cnt, _ = np.histogram(phi, bins=36, range=(0, 360))
    chi2 = ((cnt - cnt.mean()) ** 2 / cnt.mean()).sum()
    print(f"  azimuthal uniformity over 8-13.5 deg: chi2/ndf = {chi2/35:.2f} "
          f"({len(phi):,} particles, 36 bins)")

    # A(theta) weighting reproduced against the exact test
    edges = np.arange(8.0, 13.5001, 0.25)
    A = bin_acceptance(edges)
    idx = np.digitize(th, edges) - 1
    ok = (idx >= 0) & (idx < len(A))
    w = np.zeros(len(d)); w[ok] = A[idx[ok]]
    print(f"\n  {'quantity':<26}{'exact':>12}{'A(theta) weighted':>20}{'diff':>9}")
    for nm, m in (("nucleons", np.ones(len(d), bool)),
                  ("neutrons", isn), ("protons", isp)):
        e = float(exact[m].sum()); a_ = float(w[m].sum())
        print(f"  {nm:<26}{e:>12,.0f}{a_:>20,.1f}{(a_-e)/max(e,1)*100:>+8.2f}%")

    def R(mask, weights=None):
        hi = (mask & (ek >= EHI)); lo = (mask & (ek < ELO))
        if weights is None:
            return hi.sum() / max(lo.sum(), 1)
        return (weights * hi).sum() / max((weights * lo).sum(), 1e-9)

    print(f"\n  {'observable':<26}{'old band':>11}{'rectangle':>12}{'A-weighted':>12}")
    for nm, m in (("R_n", isn), ("R_p", isp)):
        print(f"  {nm:<26}{R(m & band):>11.5f}{R(m & exact):>12.5f}{R(m, w):>12.5f}")
    npb = (isn & band).sum() / max((isp & band).sum(), 1)
    npe = (isn & exact).sum() / max((isp & exact).sum(), 1)
    npw = (w * isn).sum() / max((w * isp).sum(), 1e-9)
    print(f"  {'n/p':<26}{npb:>11.5f}{npe:>12.5f}{npw:>12.5f}")
    rn_b, rp_b = R(isn & band), R(isp & band)
    rn_e, rp_e = R(isn & exact), R(isp & exact)
    rn_w, rp_w = R(isn, w), R(isp, w)
    print(f"  {'R_n/R_p':<26}{rn_b/rp_b:>11.5f}{rn_e/rp_e:>12.5f}{rn_w/rp_w:>12.5f}")


if __name__ == "__main__":
    main()
