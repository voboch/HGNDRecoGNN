#!/usr/bin/env python3
"""Response of the hardness observables to S_pot, from the three-point scan.

With samples at S_pot = 0, 18 and 90 MeV the scan can do three things the
two-point comparison could not: check that the middle point lies between the
outer two (a signal that is not monotonic in the parameter is not a response to
it), test whether the response is linear over the range, and convert the
measured slope into the S_pot resolution the observable would support.

The resolution quoted is statistical and truth-level.  It uses the null test's
measured noise floor -- the median difference between two halves of one
production -- as the smallest difference the method can resolve, and it ignores
every systematic, including the reconstruction response studied separately.
"""
from __future__ import annotations
import argparse, json
import numpy as np

KEY = ["Rn_over_Rp_band", "R_n_band", "R_p_band", "R_n_mid", "np_band", "np_4pi"]
ORDER = [("zeroSpot", 0.0), ("defaultSpot", 18.0), ("bigSpot", 90.0)]


def wfit(x, y, e):
    """Weighted straight-line fit; returns (intercept, slope, slope error)."""
    w = 1.0 / np.asarray(e) ** 2
    S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
    Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
    d = S * Sxx - Sx * Sx
    a = (Sxx * Sy - Sx * Sxy) / d
    b = (S * Sxy - Sx * Sy) / d
    return a, b, np.sqrt(S / d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--nulls", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rep = json.load(open(a.report))
    floors = {}
    for f in a.nulls:
        d = json.load(open(f))
        for k, v in d.items():
            floors.setdefault(k, []).append(v["median_abs_rel"])
    floor = {k: float(np.mean(v)) for k, v in floors.items()}

    x = np.array([u for _, u in ORDER])
    out = {}
    print("=" * 92)
    print("  Response to S_pot across the three-point scan (reweighted, job-level errors)")
    print("=" * 92)
    print(f"{'observable':<18}{'U=0':>10}{'U=18':>10}{'U=90':>10}"
          f"{'slope %/MeV':>13}{'linearity':>11}{'floor %':>9}{'dS_pot MeV':>12}")
    for k in KEY:
        if k not in rep["integrated"]:
            continue
        ig = rep["integrated"][k]
        y = np.array([ig["central"][s] for s, _ in ORDER])
        e = np.array([ig["err"][s] for s, _ in ORDER])
        a0, b0, be = wfit(x, y, e)

        # linearity: the middle point against the line through the outer two
        lin = y[0] + (y[2] - y[0]) * x[1] / x[2]
        lin_e = np.hypot((1 - x[1] / x[2]) * e[0], (x[1] / x[2]) * e[2])
        dev = y[1] - lin
        dev_s = abs(dev) / np.hypot(lin_e, e[1])

        rel_slope = b0 / y[0] * 100.0             # % of the U=0 value per MeV
        fl = floor.get(k, np.nan) * 100.0
        dspot = fl / abs(rel_slope) if rel_slope else np.nan
        mono = (y[0] < y[1] < y[2]) or (y[0] > y[1] > y[2])
        out[k] = {"values": y.tolist(), "errors": e.tolist(),
                  "slope_per_MeV": float(b0), "slope_err": float(be),
                  "rel_slope_pct_per_MeV": float(rel_slope),
                  "monotonic": bool(mono),
                  "linearity_dev": float(dev), "linearity_sigma": float(dev_s),
                  "noise_floor_pct": float(fl), "S_pot_resolution_MeV": float(dspot)}
        print(f"{k:<18}{y[0]:>10.5f}{y[1]:>10.5f}{y[2]:>10.5f}"
              f"{rel_slope:>+13.4f}{dev_s:>10.1f}σ{fl:>9.3f}{dspot:>12.1f}"
              + ("" if mono else "   NOT MONOTONIC"))

    print("\n  slope is per MeV of S_pot, as a percentage of the S_pot = 0 value.")
    print("  linearity is the middle point's deviation from the line through the outer two.")
    print("  dS_pot is the noise floor divided by the slope: the smallest S_pot difference")
    print("  this observable resolves at truth level, before any systematic.")
    with open(a.out, "w") as f:
        json.dump({"floors": floor, "response": out}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
