"""S_pot sensitivity with the working point chosen on train/val and applied to test.

The quantity is the reconstructed neutron hardness ratio

    R_n = N(E_reco >= E_hi) / N(E_lo_a <= E_reco < E_lo_b)

and both the cluster score threshold and the energy window are free.  Choosing
them on the same events used to quote the result would be circular, so the scan
runs on train+val and exactly one configuration is carried to test.

The figure of merit requires the three samples to be *ordered* in S_pot before
it rewards separation.  A large 90-vs-0 difference with the 18 MeV point out of
place is not a response to the parameter, which is the same argument that
rejected the n/p yield ratio at truth level.

Energy calibration is fitted on the training split only and applied unchanged.
"""
from __future__ import annotations
import argparse, itertools, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reco_truth_join import row_offsets, to_local
from reco_experiment import fit_energy_calibration, apply_calibration

TAGS = [("zeroSpot", 0), ("defaultSpot", 18), ("bigSpot", 90)]
SCORES = np.round(np.arange(0.10, 0.91, 0.05), 2)
# The hardness window is a free parameter, chosen to suit the detector and the
# reconstruction rather than inherited from the truth-level definition.  The
# legacy pair -- denominator below 1 GeV, numerator above 2 GeV -- puts the
# denominator where detection efficiency is 2 %, so it is excluded here on
# purpose; every window below keeps both arms inside the efficiency plateau and
# inside the energy estimator's usable range.
WINDOWS = [((0.7, 1.5), 2.0), ((0.7, 1.5), 2.5),
           ((0.8, 1.6), 2.0), ((0.8, 1.6), 2.4),
           ((1.0, 1.5), 2.0), ((1.0, 1.5), 2.2),
           ((1.0, 1.8), 2.0), ((1.0, 1.8), 2.2), ((1.0, 1.8), 2.6),
           ((1.0, 2.0), 2.0), ((1.0, 2.0), 2.5),
           ((1.2, 2.0), 2.0), ((1.2, 2.0), 2.4),
           ((1.5, 2.0), 2.0), ((1.5, 2.0), 2.5),
           ((1.5, 2.2), 2.4), ((1.8, 2.4), 2.6)]


def load_split(pred_dir, raw_dir, split, cal):
    """Predictions for one split, tagged by sample and job file."""
    c = pd.read_pickle(os.path.join(pred_dir, split, "pred_clusters_smash.pkl"))
    offs = row_offsets(os.path.join(raw_dir, split))
    loc = to_local(c.Row.to_numpy(), offs)
    if loc.unmapped.any():
        raise AssertionError(f"{split}: {int(loc.unmapped.sum())} unmapped rows")
    # run is the sample tag; stems repeat across samples so the job-file key
    # must carry both
    c = c.assign(stem=loc.stem.to_numpy(), **{"sample": loc.run.to_numpy()})
    c["key"] = c["sample"] + "/" + c["stem"]
    if cal is not None:
        c["e_cal"] = apply_calibration(c.e_pred.to_numpy(), cal)
    return c


def ratios(c, score, win, rng=None, n_boot=0, ecol="e_cal"):
    (lo_a, lo_b), hi = win
    sel = c[c.cl_score > score]
    out, boot = {}, {}
    for tag, _ in TAGS:
        d = sel[sel["sample"] == tag]
        keys = np.array(sorted(d.key.unique()))
        if len(keys) == 0:
            return None, None
        e = d[ecol].to_numpy()
        k = d.key.to_numpy()
        hi_by = pd.Series(e >= hi).groupby(k).sum().reindex(keys, fill_value=0).to_numpy(float)
        lo_by = pd.Series((e >= lo_a) & (e < lo_b)).groupby(k).sum().reindex(keys, fill_value=0).to_numpy(float)
        if lo_by.sum() < 50 or hi_by.sum() < 50:
            return None, None
        out[tag] = hi_by.sum() / lo_by.sum()
        if n_boot:
            idx = rng.integers(0, len(keys), (n_boot, len(keys)))
            boot[tag] = hi_by[idx].sum(axis=1) / np.maximum(lo_by[idx].sum(axis=1), 1e-9)
    return out, boot


def fom(vals, boot, require_increasing=True):
    """Separation significance, but only for an ordered scan.

    `require_increasing` demands the ordering run the way the truth-level scan
    established it -- R_n rises with S_pot, +4.27 % at 21.7 sigma.  That is prior
    physics, fixed before the test set is opened, not a choice made on the test
    data.  Without it the figure of merit rewards a monotonic *decrease* just as
    much, and a reconstructed observable moving opposite to the truth is not
    measuring the response.
    """
    v = [vals[t] for t, _ in TAGS]
    ordered = (v[0] < v[1] < v[2]) if require_increasing else \
              ((v[0] < v[1] < v[2]) or (v[0] > v[1] > v[2]))
    rel = v[2] / v[0] - 1
    b = boot["bigSpot"] / boot["zeroSpot"] - 1
    sd = float(np.nanstd(b, ddof=1))
    sig = abs(rel) / sd if sd > 0 else 0.0
    return (sig if ordered else 0.0), ordered, rel, sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", required=True, help="dir with train/ val/ test/ subdirs")
    p.add_argument("--raw-dir", required=True, help="dir with train/ val/ test/ CSV trees")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--any-direction", action="store_true",
                   help="accept a monotonic decrease as well; the default "
                        "requires the direction the truth-level scan established")
    p.add_argument("--raw-energy", action="store_true",
                   help="use e_pred directly instead of the calibrated energy")
    a = p.parse_args()

    rng = np.random.default_rng(99)
    ecol = "e_pred" if a.raw_energy else "e_cal"
    cal = None
    if not a.raw_energy:
        tr = pd.read_pickle(os.path.join(a.pred_dir, "train", "pred_clusters_smash.pkl"))
        cal = fit_energy_calibration(tr)
        print(f"calibration fitted on train: e_pred {cal[0][0]:.2f}..{cal[0][-1]:.2f}"
              f" -> e_true {cal[1][0]:.2f}..{cal[1][-1]:.2f} GeV")

    data = {s: load_split(a.pred_dir, a.raw_dir, s, cal) for s in ("train", "val", "test")}
    dev = pd.concat([data["train"], data["val"]], ignore_index=True)
    print(f"development set: {len(dev):,} clusters   test: {len(data['test']):,}")

    rows = []
    for score, win in itertools.product(SCORES, WINDOWS):
        vals, boot = ratios(dev, score, win, rng, a.n_boot, ecol)
        if vals is None:
            continue
        f, ordered, rel, sd = fom(vals, boot, not a.any_direction)
        rows.append({"score": float(score), "window": str(win), "fom": f,
                     "ordered": ordered, "rel": rel, "err": sd,
                     **{f"R_{t}": vals[t] for t, _ in TAGS}})
    scan = pd.DataFrame(rows).sort_values("fom", ascending=False)
    print(f"\n  configurations scanned on train+val: {len(scan)}"
          f"   ordered: {int(scan.ordered.sum())}")
    print(f"\n  {'score':>6}{'window':>18}{'R(0)':>9}{'R(18)':>9}{'R(90)':>9}"
          f"{'90/0-1':>10}{'FOM':>7}")
    for _, r in scan.head(8).iterrows():
        print(f"  {r.score:>6.2f}{r.window:>18}{r.R_zeroSpot:>9.4f}"
              f"{r.R_defaultSpot:>9.4f}{r.R_bigSpot:>9.4f}{r.rel*100:>+9.2f}%{r.fom:>7.2f}")

    if not len(scan) or scan.iloc[0].fom == 0:
        print("\n  no ordered configuration on train+val — nothing carried to test")
        best = None
    else:
        best = scan.iloc[0]
        win = eval(best.window)
        print(f"\n  chosen on train+val: score > {best.score:.2f}, window {best.window}")
        vals, boot = ratios(data["test"], best.score, win, rng, max(a.n_boot, 400), ecol)
        if vals is None:
            print("  test set too small for this configuration")
        else:
            f, ordered, rel, sd = fom(vals, boot, not a.any_direction)
            print("\n" + "=" * 70)
            print("  TEST SET — configuration fixed beforehand, not tuned here")
            print("=" * 70)
            for t, u in TAGS:
                print(f"    S_pot = {u:>2} MeV   R_n = {vals[t]:.4f}")
            print(f"    90/0 - 1 = {rel*100:+.2f} % +- {sd*100:.2f} %  ({abs(rel)/sd:.2f} sigma)")
            print(f"    ordered in S_pot: {ordered}")

    out = {"scan": scan.to_dict("records"),
           "chosen": None if best is None else {
               "score": float(best.score), "window": best.window},
           "test": None if best is None or vals is None else {
               "R": {t: float(vals[t]) for t, _ in TAGS},
               "rel": float(rel), "err": float(sd),
               "sigma": float(abs(rel) / sd) if sd else None, "ordered": bool(ordered)}}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
