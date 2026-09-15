#!/usr/bin/env python
"""Compare the pre-bugfix and re-simulated ("fixed") SMASH generations.

Runs on raw CSV trees, so it needs no cache and no GPU. Both generations must
be extracted side by side; matched event files (same basename) are compared.

    python -m HGNDRecoGNN.scripts.compare_data_versions \
        --old-root  data \
        --new-root  data_fixed \
        --out-dir   results/data_version_comparison \
        --n-files   150

Note on the CSV schema: `*_vacs.csv` is *hit-aligned* — one row per hit,
identifying the MC particle (`Id`) that produced it. A particle that makes
five hits appears five times. True per-event multiplicities therefore require
de-duplication on (`Row`, `Id`); counting raw vacs rows yields hit-weighted
composition instead, which is a different (and usually not the intended)
quantity.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

DATASETS = ("zeroSpot", "defaultSpot", "bigSpot")
U_SYM = {"zeroSpot": 0.0, "defaultSpot": 18.0, "bigSpot": 90.0}   # MeV
PDG_NAMES = {2112: "n", 2212: "p", 22: "gam", 211: "pip", -211: "pim"}
RUNID = "10943245"


def _tree(root, ds):
    return os.path.join(root, f"smash_xecs_2.87gev_hardSkyrme_{ds}", RUNID)


def matched_stems(old_root, new_root, ds, n_files):
    """Event-file basenames present in both generations, evenly sampled."""
    def stems(root):
        d = _tree(root, ds)
        if not os.path.isdir(d):
            return set()
        return {f[:-len("_hits.csv")] for f in os.listdir(d) if f.endswith("_hits.csv")}
    common = sorted(stems(old_root) & stems(new_root))
    if not common:
        return []
    step = max(len(common) // n_files, 1)
    return common[::step][:n_files]


def scan(root, ds, stems):
    """Per-event hit and MC-particle aggregates, plus the neutron pool."""
    per_ev, neutrons = [], []
    for st in stems:
        ph = os.path.join(_tree(root, ds), f"{st}_hits.csv")
        pv = os.path.join(_tree(root, ds), f"{st}_vacs.csv")
        if not (os.path.exists(ph) and os.path.exists(pv)):
            continue
        h = pd.read_csv(ph, usecols=["Row", "fELoss", "fTime", "LayerId"])
        v = pd.read_csv(pv, usecols=["Row", "Id", "PDG", "Ekin", "Rapid", "Weight"])
        hg = h.groupby("Row").agg(n_hits=("fELoss", "size"), eloss=("fELoss", "sum"),
                                  t_min=("fTime", "min"), layer=("LayerId", "mean"))
        uniq = v.drop_duplicates(subset=["Row", "Id"])          # true multiplicity
        mult = uniq.groupby(["Row", "PDG"]).size().unstack(fill_value=0)
        d = hg.join(mult.rename(columns=PDG_NAMES), how="left").fillna(0.0)
        d["n_part"] = uniq.groupby("Row").size()
        per_ev.append(d)
        nn = uniq[uniq.PDG == 2112]
        if len(nn):
            neutrons.append(nn[["Ekin", "Rapid", "Weight"]].to_numpy())
    if not per_ev:
        return None, np.empty((0, 3))
    return pd.concat(per_ev), (np.vstack(neutrons) if neutrons else np.empty((0, 3)))


def summarise(df, neu, ds, side):
    r = {"dataset": ds, "side": side, "U_sym": U_SYM[ds], "n_events": len(df)}
    r["hits_per_ev"] = df.n_hits.mean()
    r["eloss_per_ev"] = df.eloss.mean()
    r["tmin_mean"] = df.t_min.mean()
    r["layer_mean"] = df.layer.mean()
    r["part_per_ev"] = df.n_part.mean()
    for nm in PDG_NAMES.values():
        r[f"M_{nm}"] = df[nm].mean() if nm in df else 0.0
    r["M_n_std"] = df["n"].std() if "n" in df else np.nan
    if len(neu):
        r["neu_Ekin_mean"] = neu[:, 0].mean()
        r["neu_Ekin_med"] = float(np.median(neu[:, 0]))
        r["neu_Rapid_mean"] = neu[:, 1].mean()
        # the observable that actually carries the U_sym signal
        nhi, nlo = int((neu[:, 0] >= 2.0).sum()), int((neu[:, 0] < 1.0).sum())
        r["hardness_R"] = nhi / nlo if nlo else np.nan
        r["hardness_R_err"] = (r["hardness_R"] * np.sqrt(1 / max(nhi, 1) + 1 / max(nlo, 1))
                               if nlo else np.nan)
        r["n_neutrons"] = len(neu)
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old-root", required=True, help="raw CSV tree, pre-bugfix generation")
    p.add_argument("--new-root", required=True, help="raw CSV tree, re-simulated generation")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-files", type=int, default=150,
                   help="matched event files per dataset (default 150)")
    a = p.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    rows, pools = [], {}
    for ds in DATASETS:
        stems = matched_stems(a.old_root, a.new_root, ds, a.n_files)
        if not stems:
            print(f"  {ds}: no matched files in both generations — skipped", file=sys.stderr)
            continue
        for side, root in (("old", a.old_root), ("new", a.new_root)):
            df, neu = scan(root, ds, stems)
            if df is None:
                continue
            rows.append(summarise(df, neu, ds, side))
            pools[f"{ds}_{side}"] = neu
            print(f"  {ds:<12} {side}: {len(df)} events, {len(neu)} neutrons", flush=True)

    summary = pd.DataFrame(rows)
    out = os.path.join(a.out_dir, "data_version_summary.csv")
    summary.to_csv(out, index=False)
    np.savez(os.path.join(a.out_dir, "neutron_pools.npz"), **pools)

    # U_sym response of the hardness ratio. Report pairwise significances, not a
    # bare monotonicity label: at these sample sizes the 0 -> 18 MeV ordering is
    # not resolved, so the label alone flips with the sample and means nothing.
    print("\nspectral hardness R = N(Ekin>2 GeV)/N(Ekin<1 GeV)")
    for side in ("old", "new"):
        t = summary[summary.side == side].set_index("dataset")
        if not len(t):
            continue
        print(f"  --- {side} ---")
        for d in DATASETS:
            if d in t.index:
                print(f"    U={int(U_SYM[d]):>2} MeV   R = {t.loc[d,'hardness_R']:.5f}"
                      f" +- {t.loc[d,'hardness_R_err']:.5f}")
        for a_, b_ in (("zeroSpot", "defaultSpot"), ("zeroSpot", "bigSpot")):
            if a_ in t.index and b_ in t.index:
                d_ = t.loc[b_, "hardness_R"] - t.loc[a_, "hardness_R"]
                se = float(np.hypot(t.loc[a_, "hardness_R_err"], t.loc[b_, "hardness_R_err"]))
                print(f"    {b_} - {a_}: {d_:+.5f} +- {se:.5f}  ->  {abs(d_)/se:.1f} sigma")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
