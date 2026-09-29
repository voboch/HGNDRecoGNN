#!/usr/bin/env python3
"""Optimise the spectral-hardness window as a two-parameter problem.

The hardness ratio has been defined by hand throughout: first
N(E>2)/N(E<1) inherited from the truth study, then 1.8-2.4 against >2.6 chosen
by inspection.  Both are arbitrary, and the first puts its denominator where
the detector records two per cent of what arrives.

Here the window is parametrised by a centre and a half-gap,

    low  band :  E <  Rthr - delta
    high band :  E >= Rthr + delta
    R(Rthr, delta) = N_high / N_low

so one number sets where the split sits and the other sets how much of the
migration-prone region around it is discarded.  A larger `delta` buys immunity
to energy-scale and resolution error at the cost of statistics, which is
exactly the trade the optimiser should be making rather than the analyst.

The objective is the S_pot separation, but only for windows that are
admissible:

  * both bands keep at least `--min-counts` entries in every sample, so the
    ratio is not being driven by a handful of clusters;
  * the ordering runs the way the truth scan established, R rising with S_pot,
    since a reconstructed observable moving the other way is not measuring the
    response;
  * for reconstructed energy, both edges sit above `--plateau-min`, the energy
    below which detection efficiency collapses.

Selection happens on development data and the chosen point is applied once to
the held-out sample.  The scan surface is written out so the choice can be seen
to be a broad optimum rather than a spike.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAGS = [("zeroSpot", 0.0), ("defaultSpot", 18.0), ("bigSpot", 90.0)]


def load_truth(events_dir, hist_dir):
    """Per-job (species, Ekin) spectra inside the front-face acceptance."""
    out = {}
    for tag, _ in TAGS:
        p = os.path.join(hist_dir, f"{tag}_filehist.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        keys = sorted(k for k in z.files if k.endswith("|acc"))
        out[tag] = {"per_job": np.stack([z[k] for k in keys]).astype(float),
                    "edges": z["_e_edges"]}
    return out


def load_reco(pred_dir, purity):
    from reco_experiment import purity_locked_threshold
    out = {}
    for tag, _ in TAGS:
        p = os.path.join(pred_dir, tag, "pred_clusters_smash.pkl")
        if not os.path.exists(p):
            continue
        c = pd.read_pickle(p)
        t, pur, _ = purity_locked_threshold(c, purity)
        sel = c[c.cl_score > t]
        out[tag] = {"e": sel.e_pred.to_numpy(float),
                    "unit": sel.Row.to_numpy(),          # event id, the resample unit
                    "threshold": float(t), "purity": float(pur)}
    return out


def _counts_truth(per_job, edges, lo_hi, species=0):
    """Counts per job in the low and high bands."""
    (lo_edge, hi_edge) = lo_hi
    c = np.sqrt(np.maximum(edges[:-1], 1e-12) * edges[1:])
    lo_m, hi_m = c < lo_edge, c >= hi_edge
    s = per_job[:, species, :]
    return s[:, lo_m].sum(axis=1), s[:, hi_m].sum(axis=1)


def _counts_reco(e, unit, lo_hi):
    lo_edge, hi_edge = lo_hi
    uniq, inv = np.unique(unit, return_inverse=True)
    lo = np.bincount(inv, weights=(e < lo_edge).astype(float), minlength=len(uniq))
    hi = np.bincount(inv, weights=(e >= hi_edge).astype(float), minlength=len(uniq))
    return lo, hi


def evaluate(units, rng, n_boot):
    """R per sample and the 90/0 relative response, resampling the given units."""
    vals, boot = {}, {}
    for tag, _ in TAGS:
        if tag not in units:
            return None
        lo, hi = units[tag]
        if lo.sum() < 1 or hi.sum() < 1:
            return None
        vals[tag] = hi.sum() / lo.sum()
        n = len(lo)
        idx = rng.integers(0, n, (n_boot, n))
        boot[tag] = hi[idx].sum(axis=1) / np.maximum(lo[idx].sum(axis=1), 1e-9)
    rel = vals["bigSpot"] / vals["zeroSpot"] - 1.0
    b = boot["bigSpot"] / boot["zeroSpot"] - 1.0
    sd = float(np.nanstd(b, ddof=1))
    v = [vals[t] for t, _ in TAGS]
    return {"R": vals, "rel": float(rel), "err": sd,
            "sigma": float(abs(rel) / sd) if sd > 0 else 0.0,
            "ordered": bool(v[0] < v[1] < v[2]),
            "n_lo": float(sum(units[t][0].sum() for t, _ in TAGS)),
            "n_hi": float(sum(units[t][1].sum() for t, _ in TAGS)),
            "min_band": float(min(min(units[t][0].sum(), units[t][1].sum())
                                  for t, _ in TAGS))}


def scan(kind, data, grid_r, grid_d, rng, n_boot, min_counts, plateau_min):
    rows = []
    for rthr in grid_r:
        for delta in grid_d:
            lo_edge, hi_edge = rthr - delta, rthr + delta
            if lo_edge <= 0:
                continue
            if kind == "reco" and lo_edge < plateau_min:
                continue
            units = {}
            for tag, _ in TAGS:
                if tag not in data:
                    continue
                if kind == "truth":
                    units[tag] = _counts_truth(data[tag]["per_job"],
                                               data[tag]["edges"], (lo_edge, hi_edge))
                else:
                    units[tag] = _counts_reco(data[tag]["e"], data[tag]["unit"],
                                              (lo_edge, hi_edge))
            res = evaluate(units, rng, n_boot)
            if res is None or res["min_band"] < min_counts:
                continue
            res.update({"Rthr": float(rthr), "delta": float(delta),
                        "lo_edge": float(lo_edge), "hi_edge": float(hi_edge),
                        "fom": res["sigma"] if res["ordered"] else 0.0})
            rows.append(res)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["truth", "reco"], required=True)
    p.add_argument("--hist-dir", default=None)
    p.add_argument("--events-dir", default=None)
    p.add_argument("--pred-dir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--rthr", nargs=3, type=float, default=[0.8, 3.2, 0.1],
                   metavar=("LO", "HI", "STEP"))
    p.add_argument("--delta", nargs=3, type=float, default=[0.0, 1.0, 0.1],
                   metavar=("LO", "HI", "STEP"))
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--min-counts", type=float, default=2000)
    p.add_argument("--plateau-min", type=float, default=1.0,
                   help="reconstructed windows may not reach below this energy")
    p.add_argument("--purity", type=float, default=0.7)
    p.add_argument("--holdout-frac", type=float, default=0.35,
                   help="fraction of resample units reserved; the optimum is "
                        "chosen on the rest and applied once to these")
    a = p.parse_args()

    rng = np.random.default_rng(20260929)
    data = (load_truth(a.events_dir, a.hist_dir) if a.kind == "truth"
            else load_reco(a.pred_dir, a.purity))
    if not data:
        raise SystemExit("no input data found")

    # split the resample units into development and held-out halves
    dev, hold = {}, {}
    for tag, _ in TAGS:
        if tag not in data:
            continue
        if a.kind == "truth":
            n = len(data[tag]["per_job"])
            perm = rng.permutation(n); cut = int(n * (1 - a.holdout_frac))
            dev[tag] = {"per_job": data[tag]["per_job"][perm[:cut]],
                        "edges": data[tag]["edges"]}
            hold[tag] = {"per_job": data[tag]["per_job"][perm[cut:]],
                         "edges": data[tag]["edges"]}
        else:
            u = np.unique(data[tag]["unit"])
            perm = rng.permutation(len(u)); cut = int(len(u) * (1 - a.holdout_frac))
            dset, hset = set(u[perm[:cut]]), set(u[perm[cut:]])
            m = np.isin(data[tag]["unit"], list(dset))
            dev[tag] = {"e": data[tag]["e"][m], "unit": data[tag]["unit"][m]}
            hold[tag] = {"e": data[tag]["e"][~m], "unit": data[tag]["unit"][~m]}

    gr = np.round(np.arange(a.rthr[0], a.rthr[1] + 1e-9, a.rthr[2]), 3)
    gd = np.round(np.arange(a.delta[0], a.delta[1] + 1e-9, a.delta[2]), 3)
    print(f"scanning {len(gr)} x {len(gd)} = {len(gr)*len(gd)} windows "
          f"({a.kind}), development units only")
    df = scan(a.kind, dev, gr, gd, rng, a.n_boot, a.min_counts, a.plateau_min)
    if df.empty:
        raise SystemExit("no admissible window: relax --min-counts or the grid")
    df = df.sort_values("fom", ascending=False).reset_index(drop=True)
    n_ord = int(df.ordered.sum())
    print(f"admissible: {len(df)}   ordered in S_pot: {n_ord}\n")
    print(f"  {'Rthr':>6}{'delta':>7}{'low band':>12}{'high band':>12}"
          f"{'90/0-1':>10}{'sigma':>7}")
    for _, r in df.head(8).iterrows():
        print(f"  {r.Rthr:>6.2f}{r.delta:>7.2f}{'E < %.2f' % r.lo_edge:>12}"
              f"{'E >= %.2f' % r.hi_edge:>12}{r.rel*100:>+9.2f}%{r.sigma:>7.2f}")

    if n_ord == 0:
        # Sorting by a figure of merit that is zero everywhere would hand back
        # an arbitrary row, and carrying it to the held-out sample would dress
        # a null up as a measurement. Report the failure instead.
        print("  NO ADMISSIBLE WINDOW ORDERS THE THREE SAMPLES.")
        print("  The scan covered every window in the plateau with enough")
        print("  statistics; none has R rising with S_pot. There is nothing to")
        print("  carry to the held-out sample, and the best-looking window here")
        print("  would be a selection artefact.")
        best_pos = max((r for _, r in df.iterrows() if r.rel > 0),
                       key=lambda r: r.sigma, default=None)
        if best_pos is not None:
            print(f"\n  for the record, the strongest positive-but-unordered "
                  f"window is Rthr = {best_pos.Rthr:.2f}, delta = "
                  f"{best_pos.delta:.2f}: {best_pos.rel*100:+.2f} % at "
                  f"{best_pos.sigma:.2f} sigma")
        json.dump({"kind": a.kind, "ordered_windows": 0,
                   "admissible_windows": int(len(df)),
                   "grid": {"rthr": gr.tolist(), "delta": gd.tolist()},
                   "min_counts": a.min_counts, "plateau_min": a.plateau_min,
                   "chosen": None, "held_out": None,
                   "surface": df.to_dict("records")},
                  open(a.out, "w"), indent=1, default=float)
        print(f"\nwrote {a.out}")
        return

    best = df.iloc[0]
    print(f"\n  chosen on development data: Rthr = {best.Rthr:.2f}, "
          f"delta = {best.delta:.2f}")

    units = {}
    for tag, _ in TAGS:
        if tag not in hold:
            continue
        if a.kind == "truth":
            units[tag] = _counts_truth(hold[tag]["per_job"], hold[tag]["edges"],
                                       (best.lo_edge, best.hi_edge))
        else:
            units[tag] = _counts_reco(hold[tag]["e"], hold[tag]["unit"],
                                      (best.lo_edge, best.hi_edge))
    ho = evaluate(units, rng, max(a.n_boot, 500))
    print("\n" + "=" * 68)
    print("  HELD-OUT UNITS — window fixed above, not tuned here")
    print("=" * 68)
    if ho is None:
        print("  held-out sample too small for this window")
    else:
        for tag, u in TAGS:
            print(f"    S_pot = {u:>4.0f} MeV   R = {ho['R'][tag]:.4f}")
        print(f"    90/0 - 1 = {ho['rel']*100:+.2f} % +- {ho['err']*100:.2f} % "
              f"({ho['sigma']:.2f} sigma)   ordered: {ho['ordered']}")

    json.dump({"kind": a.kind,
               "grid": {"rthr": gr.tolist(), "delta": gd.tolist()},
               "min_counts": a.min_counts, "plateau_min": a.plateau_min,
               "holdout_frac": a.holdout_frac,
               "chosen": {k: float(best[k]) for k in
                          ("Rthr", "delta", "lo_edge", "hi_edge", "rel", "err",
                           "sigma")},
               "held_out": ho,
               "surface": df.to_dict("records")},
              open(a.out, "w"), indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
