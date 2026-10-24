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

The objective is the U_sym separation, but only for windows that are
admissible:

  * both bands keep at least `--min-counts` entries in every sample, so the
    ratio is not being driven by a handful of clusters;
  * the ordering runs the way the truth scan established, R rising with U_sym,
    since a reconstructed observable moving the other way is not measuring the
    response;
  * for reconstructed energy, the low-band upper edge sits above
    `--plateau-min`. This legacy constraint does not remove soft entries from
    the denominator and does not establish efficiency across either band.

Selection happens on development data and the chosen point is applied once to
the held-out sample.  The scan surface is written out so the choice can be seen
to be a broad optimum rather than a spike.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from HGNDRecoGNN.analysis.energy_bands import EnergyBands

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


def load_reco(pred_dir, purity, threshold=None):
    from reco_experiment import purity_locked_threshold
    out = {}
    for tag, _ in TAGS:
        p = os.path.join(pred_dir, tag, "pred_clusters_smash.pkl")
        if not os.path.exists(p):
            continue
        c = pd.read_pickle(p)
        if threshold is None:
            t, pur, _ = purity_locked_threshold(c, purity)
        else:
            t = float(threshold)
            pur = float(c.cl_label[c.cl_score > t].mean())
        sel = c[c.cl_score > t]
        out[tag] = {"e": sel.e_pred.to_numpy(float),
                    "unit": sel.Row.to_numpy(),          # event id, the resample unit
                    # Every event in the sample, including the ~81 % with no
                    # candidate above the cut. A yield per event needs them in
                    # the denominator; averaging only over events that survived
                    # the selection measures candidates per *non-empty* event,
                    # which is a different quantity and is much flatter.
                    "all_units": np.unique(c.Row.to_numpy()),
                    "threshold": float(t), "purity": float(pur)}
    return out


def _counts_truth(per_job, edges, lo_hi, species=0, *, energy_bounds=None):
    """Counts per job in the low and high bands."""
    (lo_edge, hi_edge) = lo_hi
    if energy_bounds is None:
        c = np.sqrt(np.maximum(edges[:-1], 1e-12) * edges[1:])
        lo_m, hi_m = c < lo_edge, c >= hi_edge
    else:
        lo_m, hi_m = EnergyBands(energy_bounds[0], lo_edge, hi_edge,
                                energy_bounds[1]).histogram_masks(edges)
    s = per_job[:, species, :]
    return s[:, lo_m].sum(axis=1), s[:, hi_m].sum(axis=1)


def _counts_reco(e, unit, lo_hi, *, energy_bounds=None, all_units=None):
    lo_edge, hi_edge = lo_hi
    if energy_bounds is not None:
        return EnergyBands(energy_bounds[0], lo_edge, hi_edge,
                           energy_bounds[1]).counts(e, unit, all_units)
    uniq, inv = np.unique(unit, return_inverse=True)
    lo = np.bincount(inv, weights=(e < lo_edge).astype(float), minlength=len(uniq))
    hi = np.bincount(inv, weights=(e >= hi_edge).astype(float), minlength=len(uniq))
    return lo, hi


def slope_response(vals, boot):
    """Response from all three U_sym points, not the two endpoints.

    The 90/0 comparison throws away the 18 MeV sample and is a two-point
    difference; a straight-line fit over all three uses every point, tests
    linearity rather than assuming it, and is the quantity a measurement would
    actually report. The slope is normalised to the U_sym = 0 value so it is a
    relative response per MeV and comparable across observables.

    Returns the slope, its bootstrap error, the separation, and the chi-square
    of the straight line, which is the linearity test the two-point difference
    cannot provide.
    """
    u = np.array([mev for _, mev in TAGS], dtype=float)
    y = np.array([vals[t] for t, _ in TAGS], dtype=float)
    B = np.stack([boot[t] for t, _ in TAGS], axis=1)      # (n_boot, 3)

    def _fit(row):
        a, b = np.polyfit(u, row, 1)
        return a

    sl = _fit(y)
    sl_b = np.array([_fit(B[i]) for i in range(B.shape[0])])
    err = float(np.nanstd(sl_b, ddof=1))
    norm = float(y[0]) if y[0] != 0 else np.nan

    # Linearity: residual of the three points about the fitted line, in units
    # of each point's own bootstrap spread. The line has to be the one that
    # *minimises* this chi-square, i.e. weighted least squares, or the reported
    # value is not a minimum and is biased high. The three per-point errors
    # here differ by only 7-16 %, so the two fits agree to about 0.05 in
    # chi-square, but scoring an unweighted fit against weighted residuals is
    # wrong whatever the size of the effect.
    sig = np.array([np.nanstd(boot[t], ddof=1) for t, _ in TAGS])
    if np.all(sig > 0):
        w = 1.0 / sig ** 2
        A = np.vstack([u, np.ones_like(u)]).T
        WA = A * w[:, None]
        coef = np.linalg.solve(A.T @ WA, WA.T @ y)
        resid = y - A @ coef
        chi2 = float(np.sum((resid / sig) ** 2))
    else:
        chi2 = float("nan")

    return {"slope_per_MeV": float(sl),
            "slope_err": err,
            "rel_slope_per_MeV": float(sl / norm) if np.isfinite(norm) else None,
            "sigma_slope": float(abs(sl) / err) if err > 0 else 0.0,
            "chi2_linear_ndf1": chi2}


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
    out_slope = slope_response(vals, boot)
    return {"R": vals,
            "R_err": {t: float(np.nanstd(boot[t], ddof=1)) for t, _ in TAGS},
            "slope": out_slope,
            "rel": float(rel), "err": sd,
            "sigma": float(abs(rel) / sd) if sd > 0 else 0.0,
            "ordered": bool(v[0] < v[1] < v[2]),
            "n_lo": float(sum(units[t][0].sum() for t, _ in TAGS)),
            "n_hi": float(sum(units[t][1].sum() for t, _ in TAGS)),
            "min_band": float(min(min(units[t][0].sum(), units[t][1].sum())
                                  for t, _ in TAGS))}


def _split_single(data, kind, rng, holdout_frac):
    """Development / held-out split of one species' resample units."""
    dev, hold = {}, {}
    for tag, _ in TAGS:
        if tag not in data:
            continue
        if kind == "truth":
            n = len(data[tag]["per_job"])
            perm = rng.permutation(n)
            cut = int(n * (1 - holdout_frac))
            dev[tag] = {"per_job": data[tag]["per_job"][perm[:cut]],
                        "edges": data[tag]["edges"]}
            hold[tag] = {"per_job": data[tag]["per_job"][perm[cut:]],
                         "edges": data[tag]["edges"]}
        else:
            # Split over EVERY event, not only those with a surviving
            # candidate. Partitioning the selected events alone would both
            # bias a per-event yield and make the two halves unequal in the
            # population they represent.
            au = data[tag].get("all_units")
            u = au if au is not None else np.unique(data[tag]["unit"])
            perm = rng.permutation(len(u))
            cut = int(len(u) * (1 - holdout_frac))
            dev_u = np.sort(u[perm[:cut]])
            hold_u = np.sort(u[perm[cut:]])
            m = np.isin(data[tag]["unit"], dev_u)
            dev[tag] = {"e": data[tag]["e"][m], "unit": data[tag]["unit"][m]}
            hold[tag] = {"e": data[tag]["e"][~m], "unit": data[tag]["unit"][~m]}
            if au is not None:
                dev[tag]["all_units"] = dev_u
                hold[tag]["all_units"] = hold_u
    return dev, hold


def _split_hybrid(data, rng, holdout_frac, n_kind="reco"):
    """Split both species, so the held-out double ratio shares no unit with
    the development one on either side."""
    dn, hn = _split_single(data["n"], n_kind, rng, holdout_frac)
    dp, hp = _split_single(data["p"], "truth", rng, holdout_frac)
    return {"n": dn, "p": dp}, {"n": hn, "p": hp}


def evaluate_hybrid(units_n, units_p, rng, n_boot):
    """Double ratio D = R_n(reco) / R_p(truth) and its 90/0 response.

    The two species come from different resample units — reconstructed neutrons
    are counted per event, truth protons per generation job — and the samples
    are not event matched, so the two are resampled independently and their
    fluctuations do not cancel. That makes this a conservative error on D
    relative to a matched double ratio, not an optimistic one.

    R_p is a truth quantity: the HGND does not measure protons. This observable
    is a feasibility bound on what a neutron detector plus an external proton
    measurement could reach, not something this detector can produce alone.
    """
    vals, boot = {}, {}
    for tag, _ in TAGS:
        if tag not in units_n or tag not in units_p:
            return None
        lo_n, hi_n = units_n[tag]
        lo_p, hi_p = units_p[tag]
        if lo_n.sum() < 1 or hi_n.sum() < 1 or lo_p.sum() < 1 or hi_p.sum() < 1:
            return None
        vals[tag] = (hi_n.sum() / lo_n.sum()) / (hi_p.sum() / lo_p.sum())

        i_n = rng.integers(0, len(lo_n), (n_boot, len(lo_n)))
        i_p = rng.integers(0, len(lo_p), (n_boot, len(lo_p)))
        b_n = hi_n[i_n].sum(axis=1) / np.maximum(lo_n[i_n].sum(axis=1), 1e-9)
        b_p = hi_p[i_p].sum(axis=1) / np.maximum(lo_p[i_p].sum(axis=1), 1e-9)
        boot[tag] = b_n / np.maximum(b_p, 1e-12)

    rel = vals["bigSpot"] / vals["zeroSpot"] - 1.0
    b = boot["bigSpot"] / np.maximum(boot["zeroSpot"], 1e-12) - 1.0
    sd = float(np.nanstd(b, ddof=1))
    v = [vals[t] for t, _ in TAGS]
    out_slope = slope_response(vals, boot)
    return {"R": vals,
            "R_err": {t: float(np.nanstd(boot[t], ddof=1)) for t, _ in TAGS},
            "slope": out_slope,
            "rel": float(rel), "err": sd,
            "sigma": float(abs(rel) / sd) if sd > 0 else 0.0,
            "ordered": bool(v[0] < v[1] < v[2]),
            "n_lo": float(sum(units_n[t][0].sum() for t, _ in TAGS)),
            "n_hi": float(sum(units_n[t][1].sum() for t, _ in TAGS)),
            "min_band": float(min(min(units_n[t][0].sum(), units_n[t][1].sum())
                                  for t, _ in TAGS))}


def evaluate_yield(data_n, data_p, rng, n_boot, need_reco):
    """Integrated-yield baselines, on the same units as the windowed scans.

    The yield observables carry no free window, so nothing is selected on them
    and no development stage is needed. They are still evaluated on the *held
    out* units alone, because quoting them on the full sample would give them
    a smaller error than the hardness ratios purely from using three times the
    statistics, and the point of the table is to compare sensitivity.

    Returns the truth n/p ratio, and when reconstructed predictions are given,
    the reconstructed neutron candidate yield per event and the hybrid
    candidate-yield-over-truth-proton-yield.
    """
    out = {}

    def _resp(vals, boot):
        rel = vals["bigSpot"] / vals["zeroSpot"] - 1.0
        b = boot["bigSpot"] / np.maximum(boot["zeroSpot"], 1e-12) - 1.0
        sd = float(np.nanstd(b, ddof=1))
        v = [vals[t] for t, _ in TAGS]
        return {"R": vals,
                "R_err": {t: float(np.nanstd(boot[t], ddof=1)) for t, _ in TAGS},
                "slope": slope_response(vals, boot),
                "rel": float(rel), "err": sd,
                "sigma": float(abs(rel) / sd) if sd > 0 else 0.0,
                "ordered": bool(v[0] < v[1] < v[2])}

    # --- truth n/p, per generation job -------------------------------------
    vals, boot, yp_vals, yp_boot = {}, {}, {}, {}
    for tag, _ in TAGS:
        per = data_p[tag]["per_job"]
        yn = per[:, 0, :].sum(axis=1)
        yp = per[:, 1, :].sum(axis=1)
        vals[tag] = float(yn.sum() / yp.sum())
        idx = rng.integers(0, len(yn), (n_boot, len(yn)))
        boot[tag] = yn[idx].sum(axis=1) / np.maximum(yp[idx].sum(axis=1), 1e-9)
        yp_vals[tag], yp_boot[tag] = yp, idx
    out["truth_np"] = _resp(vals, boot)

    if not need_reco:
        return out

    # --- reconstructed candidate yield per event ---------------------------
    vals_n, boot_n, n_evt = {}, {}, {}
    for tag, _ in TAGS:
        u = data_n[tag]["unit"]
        allu = data_n[tag].get("all_units")
        if allu is None:
            raise ValueError("load_reco must supply all_units for a yield")
        # one entry per event, zero where the selection kept nothing
        pos = np.searchsorted(allu, u)
        per_evt = np.bincount(pos, minlength=len(allu)).astype(float)
        vals_n[tag] = float(per_evt.mean())
        n_evt[tag] = len(allu)
        idx = rng.integers(0, len(per_evt), (n_boot, len(per_evt)))
        boot_n[tag] = per_evt[idx].mean(axis=1)
    out["reco_yield"] = _resp(vals_n, boot_n)

    # --- hybrid: reconstructed neutron yield over truth proton yield -------
    vh, bh = {}, {}
    for tag, _ in TAGS:
        yp = yp_vals[tag]
        idx_p = rng.integers(0, len(yp), (n_boot, len(yp)))
        vh[tag] = vals_n[tag] / float(yp.mean())
        bh[tag] = boot_n[tag] / np.maximum(yp[idx_p].mean(axis=1), 1e-9)
    out["hybrid_yield"] = _resp(vh, bh)
    return out


def scan(kind, data, grid_r, grid_d, rng, n_boot, min_counts, plateau_min):
    rows = []
    for rthr in grid_r:
        for delta in grid_d:
            lo_edge, hi_edge = rthr - delta, rthr + delta
            if lo_edge <= 0:
                continue
            if kind in ("reco", "hybrid") and lo_edge < plateau_min:
                continue
            if kind in ("hybrid", "truth_dr"):
                un, up = {}, {}
                for tag, _ in TAGS:
                    if tag not in data["n"] or tag not in data["p"]:
                        continue
                    if kind == "truth_dr":
                        un[tag] = _counts_truth(data["n"][tag]["per_job"],
                                                data["n"][tag]["edges"],
                                                (lo_edge, hi_edge), species=0)
                    else:
                        un[tag] = _counts_reco(data["n"][tag]["e"],
                                               data["n"][tag]["unit"],
                                               (lo_edge, hi_edge))
                    # species=1 is the proton row of the truth histogram
                    up[tag] = _counts_truth(data["p"][tag]["per_job"],
                                            data["p"][tag]["edges"],
                                            (lo_edge, hi_edge), species=1)
                res = evaluate_hybrid(un, up, rng, n_boot)
            else:
                units = {}
                for tag, _ in TAGS:
                    if tag not in data:
                        continue
                    if kind == "truth":
                        units[tag] = _counts_truth(data[tag]["per_job"],
                                                   data[tag]["edges"],
                                                   (lo_edge, hi_edge))
                    else:
                        units[tag] = _counts_reco(data[tag]["e"],
                                                  data[tag]["unit"],
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
    p.add_argument("--kind",
                   choices=["truth", "reco", "hybrid", "truth_dr", "yield"],
                   required=True,
                   help="hybrid = R_n(reconstructed) / R_p(truth), the same "
                        "window applied to both species. It needs --pred-dir "
                        "and --hist-dir, and it is calibrated exactly as the "
                        "other two are, on the same grid and the same "
                        "development/held-out split. truth_dr = the ideal "
                        "R_n/R_p with both species at truth, the upper bound "
                        "the hybrid is chasing.")
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
    p.add_argument("--threshold", type=float, default=None,
                   help="Fixed score cut applied to every sample, overriding "
                        "--purity. Purity-locking exists because a fixed cut "
                        "can select different physical purity in different "
                        "samples, and that difference would masquerade as an "
                        "U_sym effect. With the per-sample scaler artefact "
                        "removed the risk is not live: at q > 0.60 the three "
                        "samples agree to 0.14 pp in count purity and 0.08 pp "
                        "in recall. 0.60 is the 87 percent energy-purity "
                        "working point.")
    p.add_argument("--window-from", default=None, metavar="SCAN_JSON",
                   help="Take the window from another scan instead of "
                        "optimising it here. The window belongs to the "
                        "measurement, and the measurement is reconstructed: a "
                        "window optimised on MC truth is optimised for a "
                        "detector that does not exist, and carrying it to "
                        "reconstruction would be a different observable. So "
                        "the reconstructed scan selects, and every other row "
                        "is evaluated at that same window as a reference.")
    p.add_argument("--holdout-frac", type=float, default=0.35,
                   help="fraction of resample units reserved; the optimum is "
                        "chosen on the rest and applied once to these")
    a = p.parse_args()

    rng = np.random.default_rng(20260929)
    if a.kind == "truth":
        data = load_truth(a.events_dir, a.hist_dir)
    elif a.kind == "reco":
        data = load_reco(a.pred_dir, a.purity, a.threshold)
    elif a.kind == "yield":
        tp = load_truth(a.events_dir, a.hist_dir)
        tn = load_reco(a.pred_dir, a.purity, a.threshold) if a.pred_dir else {}
        data = {"n": tn, "p": tp}
        if not tp:
            raise SystemExit("yield needs --hist-dir")
    elif a.kind == "truth_dr":
        t = load_truth(a.events_dir, a.hist_dir)
        data = {"n": t, "p": t}
        if not t:
            raise SystemExit("truth_dr needs --hist-dir")
    else:
        # hybrid: reconstructed neutrons over truth protons, each with its own
        # resample unit
        data = {"n": load_reco(a.pred_dir, a.purity, a.threshold),
                "p": load_truth(a.events_dir, a.hist_dir)}
        if not data["n"] or not data["p"]:
            raise SystemExit("hybrid needs both --pred-dir and --hist-dir")
    if not data:
        raise SystemExit("no input data found")

    if a.kind == "yield":
        # same split code path, so the held-out units match the windowed scans
        _, hold_p = _split_single(data["p"], "truth", rng, a.holdout_frac)
        hold_n = {}
        if data["n"]:
            _, hold_n = _split_single(data["n"], "reco", rng, a.holdout_frac)
        res = evaluate_yield(hold_n, hold_p, rng, max(a.n_boot, 500),
                             need_reco=bool(data["n"]))
        print("Integrated-yield baselines, held-out units, no window to "
              "calibrate\n")
        for k, r in res.items():
            print(f"  {k:14s} {100*r['rel']:>+8.3f} % +- {100*r['err']:<6.3f} % "
                  f"({r['sigma']:.2f} sigma)   ordered: {r['ordered']}")
        with open(a.out, "w") as fh:
            json.dump({"kind": "yield", "threshold": a.threshold,
                       "holdout_frac": a.holdout_frac, "baselines": res},
                      fh, indent=1, default=float)
        print(f"\nwrote {a.out}")
        return

    if a.kind in ("hybrid", "truth_dr"):
        dev, hold = _split_hybrid(data, rng, a.holdout_frac,
                                  n_kind="truth" if a.kind == "truth_dr" else "reco")
    else:
        dev, hold = _split_single(data, a.kind, rng, a.holdout_frac)

    gr = np.round(np.arange(a.rthr[0], a.rthr[1] + 1e-9, a.rthr[2]), 3)
    gd = np.round(np.arange(a.delta[0], a.delta[1] + 1e-9, a.delta[2]), 3)
    print(f"scanning {len(gr)} x {len(gd)} = {len(gr)*len(gd)} windows "
          f"({a.kind}), development units only")
    df = scan(a.kind, dev, gr, gd, rng, a.n_boot, a.min_counts, a.plateau_min)
    if df.empty:
        raise SystemExit("no admissible window: relax --min-counts or the grid")
    df = df.sort_values("fom", ascending=False).reset_index(drop=True)
    n_ord = int(df.ordered.sum())
    print(f"admissible: {len(df)}   ordered in U_sym: {n_ord}\n")
    print(f"  {'Rthr':>6}{'delta':>7}{'low band':>12}{'high band':>12}"
          f"{'90/0-1':>10}{'sigma':>7}")
    for _, r in df.head(8).iterrows():
        print(f"  {r.Rthr:>6.2f}{r.delta:>7.2f}{'E < %.2f' % r.lo_edge:>12}"
              f"{'E >= %.2f' % r.hi_edge:>12}{r.rel*100:>+9.2f}%{r.sigma:>7.2f}")

    if n_ord == 0 and not a.window_from:
        # Sorting by a figure of merit that is zero everywhere would hand back
        # an arbitrary row, and carrying it to the held-out sample would dress
        # a null up as a measurement. Report the failure instead.
        print("  NO ADMISSIBLE WINDOW ORDERS THE THREE SAMPLES.")
        print("  The scan covered every window in the plateau with enough")
        print("  statistics; none has R rising with U_sym. There is nothing to")
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

    if a.window_from:
        ext = json.load(open(a.window_from))["chosen"]
        cand = df[(np.isclose(df.Rthr, ext["Rthr"]))
                  & (np.isclose(df.delta, ext["delta"]))]
        if cand.empty:
            # the window must still be representable on this grid, but it does
            # not have to be admissible here: a truth reference is evaluated
            # where the measurement lives, not where truth would prefer
            best = pd.Series({"Rthr": float(ext["Rthr"]),
                              "delta": float(ext["delta"]),
                              "lo_edge": float(ext["lo_edge"]),
                              "hi_edge": float(ext["hi_edge"])})
            print(f"\n  window taken from {a.window_from}: "
                  f"Rthr = {best.Rthr:.2f}, delta = {best.delta:.2f} "
                  f"(not admissible on this scan's own grid)")
        else:
            best = cand.iloc[0]
            print(f"\n  window taken from {a.window_from}: "
                  f"Rthr = {best.Rthr:.2f}, delta = {best.delta:.2f}")
    else:
        best = df.iloc[0]
        print(f"\n  chosen on development data: Rthr = {best.Rthr:.2f}, "
              f"delta = {best.delta:.2f}")

    if a.kind in ("hybrid", "truth_dr"):
        un, up = {}, {}
        for tag, _ in TAGS:
            if tag not in hold["n"] or tag not in hold["p"]:
                continue
            if a.kind == "truth_dr":
                un[tag] = _counts_truth(hold["n"][tag]["per_job"],
                                        hold["n"][tag]["edges"],
                                        (best.lo_edge, best.hi_edge), species=0)
            else:
                un[tag] = _counts_reco(hold["n"][tag]["e"], hold["n"][tag]["unit"],
                                       (best.lo_edge, best.hi_edge))
            up[tag] = _counts_truth(hold["p"][tag]["per_job"],
                                    hold["p"][tag]["edges"],
                                    (best.lo_edge, best.hi_edge), species=1)
        ho = evaluate_hybrid(un, up, rng, max(a.n_boot, 500))
    else:
        units = {}
        for tag, _ in TAGS:
            if tag not in hold:
                continue
            if a.kind == "truth":
                units[tag] = _counts_truth(hold[tag]["per_job"],
                                           hold[tag]["edges"],
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
            print(f"    U_sym = {u:>4.0f} MeV   R = {ho['R'][tag]:.4f}")
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
