"""GNN reconstruction vs Monte-Carlo truth for the S_pot hardness observables.

HGND measures neutrons; it does not measure protons, which bend out of its
acceptance in the BM@N field.  The proton arm of any n/p or double-ratio
observable therefore has to come from elsewhere, and here it comes from MC
truth: primary protons in the HGND angular band.  That makes the mixed
quantities idealised -- a real measurement would take protons from BM@N
tracking, over a different acceptance with its own systematics -- but it
isolates the question this experiment is about, which is what reconstruction
does to the neutron arm.

Three rungs are computed on the same events, so the signal can be followed:

  truth-primary  primary neutrons/protons inside theta in [8.9, 13.1) deg
                 -- the physics-level quantity from the b analysis
  truth-reaching particles that actually reach the detector (from *_vacs.csv)
                 -- adds the real acceptance, still no reconstruction
  reco           GNN clusters above a purity-locked score threshold, binned on
                 predicted energy -- adds reconstruction

Selection thresholds match the truth analysis: R = N(Ekin >= 2 GeV)/N(Ekin < 1 GeV).

Uncertainties come from resampling job files, as everywhere else in this
project: events within a job file are not independent.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reco_truth_join import row_offsets, to_local, prim_file_map

ELO, EHI = 1.0, 2.0
SAMPLES = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
VACS_COLS = ["Row", "Instance", "Id", "PDG", "Ekin", "Rapid", "fMotherId",
             "Weight", "DetectorID", "Side", "X", "Y", "Z",
             "vX", "vY", "vZ", "Px", "Py", "Pz"]


def purity_locked_threshold(clusters, target=0.7, n_scan=400):
    """Smallest score threshold reaching `target` signal purity.

    A fixed threshold selects different physical purity in different datasets,
    which would itself masquerade as an S_pot difference; locking purity
    removes that. Mirrors scripts/purity_locked_closure.py.
    """
    s = clusters.cl_score.to_numpy(float)
    y = clusters.cl_label.to_numpy(int)
    for t in np.linspace(0.001, 0.999, n_scan):
        sel = s > t
        if sel.sum() == 0:
            break
        pur = y[sel].mean()
        if pur >= target:
            return float(t), float(pur), float(sel.mean())
    return float("nan"), float("nan"), float("nan")


def fit_energy_calibration(clusters, n_bins=24):
    """Map predicted energy onto true energy, from signal clusters.

    The regression correlates well with truth but carries a scale bias that
    grows with energy, so a 2 GeV cut on the raw prediction does not select
    2 GeV neutrons.  The calibration is the median true energy in bins of
    predicted energy, which is monotone and needs no functional form.

    It is fitted on ONE sample and applied to all of them.  Fitting per sample
    would let the calibration absorb the very S_pot difference being measured.
    """
    s = clusters[clusters.cl_label == 1]
    q = np.quantile(s.e_pred, np.linspace(0, 1, n_bins + 1))
    q = np.unique(q)
    idx = np.clip(np.digitize(s.e_pred.to_numpy(), q) - 1, 0, len(q) - 2)
    xs, ys = [], []
    for k in range(len(q) - 1):
        m = idx == k
        if m.sum() < 20:
            continue
        xs.append(float(np.median(s.e_pred.to_numpy()[m])))
        ys.append(float(np.median(s.e_true.to_numpy()[m])))
    xs, ys = np.array(xs), np.array(ys)
    ys = np.maximum.accumulate(ys)          # enforce monotonicity
    return xs, ys


def apply_calibration(e_pred, cal):
    xs, ys = cal
    return np.interp(np.asarray(e_pred, float), xs, ys)


def ratio(hi, lo):
    return hi / lo if lo > 0 else np.nan


def boot_ratio(per_file_hi, per_file_lo, rng, n_boot):
    """Bootstrap a ratio of summed counters over job files."""
    hi = np.asarray(per_file_hi, float)
    lo = np.asarray(per_file_lo, float)
    n = len(hi)
    out = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.integers(0, n, n)
        d = lo[p].sum()
        out[i] = hi[p].sum() / d if d > 0 else np.nan
    return out


def load_vacs(csv_dir):
    """Particles reaching the detector, deduplicated (vacs is hit-aligned)."""
    rows, base = [], 0
    for run in sorted(d for d in os.listdir(csv_dir)
                      if os.path.isdir(os.path.join(csv_dir, d))):
        rd = os.path.join(csv_dir, run)
        for stem in sorted({f.rsplit("_", 1)[0] for f in os.listdir(rd)
                            if f.endswith("vacs.csv")}):
            v = os.path.join(rd, f"{stem}_vacs.csv")
            h = os.path.join(rd, f"{stem}_hits.csv")
            if not os.path.exists(h):
                continue
            d = pd.read_csv(v, sep=",", skiprows=[0], names=VACS_COLS,
                            usecols=["Row", "Id", "PDG", "Ekin", "fMotherId"],
                            engine="c")
            d = d.drop_duplicates(["Row", "Id"])      # one row per particle
            d["stem"] = stem
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def analyse(tag, sp, args, rng, cal=None):
    csv_dir = os.path.join(args.raw_dir, tag)
    pred_path = os.path.join(args.pred_dir, tag, "pred_clusters_smash.pkl")
    clusters = pd.read_pickle(pred_path)

    offs = row_offsets(csv_dir)
    loc = to_local(clusters.Row.to_numpy(), offs)
    if loc.unmapped.any():
        raise AssertionError(f"{tag}: {int(loc.unmapped.sum())} unmapped rows")
    clusters = clusters.assign(stem=loc.stem.to_numpy(),
                               local_row=loc.local_row.to_numpy())

    # ---- truth-primary: the reduced per-event counters, same job files ----
    ev = pd.read_pickle(os.path.join(args.events_dir, f"{tag}_events.pkl"))
    fmap = prim_file_map(os.path.join(args.events_dir, f"{tag}_progress.json"))
    ev = ev.assign(stem=ev.file_idx.map(fmap))
    present = set(offs.stem)
    ev = ev[ev.stem.isin(present)]
    missing = present - set(ev.stem)

    res = {"tag": tag, "U_MeV": sp, "n_job_files": int(len(present)),
           "n_events_reco": int(clusters.groupby(["stem", "local_row"]).ngroups),
           "n_events_truth": int(len(ev)),
           "job_files_without_truth": sorted(missing)}

    gf = ev.groupby("stem")
    tn_hi, tn_lo = gf.band_n_hi.sum(), gf.band_n_lo.sum()
    tp_hi, tp_lo = gf.band_p_hi.sum(), gf.band_p_lo.sum()
    tn, tp = gf.band_n.sum(), gf.band_p.sum()

    # ---- truth-reaching: particles that arrive at the detector ------------
    vac = load_vacs(csv_dir)
    vac = vac[vac.PDG.isin((2112, 2212))]
    vn = vac[vac.PDG == 2112]; vp = vac[vac.PDG == 2212]
    vg = lambda d, m: d[m].groupby("stem").size().reindex(sorted(present), fill_value=0)
    vn_hi, vn_lo = vg(vn, vn.Ekin >= EHI), vg(vn, vn.Ekin < ELO)
    vp_hi, vp_lo = vg(vp, vp.Ekin >= EHI), vg(vp, vp.Ekin < ELO)
    vn_all = vn.groupby("stem").size().reindex(sorted(present), fill_value=0)
    vp_all = vp.groupby("stem").size().reindex(sorted(present), fill_value=0)

    # ---- reco: purity-locked clusters, binned on predicted energy ---------
    t_star, pur, eff = purity_locked_threshold(clusters, args.purity)
    res["threshold"] = t_star
    res["purity_achieved"] = pur
    res["cluster_select_frac"] = eff
    sel = clusters[clusters.cl_score > t_star]
    rg = lambda d: d.groupby("stem").size().reindex(sorted(present), fill_value=0)
    rn_hi = rg(sel[sel.e_pred >= EHI]); rn_lo = rg(sel[sel.e_pred < ELO])
    rn_all = rg(sel)
    # Decomposition rungs.  Background clusters carry e_true = 0, so a true-energy
    # binning of the full selection would push every background object into the
    # low bin; the true-energy rungs therefore use correctly selected signal
    # clusters only, and background enters at the last rung.
    sig = sel[sel.cl_label == 1]
    sn_hi_t = rg(sig[sig.e_true >= EHI]); sn_lo_t = rg(sig[sig.e_true < ELO])
    sn_hi_p = rg(sig[sig.e_pred >= EHI]); sn_lo_p = rg(sig[sig.e_pred < ELO])
    if cal is not None:
        ec = apply_calibration(sel.e_pred.to_numpy(), cal)
        sel = sel.assign(e_cal=ec)
        rc_hi = rg(sel[sel.e_cal >= EHI]); rc_lo = rg(sel[sel.e_cal < ELO])
    else:
        rc_hi = rc_lo = rg(sel.iloc[0:0])

    def pack(name, hi, lo):
        hi = np.asarray(hi, float); lo = np.asarray(lo, float)
        c = ratio(hi.sum(), lo.sum())
        b = boot_ratio(hi, lo, rng, args.n_boot)
        return {name: {"value": float(c), "err": float(np.nanstd(b, ddof=1)),
                       "n_hi": float(hi.sum()), "n_lo": float(lo.sum())}}, (hi, lo)

    out = {}
    store = {}
    for nm, (hi, lo) in {
        "R_n_truth_primary":   (tn_hi, tn_lo),
        "R_p_truth_primary":   (tp_hi, tp_lo),
        "R_n_truth_reaching":  (vn_hi, vn_lo),
        "R_p_truth_reaching":  (vp_hi, vp_lo),
        "R_n_sig_trueE":       (sn_hi_t, sn_lo_t),
        "R_n_sig_predE":       (sn_hi_p, sn_lo_p),
        "R_n_reco":            (rn_hi, rn_lo),
        "R_n_reco_calib":      (rc_hi, rc_lo),
        "np_truth_primary":    (tn, tp),
        "np_truth_reaching":   (vn_all, vp_all),
        "np_reco_over_truthp": (rn_all, tp),
        "np_reco_over_reachp": (rn_all, vp_all),
    }.items():
        d, arrs = pack(nm, hi, lo)
        out.update(d); store[nm] = arrs
    res["observables"] = out
    res["_arrays"] = {k: (np.asarray(v[0], float).tolist(),
                          np.asarray(v[1], float).tolist())
                      for k, v in store.items()}
    res["_stems"] = sorted(present)
    return res


def compare(out, rng, n_boot=400, ref="zeroSpot", arm="bigSpot"):
    """Relative difference between two samples, bootstrapping job files in each.

    The samples are independent productions, so their job files are resampled
    independently inside each replicate.
    """
    if ref not in out or arm not in out:
        return {}
    names = list(out[ref]["observables"])
    rows = {}
    for nm in names:
        a_hi, a_lo = (np.asarray(x, float) for x in out[ref]["_arrays"][nm])
        b_hi, b_lo = (np.asarray(x, float) for x in out[arm]["_arrays"][nm])
        va = ratio(a_hi.sum(), a_lo.sum())
        vb = ratio(b_hi.sum(), b_lo.sum())
        rel = vb / va - 1.0 if va and np.isfinite(va) else np.nan
        bs = np.empty(n_boot)
        for i in range(n_boot):
            pa = rng.integers(0, len(a_hi), len(a_hi))
            pb = rng.integers(0, len(b_hi), len(b_hi))
            ra = ratio(a_hi[pa].sum(), a_lo[pa].sum())
            rb = ratio(b_hi[pb].sum(), b_lo[pb].sum())
            bs[i] = rb / ra - 1.0 if ra and np.isfinite(ra) else np.nan
        sd = float(np.nanstd(bs, ddof=1))
        rows[nm] = {"ref": float(va), "arm": float(vb), "rel": float(rel),
                    "err": sd, "sigma": float(abs(rel) / sd) if sd > 0 else np.nan}
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--events-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--purity", type=float, default=0.7)
    p.add_argument("--n-boot", type=int, default=400)
    p.add_argument("--samples", default=None)
    p.add_argument("--calib-sample", default="defaultSpot",
                   help="sample the energy calibration is fitted on (the one "
                        "the checkpoint was trained on)")
    a = p.parse_args()

    rng = np.random.default_rng(20260926)
    keys = a.samples.split(",") if a.samples else list(SAMPLES)

    # Energy calibration is fitted once, on the sample the model was trained on,
    # and reused for every sample.
    cal = None
    cal_path = os.path.join(a.pred_dir, a.calib_sample, "pred_clusters_smash.pkl")
    if os.path.exists(cal_path):
        cal = fit_energy_calibration(pd.read_pickle(cal_path))
        print(f"energy calibration fitted on {a.calib_sample}: "
              f"e_pred {cal[0][0]:.2f}..{cal[0][-1]:.2f} -> "
              f"e_true {cal[1][0]:.2f}..{cal[1][-1]:.2f} GeV")
    else:
        print(f"WARNING: no predictions for calibration sample {a.calib_sample}; "
              f"calibrated observable will be empty")
    out = {}
    for tag in keys:
        if not os.path.exists(os.path.join(a.pred_dir, tag, "pred_clusters_smash.pkl")):
            print(f"skip {tag}: no predictions"); continue
        print(f"=== {tag} ===", flush=True)
        r = analyse(tag, SAMPLES[tag], a, rng, cal=cal)
        out[tag] = r
        print(f"  {r['n_events_reco']:,} reco events, {r['n_events_truth']:,} truth events, "
              f"{r['n_job_files']} job files")
        print(f"  purity-locked threshold {r['threshold']:.3f} "
              f"(purity {r['purity_achieved']:.3f}, keeps {r['cluster_select_frac']*100:.1f}% of clusters)")
        for k, v in r["observables"].items():
            print(f"    {k:24s} {v['value']:9.5f} +- {v['err']:.5f}"
                  f"   (hi {v['n_hi']:>9,.0f}  lo {v['n_lo']:>9,.0f})")
    cmp = compare(out, rng, a.n_boot)
    if cmp:
        print("\n" + "=" * 78)
        print("  S_pot = 90 vs 0 MeV, same 20 job files per sample")
        print("  errors from independent job-file bootstrap in each sample")
        print("=" * 78)
        print(f"{'observable':<24}{'U=0':>11}{'U=90':>11}{'90/0-1':>10}{'sigma':>8}")
        for nm, r in cmp.items():
            print(f"{nm:<24}{r['ref']:>11.5f}{r['arm']:>11.5f}"
                  f"{r['rel']:>+10.4f}{r['sigma']:>8.2f}")
    payload = {"per_sample": {k: {kk: vv for kk, vv in v.items()
                                 if not kk.startswith("_")}
                              for k, v in out.items()},
               "arrays": {k: v["_arrays"] for k, v in out.items()},
               "stems": {k: v["_stems"] for k, v in out.items()},
               "comparison_90_vs_0": cmp}
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
