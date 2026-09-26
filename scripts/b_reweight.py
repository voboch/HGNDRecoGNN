"""Impact-parameter reweighting and job-level uncertainties for the S_pot comparison.

Why reweighting rather than percentile matching
-----------------------------------------------
Spectral hardness varies by a factor ~4 across centrality, so a per-cent-level
difference in the samples' b distributions manufactures a large apparent S_pot
signal.  Equal-percentile classes do not remove it: when the underlying b
distributions differ, equal percentiles select unequal b.  Common b edges are a
stratification check only, and leave a residual whenever the observable varies
inside a bin.  Reweighting instead forces every sample onto one common b
density, so the comparison is made at matched b by construction.

Why job-level errors
--------------------
b carries job-file block structure.  Within a single sample, where no S_pot
difference can exist, four job files scattered in <b> with chi2/ndf = 27.2 and
one file drifted by 0.8 fm between its own halves.  Per-event errors therefore
understate the uncertainty on any b-dependent quantity.  Every uncertainty here
comes from resampling whole job files with replacement, with the reweighting
recomputed inside each replicate so the weight-estimation error is included.

Usage:  python3 scripts/b_reweight.py <events_dir> <out_dir> [n_boot]
"""
import json, os, sys
import numpy as np
import pandas as pd

SAMPLES = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
BW = 0.25                        # reweighting bin width [fm]
MIN_COUNT = 20                   # a b bin below this is unusable in a sample
NCLASS = 10                      # centrality percentile classes (community convention)

# numerator / denominator counter pairs
OBSERVABLES = {
    "R_n_band":  ("band_n_hi", "band_n_lo"),
    "R_p_band":  ("band_p_hi", "band_p_lo"),
    "R_n_mid":   ("mid_n_hi",  "mid_n_lo"),
    "R_n_4pi":   ("all_n_hi",  "all_n_lo"),
    "np_band":   ("band_n",    "band_p"),
    "np_mid":    ("mid_n",     "mid_p"),
    "np_4pi":    ("all_n",     "all_p"),
}


def load(events_dir):
    d = {}
    for s in SAMPLES:
        p = os.path.join(events_dir, f"{s}_events.pkl")
        d[s] = pd.read_pickle(p)
    return d


def ref_density(bs_list, edges):
    """Reference b density: the mean of the per-sample densities.

    Averaging densities rather than pooling events keeps a sample with more
    events from defining the target it is then weighted towards.
    """
    dens = []
    for b in bs_list:
        c, _ = np.histogram(b, bins=edges)
        dens.append(c / max(c.sum(), 1))
    return np.mean(dens, axis=0)


def weights(b, edges, ref, min_count=MIN_COUNT):
    """Per-event weights taking this sample's b density onto `ref`."""
    nb = len(ref)
    idx = np.digitize(b, edges) - 1
    inside = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[inside], minlength=nb).astype(float)
    dens = cnt / max(cnt.sum(), 1)
    usable = (cnt >= min_count) & (ref > 0)
    ratio = np.zeros(nb)
    ratio[usable] = ref[usable] / dens[usable]
    w = np.zeros(len(b))
    w[inside] = ratio[idx[inside]]
    return w


def wratio(df, w, num, den):
    n = float(np.dot(w, df[num].to_numpy()))
    d = float(np.dot(w, df[den].to_numpy()))
    return n / d if d > 0 else np.nan


def n_eff(w):
    w = w[w > 0]
    return float(w.sum() ** 2 / np.dot(w, w)) if len(w) else 0.0


def build_edges(data):
    bmax = max(d.B.max() for d in data.values())
    return np.arange(0.0, np.ceil(bmax / BW) * BW + BW, BW)


def class_edges(data, nclass=NCLASS):
    """Centrality class edges in b, from the pooled distribution.

    b is not an observable; these stand in for the experimental convention of
    percentile classes built on a measured multiplicity or spectator signal.
    Pooling the samples gives one common set of edges, so the classes select the
    same b in every sample.
    """
    allb = np.concatenate([d.B.to_numpy() for d in data.values()])
    return np.quantile(allb, np.linspace(0, 1, nclass + 1))


def replicate(data, edges, rng=None, resample=True):
    """One (optionally bootstrapped) realisation: returns per-sample frames+weights.

    Job files are resampled with replacement and the reference density is
    rebuilt inside the replicate, so both the sampling error and the
    weight-estimation error are propagated.
    """
    out = {}
    for s, d in data.items():
        if resample:
            files = d.file_idx.unique()
            pick = rng.choice(files, size=len(files), replace=True)
            d = pd.concat([d[d.file_idx == f] for f in pick], ignore_index=True)
        out[s] = d
    ref = ref_density([out[s].B.to_numpy() for s in SAMPLES], edges)
    return out, {s: weights(out[s].B.to_numpy(), edges, ref) for s in out}, ref


def main(events_dir, out_dir, n_boot=400):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(20260926)
    data = load(events_dir)
    edges = build_edges(data)
    cedges = class_edges(data)
    frames, W, ref = replicate(data, edges, resample=False)

    rep = {"n_boot": int(n_boot), "bin_width_fm": BW, "n_class": NCLASS,
           "b_edges_class": cedges.tolist(), "samples": {}}

    print("=" * 78)
    print("  b reweighting: closure and cost")
    print("=" * 78)
    print(f"{'sample':<13}{'n':>8}{'<b> raw':>10}{'<b> rw':>10}"
          f"{'n_eff':>9}{'n_eff/n':>9}{'w=0 frac':>10}")
    for s in SAMPLES:
        d, w = frames[s], W[s]
        b = d.B.to_numpy()
        raw, rw = b.mean(), np.dot(w, b) / w.sum()
        rep["samples"][s] = {
            "U_MeV": SAMPLES[s], "n_events": int(len(d)),
            "n_files": int(d.file_idx.nunique()),
            "mean_b_raw": float(raw), "mean_b_rw": float(rw),
            "n_eff": n_eff(w), "zero_weight_frac": float((w == 0).mean()),
        }
        print(f"{s:<13}{len(d):>8d}{raw:>10.4f}{rw:>10.4f}"
              f"{n_eff(w):>9.0f}{n_eff(w)/len(d):>9.3f}{(w==0).mean():>10.4f}")

    mb = [rep["samples"][s]["mean_b_rw"] for s in SAMPLES]
    print(f"\n  closure: reweighted <b> spread = {np.std(mb):.5f} fm "
          f"(raw spread {np.std([rep['samples'][s]['mean_b_raw'] for s in SAMPLES]):.5f} fm)")
    rep["closure_mean_b_spread_rw"] = float(np.std(mb))

    # ---- integrated observables, job-level bootstrap on the 90/0 ratio ------
    print("\n" + "=" * 78)
    print("  Reweighted observables, integrated, errors from job-file bootstrap")
    print("=" * 78)
    boot = {o: {s: [] for s in SAMPLES} for o in OBSERVABLES}
    for _ in range(n_boot):
        bf, bw, _ = replicate(data, edges, rng=rng)
        for o, (nu, de) in OBSERVABLES.items():
            for s in SAMPLES:
                boot[o][s].append(wratio(bf[s], bw[s], nu, de))

    rep["integrated"] = {}
    print(f"{'observable':<11}{'U=0':>18}{'U=18':>18}{'U=90':>18}{'90/0-1':>12}{'sig':>7}")
    for o, (nu, de) in OBSERVABLES.items():
        cen = {s: wratio(frames[s], W[s], nu, de) for s in SAMPLES}
        err = {s: float(np.nanstd(boot[o][s], ddof=1)) for s in SAMPLES}
        rel = np.array(boot[o]["bigSpot"]) / np.array(boot[o]["zeroSpot"]) - 1.0
        d90 = cen["bigSpot"] / cen["zeroSpot"] - 1.0
        sd = float(np.nanstd(rel, ddof=1))
        sig = abs(d90) / sd if sd > 0 else np.nan
        rep["integrated"][o] = {
            "central": cen, "err": err,
            "rel_90_over_0": float(d90), "rel_err": sd, "sigma": float(sig)}
        print(f"{o:<11}" + "".join(f"{cen[s]:>11.5f}±{err[s]:<6.5f}" for s in SAMPLES)
              + f"{d90:>+12.4f}{sig:>7.2f}")

    # ---- per-centrality-class, reweighted within each class ----------------
    print("\n" + "=" * 78)
    print(f"  Per centrality class ({NCLASS} classes, common b edges from pooled quantiles)")
    print("  reweighted within each class; sigma from job-file bootstrap")
    print("=" * 78)
    rep["classes"] = {}
    for o, (nu, de) in OBSERVABLES.items():
        print(f"\n  {o}")
        print(f"  {'class':<9}{'b range [fm]':>16}{'U=0':>10}{'U=18':>10}"
              f"{'U=90':>10}{'90/0-1':>10}{'sig':>7}")
        rows = []
        for k in range(NCLASS):
            lo, hi = cedges[k], cedges[k + 1]
            sub = {s: frames[s][(frames[s].B >= lo) & (frames[s].B < hi)] for s in SAMPLES}
            sedg = np.arange(lo, hi + BW, BW)
            if len(sedg) < 3:
                sedg = np.linspace(lo, hi, 3)
            sref = ref_density([sub[s].B.to_numpy() for s in SAMPLES], sedg)
            sw = {s: weights(sub[s].B.to_numpy(), sedg, sref, min_count=5) for s in SAMPLES}
            cen = {s: wratio(sub[s], sw[s], nu, de) for s in SAMPLES}
            bs = []
            for _ in range(max(n_boot // 4, 50)):
                r = {}
                for s in SAMPLES:
                    files = sub[s].file_idx.unique()
                    pick = rng.choice(files, size=len(files), replace=True)
                    r[s] = pd.concat([sub[s][sub[s].file_idx == f] for f in pick],
                                     ignore_index=True)
                rr = ref_density([r[s].B.to_numpy() for s in SAMPLES], sedg)
                rw = {s: weights(r[s].B.to_numpy(), sedg, rr, min_count=5) for s in SAMPLES}
                v0, v9 = wratio(r["zeroSpot"], rw["zeroSpot"], nu, de), \
                         wratio(r["bigSpot"], rw["bigSpot"], nu, de)
                bs.append(v9 / v0 - 1.0 if v0 and np.isfinite(v0) else np.nan)
            d90 = cen["bigSpot"] / cen["zeroSpot"] - 1.0 if cen["zeroSpot"] else np.nan
            sd = float(np.nanstd(bs, ddof=1))
            sig = abs(d90) / sd if sd > 0 and np.isfinite(d90) else np.nan
            rows.append({"class": k, "b_lo": float(lo), "b_hi": float(hi),
                         "central": cen, "rel_90_over_0": float(d90),
                         "rel_err": sd, "sigma": float(sig)})
            print(f"  {f'{k*10}-{(k+1)*10}%':<9}{f'{lo:.2f}-{hi:.2f}':>16}"
                  f"{cen['zeroSpot']:>10.4f}{cen['defaultSpot']:>10.4f}"
                  f"{cen['bigSpot']:>10.4f}{d90:>+10.4f}{sig:>7.2f}")
        rep["classes"][o] = rows
        nsig = sum(1 for r in rows if np.isfinite(r["sigma"]) and r["sigma"] >= 3)
        ordered = sum(1 for r in rows
                      if np.isfinite(r["central"]["defaultSpot"])
                      and (r["central"]["zeroSpot"] < r["central"]["defaultSpot"]
                           < r["central"]["bigSpot"]
                           or r["central"]["zeroSpot"] > r["central"]["defaultSpot"]
                           > r["central"]["bigSpot"]))
        print(f"  -> {nsig}/{NCLASS} classes at >=3 sigma; "
              f"{ordered}/{NCLASS} monotonic in U_sym (chance {NCLASS/3:.1f})")
        rep["classes_summary"] = rep.get("classes_summary", {})
        rep["classes_summary"][o] = {"n_ge_3sigma": nsig, "n_ordered": ordered}

    with open(os.path.join(out_dir, "b_reweight.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"\nwrote {out_dir}/b_reweight.json")
    return rep


def _cli():
    if sys.argv[1] == "null":
        # null <events_dir> <out_dir> [sample] [n_splits] [n_boot]
        a = sys.argv[2:]
        null_test(a[0], a[2] if len(a) > 2 else "zeroSpot",
                  int(a[3]) if len(a) > 3 else 20,
                  int(a[4]) if len(a) > 4 else 200, out_dir=a[1])
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 400)


# ---------------------------------------------------------------------------
# Null test
# ---------------------------------------------------------------------------
def null_test(events_dir, sample="zeroSpot", n_splits=20, n_boot=200, out_dir=None):
    """Split one sample's own job files in two and run the identical pipeline.

    Both halves have the same S_pot, so every apparent difference is manufactured
    by the method: by the reweighting, by the job-level bootstrap, or by the
    block structure in b.  The distribution of |sigma| over many random splits is
    the procedure's false-positive rate, and a calibrated 3-sigma threshold
    should be exceeded in about 0.3 % of splits.  This is the check that decides
    whether a 3-4 sigma reweighted result on the real samples is believable.
    """
    rng = np.random.default_rng(31415)
    d = pd.read_pickle(os.path.join(events_dir, f"{sample}_events.pkl"))
    files = np.sort(d.file_idx.unique())
    if len(files) < 4:
        raise SystemExit(f"need >=4 job files to split, {sample} has {len(files)}")
    edges = np.arange(0.0, np.ceil(d.B.max() / BW) * BW + BW, BW)

    rows = {o: [] for o in OBSERVABLES}
    print("=" * 78)
    print(f"  NULL TEST: {sample} split into two pseudo-samples, {n_splits} random splits")
    print(f"  {len(files)} job files, {len(d)} events; both halves have identical S_pot")
    print("=" * 78)
    for it in range(n_splits):
        perm = rng.permutation(files)
        A, B = perm[: len(files) // 2], perm[len(files) // 2:]
        fa, fb = d[d.file_idx.isin(A)], d[d.file_idx.isin(B)]
        ref = ref_density([fa.B.to_numpy(), fb.B.to_numpy()], edges)
        wa, wb = weights(fa.B.to_numpy(), edges, ref), weights(fb.B.to_numpy(), edges, ref)

        boot = {o: [] for o in OBSERVABLES}
        for _ in range(n_boot):
            ra = pd.concat([fa[fa.file_idx == f] for f in rng.choice(A, len(A), replace=True)],
                           ignore_index=True)
            rb = pd.concat([fb[fb.file_idx == f] for f in rng.choice(B, len(B), replace=True)],
                           ignore_index=True)
            rr = ref_density([ra.B.to_numpy(), rb.B.to_numpy()], edges)
            wra, wrb = weights(ra.B.to_numpy(), edges, rr), weights(rb.B.to_numpy(), edges, rr)
            for o, (nu, de) in OBSERVABLES.items():
                va, vb = wratio(ra, wra, nu, de), wratio(rb, wrb, nu, de)
                boot[o].append(vb / va - 1.0 if va and np.isfinite(va) else np.nan)
        for o, (nu, de) in OBSERVABLES.items():
            va, vb = wratio(fa, wa, nu, de), wratio(fb, wb, nu, de)
            rel = vb / va - 1.0 if va else np.nan
            sd = float(np.nanstd(boot[o], ddof=1))
            rows[o].append({"rel": float(rel), "err": sd,
                            "sigma": float(abs(rel) / sd) if sd > 0 else np.nan})

    print(f"{'observable':<11}{'median |sig|':>13}{'p90':>8}{'max':>8}"
          f"{'frac>=2':>9}{'frac>=3':>9}{'median |rel|':>14}")
    rep = {}
    for o in OBSERVABLES:
        s = np.array([r["sigma"] for r in rows[o]], dtype=float)
        rl = np.abs([r["rel"] for r in rows[o]])
        s = s[np.isfinite(s)]
        rep[o] = {"median_sigma": float(np.median(s)), "p90_sigma": float(np.percentile(s, 90)),
                  "max_sigma": float(s.max()), "frac_ge_2": float((s >= 2).mean()),
                  "frac_ge_3": float((s >= 3).mean()),
                  "median_abs_rel": float(np.nanmedian(rl)), "n_splits": int(len(s))}
        print(f"{o:<11}{np.median(s):>13.2f}{np.percentile(s,90):>8.2f}{s.max():>8.2f}"
              f"{(s>=2).mean():>9.2f}{(s>=3).mean():>9.2f}{np.nanmedian(rl):>+14.4f}")
    print("\n  A calibrated procedure gives median |sigma| ~ 0.7, frac>=3 ~ 0.003.")
    print("  frac>=3 well above that means the quoted sigmas are too small and any")
    print("  3-sigma result on the real samples is within the method's own noise.")
    if out_dir:
        with open(os.path.join(out_dir, f"null_test_{sample}.json"), "w") as f:
            json.dump(rep, f, indent=1)
    return rep


if __name__ == "__main__":
    _cli()
