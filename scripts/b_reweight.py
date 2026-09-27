"""Impact-parameter reweighting and job-level uncertainties for the S_pot comparison.

Why reweighting rather than percentile matching
-----------------------------------------------
Spectral hardness varies by a factor ~4 across centrality, so a per-cent-level
difference in the samples' b distributions manufactures a large apparent S_pot
signal.  Equal-percentile classes do not remove it: when the underlying b
distributions differ, equal percentiles select unequal b.  Common b edges are a
stratification check only, and leave a residual whenever the observable varies
inside a bin.  Reweighting instead forces every sample onto one common b
density event by event, so the comparison is made at matched b by construction.

Why job-level errors
--------------------
b carries job-file block structure: within a single sample, where no S_pot
difference can exist, job-file means of b scatter at chi2/ndf = 6.7 over 30
complete files, and the job-level error on <b> is 2.6x the per-event one.
Per-event errors therefore understate the uncertainty on any b-dependent
quantity.  Every uncertainty here comes from resampling whole job files with
replacement, with the reweighting rebuilt inside each replicate so the
weight-estimation error is propagated too.

How it is made affordable
-------------------------
A bootstrap replicate is a multiset of job files, and the reweighting factor
depends only on the b bin.  Both facts let a replicate be evaluated as a small
matrix product over a precomputed (file x b-bin x counter) tensor rather than by
gathering hundreds of thousands of event rows.

Usage:
  python3 scripts/b_reweight.py <events_dir> <out_dir> [n_boot]
  python3 scripts/b_reweight.py null <events_dir> <out_dir> [sample] [splits] [boot]
"""
import json, os, sys
import numpy as np
import pandas as pd

SAMPLES = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
REF_SAMPLE = "zeroSpot"
BW = 0.10                        # reweighting bin width [fm]
MIN_COUNT = 20                   # a b bin below this count is unusable in a sample
NCLASS = 10                      # centrality percentile classes (community convention)

# acc_* select on the per-particle front-face acceptance test; band_* are the
# legacy polar-angle band, kept so the two can be compared directly on the same
# events.  The band is 7.7x the solid angle and spans azimuths the detector does
# not cover, so acc_* are the physics numbers and band_* the continuity ones.
OBSERVABLES = {
    "R_n_acc":   ("acc_n_hi",  "acc_n_lo"),
    "R_p_acc":   ("acc_p_hi",  "acc_p_lo"),
    "np_acc":    ("acc_n",     "acc_p"),
    "R_n_band":  ("band_n_hi", "band_n_lo"),
    "R_p_band":  ("band_p_hi", "band_p_lo"),
    "R_n_mid":   ("mid_n_hi",  "mid_n_lo"),
    "R_n_4pi":   ("all_n_hi",  "all_n_lo"),
    "np_band":   ("band_n",    "band_p"),
    "np_mid":    ("mid_n",     "mid_p"),
    "np_4pi":    ("all_n",     "all_p"),
}
# Derived quantities: ratios of two primary observables.  The neutron-to-proton
# double ratio of spectral hardness cancels effects common to both species
# (centrality residual, acceptance, the overall spectral slope) and keeps only
# the isovector part, which is what a symmetry-potential measurement wants.
DERIVED = {
    "Rn_over_Rp_acc":  ("R_n_acc",  "R_p_acc"),
    "Rn_over_Rp_band": ("R_n_band", "R_p_band"),
}
COUNTERS = sorted({c for pair in OBSERVABLES.values() for c in pair})
ALL_OBS = list(OBSERVABLES) + list(DERIVED)
CIDX = {c: i for i, c in enumerate(COUNTERS)}


def load(events_dir):
    """Load the per-event tables, dropping observables the tables do not carry.

    Tables reduced before the acceptance test existed have only the band_*
    counters.  Rather than fail, the acceptance observables are dropped and the
    run reports what it actually computed.
    """
    d = {s: pd.read_pickle(os.path.join(events_dir, f"{s}_events.pkl")) for s in SAMPLES}
    have = set.intersection(*(set(v.columns) for v in d.values()))
    global OBSERVABLES, DERIVED, COUNTERS, CIDX, ALL_OBS
    missing = {o for o, (a, b) in OBSERVABLES.items() if a not in have or b not in have}
    if missing:
        print(f"note: {sorted(missing)} not in these event tables; skipping")
        OBSERVABLES = {k: v for k, v in OBSERVABLES.items() if k not in missing}
        DERIVED = {k: v for k, v in DERIVED.items()
                   if v[0] not in missing and v[1] not in missing}
        COUNTERS = sorted({c for pair in OBSERVABLES.values() for c in pair})
        CIDX = {c: i for i, c in enumerate(COUNTERS)}
        ALL_OBS = list(OBSERVABLES) + list(DERIVED)
    return d


def grid(data):
    bmax = max(float(d.B.max()) for d in data.values())
    return np.arange(0.0, np.ceil(bmax / BW) * BW + BW, BW)


class Binned:
    """A sample reduced to (job file) x (b bin) sums, the unit the bootstrap needs."""

    def __init__(self, df, edges, counters=None):
        # resolved at call time, not definition time: load() may drop
        # observables the event tables do not carry, which rebinds COUNTERS
        counters = COUNTERS if counters is None else counters
        fi = df.file_idx.to_numpy()
        self.files = np.unique(fi)
        nf, nb = len(self.files), len(edges) - 1
        fpos = np.searchsorted(self.files, fi)
        bpos = np.clip(np.digitize(df.B.to_numpy(), edges) - 1, 0, nb - 1)
        flat = fpos * nb + bpos
        self.n = np.bincount(flat, minlength=nf * nb).reshape(nf, nb).astype(np.float64)
        self.bsum = np.bincount(flat, weights=df.B.to_numpy().astype(np.float64),
                                minlength=nf * nb).reshape(nf, nb)
        C = np.empty((nf, nb, len(counters)))
        for k, c in enumerate(counters):
            C[:, :, k] = np.bincount(flat, weights=df[c].to_numpy().astype(np.float64),
                                     minlength=nf * nb).reshape(nf, nb)
        self.nf, self.nb, self.nc = nf, nb, len(counters)
        self.Cflat = C.reshape(nf, nb * len(counters))    # BLAS-friendly
        self.n_events = len(df)

    def unit(self):
        return np.ones(self.nf)

    def resample(self, rng):
        pick = rng.integers(0, self.nf, self.nf)
        return np.bincount(pick, minlength=self.nf).astype(np.float64)

    def aggregate(self, m):
        return m @ self.n, (m @ self.Cflat).reshape(self.nb, self.nc), m @ self.bsum


def ref_density(dens_list):
    """Reference b density: the mean of the per-sample densities."""
    return np.mean([d / max(d.sum(), 1) for d in dens_list], axis=0)


def bin_weights(N, ref, min_count=MIN_COUNT):
    dens = N / max(N.sum(), 1)
    w = np.zeros_like(ref)
    ok = (N >= min_count) & (ref > 0) & (dens > 0)
    w[ok] = ref[ok] / dens[ok]
    return w


def evaluate(aggs, keys, sl=None, min_count=MIN_COUNT, reweight=True):
    """Reweight every sample onto their common reference and read the observables.

    `sl` restricts to a slice of b bins, i.e. one centrality class.
    """
    N, T, Bs = {}, {}, {}
    for s in keys:
        n, t, b = aggs[s]
        if sl is not None:
            n, t, b = n[sl], t[sl], b[sl]
        N[s], T[s], Bs[s] = n, t, b
    ref = ref_density([N[s] for s in keys])
    res, meta = {}, {}
    for s in keys:
        # reweight=False evaluates the same samples with unit weights, which is
        # what an uncorrected comparison would report
        w = bin_weights(N[s], ref, min_count) if reweight \
            else (N[s] > 0).astype(float)
        tot = w @ T[s]
        res[s] = {o: (tot[CIDX[nu]] / tot[CIDX[de]] if tot[CIDX[de]] > 0 else np.nan)
                  for o, (nu, de) in OBSERVABLES.items()}
        for d, (a, b_) in DERIVED.items():
            res[s][d] = (res[s][a] / res[s][b_]
                         if res[s][b_] and np.isfinite(res[s][b_]) else np.nan)
        wn = w @ N[s]
        meta[s] = {
            "mean_b": float((w * Bs[s]).sum() / wn) if wn > 0 else np.nan,
            "n_eff": float(wn ** 2 / ((w ** 2) @ N[s])) if (w ** 2) @ N[s] > 0 else 0.0,
            "zero_w_frac": float(1 - N[s][w > 0].sum() / max(N[s].sum(), 1e-30)),
            "n_raw": float(N[s].sum()),
        }
    return res, meta


def class_slices(data, edges, nclass=NCLASS):
    """Centrality classes as slices of the reweighting grid.

    b is not an observable; these stand in for the experimental convention of
    percentile classes built on a measured multiplicity or spectator signal.
    Edges come from the pooled b quantiles and are snapped to the reweighting
    grid, so every sample's class k covers exactly the same b interval.
    """
    allb = np.concatenate([d.B.to_numpy() for d in data.values()])
    q = np.quantile(allb, np.linspace(0, 1, nclass + 1))
    j = np.clip(np.searchsorted(edges, q), 0, len(edges) - 1)
    j[0], j[-1] = 0, len(edges) - 1
    for k in range(1, len(j)):                      # keep every class non-empty
        j[k] = max(j[k], j[k - 1] + 1)
    return [(slice(j[k], j[k + 1]), float(edges[j[k]]), float(edges[j[k + 1]]))
            for k in range(nclass)]


def rel_90_0(res):
    out = {}
    for o in ALL_OBS:
        a, b = res[REF_SAMPLE][o], res["bigSpot"][o]
        out[o] = b / a - 1.0 if a and np.isfinite(a) and np.isfinite(b) else np.nan
    return out


def main(events_dir, out_dir, n_boot=400, samples=None):
    """`samples` restricts the comparison, e.g. when one production is still
    being reduced.  The reference and the high-S_pot arm must both be present."""
    global SAMPLES
    if samples:
        SAMPLES = {k: v for k, v in SAMPLES.items() if k in samples}
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(20260926)
    data = load(events_dir)
    edges = grid(data)
    bins = {s: Binned(data[s], edges) for s in SAMPLES}
    keys = list(SAMPLES)
    classes = class_slices(data, edges)

    cen_agg = {s: bins[s].aggregate(bins[s].unit()) for s in keys}
    cen, meta = evaluate(cen_agg, keys)
    cen_cls = [evaluate(cen_agg, keys, sl=sl, min_count=5)[0] for sl, _, _ in classes]

    rep = {"n_boot": int(n_boot), "bin_width_fm": BW, "n_class": NCLASS,
           "min_count": MIN_COUNT, "samples": {}, "classes": {}, "integrated": {}}

    print("=" * 78)
    print("  b reweighting: closure and cost")
    print("=" * 78)
    print(f"{'sample':<13}{'events':>9}{'files':>7}{'<b> raw':>10}{'<b> rw':>10}"
          f"{'n_eff/n':>9}{'w=0 frac':>10}")
    for s in keys:
        raw = float(data[s].B.mean())
        m = meta[s]
        rep["samples"][s] = {"U_MeV": SAMPLES[s], "n_events": int(bins[s].n_events),
                             "n_files": int(bins[s].nf), "mean_b_raw": raw,
                             "mean_b_rw": m["mean_b"], "n_eff": m["n_eff"],
                             "zero_weight_frac": m["zero_w_frac"]}
        print(f"{s:<13}{bins[s].n_events:>9d}{bins[s].nf:>7d}{raw:>10.4f}"
              f"{m['mean_b']:>10.4f}{m['n_eff']/bins[s].n_events:>9.3f}"
              f"{m['zero_w_frac']:>10.5f}")
    sp_raw = float(np.std([rep["samples"][s]["mean_b_raw"] for s in keys]))
    sp_rw = float(np.std([rep["samples"][s]["mean_b_rw"] for s in keys]))
    rep["closure"] = {"mean_b_spread_raw": sp_raw, "mean_b_spread_rw": sp_rw}
    print(f"\n  closure: <b> spread across samples {sp_raw:.5f} fm raw "
          f"-> {sp_rw:.5f} fm reweighted")

    # ---- bootstrap: one set of replicate multiplicities drives every result ---
    bi = {o: {s: [] for s in keys} for o in ALL_OBS}
    brel = {o: [] for o in ALL_OBS}
    bcls = [{o: [] for o in ALL_OBS} for _ in classes]
    for _ in range(n_boot):
        mult = {s: bins[s].resample(rng) for s in keys}
        agg = {s: bins[s].aggregate(mult[s]) for s in keys}
        r, _ = evaluate(agg, keys)
        for o in ALL_OBS:
            for s in keys:
                bi[o][s].append(r[s][o])
        for o, v in rel_90_0(r).items():
            brel[o].append(v)
        for k, (sl, _, _) in enumerate(classes):
            rc, _ = evaluate(agg, keys, sl=sl, min_count=5)
            for o, v in rel_90_0(rc).items():
                bcls[k][o].append(v)

    print("\n" + "=" * 78)
    print("  Reweighted observables, integrated; errors from job-file bootstrap")
    print("=" * 78)
    print(f"{'observable':<17}" + "".join(f"{'U='+str(SAMPLES[s]):>17}" for s in keys)
          + f"{'90/0-1':>11}{'sig':>7}")
    for o in ALL_OBS:
        err = {s: float(np.nanstd(bi[o][s], ddof=1)) for s in keys}
        d = cen["bigSpot"][o] / cen[REF_SAMPLE][o] - 1.0
        sd = float(np.nanstd(brel[o], ddof=1))
        sig = abs(d) / sd if sd > 0 else np.nan
        rep["integrated"][o] = {"central": {s: float(cen[s][o]) for s in keys},
                                "err": err, "rel_90_over_0": float(d),
                                "rel_err": sd, "sigma": float(sig)}
        print(f"{o:<17}" + "".join(f"{cen[s][o]:>10.5f}±{err[s]:<6.5f}" for s in keys)
              + f"{d:>+11.4f}{sig:>7.2f}")

    print("\n" + "=" * 78)
    print(f"  Per centrality class ({NCLASS} classes, common b edges, reweighted within class)")
    print("=" * 78)
    summary = {}
    for o in ALL_OBS:
        print(f"\n  {o}")
        print(f"  {'class':<9}{'b [fm]':>14}"
              + "".join(f"{'U='+str(SAMPLES[s]):>10}" for s in keys)
              + f"{'90/0-1':>10}{'sig':>7}")
        rows = []
        for k, (sl, lo, hi) in enumerate(classes):
            c = cen_cls[k]
            d = c["bigSpot"][o] / c[REF_SAMPLE][o] - 1.0 if c[REF_SAMPLE][o] else np.nan
            sd = float(np.nanstd(bcls[k][o], ddof=1))
            sig = abs(d) / sd if sd > 0 and np.isfinite(d) else np.nan
            rows.append({"class": k, "b_lo": lo, "b_hi": hi,
                         "central": {s: float(c[s][o]) for s in keys},
                         "rel_90_over_0": float(d), "rel_err": sd, "sigma": float(sig)})
            print(f"  {f'{k*10}-{(k+1)*10}%':<9}{f'{lo:.2f}-{hi:.2f}':>14}"
                  + "".join(f"{c[s][o]:>10.4f}" for s in keys)
                  + f"{d:>+10.4f}{sig:>7.2f}")
        rep["classes"][o] = rows
        ns = sum(1 for r in rows if np.isfinite(r["sigma"]) and r["sigma"] >= 3)
        if "defaultSpot" in keys:
            om = sum(1 for r in rows
                     if r["central"][REF_SAMPLE] < r["central"]["defaultSpot"] < r["central"]["bigSpot"]
                     or r["central"][REF_SAMPLE] > r["central"]["defaultSpot"] > r["central"]["bigSpot"])
        else:
            om = -1        # monotonicity needs the intermediate sample
        summary[o] = {"n_ge_3sigma": ns, "n_ordered": om}
        print(f"  -> {ns}/{NCLASS} classes at >=3 sigma; {om}/{NCLASS} monotonic "
              f"in U_sym (chance {NCLASS/3:.1f})")
    rep["classes_summary"] = summary

    with open(os.path.join(out_dir, "b_reweight.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(f"\nwrote {out_dir}/b_reweight.json")
    return rep


# ---------------------------------------------------------------------------
def null_test(events_dir, out_dir, sample="zeroSpot", n_splits=40, n_boot=200):
    """Split one sample's own job files in two and run the identical pipeline.

    Both halves have the same S_pot, so every apparent difference is manufactured
    by the method.  The distribution of |sigma| over random splits is the
    procedure's false-positive rate: a calibrated procedure gives a median |sigma|
    near 0.7 and exceeds 3 sigma in about 0.3 % of splits.  This is the check
    that decides whether a 3-4 sigma reweighted result is believable.
    """
    rng = np.random.default_rng(31415)
    df = pd.read_pickle(os.path.join(events_dir, f"{sample}_events.pkl"))
    edges = np.arange(0.0, np.ceil(float(df.B.max()) / BW) * BW + BW, BW)
    files = np.sort(df.file_idx.unique())
    if len(files) < 4:
        raise SystemExit(f"{sample} has only {len(files)} job files")
    keys = ["A", "B"]

    print("=" * 78)
    print(f"  NULL TEST: {sample}, {len(files)} job files, {len(df)} events")
    print(f"  {n_splits} random half/half splits; both halves share the same S_pot")
    print("=" * 78)
    got = {o: [] for o in ALL_OBS}
    for _ in range(n_splits):
        perm = rng.permutation(files)
        half = {"A": perm[: len(files) // 2], "B": perm[len(files) // 2:]}
        bns = {k: Binned(df[df.file_idx.isin(half[k])], edges) for k in keys}
        agg = {k: bns[k].aggregate(bns[k].unit()) for k in keys}
        cen, _ = evaluate(agg, keys)
        boot = {o: [] for o in ALL_OBS}
        for _ in range(n_boot):
            m = {k: bns[k].resample(rng) for k in keys}
            r, _ = evaluate({k: bns[k].aggregate(m[k]) for k in keys}, keys)
            for o in ALL_OBS:
                a, b = r["A"][o], r["B"][o]
                boot[o].append(b / a - 1.0 if a and np.isfinite(a) else np.nan)
        for o in ALL_OBS:
            a, b = cen["A"][o], cen["B"][o]
            rel = b / a - 1.0 if a and np.isfinite(a) else np.nan
            sd = float(np.nanstd(boot[o], ddof=1))
            got[o].append((float(rel), sd, abs(rel) / sd if sd > 0 else np.nan))

    print(f"{'observable':<17}{'median|sig|':>12}{'p90':>7}{'max':>7}"
          f"{'frac>=2':>9}{'frac>=3':>9}{'median|rel|':>13}")
    out = {}
    for o in ALL_OBS:
        sg = np.array([g[2] for g in got[o]], float); sg = sg[np.isfinite(sg)]
        rl = np.abs([g[0] for g in got[o]])
        out[o] = {"median_sigma": float(np.median(sg)),
                  "p90_sigma": float(np.percentile(sg, 90)),
                  "max_sigma": float(sg.max()),
                  "frac_ge_2": float((sg >= 2).mean()),
                  "frac_ge_3": float((sg >= 3).mean()),
                  "median_abs_rel": float(np.nanmedian(rl)),
                  "n_splits": int(len(sg))}
        print(f"{o:<17}{np.median(sg):>12.2f}{np.percentile(sg,90):>7.2f}{sg.max():>7.2f}"
              f"{(sg>=2).mean():>9.3f}{(sg>=3).mean():>9.3f}{np.nanmedian(rl):>+13.4f}")
    print("\n  Calibrated: median |sigma| ~ 0.7, frac>=3 ~ 0.003.  A larger frac>=3")
    print("  means the quoted sigmas are too small and a 3-sigma S_pot result sits")
    print("  inside the method's own noise.")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"null_test_{sample}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def _cli():
    if sys.argv[1] == "inject":
        a = sys.argv[2:]
        injection_test(a[0], a[1], a[2] if len(a) > 2 else "zeroSpot",
                       float(a[3]) if len(a) > 3 else 0.15,
                       int(a[4]) if len(a) > 4 else 300)
    elif sys.argv[1] == "null":
        a = sys.argv[2:]
        null_test(a[0], a[1], a[2] if len(a) > 2 else "zeroSpot",
                  int(a[3]) if len(a) > 3 else 40, int(a[4]) if len(a) > 4 else 200)
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 400,
             samples=sys.argv[4].split(",") if len(sys.argv) > 4 else None)



# ---------------------------------------------------------------------------
def injection_test(events_dir, out_dir, sample="zeroSpot", tilt=0.15, n_boot=300):
    """Inject a known b bias into one half of a sample and try to remove it.

    The null test shows the method invents nothing when nothing is there.  This
    shows the converse: that it *removes* a b-induced difference that really is
    there.  One half of a single sample is subsampled with acceptance
    exp(-tilt * b), which pulls it towards central collisions by roughly the
    offset claimed between productions.  Both halves still share the same S_pot,
    so the uncorrected comparison measures the size of the artefact and the
    reweighted comparison measures what survives correction.
    """
    rng = np.random.default_rng(2718)
    df = pd.read_pickle(os.path.join(events_dir, f"{sample}_events.pkl"))
    edges = np.arange(0.0, np.ceil(float(df.B.max()) / BW) * BW + BW, BW)
    files = np.sort(df.file_idx.unique())
    perm = rng.permutation(files)
    A = df[df.file_idx.isin(perm[: len(files) // 2])]
    Bfull = df[df.file_idx.isin(perm[len(files) // 2:])]
    acc = np.exp(-tilt * Bfull.B.to_numpy())
    B = Bfull[rng.random(len(Bfull)) < acc / acc.max()]

    keys = ["A", "B"]
    bns = {"A": Binned(A, edges), "B": Binned(B, edges)}
    agg = {k: bns[k].aggregate(bns[k].unit()) for k in keys}

    print("=" * 78)
    print(f"  INJECTION TEST: {sample}, acceptance exp(-{tilt} b) applied to half B")
    print("=" * 78)
    print(f"  half A: {len(A):7d} events, <b> = {A.B.mean():.4f} fm")
    print(f"  half B: {len(B):7d} events, <b> = {B.B.mean():.4f} fm  "
          f"(shifted {B.B.mean()-A.B.mean():+.4f} fm)")
    out = {"tilt": tilt, "mean_b_A": float(A.B.mean()), "mean_b_B": float(B.B.mean()),
           "n_A": int(len(A)), "n_B": int(len(B)), "observables": {}}

    res = {}
    for mode, rw in (("uncorrected", False), ("reweighted", True)):
        cen, meta = evaluate(agg, keys, reweight=rw)
        boot = {o: [] for o in ALL_OBS}
        for _ in range(n_boot):
            m = {k: bns[k].resample(rng) for k in keys}
            r, _ = evaluate({k: bns[k].aggregate(m[k]) for k in keys}, keys, reweight=rw)
            for o in ALL_OBS:
                a, b = r["A"][o], r["B"][o]
                boot[o].append(b / a - 1.0 if a and np.isfinite(a) else np.nan)
        res[mode] = (cen, {o: float(np.nanstd(boot[o], ddof=1)) for o in ALL_OBS}, meta)

    print(f"\n  <b> after reweighting: A = {res['reweighted'][2]['A']['mean_b']:.4f}, "
          f"B = {res['reweighted'][2]['B']['mean_b']:.4f} fm")
    print(f"\n{'observable':<11}{'uncorrected B/A-1':>20}{'sig':>7}"
          f"{'reweighted B/A-1':>20}{'sig':>7}")
    for o in ALL_OBS:
        row = {}
        for mode in ("uncorrected", "reweighted"):
            cen, err, _ = res[mode]
            d = cen["B"][o] / cen["A"][o] - 1.0 if cen["A"][o] else np.nan
            sd = err[o]
            row[mode] = {"rel": float(d), "err": sd,
                         "sigma": float(abs(d) / sd) if sd > 0 else np.nan}
        out["observables"][o] = row
        u, r = row["uncorrected"], row["reweighted"]
        print(f"{o:<17}{u['rel']:>+14.4f}±{u['err']:<5.4f}{u['sigma']:>7.2f}"
              f"{r['rel']:>+14.4f}±{r['err']:<5.4f}{r['sigma']:>7.2f}")
    print("\n  The injected bias should be large and significant uncorrected, and")
    print("  consistent with zero after reweighting.  Anything left is the residual")
    print("  the correction cannot reach, and bounds what a real result must exceed.")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"injection_test_{sample}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out

if __name__ == "__main__":
    _cli()
