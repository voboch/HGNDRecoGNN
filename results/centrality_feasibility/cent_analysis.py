"""Feasibility of the S_pot measurement in centrality percentile classes.

b is not an observable; experiments bin in percentile classes of the inelastic
cross section (docs/CENTRALITY_CONVENTION.md). With minimum-bias sampling the
percentile follows from b, so deciles of b reproduce the classes a perfect
estimator would give -- the MC-truth limit of the experimental procedure.

Quantiles are computed PER SAMPLE, as an experiment calibrates each dataset
against its own distribution.
"""
import glob, os, numpy as np, pandas as pd

SP = os.environ["SP"]
U = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
TH_LO, TH_HI = 8.9, 13.1          # HGND acceptance, from hit geometry
NBINS = 10                        # community convention: ten 10% classes
ELO, EHI = 1.0, 2.0               # spectral hardness thresholds

def load(ds):
    ev_b, rows = [], []
    for f in sorted(glob.glob(f"{SP}/cent/{ds}_*_prim.csv")):
        d = pd.read_csv(f, usecols=["Row","PDG","Ekin","Pt","Pz","B"], engine="pyarrow")
        tag = os.path.basename(f).split("_")[1]
        # Row restarts per file -> key events by (file, Row)
        d["ev"] = tag + ":" + d.Row.astype(str)
        # drop the last (truncated) event of each file
        last = d.ev.iloc[-1]
        d = d[d.ev != last]
        rows.append(d)
    if not rows: return None
    d = pd.concat(rows, ignore_index=True)
    d["theta"] = np.degrees(np.arctan2(d.Pt.to_numpy(), d.Pz.to_numpy()))
    return d

def stats(d):
    """per-event b, and per-event counts of the observables"""
    g = d.groupby("ev")
    b = g.B.first()
    inband = (d.theta >= TH_LO) & (d.theta < TH_HI)
    isn, isp = d.PDG == 2112, d.PDG == 2212
    per = pd.DataFrame({
        "b":     b,
        "n_band": d[inband & isn].groupby("ev").size().reindex(b.index, fill_value=0),
        "p_band": d[inband & isp].groupby("ev").size().reindex(b.index, fill_value=0),
        "n_hi":   d[isn & (d.Ekin >= EHI)].groupby("ev").size().reindex(b.index, fill_value=0),
        "n_lo":   d[isn & (d.Ekin <  ELO)].groupby("ev").size().reindex(b.index, fill_value=0),
    })
    return per

D = {}
for ds in U:
    d = load(ds)
    if d is None: print(f"{ds}: no data"); continue
    D[ds] = stats(d)
    b = D[ds].b
    print(f"{ds:12s} events={len(b):>7,}  b: {b.min():.2f}-{b.max():.2f} fm  "
          f"mean {b.mean():.3f}  median {np.median(b):.3f}", flush=True)
    del d

print("\n" + "="*78)
print("  [1] do the three samples share a b distribution?")
print("      (a mismatch invalidates any centrality-INTEGRATED comparison)")
print("="*78)
ref = D["zeroSpot"].b.to_numpy()
qs = [10,25,50,75,90,99]
print(f"{'sample':<13}" + "".join(f"{f'p{q}':>9}" for q in qs) + f"{'KS vs U=0':>12}")
from scipy import stats as st
for ds in D:
    b = D[ds].b.to_numpy()
    line = f"{ds:<13}" + "".join(f"{np.percentile(b,q):>9.3f}" for q in qs)
    if ds != "zeroSpot":
        ks = st.ks_2samp(ref, b)
        line += f"{ks.statistic:>8.4f} p={ks.pvalue:.2g}"
    print(line)

print("\n" + "="*78)
print(f"  [2] centrality classes: {NBINS} bins of {100//NBINS}%, quantiles per sample")
print("="*78)
edges = {ds: np.quantile(D[ds].b, np.linspace(0,1,NBINS+1)) for ds in D}
print(f"{'class':<12}" + "".join(f"{f'b edge U={U[ds]}':>16}" for ds in D))
for i in range(NBINS):
    lab = f"{i*100//NBINS}-{(i+1)*100//NBINS}%"
    print(f"{lab:<12}" + "".join(f"{edges[ds][i]:>7.2f}-{edges[ds][i+1]:<8.2f}" for ds in D))

def ratio(a, b_):
    if b_ <= 0 or a <= 0: return np.nan, np.nan
    r = a/b_; return r, r*np.sqrt(1/a + 1/b_)

for obs, num, den, name in (("np","n_band","p_band","n/p in HGND band"),
                            ("R","n_hi","n_lo","spectral hardness R (all neutrons)")):
    print("\n" + "="*78)
    print(f"  [3] {name}  by centrality class")
    print("="*78)
    print(f"{'class':<12}{'U=0':>12}{'U=18':>12}{'U=90':>12}{'90/0':>10}{'sigma':>8}{'ord':>5}")
    tot = {ds: [0,0] for ds in D}
    for i in range(NBINS):
        vals = {}
        for ds in D:
            p = D[ds]
            m = (p.b >= edges[ds][i]) & (p.b < edges[ds][i+1] if i < NBINS-1 else p.b <= edges[ds][i+1])
            A, B_ = p[num][m].sum(), p[den][m].sum()
            tot[ds][0] += A; tot[ds][1] += B_
            vals[ds] = ratio(A, B_)
        (r0,s0),(r18,_),(r9,s9) = vals["zeroSpot"], vals["defaultSpot"], vals["bigSpot"]
        if not np.isfinite(r0) or not np.isfinite(r9):
            print(f"{i*10}-{(i+1)*10}%".ljust(12) + f"{'low stat':>50}"); continue
        dr = r9/r0; sd = dr*np.sqrt((s0/r0)**2 + (s9/r9)**2)
        mono = "yes" if r0 < r18 < r9 else ("rev" if r0 > r18 > r9 else "no")
        print(f"{f'{i*10}-{(i+1)*10}%':<12}{r0:>12.4f}{r18:>12.4f}{r9:>12.4f}"
              f"{dr:>10.4f}{abs(dr-1)/sd:>8.1f}{mono:>5}")
    v = {ds: ratio(*tot[ds]) for ds in D}
    (r0,s0),(r18,_),(r9,s9) = v["zeroSpot"], v["defaultSpot"], v["bigSpot"]
    dr = r9/r0; sd = dr*np.sqrt((s0/r0)**2 + (s9/r9)**2)
    mono = "yes" if r0 < r18 < r9 else ("rev" if r0 > r18 > r9 else "no")
    print(f"{'INTEGRATED':<12}{r0:>12.4f}{r18:>12.4f}{r9:>12.4f}{dr:>10.4f}"
          f"{abs(dr-1)/sd:>8.1f}{mono:>5}")
