"""First look at the neutron/proton ratio as a U_sym probe, from the _v3 parquets.

CAVEAT built into the numbers: the parquet is hit-aligned, so "protons" here are
protons that deposited energy in the HGND, not emitted protons. The HGND sits
behind absorber and vetoes charge, so this is a detector-response ratio, not the
physics n/p of the emitted source. It is still worth measuring: if even this
biased ratio separates the potentials better than the neutron spectrum alone,
the true n/p is worth the work of extracting.

Also note load_hits() applies np.abs() to PDG, so antiprotons fold into protons
(negligible at 2.87 GeV, but it is why sign must be preserved going forward).
"""
import glob
import numpy as np
import pandas as pd

CACHE = "/scratch/vbocharnikov/hgnd/cache"
U = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
EDGES = np.array([0, .25, .5, .75, 1., 1.5, 2., 3., 5.])
res = {}

for ds in ("zeroSpot", "defaultSpot", "bigSpot"):
    g = glob.glob(f"{CACHE}/ndet_dataset_smash_{ds}_v3/processed/_hits_cache_*.parquet")
    if not g:
        print(f"{ds}: no parquet"); continue
    df = pd.read_parquet(g[0], columns=["Row", "Id", "PDG", "Ekin"])
    n_ev = int(df.Row.nunique())
    u = df.drop_duplicates(subset=["Row", "Id"])
    n = u[u.PDG == 2112]
    p = u[u.PDG == 2212]
    ratio = len(n) / len(p)
    se = ratio * np.sqrt(1/len(n) + 1/len(p))
    hn, _ = np.histogram(n.Ekin.to_numpy(), bins=EDGES)
    hp, _ = np.histogram(p.Ekin.to_numpy(), bins=EDGES)
    res[ds] = dict(ev=n_ev, n=len(n), p=len(p), R=ratio, se=se, hn=hn, hp=hp)
    print(f"{ds:12s} U={U[ds]:>2}  events={n_ev:>9,}  n={len(n):>9,}  p={len(p):>9,}  "
          f"n/p={ratio:.4f}+-{se:.4f}", flush=True)
    del df, u, n, p

a = res["zeroSpot"]
print("\n=== integrated n/p vs zeroSpot ===")
for ds in ("defaultSpot", "bigSpot"):
    b = res[ds]
    d = b["R"] - a["R"]; s = float(np.hypot(a["se"], b["se"]))
    print(f"  U={U[ds]:>2} vs 0:  {d:+.4f} +- {s:.4f}  ->  {abs(d)/s:5.1f} sigma "
          f"({100*d/a['R']:+.2f} %)")

print("\n=== n/p differential in Ekin ===")
print(f"{'Ekin [GeV]':<13}{'U=0':>9}{'U=18':>9}{'U=90':>9}{'90/0':>9}{'sigma':>8}")
for i in range(len(EDGES)-1):
    vals = {}
    for ds in ("zeroSpot", "defaultSpot", "bigSpot"):
        r = res[ds]
        vals[ds] = (r["hn"][i]/r["hp"][i] if r["hp"][i] else np.nan,
                    r["hn"][i], r["hp"][i])
    (r0,n0,p0), (r18,_,_), (r9,n9,p9) = vals["zeroSpot"], vals["defaultSpot"], vals["bigSpot"]
    dr = r9/r0 if r0 else np.nan
    se = dr*np.sqrt(1/max(n0,1)+1/max(p0,1)+1/max(n9,1)+1/max(p9,1))
    print(f"{f'[{EDGES[i]:g},{EDGES[i+1]:g})':<13}{r0:>9.3f}{r18:>9.3f}{r9:>9.3f}"
          f"{dr:>9.4f}{abs(dr-1)/se:>8.1f}")
