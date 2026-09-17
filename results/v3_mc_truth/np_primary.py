"""n/p restricted to PRIMARY particles (fMotherId == -1), to isolate how much of
the earlier hit-level n/p was secondary contamination.

Primary protons are largely swept out of the HGND by the BM@N analysing magnet,
so even this is not an acceptance-matched ratio -- it only separates the
secondary-production effect from whatever primary signal survives.
"""
import glob
import numpy as np, pandas as pd

CACHE="/scratch/vbocharnikov/hgnd/cache"
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
EDGES=np.array([0,.25,.5,.75,1.,1.5,2.,3.,5.])
res={}
for ds in ("zeroSpot","defaultSpot","bigSpot"):
    g=glob.glob(f"{CACHE}/ndet_dataset_smash_{ds}_v3/processed/_hits_cache_*.parquet")
    if not g: continue
    df=pd.read_parquet(g[0], columns=["Row","Id","PDG","Ekin","fMotherId"])
    u=df.drop_duplicates(subset=["Row","Id"])
    pri=u[u.fMotherId==-1]
    n=pri[pri.PDG==2112]; p=pri[pri.PDG==2212]
    r=len(n)/len(p); se=r*np.sqrt(1/len(n)+1/len(p))
    res[ds]=dict(n=len(n),p=len(p),R=r,se=se,
                 hn=np.histogram(n.Ekin.to_numpy(),bins=EDGES)[0],
                 hp=np.histogram(p.Ekin.to_numpy(),bins=EDGES)[0],
                 ev=int(df.Row.nunique()))
    print(f"{ds:12s} U={U[ds]:>2}  PRIMARY n={len(n):>9,}  p={len(p):>8,}  "
          f"n/p={r:.4f}+-{se:.4f}", flush=True)
    del df,u,pri

a=res["zeroSpot"]
print("\n=== primary-only n/p vs zeroSpot ===")
for ds in ("defaultSpot","bigSpot"):
    b=res[ds]; d=b["R"]-a["R"]; s=float(np.hypot(a["se"],b["se"]))
    print(f"  U={U[ds]:>2} vs 0: {d:+.4f} +- {s:.4f} -> {abs(d)/s:5.1f} sigma ({100*d/a['R']:+.2f} %)")

print("\n=== primary-only n/p differential ===")
print(f"{'Ekin [GeV]':<13}{'U=0':>9}{'U=18':>9}{'U=90':>9}{'90/0':>9}{'sigma':>8}")
for i in range(len(EDGES)-1):
    v={ds:(res[ds]['hn'][i]/res[ds]['hp'][i] if res[ds]['hp'][i] else np.nan,
           res[ds]['hn'][i],res[ds]['hp'][i]) for ds in res}
    (r0,n0,p0),(r18,_,_),(r9,n9,p9)=v["zeroSpot"],v["defaultSpot"],v["bigSpot"]
    if not (p0 and p9): 
        print(f"{f'[{EDGES[i]:g},{EDGES[i+1]:g})':<13}{'—':>9}{'—':>9}{'—':>9}{'no protons':>18}")
        continue
    dr=r9/r0; se=dr*np.sqrt(1/max(n0,1)+1/max(p0,1)+1/max(n9,1)+1/max(p9,1))
    print(f"{f'[{EDGES[i]:g},{EDGES[i+1]:g})':<13}{r0:>9.3f}{r18:>9.3f}{r9:>9.3f}{dr:>9.4f}{abs(dr-1)/se:>8.1f}")

print("\n=== proton survival: primary fraction by Ekin (zeroSpot) ===")
g=glob.glob(f"{CACHE}/ndet_dataset_smash_zeroSpot_v3/processed/_hits_cache_*.parquet")[0]
df=pd.read_parquet(g, columns=["Row","Id","PDG","Ekin","fMotherId"])
u=df.drop_duplicates(subset=["Row","Id"])
for name,pdg in (("neutron",2112),("proton",2212)):
    s=u[u.PDG==pdg]
    out=[]
    for i in range(len(EDGES)-1):
        m=(s.Ekin>=EDGES[i])&(s.Ekin<EDGES[i+1])
        out.append(f"{100*(s[m].fMotherId==-1).mean():.0f}%" if m.sum() else "—")
    print(f"  {name:8s} primary%: " + "  ".join(f"{EDGES[i]:g}-{EDGES[i+1]:g}:{o}" for i,o in enumerate(out)))
