"""Event-level bootstrap for spectral hardness R, per plotting_style_protocol
section 3: per-particle standard errors treat tracks within an event as
independent and understate the error.

Measures the inflation factor between the per-particle (naive Poisson) error and
the event-level bootstrap error, so the full-production significances -- which
were computed from aggregate bin counts and therefore per-particle -- can be
corrected rather than quoted as they stand.
"""
import glob, os, numpy as np, pandas as pd
SP=os.environ["SP"]
rng=np.random.default_rng(42)
NBOOT=400
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
ELO,EHI=1.0,2.0
YSHIFT=0.9863

for ds in U:
    hi_pe=[]; lo_pe=[]   # per-event counts
    for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv")):
        x=pd.read_csv(f,usecols=["Row","PDG","Ekin","Rapid"],engine="pyarrow")
        n=x[(x.PDG==2112)&(np.abs(x.Rapid.to_numpy()-YSHIFT)<0.5)]
        g=n.groupby("Row")["Ekin"]
        hi_pe.append(g.apply(lambda s:(s>=EHI).sum()).to_numpy())
        lo_pe.append(g.apply(lambda s:(s<ELO).sum()).to_numpy())
        del x,n
    hi=np.concatenate(hi_pe); lo=np.concatenate(lo_pe)
    nev=len(hi)
    H,L=hi.sum(),lo.sum()
    R=H/L
    se_pp=R*np.sqrt(1/H+1/L)                       # per-particle Poisson
    idx=rng.integers(0,nev,size=(NBOOT,nev))       # resample EVENTS
    bs=np.array([hi[i].sum()/max(lo[i].sum(),1) for i in idx])
    se_bs=bs.std(ddof=1)
    print(f"{ds:12s} events={nev:>6,}  N(E>{EHI})={H:>7,}  N(E<{ELO})={L:>7,}  R={R:.4f}")
    print(f"{'':12s}   per-particle SE = {se_pp:.5f}   event bootstrap SE = {se_bs:.5f}"
          f"   inflation = {se_bs/se_pp:.2f}x")
