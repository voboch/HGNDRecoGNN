"""Percentile matching vs matching in b.

S_pot is applied AFTER the impact parameter is sampled, so it cannot change
dsigma/db: the three samples must share a b distribution. They do not. That is a
generation-level sampling difference, and it matters for how centrality is
matched:

  * percentile classes align the 10th percentile of one sample with the 10th of
    another -- but if the b distributions differ, those are DIFFERENT b, so the
    artefact survives the matching;
  * common b edges align the physics, at the cost of not being reproducible on
    data.

Running both separates a genuine S_pot effect from the sampling difference.
"""
import glob, os, numpy as np, pandas as pd
SP=os.environ["SP"]
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
TH_LO,TH_HI=8.9,13.1; ELO,EHI=1.0,2.0; NB=10

def load(ds):
    rows=[]
    for f in sorted(glob.glob(f"{SP}/cent/{ds}_*_prim.csv")):
        d=pd.read_csv(f,usecols=["Row","PDG","Ekin","Pt","Pz","B"],engine="pyarrow")
        d["ev"]=os.path.basename(f).split("_")[1]+":"+d.Row.astype(str)
        d=d[d.ev!=d.ev.iloc[-1]]
        rows.append(d)
    d=pd.concat(rows,ignore_index=True)
    th=np.degrees(np.arctan2(d.Pt.to_numpy(),d.Pz.to_numpy()))
    inb=(th>=TH_LO)&(th<TH_HI); isn=(d.PDG==2112); isp=(d.PDG==2212)
    b=d.groupby("ev").B.first()
    out=pd.DataFrame({"b":b})
    for nm,msk in (("n_band",inb&isn),("p_band",inb&isp),
                   ("n_hi",isn&(d.Ekin>=EHI)),("n_lo",isn&(d.Ekin<ELO))):
        out[nm]=d[msk].groupby("ev").size().reindex(b.index,fill_value=0)
    return out

D={ds:load(ds) for ds in U}
print("b distribution (S_pot cannot affect this — differences are sampling):")
for ds in D:
    b=D[ds].b
    print(f"  {ds:12s} N={len(b):>6,}  mean {b.mean():.4f}  median {np.median(b):.4f}"
          f"  <b^2> {np.mean(b**2):.3f}")
mb={ds:D[ds].b.mean() for ds in D}
print(f"  -> bigSpot is {100*(1-mb['bigSpot']/mb['zeroSpot']):.2f}% more central than zeroSpot on average\n")

def rat(a,b_):
    if a<=0 or b_<=0: return np.nan,np.nan
    r=a/b_; return r,r*np.sqrt(1/a+1/b_)

def run(mode, num, den, title):
    print("="*74); print(f"  {title}   [{mode}]"); print("="*74)
    if mode=="percentile":
        edges={ds:np.quantile(D[ds].b,np.linspace(0,1,NB+1)) for ds in D}
    else:
        allb=np.concatenate([D[ds].b.to_numpy() for ds in D])
        e=np.quantile(allb,np.linspace(0,1,NB+1))
        edges={ds:e for ds in D}
    print(f"{'class':<11}{'U=0':>11}{'U=18':>11}{'U=90':>11}{'90/0':>10}{'sig':>7}{'ord':>5}{'Nev U=0':>9}")
    tot={ds:[0,0] for ds in D}
    for i in range(NB):
        v={}
        nev0=0
        for ds in D:
            p=D[ds]; lo,hi=edges[ds][i],edges[ds][i+1]
            m=(p.b>=lo)&((p.b<hi) if i<NB-1 else (p.b<=hi))
            if ds=="zeroSpot": nev0=int(m.sum())
            A,B_=p[num][m].sum(),p[den][m].sum()
            tot[ds][0]+=A; tot[ds][1]+=B_
            v[ds]=rat(A,B_)
        (r0,s0),(r18,_),(r9,s9)=v["zeroSpot"],v["defaultSpot"],v["bigSpot"]
        if not np.isfinite(r0) or not np.isfinite(r9): continue
        dr=r9/r0; sd=dr*np.sqrt((s0/r0)**2+(s9/r9)**2)
        o="yes" if r0<r18<r9 else ("rev" if r0>r18>r9 else "no")
        print(f"{f'{i*10}-{(i+1)*10}%':<11}{r0:>11.4f}{r18:>11.4f}{r9:>11.4f}{dr:>10.4f}{abs(dr-1)/sd:>7.1f}{o:>5}{nev0:>9,}")
    v={ds:rat(*tot[ds]) for ds in D}
    (r0,s0),(r18,_),(r9,s9)=v["zeroSpot"],v["defaultSpot"],v["bigSpot"]
    dr=r9/r0; sd=dr*np.sqrt((s0/r0)**2+(s9/r9)**2)
    o="yes" if r0<r18<r9 else ("rev" if r0>r18>r9 else "no")
    print(f"{'ALL':<11}{r0:>11.4f}{r18:>11.4f}{r9:>11.4f}{dr:>10.4f}{abs(dr-1)/sd:>7.1f}{o:>5}")
    print()

for mode in ("percentile","common-b"):
    run(mode,"n_hi","n_lo","spectral hardness R, all primary neutrons")
