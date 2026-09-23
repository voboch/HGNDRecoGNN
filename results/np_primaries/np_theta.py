"""n/p as a function of polar angle: where does the U_sym signal actually live,
and is any of it inside the HGND acceptance?"""
import glob, numpy as np, pandas as pd
SP="/private/tmp/claude-501/-Users-vovvy-Project-BM-N-HGND/14d99a58-559a-4867-a447-5040e9433acb/scratchpad"
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
EDG=np.array([0,1,2,3,4,6,8.9,13.1,18,25,40,90,180])
d={}
for ds in U:
    tn=[];tp=[];nev=0
    for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv")):
        x=pd.read_csv(f,usecols=["Row","PDG","Pt","Pz"],engine="pyarrow")
        nev+=int(x.Row.nunique())
        t=np.degrees(np.arctan2(x.Pt.to_numpy(),x.Pz.to_numpy()))
        tn.append(t[(x.PDG==2112).to_numpy()]); tp.append(t[(x.PDG==2212).to_numpy()])
        del x
    d[ds]=dict(n=np.concatenate(tn),p=np.concatenate(tp),ev=nev)

print("n/p vs polar angle.  '*' = HGND acceptance band\n")
print(f"{'theta [deg]':<14}{'n/p U=0':>10}{'n/p U=18':>10}{'n/p U=90':>10}"
      f"{'(90-0)':>9}{'sigma':>7}{'mono':>6}")
rows=[]
for i in range(len(EDG)-1):
    v={}
    for k in U:
        n=int(((d[k]["n"]>=EDG[i])&(d[k]["n"]<EDG[i+1])).sum())
        p=int(((d[k]["p"]>=EDG[i])&(d[k]["p"]<EDG[i+1])).sum())
        v[k]=(n,p, n/p if p else np.nan, (n/p)*np.sqrt(1/max(n,1)+1/max(p,1)) if p else np.nan)
    r0,s0=v["zeroSpot"][2],v["zeroSpot"][3]
    r18   =v["defaultSpot"][2]
    r9,s9 =v["bigSpot"][2],v["bigSpot"][3]
    dd=r9-r0; se=float(np.hypot(s0,s9)); sig=abs(dd)/se
    mono = "yes" if (r0<r18<r9 or r0>r18>r9) else "no"
    star=" *" if EDG[i]==8.9 else "  "
    print(f"{f'[{EDG[i]:g},{EDG[i+1]:g})'+star:<14}{r0:>10.4f}{r18:>10.4f}{r9:>10.4f}"
          f"{dd:>+9.4f}{sig:>7.1f}{mono:>6}")
    rows.append((EDG[i],EDG[i+1],sig,mono,dd))

print("\nsummary")
best=max(rows,key=lambda r:r[2])
print(f"  strongest bin      : theta [{best[0]:g},{best[1]:g}) at {best[2]:.1f} sigma, monotonic={best[3]}")
inband=[r for r in rows if r[0]==8.9][0]
print(f"  HGND band [8.9,13.1): {inband[2]:.1f} sigma, monotonic={inband[3]}")
# integrated forward cone vs band
for lab,lo,hi in (("forward cone theta<2", 0,2), ("HGND band", 8.9,13.1)):
    tot={}
    for k in U:
        n=int(((d[k]["n"]>=lo)&(d[k]["n"]<hi)).sum()); p=int(((d[k]["p"]>=lo)&(d[k]["p"]<hi)).sum())
        tot[k]=(n,p,n/p,(n/p)*np.sqrt(1/n+1/p))
    r0,s0=tot["zeroSpot"][2],tot["zeroSpot"][3]; r9,s9=tot["bigSpot"][2],tot["bigSpot"][3]
    print(f"  {lab:<22} n/p: {r0:.4f} -> {tot['defaultSpot'][2]:.4f} -> {r9:.4f}   "
          f"d={r9-r0:+.4f} ({abs(r9-r0)/np.hypot(s0,s9):.1f} sigma, {100*(r9-r0)/r0:+.2f} %)")
