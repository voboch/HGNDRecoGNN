"""R for primary NEUTRONS inside the HGND angular band, and split by the
spectator proxy — is R's sensitivity participant physics or spectator leakage?"""
import glob, numpy as np, pandas as pd, os
SP=os.environ["SP"]
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
YB=1.9727; TH_LO,TH_HI=8.9,13.1
res={}
for ds in U:
    E=[];TH=[];SP_=[];PDG=[]
    for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv")):
        x=pd.read_csv(f,usecols=["PDG","Ekin","Pt","Pz","Rapid"],engine="pyarrow")
        th=np.degrees(np.arctan2(x.Pt.to_numpy(),x.Pz.to_numpy()))
        y=x.Rapid.to_numpy(); pt=x.Pt.to_numpy()
        spec=((np.abs(y-YB)<0.25)|(np.abs(y)<0.25))&(pt<0.25)
        E.append(x.Ekin.to_numpy()); TH.append(th); SP_.append(spec); PDG.append(x.PDG.to_numpy()); del x
    res[ds]=dict(E=np.concatenate(E),th=np.concatenate(TH),
                 spec=np.concatenate(SP_),pdg=np.concatenate(PDG))

def R(sel,E,Elo=1.0,Ehi=2.0):
    hi=int((sel&(E>=Ehi)).sum()); lo=int((sel&(E<Elo)).sum())
    if hi<10 or lo<10: return None
    r=hi/lo; return r, r*np.sqrt(1/hi+1/lo), hi, lo

print("R = N(Ekin>2)/N(Ekin<1) for PRIMARY NEUTRONS in the HGND band "
      f"(theta {TH_LO}-{TH_HI} deg)\n")
for lab,mode in (("all neutrons in band","all"),
                 ("participant-like only","part"),
                 ("spectator-like only","spec")):
    out=[]
    for ds in U:
        d=res[ds]
        sel=(d["pdg"]==2112)&(d["th"]>=TH_LO)&(d["th"]<TH_HI)
        if mode=="part": sel&=~d["spec"]
        elif mode=="spec": sel&=d["spec"]
        out.append(R(sel,d["E"]))
    if any(o is None for o in out):
        print(f"  {lab:<24} insufficient statistics in this pilot sample"); continue
    (r0,s0,h0,l0),(r18,_,_,_),(r9,s9,h9,l9)=out
    dr=r9/r0; sd=dr*np.sqrt((s0/r0)**2+(s9/r9)**2)
    mono="yes" if r0<r18<r9 else "no"
    print(f"  {lab:<24} {r0:.4f} / {r18:.4f} / {r9:.4f}   "
          f"90/0={dr:.4f}±{sd:.4f} ({abs(dr-1)/sd:4.1f}σ) ordered={mono}   "
          f"[N hi/lo = {h0}/{l0}]")
print("\n  (pilot sample: 2,225 events per Spot — indicative only;"
      "\n   the full-production numbers are in the report)")
