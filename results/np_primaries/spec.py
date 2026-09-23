"""Do the primaries include spectator nucleons? Xe+Cs has A=131+133=264.
If the per-event nucleon count approaches 264, the file holds every nucleon of
both nuclei, not just participants -- and spectators would dilute any U_sym
signal in the participant region while dominating forward angles."""
import glob, numpy as np, pandas as pd
SP="/private/tmp/claude-501/-Users-vovvy-Project-BM-N-HGND/14d99a58-559a-4867-a447-5040e9433acb/scratchpad"
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
YB=1.9727
for ds in U:
    E=[];Y=[];P=[];PT=[];R=[]
    for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv")):
        x=pd.read_csv(f,usecols=["Row","PDG","Ekin","Rapid","Pt","Pz"],engine="pyarrow")
        E.append(x.Ekin.to_numpy());Y.append(x.Rapid.to_numpy());P.append(x.PDG.to_numpy())
        PT.append(x.Pt.to_numpy());R.append(x.Row.to_numpy()); del x
    E=np.concatenate(E);Y=np.concatenate(Y);P=np.concatenate(P);PT=np.concatenate(PT)
    nev=len(np.unique(np.concatenate(R)))
    n=(P==2112).sum(); p=(P==2212).sum()
    print(f"{ds:12s} nucleons/event = {(n+p)/nev:6.1f}   (Xe+Cs total A = 264)"
          f"   n/ev={n/nev:5.1f}  p/ev={p/nev:5.1f}")
    if ds=="zeroSpot":
        # spectator proxy: low pT and rapidity near target (0) or projectile (y_beam)
        proj = (np.abs(Y-YB)<0.25)&(PT<0.25)
        targ = (np.abs(Y)<0.25)&(PT<0.25)
        part = ~(proj|targ)
        print(f"\n  composition (zeroSpot), spectator proxy |y-y_0|<0.25 & pT<0.25 GeV/c:")
        print(f"    projectile-like : {100*proj.mean():5.1f}%   n/p = {(P[proj]==2112).sum()/max((P[proj]==2212).sum(),1):.3f}")
        print(f"    target-like     : {100*targ.mean():5.1f}%   n/p = {(P[targ]==2112).sum()/max((P[targ]==2212).sum(),1):.3f}")
        print(f"    remainder       : {100*part.mean():5.1f}%   n/p = {(P[part]==2112).sum()/max((P[part]==2212).sum(),1):.3f}")
        th=np.degrees(np.arctan2(PT,np.concatenate([np.array([])]*0+[np.zeros(0)]) if False else np.zeros(len(PT))))
print()
# spectator fraction vs polar angle, all three
print("spectator-proxy fraction vs polar angle (zeroSpot):")
ds="zeroSpot"
E=[];Y=[];P=[];PT=[];PZ=[]
for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv")):
    x=pd.read_csv(f,usecols=["PDG","Rapid","Pt","Pz"],engine="pyarrow")
    Y.append(x.Rapid.to_numpy());P.append(x.PDG.to_numpy());PT.append(x.Pt.to_numpy());PZ.append(x.Pz.to_numpy()); del x
Y=np.concatenate(Y);P=np.concatenate(P);PT=np.concatenate(PT);PZ=np.concatenate(PZ)
th=np.degrees(np.arctan2(PT,PZ))
spec=((np.abs(Y-YB)<0.25)|(np.abs(Y)<0.25))&(PT<0.25)
for lo,hi in ((0,1),(1,2),(2,3),(3,4),(4,6),(6,8.9),(8.9,13.1),(13.1,25),(25,90)):
    m=(th>=lo)&(th<hi)
    if m.sum()<50: continue
    print(f"   theta [{lo:g},{hi:g})  : {100*spec[m].mean():5.1f}% spectator-like   (n={m.sum():,})")
