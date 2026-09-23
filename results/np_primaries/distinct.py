"""Are the three S_pot samples genuinely different, and do they differ in the
PARTICIPANT region where the slides see the effect?"""
import glob, numpy as np, pandas as pd, hashlib
SP="/private/tmp/claude-501/-Users-vovvy-Project-BM-N-HGND/14d99a58-559a-4867-a447-5040e9433acb/scratchpad"
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
YB=1.9727; YS=0.9863
print("file-content check (first 2 MB of each sampled file):")
for ds in U:
    for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv"))[:1]:
        h=hashlib.md5(open(f,'rb').read(2*1024*1024)).hexdigest()[:16]
        print(f"  {ds:12s} {h}")
print()
D={}
for ds in U:
    fr=[]
    for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv")):
        x=pd.read_csv(f,usecols=["Row","PDG","Ekin","Rapid","Pt","Pz"],engine="pyarrow")
        x["ev"]=f.split("_")[-2]+"_"+x.Row.astype(str)
        fr.append(x)
    D[ds]=pd.concat(fr,ignore_index=True); del fr

# participant selection: veto spectator-like nucleons
print("PARTICIPANT region (spectator veto: not(|y-0|<0.25 or |y-y_beam|<0.25 with pT<0.25))")
print(f"{'':12s}{'N part':>10}{'<pT>':>8}{'<y_cm>':>9}{'n/p':>9}{'±':>8}")
base=None
for ds in U:
    d=D[ds]
    y=d.Rapid.to_numpy(); pt=d.Pt.to_numpy()
    spec=((np.abs(y-YB)<0.25)|(np.abs(y)<0.25))&(pt<0.25)
    part=~spec
    n=int((part&(d.PDG==2112)).sum()); p=int((part&(d.PDG==2212)).sum())
    r=n/p; se=r*np.sqrt(1/n+1/p)
    print(f"  {ds:10s}{n+p:>10,}{pt[part].mean():>8.3f}{(y[part]-YS).mean():>9.3f}{r:>9.4f}{se:>8.4f}")
    if base is None: base=(r,se)
print(f"    -> 90/0 = {r/base[0]:.4f} ± {(r/base[0])*np.sqrt((se/r)**2+(base[1]/base[0])**2):.4f}")

print("\nSPECTATOR-like nucleons only")
print(f"{'':12s}{'N spec':>10}{'n/p':>9}{'±':>8}")
b2=None
for ds in U:
    d=D[ds]; y=d.Rapid.to_numpy(); pt=d.Pt.to_numpy()
    spec=((np.abs(y-YB)<0.25)|(np.abs(y)<0.25))&(pt<0.25)
    n=int((spec&(d.PDG==2112)).sum()); p=int((spec&(d.PDG==2212)).sum())
    r=n/p; se=r*np.sqrt(1/n+1/p)
    print(f"  {ds:10s}{n+p:>10,}{r:>9.4f}{se:>8.4f}")
    if b2 is None: b2=(r,se)
print(f"    -> 90/0 = {r/b2[0]:.4f} ± {(r/b2[0])*np.sqrt((se/r)**2+(b2[1]/b2[0])**2):.4f}")

print("\nmean kinematics per sample (should differ if S_pot differs):")
print(f"{'':12s}{'<pT> all':>10}{'<pT> part':>11}{'<Ekin> part':>13}")
for ds in U:
    d=D[ds]; y=d.Rapid.to_numpy(); pt=d.Pt.to_numpy()
    spec=((np.abs(y-YB)<0.25)|(np.abs(y)<0.25))&(pt<0.25); part=~spec
    print(f"  {ds:10s}{pt.mean():>10.4f}{pt[part].mean():>11.4f}{d.Ekin.to_numpy()[part].mean():>13.4f}")
