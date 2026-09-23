"""If my rapidity selection were misaligned with the slides', the |y|<0.5 and
|y|>0.5 windows would sample the wrong physical regions. Scan the assumed
frame shift and see where the slide-like ordering appears at mid-rapidity."""
import glob, numpy as np, pandas as pd
SP="/private/tmp/claude-501/-Users-vovvy-Project-BM-N-HGND/14d99a58-559a-4867-a447-5040e9433acb/scratchpad"
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
d={}
for ds in U:
    E=[];Y=[];P=[]
    for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv")):
        x=pd.read_csv(f,usecols=["PDG","Ekin","Rapid"],engine="pyarrow")
        E.append(x.Ekin.to_numpy()); Y.append(x.Rapid.to_numpy()); P.append(x.PDG.to_numpy()); del x
    d[ds]=dict(E=np.concatenate(E),y=np.concatenate(Y),pdg=np.concatenate(P))

print("raw Rapid distribution (zeroSpot):")
y=d["zeroSpot"]["y"]
for q in (0.5,5,25,50,75,95,99.5):
    print(f"   p{q:<5} = {np.percentile(y,q):+.3f}")
print(f"   mean={y.mean():+.3f}  y_beam(lab)=1.973  y_cm(lab)=0.986\n")

print("scan of assumed frame shift: n/p double ratio 90/0 at |y-shift|<0.5, Ekin 0.9-1.3 GeV")
print("(slide 3 mid-rapidity expectation ~1.13)\n")
print(f"{'shift':>7}{'n(S=0)':>9}{'n/p S=0':>10}{'n/p S=90':>10}{'90/0':>9}{'±':>8}")
for sh in (0.0,0.25,0.5,0.75,0.9863,1.25,1.5,1.75):
    vals={}
    for ds in U:
        ycm=d[ds]["y"]-sh
        m=(d[ds]["E"]>=0.9)&(d[ds]["E"]<1.3)&(np.abs(ycm)<0.5)
        n=int((m&(d[ds]["pdg"]==2112)).sum()); p=int((m&(d[ds]["pdg"]==2212)).sum())
        vals[ds]=(n,p)
    n0,p0=vals["zeroSpot"]; n9,p9=vals["bigSpot"]
    if min(n0,p0,n9,p9)<20:
        print(f"{sh:>7.3f}{n0:>9,}{'—':>10}{'—':>10}{'low stat':>17}"); continue
    r0=n0/p0; r9=n9/p9; dr=r9/r0; sd=dr*np.sqrt(1/n0+1/p0+1/n9+1/p9)
    print(f"{sh:>7.3f}{n0:>9,}{r0:>10.3f}{r9:>10.3f}{dr:>9.3f}{sd:>8.3f}")

print("\nsame scan, |y-shift| > 0.5 (where my result already matched the slides)")
print(f"{'shift':>7}{'n(S=0)':>9}{'n/p S=0':>10}{'n/p S=90':>10}{'90/0':>9}{'±':>8}")
for sh in (0.0,0.5,0.9863,1.5):
    vals={}
    for ds in U:
        ycm=d[ds]["y"]-sh
        m=(d[ds]["E"]>=0.9)&(d[ds]["E"]<1.3)&(np.abs(ycm)>=0.5)
        n=int((m&(d[ds]["pdg"]==2112)).sum()); p=int((m&(d[ds]["pdg"]==2212)).sum())
        vals[ds]=(n,p)
    n0,p0=vals["zeroSpot"]; n9,p9=vals["bigSpot"]
    r0=n0/p0; r9=n9/p9; dr=r9/r0; sd=dr*np.sqrt(1/n0+1/p0+1/n9+1/p9)
    print(f"{sh:>7.3f}{n0:>9,}{r0:>10.3f}{r9:>10.3f}{dr:>9.3f}{sd:>8.3f}")
