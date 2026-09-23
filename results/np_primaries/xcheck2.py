"""Is my null result at mid-rapidity a real disagreement with the slides, or
just statistics? Slides (500k events) show at |y|<0.5 a clear rise of n/p for
S_pot=90 with Ekin, reaching ~1.40 at Ekin~1.5 GeV against ~1.15 for S_pot=0,
i.e. a 90/0 double ratio of ~1.20."""
import glob, numpy as np, pandas as pd
SP="/private/tmp/claude-501/-Users-vovvy-Project-BM-N-HGND/14d99a58-559a-4867-a447-5040e9433acb/scratchpad"
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
YS=0.9863
d={}
for ds in U:
    E=[];Y=[];P=[]
    for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv")):
        x=pd.read_csv(f,usecols=["PDG","Ekin","Rapid"],engine="pyarrow")
        E.append(x.Ekin.to_numpy()); Y.append(x.Rapid.to_numpy()); P.append(x.PDG.to_numpy()); del x
    d[ds]=dict(E=np.concatenate(E),y=np.concatenate(Y)-YS,pdg=np.concatenate(P))

print("counts and precision per (y, Ekin) cell — can this sample even see the slide effect?\n")
print(f"{'selection':<28}{'n(S=0)':>9}{'p(S=0)':>9}{'n/p':>8}{'±':>8}{'90/0':>8}{'±':>8}{'slides':>9}{'tension':>9}")
cases=[("|y|<0.5, Ekin 1.3-1.7",True,1.3,1.7,1.20),
       ("|y|<0.5, Ekin 0.9-1.3",True,0.9,1.3,1.13),
       ("|y|<0.5, Ekin 0.3-0.7",True,0.3,0.7,1.04),
       ("|y|>0.5, Ekin 0.3-0.7",False,0.3,0.7,1.10),
       ("|y|>0.5, Ekin 0.9-1.3",False,0.9,1.3,1.25)]
for lab,mid,e0,e1,slide in cases:
    vals={}
    for ds in U:
        m=(d[ds]["E"]>=e0)&(d[ds]["E"]<e1)
        m&= (np.abs(d[ds]["y"])<0.5) if mid else (np.abs(d[ds]["y"])>=0.5)
        n=int((m&(d[ds]["pdg"]==2112)).sum()); p=int((m&(d[ds]["pdg"]==2212)).sum())
        vals[ds]=(n,p)
    n0,p0=vals["zeroSpot"]; n9,p9=vals["bigSpot"]
    if min(n0,p0,n9,p9)<5:
        print(f"{lab:<28}{n0:>9}{p0:>9}{'—':>8}{'—':>8}{'—':>8}{'—':>8}{slide:>9.2f}{'low stat':>9}"); continue
    r0=n0/p0; s0=r0*np.sqrt(1/n0+1/p0)
    r9=n9/p9; s9=r9*np.sqrt(1/n9+1/p9)
    dr=r9/r0; sdr=dr*np.sqrt(1/n0+1/p0+1/n9+1/p9)
    tension=abs(dr-slide)/sdr
    print(f"{lab:<28}{n0:>9,}{p0:>9,}{r0:>8.3f}{s0:>8.3f}{dr:>8.3f}{sdr:>8.3f}{slide:>9.2f}{tension:>8.1f}σ")

print("\ninterpretation")
print("  'slides' column = double ratio read off slide 3 for that cell.")
print("  tension = |my 90/0 - slide 90/0| / my statistical error.")
print("  tension < 2 sigma => consistent, my sample is simply too small to resolve it.")
