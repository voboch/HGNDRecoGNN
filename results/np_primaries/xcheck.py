"""Cross-check against 'n/p - SMASH check' slide 3:
Xe+Cs @ 2.5A GeV (= sqrt(s_NN) 2.87 GeV), n/p vs Ekin, |y|<0.5 and |y|>0.5,
Ekin > 300 MeV. Slides report clear S_pot ordering 0 < 18 < 90 growing with Ekin.
"""
import glob, numpy as np, pandas as pd
SP="/private/tmp/claude-501/-Users-vovvy-Project-BM-N-HGND/14d99a58-559a-4867-a447-5040e9433acb/scratchpad"
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
YSHIFT=0.9863                      # y_cm = y_lab - y_beam/2
EDG=np.array([0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.9,2.3,3.0])
d={}
for ds in U:
    E=[];Y=[];P=[]
    for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv")):
        x=pd.read_csv(f,usecols=["PDG","Ekin","Rapid"],engine="pyarrow")
        E.append(x.Ekin.to_numpy()); Y.append(x.Rapid.to_numpy()); P.append(x.PDG.to_numpy())
        del x
    d[ds]=dict(E=np.concatenate(E),y=np.concatenate(Y),pdg=np.concatenate(P))

print("=== is Rapid lab or cm? ===")
for ds in ("zeroSpot",):
    y=d[ds]["y"]
    print(f"  Rapid: min={y.min():.2f} max={y.max():.2f} mean={y.mean():.3f} median={np.median(y):.3f}")
    print(f"  after -{YSHIFT}: mean={y.mean()-YSHIFT:.3f} median={np.median(y)-YSHIFT:.3f}")
    print("  -> slides show y symmetric about 0 (cm); shift applied below")

for ycut,lab in ((0.5,"|y_cm| < 0.5"),(None,"|y_cm| > 0.5")):
    print(f"\n{'='*74}\n  n/p vs Ekin,  {lab},  Ekin > 0.3 GeV   [slide 3 comparison]\n{'='*74}")
    print(f"{'Ekin [GeV]':<13}{'S=0':>9}{'S=18':>9}{'S=90':>9}{'90/0':>9}{'ordered':>9}")
    for i in range(len(EDG)-1):
        row={}
        for ds in U:
            ycm=d[ds]["y"]-YSHIFT
            m=(d[ds]["E"]>=EDG[i])&(d[ds]["E"]<EDG[i+1])
            m &= (np.abs(ycm)<0.5) if ycut else (np.abs(ycm)>=0.5)
            n=int((m&(d[ds]["pdg"]==2112)).sum()); p=int((m&(d[ds]["pdg"]==2212)).sum())
            row[ds]=(n/p if p>5 else np.nan,n,p)
        r0,r18,r9=row["zeroSpot"][0],row["defaultSpot"][0],row["bigSpot"][0]
        if not np.isfinite(r0) or not np.isfinite(r9):
            print(f"{f'[{EDG[i]:g},{EDG[i+1]:g})':<13}{'—':>9}{'—':>9}{'—':>9}{'low stat':>18}"); continue
        mono="yes" if (r0<r18<r9) else ("rev" if (r0>r18>r9) else "no")
        print(f"{f'[{EDG[i]:g},{EDG[i+1]:g})':<13}{r0:>9.3f}{r18:>9.3f}{r9:>9.3f}{r9/r0:>9.4f}{mono:>9}")

print(f"\n{'='*74}\n  integrated, Ekin > 0.3 GeV  (slides' working cut)\n{'='*74}")
for ycut,lab in ((True,"|y_cm| < 0.5"),(False,"|y_cm| > 0.5"),(None,"all y")):
    out=f"  {lab:<14}"
    vals=[]
    for ds in U:
        ycm=d[ds]["y"]-YSHIFT
        m=d[ds]["E"]>0.3
        if ycut is True: m&=np.abs(ycm)<0.5
        elif ycut is False: m&=np.abs(ycm)>=0.5
        n=int((m&(d[ds]["pdg"]==2112)).sum()); p=int((m&(d[ds]["pdg"]==2212)).sum())
        r=n/p; se=r*np.sqrt(1/n+1/p); vals.append((r,se,n,p))
        out+=f"  S={U[ds]}: {r:.4f}±{se:.4f}"
    d0,d9=vals[0],vals[2]
    sig=abs(d9[0]-d0[0])/np.hypot(d0[1],d9[1])
    mono = "yes" if vals[0][0]<vals[1][0]<vals[2][0] else "no"
    print(out+f"   90/0={d9[0]/d0[0]:.4f} ({sig:.1f}σ, ordered={mono})")
