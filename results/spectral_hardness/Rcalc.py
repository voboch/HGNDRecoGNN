"""Spectral hardness R = N(Ekin > Ehi) / N(Ekin < Elo) for NEUTRONS only.
Computed from the full-production primary spectra (np_ratio_binned.csv), and
compared with the HGND-detected-neutron measurement from the _v3 caches."""
import pandas as pd, numpy as np
d=pd.read_csv("results/np_ratio_smash_check_full/np_ratio_binned.csv")
ek=d[(d.figure=="ekin")&(d.valid)]
S={0:"0 MeV",18:"18 MeV",90:"90 MeV"}
print("full-production PRIMARY neutrons, spectral hardness R\n")
for panel,plab in (("abs_y_lt_0p5","|y_cm| < 0.5"),("abs_y_gt_0p5","|y_cm| > 0.5")):
    q=ek[ek.panel==panel]
    print(f"--- {plab} ---")
    for Elo,Ehi in ((1.0,2.0),(0.8,1.5),(1.0,1.5)):
        res={}
        for s in S:
            g=q[q.spot_mev==s]
            lo=g[g.bin_hi<=Elo]["n"].sum()
            hi=g[g.bin_lo>=Ehi]["n"].sum()
            if lo<=0 or hi<=0: res=None; break
            R=hi/lo; se=R*np.sqrt(1/hi+1/lo)
            res[s]=(R,se,hi,lo)
        if not res: print(f"  Ehi>{Ehi} / Elo<{Elo}: insufficient bins"); continue
        r0,s0,_,_=res[0]; r9,s9,_,_=res[90]
        dr=r9/r0; sd=dr*np.sqrt((s0/r0)**2+(s9/r9)**2)
        mono = "yes" if res[0][0]<res[18][0]<res[90][0] else "no"
        print(f"  R = N(E>{Ehi})/N(E<{Elo}):  "
              f"{res[0][0]:.4f} / {res[18][0]:.4f} / {res[90][0]:.4f}   "
              f"90/0 = {dr:.4f} ± {sd:.4f}  ({abs(dr-1)/sd:5.1f}σ)  ordered={mono}")
    print()

print("="*72)
print("  HGND-DETECTED neutrons, _v3 caches (the actual detector observable)")
print("="*72)
print("  all detected neutrons, full statistics:")
print("    R = 0.36155 ± 0.00065 / 0.36219 ± 0.00068 / 0.37179 ± 0.00080")
d0,e0,d9,e9=0.36155,0.00065,0.37179,0.00080
print(f"    90/0 = {d9/d0:.4f} ± {(d9/d0)*np.sqrt((e0/d0)**2+(e9/d9)**2):.4f}"
      f"  ({abs(d9-d0)/np.hypot(e0,e9):.1f}σ)  ordered=yes")
print("  signal neutrons, seed-42 test half (reconstruction target):")
print("    R = 1.23848 ± 0.00416 / 1.25001 ± 0.00439 / 1.28051 ± 0.00514")
a0,b0,a9,b9=1.23848,0.00416,1.28051,0.00514
print(f"    90/0 = {a9/a0:.4f} ± {(a9/a0)*np.sqrt((b0/a0)**2+(b9/a9)**2):.4f}"
      f"  ({abs(a9-a0)/np.hypot(b0,b9):.1f}σ)  ordered=yes")
