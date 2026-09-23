"""n/p from PRIMARY nucleons: full 4pi vs the HGND angular acceptance.

Does the Spot dependence of n/p come from the proton spectrum, and is it
confined to the HGND acceptance region?

  4pi        every primary p and n -- the physics ratio
  HGND band  primaries whose EMISSION direction points into the HGND annulus,
             theta in [8.9, 13.1] deg (derived from hit positions). This is what
             you would measure if protons flew straight; they do not, because the
             analysing magnet bends them, which is why the ratio measured from
             HGND hits is not this number.

Decomposition: R = n/p  =>  dR/R = dn/n - dp/p, so a ratio change is attributed
to numerator or denominator directly.
"""
import glob, numpy as np, pandas as pd

SP = "/private/tmp/claude-501/-Users-vovvy-Project-BM-N-HGND/14d99a58-559a-4867-a447-5040e9433acb/scratchpad"
U = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
TH_LO, TH_HI = 8.9, 13.1
EDGES = np.array([0, .25, .5, .75, 1., 1.5, 2., 3., 5., 8.])

res = {}
for ds in ("zeroSpot", "defaultSpot", "bigSpot"):
    ns4=[]; ps4=[]; nsA=[]; psA=[]; yA_n=[]; yA_p=[]; nev=0
    for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv")):
        d = pd.read_csv(f, usecols=["Row","PDG","Ekin","Pt","Pz","Rapid"], engine="pyarrow")
        nev += int(d.Row.nunique())
        th = np.degrees(np.arctan2(d.Pt.to_numpy(), d.Pz.to_numpy()))
        acc = (th >= TH_LO) & (th <= TH_HI)
        isn = (d.PDG==2112).to_numpy(); isp = (d.PDG==2212).to_numpy()
        E = d.Ekin.to_numpy(); y = d.Rapid.to_numpy()
        ns4.append(E[isn]); ps4.append(E[isp])
        nsA.append(E[isn&acc]); psA.append(E[isp&acc])
        yA_n.append(y[isn&acc]); yA_p.append(y[isp&acc])
        del d
    res[ds] = dict(ev=nev,
                   n4=np.concatenate(ns4), p4=np.concatenate(ps4),
                   nA=np.concatenate(nsA), pA=np.concatenate(psA),
                   yn=np.concatenate(yA_n), yp=np.concatenate(yA_p))
    r = res[ds]
    print(f"{ds:12s} U={U[ds]:>2}  events={nev:>6,}   "
          f"4pi n={len(r['n4']):>7,} p={len(r['p4']):>7,}   "
          f"band n={len(r['nA']):>6,} p={len(r['pA']):>6,}  "
          f"(band keeps {100*len(r['nA'])/len(r['n4']):.2f}% of n, "
          f"{100*len(r['pA'])/len(r['p4']):.2f}% of p)", flush=True)

def rat(a,b):
    r=len(a)/len(b); return r, r*np.sqrt(1/max(len(a),1)+1/max(len(b),1))

z = res["zeroSpot"]
print("\n"+"="*80)
print("  A.  n/p and its U_sym response")
print("="*80)
for lab,kn,kp in (("4pi  (physics ratio)","n4","p4"), ("HGND band (theta 8.9-13.1 deg)","nA","pA")):
    r0,s0 = rat(z[kn], z[kp])
    print(f"\n{lab}\n  U= 0   n/p = {r0:.4f} +- {s0:.4f}")
    for ds in ("defaultSpot","bigSpot"):
        b=res[ds]; r,s = rat(b[kn],b[kp]); d=r-r0; se=float(np.hypot(s0,s))
        print(f"  U={U[ds]:>2}   n/p = {r:.4f} +- {s:.4f}    d={d:+.4f}  "
              f"({abs(d)/se:4.1f} sigma, {100*d/r0:+.2f} %)")

print("\n"+"="*80)
print("  B.  where does the change come from?   dR/R = dn/n - dp/p")
print("="*80)
for lab,kn,kp in (("4pi","n4","p4"), ("HGND band","nA","pA")):
    print(f"\n{lab}:")
    print(f"  {'pair':<10}{'dR/R %':>9}{'dn/n %':>9}{'dp/p %':>9}   attribution")
    for ds in ("defaultSpot","bigSpot"):
        b=res[ds]
        r0,_=rat(z[kn],z[kp]); r,_=rat(b[kn],b[kp])
        dR=100*(r-r0)/r0
        yn0=len(z[kn])/z["ev"]; yp0=len(z[kp])/z["ev"]
        dn=100*((len(b[kn])/b["ev"])-yn0)/yn0
        dp=100*((len(b[kp])/b["ev"])-yp0)/yp0
        tot=abs(dn)+abs(dp)
        att = f"{100*abs(dp)/tot:.0f}% proton" if tot>0 else "n/a"
        print(f"  U={U[ds]:<7}{dR:>9.2f}{dn:>9.2f}{dp:>9.2f}   {att}")

print("\n"+"="*80)
print("  C.  n/p vs Ekin  —  4pi (left) and HGND band (right)")
print("="*80)
print(f"{'Ekin [GeV]':<12}|{'4pi: U=0':>10}{'U=90':>9}{'90/0':>8}{'sig':>6}  |"
      f"{'band: U=0':>11}{'U=90':>9}{'90/0':>8}{'sig':>6}")
for i in range(len(EDGES)-1):
    line=f"[{EDGES[i]:g},{EDGES[i+1]:g})"
    out=f"{line:<12}|"
    for kn,kp in (("n4","p4"),("nA","pA")):
        vals={}
        for ds in res:
            b=res[ds]
            nn=int(((b[kn]>=EDGES[i])&(b[kn]<EDGES[i+1])).sum())
            pp=int(((b[kp]>=EDGES[i])&(b[kp]<EDGES[i+1])).sum())
            vals[ds]=(nn,pp)
        n0,p0=vals["zeroSpot"]; n9,p9=vals["bigSpot"]
        if p0<20 or p9<20 or n0<20 or n9<20:
            out+=f"{'—':>10}{'—':>9}{'—':>8}{'—':>6}  |" if kn=="n4" else f"{'—':>11}{'—':>9}{'—':>8}{'—':>6}"
            continue
        r0=n0/p0; r9=n9/p9; dr=r9/r0
        se=dr*np.sqrt(1/n0+1/p0+1/n9+1/p9)
        if kn=="n4": out+=f"{r0:>10.3f}{r9:>9.3f}{dr:>8.4f}{abs(dr-1)/se:>6.1f}  |"
        else:        out+=f"{r0:>11.3f}{r9:>9.3f}{dr:>8.4f}{abs(dr-1)/se:>6.1f}"
    print(out)

print("\n"+"="*80)
print("  D.  what the band actually contains (rapidity of accepted primaries)")
print("="*80)
for ds in res:
    b=res[ds]
    print(f"  {ds:12s} n: y={np.mean(b['yn']):.3f}+-{np.std(b['yn']):.3f}  "
          f"Ekin med={np.median(b['nA']):.3f} | "
          f"p: y={np.mean(b['yp']):.3f}+-{np.std(b['yp']):.3f}  "
          f"Ekin med={np.median(b['pA']):.3f}")
