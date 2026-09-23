"""Angular distributions of primary n and p, and the significance of the
band-yield differences. The HGND band is narrow, so what matters is how the
U_sym potential redistributes nucleons in polar angle."""
import glob, numpy as np, pandas as pd
SP="/private/tmp/claude-501/-Users-vovvy-Project-BM-N-HGND/14d99a58-559a-4867-a447-5040e9433acb/scratchpad"
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
EDG=np.array([0,2,4,6,8.9,13.1,18,25,40,90,180])
d={}
for ds in U:
    th_n=[];th_p=[];nev=0
    for f in sorted(glob.glob(f"{SP}/npsample/{ds}_*_prim.csv")):
        x=pd.read_csv(f,usecols=["Row","PDG","Pt","Pz"],engine="pyarrow")
        nev+=int(x.Row.nunique())
        t=np.degrees(np.arctan2(x.Pt.to_numpy(),x.Pz.to_numpy()))
        isn=(x.PDG==2112).to_numpy()
        th_n.append(t[isn]); th_p.append(t[(x.PDG==2212).to_numpy()])
        del x
    d[ds]=dict(n=np.concatenate(th_n),p=np.concatenate(th_p),ev=nev)

print("primary nucleons per event by polar angle (HGND band = 8.9-13.1 deg)\n")
print(f"{'theta [deg]':<14}" + "".join(f"{f'n U={U[k]}':>11}" for k in U) + "   |" + "".join(f"{f'p U={U[k]}':>11}" for k in U))
for i in range(len(EDG)-1):
    lab=f"[{EDG[i]:g},{EDG[i+1]:g})"
    mark=" *" if (EDG[i]==8.9) else "  "
    row=f"{lab+mark:<14}"
    for sp in ("n","p"):
        for k in U:
            c=int(((d[k][sp]>=EDG[i])&(d[k][sp]<EDG[i+1])).sum())
            row+=f"{c/d[k]['ev']:>11.3f}"
        row+="   |" if sp=="n" else ""
    print(row)

print("\nband occupancy vs zeroSpot (Poisson errors):")
for sp,name in (("n","neutrons"),("p","protons")):
    c0=int(((d["zeroSpot"][sp]>=8.9)&(d["zeroSpot"][sp]<13.1)).sum())
    print(f"  {name}:  U=0 -> {c0:,}")
    for k in ("defaultSpot","bigSpot"):
        c=int(((d[k][sp]>=8.9)&(d[k][sp]<13.1)).sum())
        diff=c-c0; se=np.sqrt(c+c0)
        print(f"    U={U[k]:>2}  {c:,}   d={diff:+,} +- {se:.0f}  ({diff/se:+.1f} sigma, {100*diff/c0:+.2f} %)")

print("\ntotal 4pi yields per event (sanity: same system, should agree):")
for sp,name in (("n","neutrons"),("p","protons")):
    vals=[len(d[k][sp])/d[k]["ev"] for k in U]
    print(f"  {name}: " + "  ".join(f"U={U[k]}:{len(d[k][sp])/d[k]['ev']:.2f}" for k in U)
          + f"   spread={100*(max(vals)-min(vals))/np.mean(vals):.2f} %")
