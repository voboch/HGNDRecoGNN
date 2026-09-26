"""Feasibility for the HGND-relevant selections, with centrality matched in b.

The 4pi result showed that an apparent R signal is produced by a 2.9% mean-b
difference between productions. This repeats the test for the selections the
HGND actually measures.
"""
import glob, os, numpy as np, pandas as pd
SP=os.environ["SP"]
U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
TH_LO,TH_HI=8.9,13.1; ELO,EHI=1.0,2.0; NB=10; YS=0.9863

def load(ds):
    rows=[]
    for f in sorted(glob.glob(f"{SP}/cent/{ds}_*_prim.csv")):
        d=pd.read_csv(f,usecols=["Row","PDG","Ekin","Pt","Pz","Rapid","B"],engine="pyarrow")
        d["ev"]=os.path.basename(f).split("_")[1]+":"+d.Row.astype(str)
        d=d[d.ev!=d.ev.iloc[-1]]; rows.append(d)
    d=pd.concat(rows,ignore_index=True)
    th=np.degrees(np.arctan2(d.Pt.to_numpy(),d.Pz.to_numpy()))
    ycm=d.Rapid.to_numpy()-YS
    inb=(th>=TH_LO)&(th<TH_HI); mid=np.abs(ycm)<0.5
    isn=(d.PDG==2112); isp=(d.PDG==2212)
    b=d.groupby("ev").B.first()
    o=pd.DataFrame({"b":b})
    cols={
      "band_n_hi": inb&isn&(d.Ekin>=EHI), "band_n_lo": inb&isn&(d.Ekin<ELO),
      "mid_n_hi":  mid&isn&(d.Ekin>=EHI), "mid_n_lo":  mid&isn&(d.Ekin<ELO),
      "band_n": inb&isn, "band_p": inb&isp,
    }
    for nm,m in cols.items():
        o[nm]=d[m].groupby("ev").size().reindex(b.index,fill_value=0)
    return o

D={ds:load(ds) for ds in U}
allb=np.concatenate([D[ds].b.to_numpy() for ds in D])
EDG=np.quantile(allb,np.linspace(0,1,NB+1))

def rat(a,b_):
    if a<=0 or b_<=0: return np.nan,np.nan
    r=a/b_; return r,r*np.sqrt(1/a+1/b_)

def run(num,den,title):
    print("="*72); print(f"  {title}"); print("  centrality matched in b (common edges across samples)")
    print("="*72)
    print(f"{'class':<11}{'U=0':>11}{'U=18':>11}{'U=90':>11}{'90/0':>10}{'sig':>7}{'ord':>5}")
    tot={ds:[0,0] for ds in D}; nsig=0; nord=0
    for i in range(NB):
        v={}
        for ds in D:
            p=D[ds]; m=(p.b>=EDG[i])&((p.b<EDG[i+1]) if i<NB-1 else (p.b<=EDG[i+1]))
            A,B_=p[num][m].sum(),p[den][m].sum(); tot[ds][0]+=A; tot[ds][1]+=B_
            v[ds]=rat(A,B_)
        (r0,s0),(r18,_),(r9,s9)=v["zeroSpot"],v["defaultSpot"],v["bigSpot"]
        if not np.isfinite(r0) or not np.isfinite(r9):
            print(f"{f'{i*10}-{(i+1)*10}%':<11}{'insufficient statistics':>50}"); continue
        dr=r9/r0; sd=dr*np.sqrt((s0/r0)**2+(s9/r9)**2); sig=abs(dr-1)/sd
        o="yes" if r0<r18<r9 else ("rev" if r0>r18>r9 else "no")
        if sig>3: nsig+=1
        if o in ("yes","rev"): nord+=1
        print(f"{f'{i*10}-{(i+1)*10}%':<11}{r0:>11.4f}{r18:>11.4f}{r9:>11.4f}{dr:>10.4f}{sig:>7.1f}{o:>5}")
    v={ds:rat(*tot[ds]) for ds in D}
    (r0,s0),(r18,_),(r9,s9)=v["zeroSpot"],v["defaultSpot"],v["bigSpot"]
    dr=r9/r0; sd=dr*np.sqrt((s0/r0)**2+(s9/r9)**2)
    o="yes" if r0<r18<r9 else ("rev" if r0>r18>r9 else "no")
    print(f"{'ALL':<11}{r0:>11.4f}{r18:>11.4f}{r9:>11.4f}{dr:>10.4f}{abs(dr-1)/sd:>7.1f}{o:>5}")
    print(f"  -> classes above 3 sigma: {nsig}/{NB};  ordered in S_pot: {nord}/{NB}\n")

run("band_n_hi","band_n_lo","spectral hardness R, neutrons in the HGND band")
run("mid_n_hi","mid_n_lo","spectral hardness R, neutrons at |y_cm| < 0.5")
run("band_n","band_p","n/p in the HGND band")
