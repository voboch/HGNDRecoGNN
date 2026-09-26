import glob,os,json,numpy as np,pandas as pd
SP=os.environ["SP"]; U={"zeroSpot":0,"defaultSpot":18,"bigSpot":90}
TH_LO,TH_HI=8.9,13.1; ELO,EHI=1.0,2.0; NB=10
def load(ds):
    rows=[]
    for f in sorted(glob.glob(f"{SP}/cent/{ds}_*_prim.csv")):
        d=pd.read_csv(f,usecols=["Row","PDG","Ekin","Pt","Pz","B"],engine="pyarrow")
        d["ev"]=os.path.basename(f).split("_")[1]+":"+d.Row.astype(str)
        d=d[d.ev!=d.ev.iloc[-1]]; rows.append(d)
    d=pd.concat(rows,ignore_index=True)
    th=np.degrees(np.arctan2(d.Pt.to_numpy(),d.Pz.to_numpy()))
    inb=(th>=TH_LO)&(th<TH_HI); isn=(d.PDG==2112)
    b=d.groupby("ev").B.first(); o=pd.DataFrame({"b":b})
    o["n_hi"]=d[isn&(d.Ekin>=EHI)].groupby("ev").size().reindex(b.index,fill_value=0)
    o["n_lo"]=d[isn&(d.Ekin<ELO)].groupby("ev").size().reindex(b.index,fill_value=0)
    o["b_hi"]=d[inb&isn&(d.Ekin>=EHI)].groupby("ev").size().reindex(b.index,fill_value=0)
    o["b_lo"]=d[inb&isn&(d.Ekin<ELO)].groupby("ev").size().reindex(b.index,fill_value=0)
    return o
D={ds:load(ds) for ds in U}
out={"U":U,"nev":{ds:int(len(D[ds])) for ds in D}}
# b distribution, common bins
be=np.linspace(0,14,29)
out["b_edges"]=be.tolist()
out["b_hist"]={ds:np.histogram(D[ds].b,bins=be)[0].tolist() for ds in D}
out["b_mean"]={ds:float(D[ds].b.mean()) for ds in D}
# R vs centrality, both matchings
allb=np.concatenate([D[ds].b.to_numpy() for ds in D])
common=np.quantile(allb,np.linspace(0,1,NB+1))
def series(edges_fn,num,den):
    res={ds:[] for ds in D}
    for i in range(NB):
        for ds in D:
            e=edges_fn(ds); p=D[ds]
            m=(p.b>=e[i])&((p.b<e[i+1]) if i<NB-1 else (p.b<=e[i+1]))
            A,B_=p[num][m].sum(),p[den][m].sum()
            r=A/B_ if B_>0 else float("nan")
            se=r*np.sqrt(1/max(A,1)+1/max(B_,1)) if B_>0 else float("nan")
            res[ds].append({"r":r,"se":se,"A":int(A),"B":int(B_)})
    return res
pct={ds:np.quantile(D[ds].b,np.linspace(0,1,NB+1)) for ds in D}
out["R_pct"]=series(lambda ds:pct[ds],"n_hi","n_lo")
out["R_com"]=series(lambda ds:common,"n_hi","n_lo")
out["Rband_com"]=series(lambda ds:common,"b_hi","b_lo")
json.dump(out,open(SP+"/cent_fig.json","w"))
print("events:",out["nev"]); print("mean b:",{k:round(v,4) for k,v in out["b_mean"].items()})
print("wrote cent_fig.json")
