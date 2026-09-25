import pandas as pd, numpy as np, json
d=pd.read_csv("results/np_ratio_smash_check_full/np_ratio_binned.csv")
ek=d[(d.figure=="ekin")&(d.valid)&(d.panel=="abs_y_lt_0p5")]
out={"bins":[]}
for (lo,hi),g in ek.groupby(["bin_lo","bin_hi"]):
    row={"lo":float(lo),"hi":float(hi)}
    ok=True
    for s in (0,18,90):
        gg=g[g.spot_mev==s]
        if gg.empty: ok=False; break
        row[str(s)]={"n":float(gg.n.iloc[0]),"p":float(gg.p.iloc[0]),
                     "ev":int(gg.events_total.iloc[0])}
    if ok and row["0"]["n"]>0: out["bins"].append(row)
out["bins"].sort(key=lambda r:r["lo"])
print(f"usable Ekin bins: {len(out['bins'])}  range {out['bins'][0]['lo']}–{out['bins'][-1]['hi']} GeV")
# per-event neutron yield + 90/0 ratio
print(f"\n{'Ekin':<12}{'Y(0)':>10}{'Y(18)':>10}{'Y(90)':>10}{'90/0':>9}{'σ':>7}")
for b in out["bins"]:
    if b["hi"]>4.0: continue
    y={s:b[str(s)]["n"]/b[str(s)]["ev"] for s in (0,18,90)}
    n0=b["0"]["n"]; n9=b["90"]["n"]
    r=y[90]/y[0]; se=r*np.sqrt(1/n0+1/n9)
    print(f"[{b['lo']:.1f},{b['hi']:.1f})".ljust(12)
          +f"{y[0]:>10.4f}{y[18]:>10.4f}{y[90]:>10.4f}{r:>9.4f}{abs(r-1)/se:>7.1f}")
json.dump(out,open("/tmp/spec.json","w"))
print("\nwrote /tmp/spec.json")
