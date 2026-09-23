"""513 nucleons/event vs A=264 for Xe+Cs. Where does the factor ~2 come from?"""
import pandas as pd, numpy as np
SP="/private/tmp/claude-501/-Users-vovvy-Project-BM-N-HGND/14d99a58-559a-4867-a447-5040e9433acb/scratchpad"
d=pd.read_csv(f"{SP}/npsample/zeroSpot_1_prim.csv",engine="pyarrow")
print(f"rows={len(d):,}  events={d.Row.nunique():,}  rows/event={len(d)/d.Row.nunique():.1f}")
ev=d[d.Row==d.Row.iloc[0]]
print(f"\nevent {ev.Row.iloc[0]}: {len(ev)} rows, {ev.Id.nunique()} unique Id")
print(f"  duplicate Id within event: {len(ev)-ev.Id.nunique()}")
print(f"  exact duplicate rows     : {len(ev)-len(ev.drop_duplicates())}")
key=["PDG","Px","Py","Pz"]
print(f"  duplicate (PDG,Px,Py,Pz) : {len(ev)-len(ev.drop_duplicates(subset=key))}")
print(f"\n  Instance range: {ev.Instance.min()}–{ev.Instance.max()}   Id range: {ev.Id.min()}–{ev.Id.max()}")
print(f"  PDG counts: {ev.PDG.value_counts().to_dict()}")
# are the duplicates identical or mirror pairs?
g=ev.groupby(key).size()
print(f"\n  momentum-multiplicity histogram: {g.value_counts().to_dict()}")
sample=ev[ev.duplicated(subset=key,keep=False)].sort_values(key).head(6)
print("\n  first duplicated momentum group:")
print(sample[["Row","Instance","Id","PDG","Ekin","Px","Py","Pz","Rapid"]].to_string(index=False))
# vertex check: two nuclei?
print(f"\n  vertex vZ unique values (first 8): {sorted(ev.vZ.unique())[:8]}")
print(f"  vertex vZ counts: {ev.vZ.value_counts().head(4).to_dict()}")
