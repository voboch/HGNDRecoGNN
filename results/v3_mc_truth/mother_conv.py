"""What does fMotherId actually encode? Check before drawing conclusions from it."""
import glob
import numpy as np, pandas as pd
g = glob.glob("/scratch/vbocharnikov/hgnd/cache/ndet_dataset_smash_zeroSpot_v3/processed/_hits_cache_*.parquet")[0]
df = pd.read_parquet(g, columns=["Row","Id","PDG","Ekin","fMotherId","n0_label","Z"])
u = df.drop_duplicates(subset=["Row","Id"])
m = u.fMotherId
print("fMotherId over all unique MC particles:")
print(f"  min={m.min()}  max={m.max()}  n={len(m):,}")
for v in (-1, 0, 1):
    print(f"  == {v:>2}: {(m==v).sum():>10,}  ({100*(m==v).mean():5.2f}%)")
print(f"  <  0 : {(m<0).sum():>10,}  ({100*(m<0).mean():5.2f}%)")
print(f"  >  0 : {(m>0).sum():>10,}  ({100*(m>0).mean():5.2f}%)")
print()
print("cross-check against n0_label (pipeline's 'signal neutron' flag):")
sig = df[df.n0_label > 0].drop_duplicates(subset=["Row","Id"])
print(f"  signal neutrons: {len(sig):,}")
for v in (-1, 0):
    print(f"    fMotherId == {v:>2}: {(sig.fMotherId==v).sum():>9,} ({100*(sig.fMotherId==v).mean():5.2f}%)")
print(f"    fMotherId >  0 : {(sig.fMotherId>0).sum():>9,} ({100*(sig.fMotherId>0).mean():5.2f}%)")
print()
for name,pdg in (("neutron",2112),("proton",2212)):
    s = u[u.PDG==pdg]
    print(f"{name}: fMotherId==0 {100*(s.fMotherId==0).mean():5.2f}% | "
          f"<0 {100*(s.fMotherId<0).mean():5.2f}% | >0 {100*(s.fMotherId>0).mean():5.2f}%")
