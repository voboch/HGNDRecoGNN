"""MC-truth performance ceiling for the U_sym (Spot) sensitivity study.

Reconstruction can only ever recover *signal* neutrons (n0_label==1: a neutron
whose Ekin is consistent with its time-of-flight and which hits the detector
face). So the achievable significance is bounded by the U_sym response measured
on that subset, in the seed-42 test half the sensitivity stage evaluates, and
then degraded by reconstruction efficiency and purity.

Read from the _v3 parquets (loader v2, per-file Row blocks).
"""
import glob
import numpy as np
import pandas as pd

CACHE = "/scratch/vbocharnikov/hgnd/cache"
U = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
EDGES = np.array([0, .25, .5, .75, 1., 1.5, 2., 3., 5.])
SEED, TRAIN_FRAC = 42, 0.5

stat, spec = {}, {}
for ds in ("zeroSpot", "defaultSpot", "bigSpot"):
    g = glob.glob(f"{CACHE}/ndet_dataset_smash_{ds}_v3/processed/_hits_cache_*.parquet")
    if not g:
        print(f"{ds}: no parquet", flush=True); continue
    df = pd.read_parquet(g[0], columns=["Row", "Id", "PDG", "Ekin", "n0_label", "Ncl"])

    rows = np.sort(df.Row.unique())
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(rows))
    test_rows = set(rows[perm[int(TRAIN_FRAC * len(rows)):]].tolist())
    te = df[df.Row.isin(test_rows)]

    def block(d, label):
        n_ev = int(d.Row.nunique())
        u = d.drop_duplicates(subset=["Row", "Id"])
        alln = u[u.PDG == 2112]
        sig = d[d.n0_label > 0].drop_duplicates(subset=["Row", "Id"])
        E = sig.Ekin.to_numpy()
        nhi, nlo = int((E >= 2.0).sum()), int((E < 1.0).sum())
        R = nhi / nlo if nlo else np.nan
        seR = R * np.sqrt(1/max(nhi,1) + 1/max(nlo,1))
        return dict(label=label, events=n_ev, neutrons=len(alln), signal=len(E),
                    sig_frac=len(E)/max(len(alln),1),
                    M_sig=len(E)/n_ev, R=R, seR=seR,
                    hist=np.histogram(E, bins=EDGES)[0])

    full, test = block(df, "full"), block(te, "test half")
    stat[ds] = dict(full=full, test=test,
                    ncl=float(df.groupby("Row").Ncl.first().mean()))
    spec[ds] = test["hist"], test["events"]
    print(f"{ds:12s} U={U[ds]:>2}  full: {full['events']:>9,} ev  "
          f"signal={full['signal']:>9,} ({100*full['sig_frac']:.1f}% of neutrons)  "
          f"| test half: {test['events']:>9,} ev  signal={test['signal']:>8,}  "
          f"R={test['R']:.5f}+-{test['seR']:.5f}", flush=True)
    del df, te

a = stat["zeroSpot"]
print("\n=== signal-neutron U_sym response, TEST HALF (the reconstruction target) ===")
print(f"{'pair':<28}{'dR':>12}{'+-':>10}{'sigma':>8}{'rel %':>9}")
for ds in ("defaultSpot", "bigSpot"):
    b = stat[ds]
    d = b["test"]["R"] - a["test"]["R"]
    se = float(np.hypot(a["test"]["seR"], b["test"]["seR"]))
    print(f"{'U=%d vs U=0' % U[ds]:<28}{d:>+12.5f}{se:>10.5f}{abs(d)/se:>8.1f}"
          f"{100*d/a['test']['R']:>9.2f}")

sig_truth = abs(stat["bigSpot"]["test"]["R"] - a["test"]["R"]) / float(
    np.hypot(a["test"]["seR"], stat["bigSpot"]["test"]["seR"]))
print(f"\n=== projected significance of U=90 vs U=0 under reconstruction ===")
print("sigma_reco ~ sigma_truth * sqrt(eff) * dilution")
print(f"(sigma_truth = {sig_truth:.1f} on the test half)\n")
print(f"{'efficiency':>11}" + "".join(f"{f'pur {p:.1f}':>9}" for p in (1.0, 0.9, 0.8, 0.7, 0.6)))
for eff in (1.0, 0.8, 0.6, 0.4, 0.3, 0.2):
    row = f"{eff:>11.1f}"
    for pur in (1.0, 0.9, 0.8, 0.7, 0.6):
        row += f"{sig_truth * np.sqrt(eff) * pur:>9.1f}"
    print(row)

print("\n=== signal-neutron yield/event, Ekin-differential (test half) ===")
print(f"{'Ekin [GeV]':<13}{'U=0':>9}{'U=18':>9}{'U=90':>9}{'90/0':>8}{'sigma':>7}")
for i in range(len(EDGES) - 1):
    c0, n0 = spec["zeroSpot"]; c9, n9 = spec["bigSpot"]; c18, n18 = spec["defaultSpot"]
    y0, y9, y18 = c0[i]/n0, c9[i]/n9, c18[i]/n18
    r = y9/y0 if y0 else np.nan
    se = r*np.sqrt(1/max(c0[i],1)+1/max(c9[i],1)) if y0 else np.nan
    print(f"{f'[{EDGES[i]:g},{EDGES[i+1]:g})':<13}{y0:>9.4f}{y18:>9.4f}{y9:>9.4f}"
          f"{r:>8.4f}{abs(r-1)/se:>7.1f}")
