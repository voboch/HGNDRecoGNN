"""MC-level ground truth on the _v3 caches (loader v2, per-file Row blocks).

These parquets are the first generation whose hits<->truth association is
trustworthy: the previous builds aliased ~8 input files onto the same event
ids, so ~31% of hits carried a truth particle from a different file.

Reports, per dataset: statistics actually available, neutron multiplicity and
spectrum, and the U_sym response of both an integrated and a spectral
observable -- with the significance of each.
"""
import glob
import numpy as np
import pandas as pd

CACHE = "/scratch/vbocharnikov/hgnd/cache"
U = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
EDGES = np.array([0, .25, .5, .75, 1., 1.5, 2., 3., 5.])
res, spec = {}, {}

for ds in ("zeroSpot", "defaultSpot", "bigSpot"):
    g = glob.glob(f"{CACHE}/ndet_dataset_smash_{ds}_v3/processed/_hits_cache_*.parquet")
    if not g:
        print(f"{ds:12s} no _v3 parquet yet — skipped", flush=True)
        continue
    df = pd.read_parquet(g[0], columns=["Row", "Id", "PDG", "Ekin", "n0_label"])
    n_ev = int(df.Row.nunique())
    # de-dup to unique MC particles: parquet rows are hit-aligned
    u = df.drop_duplicates(subset=["Row", "Id"])
    neu = u[u.PDG == 2112]
    E = neu.Ekin.to_numpy()
    # signal neutrons as the pipeline defines them
    sig = int(df.loc[df.n0_label > 0, ["Row", "Id"]].drop_duplicates().shape[0])
    nhi, nlo = int((E >= 2.0).sum()), int((E < 1.0).sum())
    R = nhi / nlo
    res[ds] = dict(
        hits=len(df), events=n_ev, particles=len(u), neutrons=len(E), signal=sig,
        M_n=len(E) / n_ev, seM=np.sqrt(len(E)) / n_ev,
        M_sig=sig / n_ev, seMs=np.sqrt(sig) / n_ev,
        Ekin_mean=E.mean(), Ekin_med=float(np.median(E)),
        R=R, seR=R * np.sqrt(1 / nhi + 1 / nlo),
    )
    c, _ = np.histogram(E, bins=EDGES)
    spec[ds] = (c, n_ev)
    r = res[ds]
    print(f"{ds:12s} U={U[ds]:>2}  events={n_ev:>9,}  hits={len(df):>11,}  "
          f"neutrons={len(E):>9,}  signal={sig:>8,}  M_n={r['M_n']:.4f}  "
          f"Ekin_med={r['Ekin_med']:.3f}  R={R:.5f}+-{r['seR']:.5f}", flush=True)
    del df, u, neu

if "zeroSpot" in res and len(res) >= 2:
    a = res["zeroSpot"]
    print("\n=== U_sym response vs zeroSpot (U=0) ===")
    hdr = f"{'observable':<34}{'U=18':>22}{'U=90':>22}"
    print(hdr); print("-" * len(hdr))
    for lbl, k, se in (("M_n  (all neutrons/event)", "M_n", "seM"),
                       ("M_sig (signal neutrons/event)", "M_sig", "seMs"),
                       ("R = N(E>2)/N(E<1)", "R", "seR")):
        cells = ""
        for ds in ("defaultSpot", "bigSpot"):
            if ds not in res:
                cells += f"{'—':>22}"; continue
            b = res[ds]
            d = b[k] - a[k]
            s = float(np.hypot(a[se], b[se]))
            cells += f"{f'{d:+.5f} ({abs(d)/s:.1f}s)':>22}"
        print(f"{lbl:<34}{cells}")

    print("\n=== neutron yield per event, Ekin-differential ===")
    print(f"{'Ekin [GeV]':<14}{'Y(U=0)':>10}{'Y(U=18)':>10}{'Y(U=90)':>10}"
          f"{'90/0':>9}{'sigma':>8}")
    for i in range(len(EDGES) - 1):
        row = f"[{EDGES[i]:g},{EDGES[i+1]:g})"
        ys = {}
        for ds in ("zeroSpot", "defaultSpot", "bigSpot"):
            if ds in spec:
                c, n = spec[ds]; ys[ds] = (c[i] / n, int(c[i]))
        if "zeroSpot" not in ys or "bigSpot" not in ys:
            continue
        y0, n0 = ys["zeroSpot"]; y9, n9 = ys["bigSpot"]
        y18 = ys.get("defaultSpot", (np.nan,))[0]
        rat = y9 / y0 if y0 else np.nan
        se = rat * np.sqrt(1 / max(n0, 1) + 1 / max(n9, 1)) if y0 else np.nan
        print(f"{row:<14}{y0:>10.4f}{y18:>10.4f}{y9:>10.4f}{rat:>9.4f}"
              f"{abs(rat-1)/se:>8.1f}")
