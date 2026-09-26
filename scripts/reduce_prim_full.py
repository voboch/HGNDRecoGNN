"""Stream the full B-enabled primary-nucleon production and reduce it per event.

The primary CSV exports are ~8 GB compressed per S_pot sample and cannot be
staged locally, so this reads each tarball as a single sequential gzip stream
and emits two compact products per sample:

  <tag>_events.pkl       one row per event: the job file it came from, the
                         impact parameter, and integer counters for every
                         selection the feasibility analysis uses.
  <tag>_filehist.npz     per-job-file 3-D histograms in (b, Ekin, theta) for
                         each species, for spectra figures and for job-level
                         resampling.

The per-job file index is kept deliberately, because impact parameter carries
job-level block structure: within a single sample, where no symmetry-potential
difference can exist, the four-file feasibility subset scattered in <b> with
chi2/ndf = 27.2.  Any uncertainty on a b-dependent quantity must therefore
resample job files, not events.  See docs/CENTRALITY_CONVENTION.md.

Selections follow results/centrality_feasibility/cent3.py exactly.
"""
import io, json, os, sys, tarfile, time
import numpy as np
import pandas as pd

TH_LO, TH_HI = 8.9, 13.1          # HGND angular acceptance band [deg]
ELO, EHI = 1.0, 2.0               # spectral-hardness Ekin thresholds [GeV]
YS = 0.9863                       # y_cm shift (beam rapidity / 2)
USECOLS = ["Row", "PDG", "Ekin", "Pt", "Pz", "Rapid", "B"]

# Histogram axes.  They must span the full kinematic range: the exports carry
# the whole nucleon inventory, so target spectators reach theta -> 180 deg and
# Ekin -> 0.  Axes that stopped at 60 deg / 10 GeV silently dropped 30 % of rows.
B_EDGES  = np.arange(0.0, 15.5, 0.5)                              # 30 bins
E_EDGES  = np.concatenate([[0.0], np.logspace(-4, 1.4, 54), [1e4]])   # 55 bins
TH_EDGES = np.concatenate([np.arange(0.0, 20.0, 0.5),
                           np.arange(20.0, 190.0, 10.0)])         # 56 bins
HSHAPE   = (2, len(B_EDGES) - 1, len(E_EDGES) - 1, len(TH_EDGES) - 1)
HSHAPE_F = (2, len(E_EDGES) - 1, len(TH_EDGES) - 1)   # per-file: no b axis

COUNTERS = [
    "band_n", "band_p", "band_n_hi", "band_n_lo", "band_p_hi", "band_p_lo",
    "mid_n", "mid_p", "mid_n_hi", "mid_n_lo", "mid_p_hi", "mid_p_lo",
    "all_n", "all_p", "all_n_hi", "all_n_lo", "all_p_hi", "all_p_lo",
]


def reduce_member(fobj, file_idx):
    """Per-event counters and one 3-D histogram for a single job CSV."""
    d = pd.read_csv(fobj, usecols=USECOLS, engine="c")
    row = d.Row.to_numpy()
    # events are contiguous and ascending within a job file
    uniq, start, cnt = np.unique(row, return_index=True, return_counts=True)
    if not np.all(np.diff(start) > 0):
        order = np.argsort(start)
        uniq, start, cnt = uniq[order], start[order], cnt[order]

    pt, pz = d.Pt.to_numpy(), d.Pz.to_numpy()
    th = np.degrees(np.arctan2(pt, pz))
    ycm = d.Rapid.to_numpy() - YS
    ek = d.Ekin.to_numpy()
    pdg = d.PDG.to_numpy()

    isn, isp = pdg == 2112, pdg == 2212
    inband = (th >= TH_LO) & (th < TH_HI)
    mid = np.abs(ycm) < 0.5
    hi, lo = ek >= EHI, ek < ELO

    masks = {
        "band_n": inband & isn, "band_p": inband & isp,
        "band_n_hi": inband & isn & hi, "band_n_lo": inband & isn & lo,
        "band_p_hi": inband & isp & hi, "band_p_lo": inband & isp & lo,
        "mid_n": mid & isn, "mid_p": mid & isp,
        "mid_n_hi": mid & isn & hi, "mid_n_lo": mid & isn & lo,
        "mid_p_hi": mid & isp & hi, "mid_p_lo": mid & isp & lo,
        "all_n": isn, "all_p": isp,
        "all_n_hi": isn & hi, "all_n_lo": isn & lo,
        "all_p_hi": isp & hi, "all_p_lo": isp & lo,
    }
    out = {
        "file_idx": np.full(len(uniq), file_idx, dtype=np.int16),
        "Row": uniq.astype(np.int32),
        "B": d.B.to_numpy()[start].astype(np.float32),
        "nprim": cnt.astype(np.int16),
    }
    for name in COUNTERS:
        out[name] = np.add.reduceat(masks[name].astype(np.int32), start).astype(np.int16)

    # b is an event property; broadcast it to the event's rows
    b_rows = np.repeat(d.B.to_numpy()[start], cnt)
    h = np.zeros(HSHAPE, dtype=np.int64)
    for k, m in enumerate((isn, isp)):
        if m.any():
            hk, _ = np.histogramdd(
                np.column_stack([b_rows[m], ek[m], th[m]]),
                bins=[B_EDGES, E_EDGES, TH_EDGES])
            h[k] = hk.astype(np.int64)
    lost = len(d) - int(h.sum())
    # the full (b, Ekin, theta) array is accumulated per sample; only the
    # b-integrated projection is kept per job file, so job-level resampling of
    # spectra stays affordable (23 kB rather than 2.7 MB per file)
    return pd.DataFrame(out), h, h.sum(axis=1).astype(np.int32), lost


def run(tar_path, tag, outdir):
    os.makedirs(outdir, exist_ok=True)
    ev_path = os.path.join(outdir, f"{tag}_events.pkl")
    hi_path = os.path.join(outdir, f"{tag}_filehist.npz")
    pr_path = os.path.join(outdir, f"{tag}_progress.json")

    done, frames, hists = {}, [], {}
    failed = []
    gh = np.zeros(HSHAPE, dtype=np.int64)
    lost_total = 0
    if os.path.exists(pr_path):
        done = json.load(open(pr_path))["done"]
        if os.path.exists(ev_path):
            frames.append(pd.read_pickle(ev_path))
        if os.path.exists(hi_path):
            z = np.load(hi_path)
            hists = {k: z[k] for k in z.files if not k.startswith("_")}
            if "_global" in z.files:
                gh = z["_global"]
        print(f"[{tag}] resuming, {len(done)} members already reduced", flush=True)

    total = os.path.getsize(tar_path)
    f = open(tar_path, "rb", buffering=1 << 24)
    t = tarfile.open(fileobj=f, mode="r|gz")
    t0, nev, nfile = time.time(), 0, 0

    def flush():
        if frames:
            tmp = ev_path + ".tmp"
            pd.concat(frames, ignore_index=True).to_pickle(tmp)
            os.replace(tmp, ev_path)     # atomic: a killed flush must not truncate
        if hists or gh.any():
            tmph = hi_path + ".tmp.npz"
            np.savez_compressed(tmph, _global=gh, _b_edges=B_EDGES,
                                _e_edges=E_EDGES, _th_edges=TH_EDGES, **hists)
            os.replace(tmph, hi_path)
        with open(pr_path + ".tmp", "w") as fh:
            json.dump({"done": done}, fh)
        os.replace(pr_path + ".tmp", pr_path)

    try:
        for m in t:
            if not (m.isfile() and m.name.endswith("_prim.csv")):
                continue
            key = "/".join(m.name.split("/")[-2:])
            if key in done:
                continue
            try:
                # tarfile in streaming mode returns a non-seekable proxy; the
                # pandas C parser requires seekable(), so buffer the member.
                buf = io.BytesIO(t.extractfile(m).read())
                df, h, hf, lost = reduce_member(buf, len(done))
            except Exception as e:      # a bad job must not kill the pass, but it
                                        # must stay retryable: do NOT mark it done
                print(f"[{tag}] FAILED {key}: {type(e).__name__}: {e}", flush=True)
                failed.append(key)
                continue
            frames.append(df)
            hists[key] = hf
            gh += h
            lost_total += lost
            done[key] = int(len(df))
            nev += len(df); nfile += 1
            if nfile % 10 == 0:
                pos = f.tell(); el = time.time() - t0
                rate = pos / el / 1e6
                eta = (total - pos) / (pos / el) / 60 if pos else 0
                print(f"[{tag}] {nfile:4d} files  {nev:8d} events  "
                      f"{pos/1e9:5.2f}/{total/1e9:.2f} GB  {rate:5.2f} MB/s  ETA {eta:5.1f} min",
                      flush=True)
                flush()
    except (EOFError, OSError, tarfile.TarError) as e:
        print(f"[{tag}] STREAM ENDED: {type(e).__name__}: {e}", flush=True)
    finally:
        flush()
        f.close()
    print(f"[{tag}] DONE {nfile} files this pass, {nev} events, "
          f"{len(done)} members total, {lost_total} rows outside histogram axes, "
          f"{len(failed)} failed, {time.time()-t0:.0f}s", flush=True)
    if failed:
        print(f"[{tag}] retryable failures: {failed}", flush=True)


if __name__ == "__main__":
    outdir = sys.argv[1]
    for tag in sys.argv[2:]:
        tp = f"/Users/vovvy/ncx/data/smash_xecs_2.87gev_hardSkyrme_{tag}.tar.gz"
        print(f"=== {tag}: {tp} ===", flush=True)
        run(tp, tag, outdir)
