"""Join GNN reconstruction output back to per-event Monte-Carlo truth.

The graph loader renumbers `Row` into a global space, because `Row` is only
unique within one CSV.  It walks run directories and file stems in sorted order
and gives each file a block:

    span = max(hits.Row.max(), vacs.Row.max()) + 1
    Row += row_base ;  row_base += span

Predictions therefore carry global Row, while the reduced primary-nucleon
tables are keyed by (job file, local Row).  This module rebuilds the same block
layout -- reading only the Row column -- so the two can be joined.

It is deliberately a re-derivation rather than a saved side-product: it reads
the same files the loader read, so a mismatch in file set or ordering shows up
as a failed assertion rather than as a silently wrong join.
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd


def _max_row(path):
    r = pd.read_csv(path, sep=",", usecols=[0], skiprows=[0], names=["Row"],
                    engine="c").Row
    return int(r.max()) if len(r) else -1


def row_offsets(csv_dir, runs=None):
    """Rebuild the loader's global Row blocks. Returns one row per file pair."""
    if runs is None:
        runs = sorted(d for d in os.listdir(csv_dir)
                      if os.path.isdir(os.path.join(csv_dir, d)))
    out, row_base = [], 0
    for run in runs:
        rd = os.path.join(csv_dir, run)
        stems = sorted({f.rsplit("_", 1)[0] for f in os.listdir(rd)
                        if f.endswith("hits.csv") or f.endswith("vacs.csv")})
        for stem in stems:
            h = os.path.join(rd, f"{stem}_hits.csv")
            v = os.path.join(rd, f"{stem}_vacs.csv")
            if not (os.path.exists(h) and os.path.exists(v)):
                continue
            if os.stat(h).st_size <= 100 or os.stat(v).st_size <= 100:
                continue
            mh, mv = _max_row(h), _max_row(v)
            if mh < 0 or mv < 0:
                continue
            span = max(mh, mv) + 1
            out.append({"run": run, "stem": stem, "row_base": row_base,
                        "span": span, "row_end": row_base + span})
            row_base += span
    if not out:
        raise RuntimeError(f"no usable hits/vacs pairs under {csv_dir}")
    df = pd.DataFrame(out)
    df.attrs["global_row_space"] = row_base
    return df


def to_local(global_rows, offsets):
    """Map global Row values to (run, stem, local Row).

    The run is carried because file stems repeat across samples: a split tree
    holding zeroSpot/0020 and bigSpot/0020 would otherwise collapse them.
    """
    g = np.asarray(global_rows, dtype=np.int64)
    edges = offsets.row_base.to_numpy()
    idx = np.searchsorted(edges, g, side="right") - 1
    bad = (idx < 0) | (g >= offsets.row_end.to_numpy()[np.clip(idx, 0, len(edges) - 1)])
    idx = np.clip(idx, 0, len(edges) - 1)
    stem = offsets.stem.to_numpy()[idx]
    run = offsets.run.to_numpy()[idx]
    local = g - edges[idx]
    return pd.DataFrame({"run": np.where(bad, None, run),
                         "stem": np.where(bad, None, stem),
                         "local_row": np.where(bad, -1, local),
                         "unmapped": bad})


def prim_file_map(progress_json):
    """file_idx -> job-file stem, from the reduction's insertion order."""
    done = json.load(open(progress_json))["done"]
    return {i: k.split("/")[-1].replace("_prim.csv", "")
            for i, k in enumerate(done)}


def attach_truth(pred_df, csv_dir, events_pkl, progress_json, row_col="Row"):
    """Add (stem, local_row) to a prediction frame and join the truth counters.

    Returns (joined_predictions, truth_events) where truth_events is the subset
    of the reduced per-event table covering exactly the job files present here.
    """
    offs = row_offsets(csv_dir)
    loc = to_local(pred_df[row_col].to_numpy(), offs)
    pred = pred_df.copy()
    pred["stem"] = loc.stem.to_numpy()
    pred["local_row"] = loc.local_row.to_numpy()
    if loc.unmapped.any():
        raise AssertionError(
            f"{int(loc.unmapped.sum())} predicted rows fall outside the "
            f"rebuilt global Row space — file set or ordering differs from "
            f"the one the dataset was built from")

    ev = pd.read_pickle(events_pkl)
    fmap = prim_file_map(progress_json)
    ev = ev.copy()
    ev["stem"] = ev.file_idx.map(fmap)
    ev = ev[ev.stem.isin(set(offs.stem))]
    ev = ev.rename(columns={"Row": "local_row"})
    return pred, ev, offs
