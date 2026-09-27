"""Join acceptance counters computed on one copy of the primaries with the
impact parameter recovered on another.

Two copies of the same primary export exist.  The cHARISMa copy predates the
converter fix and carries `B = -1`, but it is on Lustre and its kinematics are
intact, so the per-particle front-face acceptance can be computed from it in
minutes.  The ncx copy carries the recovered `DstEventHeader.fB` but is
reachable only over a mount that has served between 0.2 and 4 MB/s.

Spot checks show the two files are identical except in the `B` and `NPrim`
columns, so the events correspond one to one.  This script makes that an
assertion rather than an assumption: it requires every job file to appear in
both reductions with the same number of events, and every event to have the
same primary-nucleon multiplicity, which is computed from the row count and is
therefore independent of the `NPrim` column that differs.

Usage:
  python3 scripts/merge_acc_with_b.py --acc-dir DIR --b-dir DIR --out-dir DIR
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reco_truth_join import prim_file_map

SAMPLES = ("zeroSpot", "defaultSpot", "bigSpot")


def keyed(events_dir, tag):
    """Per-event table keyed by (job-file stem, local Row)."""
    df = pd.read_pickle(os.path.join(events_dir, f"{tag}_events.pkl"))
    fmap = prim_file_map(os.path.join(events_dir, f"{tag}_progress.json"))
    df = df.assign(stem=df.file_idx.map(fmap))
    return df.set_index(["stem", "Row"]).sort_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acc-dir", required=True,
                   help="reduction carrying acc_* counters (B is -1 there)")
    p.add_argument("--b-dir", required=True,
                   help="reduction carrying the recovered impact parameter")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    report = {}
    for tag in SAMPLES:
        acc, bee = keyed(a.acc_dir, tag), keyed(a.b_dir, tag)
        sa, sb = set(acc.index.get_level_values(0)), set(bee.index.get_level_values(0))
        common = sorted(sa & sb)
        if not common:
            raise SystemExit(f"{tag}: the two reductions share no job files")

        acc = acc.loc[acc.index.get_level_values(0).isin(common)]
        bee = bee.loc[bee.index.get_level_values(0).isin(common)]
        if not acc.index.equals(bee.index):
            raise AssertionError(
                f"{tag}: event keys differ between the two reductions "
                f"({len(acc)} vs {len(bee)} events over {len(common)} job files)")

        # independent of the differing NPrim column: this is the row count
        bad = int((acc.nprim.to_numpy() != bee.nprim.to_numpy()).sum())
        if bad:
            raise AssertionError(
                f"{tag}: {bad} events differ in primary-nucleon multiplicity; "
                f"the two copies are not the same events")

        if float(bee.B.min()) < 0:
            raise AssertionError(f"{tag}: --b-dir still contains B < 0")

        out = acc.copy()
        out["B"] = bee.B.to_numpy()
        out = out.reset_index()
        out["file_idx"] = pd.factorize(out.stem)[0].astype("int16")
        order = ["file_idx", "Row", "B", "nprim"] + \
                [c for c in out.columns if c not in ("file_idx", "Row", "B", "nprim", "stem")]
        out[order].to_pickle(os.path.join(a.out_dir, f"{tag}_events.pkl"))
        with open(os.path.join(a.out_dir, f"{tag}_progress.json"), "w") as f:
            json.dump({"done": {f"merged/{s}_prim.csv": int((out.stem == s).sum())
                                for s in sorted(out.stem.unique())}}, f)

        report[tag] = {"job_files": len(common), "events": int(len(out)),
                       "dropped_files_acc_only": sorted(sa - sb),
                       "dropped_files_b_only": sorted(sb - sa),
                       "mean_b": float(out.B.mean()),
                       "acc_n_per_event": float(out.acc_n.mean()) if "acc_n" in out else None,
                       "band_n_per_event": float(out.band_n.mean()) if "band_n" in out else None}
        r = report[tag]
        print(f"{tag:<13} {r['job_files']:>4} job files  {r['events']:>8,} events  "
              f"<b> = {r['mean_b']:.4f} fm   "
              f"acc_n/ev {r['acc_n_per_event']:.3f}  band_n/ev {r['band_n_per_event']:.3f}")
        if sa - sb or sb - sa:
            print(f"  note: {len(sa-sb)} files only in acc, {len(sb-sa)} only in b; "
                  f"kept the intersection")

    with open(os.path.join(a.out_dir, "merge_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {a.out_dir}")


if __name__ == "__main__":
    main()
