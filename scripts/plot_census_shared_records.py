"""What the shared-record threshold of the SMASH export census decides.

`smash_export_census.MIN_SHARED_RECORDS` links two contents when one event of
each shares at least that many hashed direct-primary records. This figure
scans the threshold from 1 to 6 on the full census. The upper panel shows
the event pairs that would link contents. The lower panel shows the content
pairs those events link, split into pairs the final grouping already joins
and pairs that would merge two different groups. If the second count is
zero across the scan, the threshold has no effect on the groups, the split
or the training sets.

Usage: python scripts/plot_census_shared_records.py --census <dir with census.json> --output <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.plot_b_full import (SINGLE, apply_style, assert_contained, hep_label,  # noqa: E402
                                             mark_provenance, save)
from HGNDRecoGNN.scripts.smash_export_census import MIN_SHARED_RECORDS, TAGS  # noqa: E402

THRESHOLDS = range(1, 7)


def scan(census_dir):
    census = json.loads((census_dir / "census.json").read_text())
    rows = []
    for tag in TAGS:
        s = census["tags"][tag]
        group = {cid: g["id"] for g in s["groups"] for cid in g["contents"]}
        records = pd.read_csv(census_dir / s["indexes"]["records"]["path"]).rename(columns={"sha256": "h"})
        pairs = records.merge(records, on="h", suffixes=("_a", "_b"))
        pairs = pairs[pairs.content_a < pairs.content_b]
        per_event = (pairs.drop_duplicates(["content_a", "Row_a", "Id_a", "content_b", "Row_b"])
                     .groupby(["content_a", "Row_a", "content_b", "Row_b"]).size().rename("n").reset_index())
        for t in THRESHOLDS:
            linked = per_event[per_event.n >= t][["content_a", "content_b"]].drop_duplicates()
            same = sum(group[a] == group[b] for a, b in linked.itertuples(index=False))
            rows.append({"tag": tag, "threshold": t, "event_pairs": int((per_event.n >= t).sum()),
                         "content_pairs": len(linked), "same_group": same, "cross_group": len(linked) - same})
    return pd.DataFrame(rows), census


def plot(table, out_dir, stem="fig_census_shared_records"):
    apply_style()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(SINGLE, 3.6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    tags = [t for t in TAGS if table[table.tag == t].event_pairs.sum()]
    silent = [t for t in TAGS if t not in tags]
    for tag, color in zip(tags, ("#08519c", "#b5651d", "#4d9221")):
        part = table[table.tag == tag]
        top.plot(part.threshold, part.event_pairs, "o-", color=color, ms=3, label=tag)
        bottom.plot(part.threshold, part.same_group, "o-", color=color, ms=3,
                    label=f"{tag}: already one group")
        bottom.plot(part.threshold, part.cross_group, "s--", color=color, mfc="white", ms=3,
                    label=f"{tag}: would merge groups")
    for ax in (top, bottom):
        ax.axvline(MIN_SHARED_RECORDS, color="#69727C", ls="--", lw=0.6)
    top.set_yscale("log")
    top.set_ylim(1, table.event_pairs.max() * 40)
    top.set_ylabel("event pairs")
    hep_label(top, extra="no shared records: " + ", ".join(silent) if silent else None,
              loc="upper right", fontsize=6.5)
    top.legend(loc="lower left", fontsize=7)
    bottom.set_ylim(-0.8, table.content_pairs.max() * 1.9)
    bottom.set_ylabel("content pairs")
    bottom.text(MIN_SHARED_RECORDS + 0.08, 0.3, "adopted", transform=bottom.get_xaxis_transform(),
                fontsize=7, color="#69727C")
    bottom.set_xlabel("threshold: shared primary records per event pair")
    bottom.set_xticks(list(THRESHOLDS))
    bottom.legend(loc="upper right", fontsize=6.5)
    mark_provenance(top, "truth", loc="upper left")
    assert_contained(top, stem)
    assert_contained(bottom, stem)
    save(fig, out_dir, stem)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--census", type=Path, required=True, help="directory holding census.json and its indexes")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    table, census = scan(args.census)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(table, args.output)
    result = {"census_sha256_of": "census.json", "min_shared_records": MIN_SHARED_RECORDS,
              "rows": table.to_dict(orient="records"),
              "cross_group_at_any_threshold": int(table.cross_group.sum())}
    (args.output / "fig_census_shared_records.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"cross_group_at_any_threshold": result["cross_group_at_any_threshold"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
