"""Group-level six-part split and three-fold cross-fitting for SMASH training.

Groups come from the exact content census (scripts/smash_export_census.py).
Per potential, groups are ranked by sha256(SALT + tag + ":" + hits_sha +
vacs_sha) of the kept content and assigned to parts P0..P5 by rank mod 6.
Fold k: test = truth-bound groups of P_{2k} and P_{2k+1}; validation =
P_{(2k+2) mod 6}; training = the other three parts. Unbound groups that land
in a test part are unused in that fold. Every truth-bound group is therefore
tested exactly once, by models that never saw it.

Why six parts and three folds: fewer folds leave truth-bound files that no
held-out model scores (zeroSpot has no unbound content to train on); more
folds shrink each fold's training set below half. Two sixths of the bound
files per fold test, one sixth validates the cut and calibration, one half
trains. `plot_projected_sd` shows what the test fraction buys in paired SD.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.data.content_guard import assert_disjoint  # noqa: E402
from HGNDRecoGNN.scripts.content_census import sha256  # noqa: E402

# Arbitrary, fixed before any split was drawn; changing it redraws the parts.
SALT = "smash-train-split-v1:"
SMOKE_SALT = "smash-train-smoke-v1:"
PARTS, FOLDS = 6, 3
TAGS = ("zeroSpot", "defaultSpot", "bigSpot")
PARTITIONS = ("train", "val", "test", "unused")
# Paired SD of C4 - T1 (percentage points) of the frozen DCM transfer on every
# pinned truth-bound file, 195 zeroSpot / 51 bigSpot: 1.5 / 1.7 / 1.6 for
# seeds 42 / 123 / 456 (reports/reco_rn_retrospective_results_20261026.md,
# 400 whole-file replicates). The projection only rescales this with file
# counts, sd ~ sqrt(1/n0 + 1/n90): it assumes exchangeable whole files and a
# per-file variance that neither the split nor the SMASH-trained arm changes.
ANCHOR_SD = (1.5, 1.7)
ANCHOR_FILES = (195, 51)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rank_digest(tag, hits_sha, vacs_sha, salt=SALT):
    return hashlib.sha256((salt + tag + ":" + hits_sha + vacs_sha).encode()).hexdigest()


def load_census(path, *, allow_smoke=False):
    path = Path(path)
    census = json.loads(path.read_text())
    if census.get("status") != "complete":
        raise ValueError("Content census is not complete")
    if census.get("coverage") != "full" and not allow_smoke:
        raise ValueError("Content census covers a subset only")
    if set(census["tags"]) != set(TAGS) and not allow_smoke:
        raise ValueError("Content census lacks a potential")
    for tag, info in census["tags"].items():
        for item in info["indexes"].values():
            if sha256(path.parent / item["path"]) != item["sha256"]:
                raise ValueError(f"Census index changed: {item['path']}")
    return census


def assign_parts(census):
    """Rank each potential's groups by the salted kept-content digest."""
    groups = []
    for tag in TAGS:
        if tag not in census["tags"]:
            continue
        info = census["tags"][tag]
        ranked = []
        for g in info["groups"]:
            kept = info["contents"][g["kept"]]
            ranked.append({"id": g["id"], "tag": tag, "kept": g["kept"],
                           "contents": g["contents"],
                           "hits_sha256": kept["hits_sha256"], "vacs_sha256": kept["vacs_sha256"],
                           "rank_digest": rank_digest(tag, kept["hits_sha256"], kept["vacs_sha256"]),
                           "truth_bound": bool(g["truth_bound"]), "primary": g["primary"],
                           "events": g["events"],
                           # The audit jitter is seeded by this relative path.
                           "hits_path": g["kept"] + "_hits.csv", "vacs_path": g["kept"] + "_vacs.csv"})
        ranked.sort(key=lambda g: g["rank_digest"])
        for i, g in enumerate(ranked):
            g["rank"], g["part"] = i, i % PARTS
        groups.extend(ranked)
    ids = [g["id"] for g in groups]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate group identifiers")
    primaries = [(g["tag"], g["primary"]) for g in groups if g["truth_bound"]]
    if len(primaries) != len(set(primaries)):
        raise ValueError("A primary file binds several groups")
    return groups


def fold_partitions(groups, k):
    test_parts = {2 * k, 2 * k + 1}
    val_part = (2 * k + 2) % PARTS
    out = {p: [] for p in PARTITIONS}
    for g in groups:
        if g["part"] in test_parts:
            out["test" if g["truth_bound"] else "unused"].append(g["id"])
        elif g["part"] == val_part:
            out["val"].append(g["id"])
        else:
            out["train"].append(g["id"])
    return {"test_parts": sorted(test_parts), "val_part": val_part,
            "train_parts": sorted(set(range(PARTS)) - test_parts - {val_part}),
            "partitions": {p: sorted(v) for p, v in out.items()}}


def write_guard_index(census, census_dir, groups, path):
    """Re-key the census event index from (tag, content) to group id."""
    owner = {(g["tag"], c): g["id"] for g in groups for c in g["contents"]}
    rows = 0
    with gzip.open(path, "wt") as out:
        out.write("group,file,Row,records,sha256\n")
        for tag in TAGS:
            if tag not in census["tags"]:
                continue
            index = Path(census_dir) / census["tags"][tag]["indexes"]["events"]["path"]
            with gzip.open(index, "rt") as stream:
                for row in csv.DictReader(stream):
                    if row["group"] != tag:
                        raise ValueError("Census index row from another potential")
                    out.write(",".join((owner[(tag, row["file"])], f"{tag}:{row['file']}", row["Row"],
                                        row["records"], row["sha256"])) + "\n")
                    rows += 1
    return rows


def build_manifest(census_path, output, *, allow_smoke=False):
    census_path, output = Path(census_path), Path(output)
    census = load_census(census_path, allow_smoke=allow_smoke)
    groups = assign_parts(census)
    output.mkdir(parents=True, exist_ok=True)
    guard = output / "guard_index.csv.gz"
    rows = write_guard_index(census, census_path.parent, groups, guard)
    expected = sum(t["indexes"]["events"]["rows"] for t in census["tags"].values())
    if rows != expected:
        raise ValueError("Guard index does not cover every census event")
    by_id = {g["id"]: g for g in groups}
    folds, tested = {}, {}
    for k in range(FOLDS):
        fold = fold_partitions(groups, k)
        parts = fold["partitions"]
        assigned = [i for p in PARTITIONS for i in parts[p]]
        if sorted(assigned) != sorted(by_id):
            raise ValueError(f"Fold {k} does not place every group exactly once")
        if any(not by_id[i]["truth_bound"] for i in parts["test"]):
            raise ValueError("Unbound group in a test partition")
        # Exact raw event content may not cross partitions (event level).
        checked = assert_disjoint(guard, parts)
        if checked != rows:
            raise ValueError("Guard checked a different number of events")
        for i in parts["test"]:
            tested[i] = tested.get(i, 0) + 1
        fold["counts"] = {p: {tag: {"groups": sum(by_id[i]["tag"] == tag for i in ids),
                                    "truth_bound": sum(by_id[i]["tag"] == tag and by_id[i]["truth_bound"]
                                                       for i in ids),
                                    "events": sum(by_id[i]["events"] for i in ids if by_id[i]["tag"] == tag)}
                              for tag in census["tags"]} for p, ids in parts.items()}
        fold["guard_events_checked"] = checked
        folds[str(k)] = fold
    bound = {g["id"] for g in groups if g["truth_bound"]}
    if set(tested) != bound or any(v != 1 for v in tested.values()):
        raise ValueError("A truth-bound group is not tested exactly once")
    manifest = {"schema": "smash-split-v1", "salt": SALT, "parts": PARTS, "folds_count": FOLDS,
                "rule": "per potential: rank by sha256(salt+tag+':'+hits_sha+vacs_sha) of the kept "
                        "content, part = rank mod 6; fold k: test = truth-bound groups of P_2k and "
                        "P_2k+1, val = P_(2k+2) mod 6, train = the other three parts",
                "census_sha256": sha256(census_path), "census_coverage": census["coverage"],
                "guard_index": {"path": guard.name, "sha256": sha256(guard), "rows": rows},
                "raw_roots": {t: census["tags"][t]["raw_root"] for t in census["tags"]},
                "groups": groups, "folds": folds,
                "truth_bound_groups": {t: sum(g["truth_bound"] for g in groups if g["tag"] == t)
                                       for t in census["tags"]}}
    write_json(output / "split_manifest.json", manifest)
    return manifest


def smoke_split(inventory_path, output):
    """Six pinned bound files, two per potential, for the GPU smoke only.

    Not a fold: the smoke never selects or evaluates anything that is reported.
    """
    inventory_path = Path(inventory_path)
    inventory = json.loads(inventory_path.read_text())
    out = {"schema": "smash-smoke-split-v1", "salt": SMOKE_SALT,
           "inventory_sha256": sha256(inventory_path),
           "scope": "GPU smoke only; pinned truth-bound files with saved audit tensors and "
                    "transfer CSVs; no reported number may come from these assignments",
           "train": [], "val": []}
    for tag in TAGS:
        info = inventory["tags"][tag]
        bound = [n for n in info["common_primary_pair_names"]
                 if info["primary_member_map"].get(n) is not None and n not in info["duplicate_exports"]]
        ranked = sorted(bound, key=lambda n: rank_digest(tag, info["input_sha256"][n]["hits"],
                                                         info["input_sha256"][n]["vacs"], SMOKE_SALT))
        for name, part in zip(ranked[:2], ("train", "val")):
            primary = info["primary_member_map"][name]
            out[part].append({"tag": tag, "member": name, "raw_root": info["raw_root"],
                              "hits_path": name + "_hits.csv", "vacs_path": name + "_vacs.csv",
                              "hits_sha256": info["input_sha256"][name]["hits"],
                              "vacs_sha256": info["input_sha256"][name]["vacs"],
                              "primary": primary, "primary_sha256": info["primary_sha256"][primary]})
    write_json(output, out)
    return out


def projected_sd(n0, n90, anchor_sd, anchor_files=ANCHOR_FILES):
    """Paired SD scaled by whole-file counts from a measured anchor."""
    scale = np.sqrt(1 / np.asarray(n0, float) + 1 / np.asarray(n90, float))
    return anchor_sd * scale / np.sqrt(1 / anchor_files[0] + 1 / anchor_files[1])


def plot_projected_sd(out_dir, bound=None, stem="fig_split_projected_sd"):
    """Projected paired SD of C4 - T1 against the per-fold test fraction.

    Scenarios differ only in how many bigSpot files are truth-bound: the 51
    pinned before the census, the census count when known, and 173 if every
    bigSpot primary binds. zeroSpot stays at 195. The band spans the anchor.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from HGNDRecoGNN.scripts.plot_b_full import (SINGLE, apply_style, assert_contained, hep_label,
                                                 mark_provenance, save)

    bound = dict(bound or {})
    n0 = bound.get("zeroSpot", ANCHOR_FILES[0])
    scenarios = [(ANCHOR_FILES[1], "51 pinned (no new binding)")]
    if bound.get("bigSpot") not in (None, ANCHOR_FILES[1], 173):
        scenarios.append((bound["bigSpot"], f"{bound['bigSpot']} after census binding"))
    scenarios.append((173, "173 (every primary bound)"))
    styles = [("#9ecae1", ":"), ("#4292c6", "--"), ("#08519c", "-")][-len(scenarios):]
    apply_style()
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    f = np.linspace(1 / PARTS, 1, 200)
    values, rows = [], []
    for (n90, label), (color, ls) in zip(scenarios, styles):
        lo = projected_sd(f * n0, f * n90, ANCHOR_SD[0])
        hi = projected_sd(f * n0, f * n90, ANCHOR_SD[1])
        ax.fill_between(f, lo, hi, color=color, alpha=0.35, lw=0)
        ax.plot(f, (lo + hi) / 2, color=color, ls=ls, label=rf"$N_{{90}}$ = {label}")
        values.extend([lo, hi])
        for frac in (1 / 3, 1.0):
            rows.append({"n90_bound": n90, "test_fraction": frac,
                         "sd_low": float(projected_sd(frac * n0, frac * n90, ANCHOR_SD[0])),
                         "sd_high": float(projected_sd(frac * n0, frac * n90, ANCHOR_SD[1]))})
    ax.axvline(1 / 3, color="#69727C", ls="--", lw=0.6)
    ax.text(1 / 3 - 0.012, 0.97, "one fold:\n2/6 test", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=7, color="#69727C")
    ax.text(0.985, 0.03, "pooled out-of-fold\nat 1.0", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#69727C")
    top = float(np.max(values))
    ax.set_xlim(1 / PARTS, 1.0)
    ax.set_ylim(0, top * 1.55)  # headroom for the in-frame labels
    ax.set_xlabel("test fraction of truth-bound files per fold")
    ax.set_ylabel(r"projected SD of C4$-$T1 (points)")
    hep_label(ax, extra=r"$U_{\mathrm{sym}}$ = 0 vs 90 MeV, $N_0$ = %d" % n0, loc="upper right",
              fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.74), fontsize=7, handlelength=2.4)
    mark_provenance(ax, "mixed", loc="lower left")
    assert_contained(ax, stem)
    save(fig, Path(out_dir), stem)
    table = {"anchor_sd": ANCHOR_SD, "anchor_files": ANCHOR_FILES, "n0_bound": n0, "rows": rows,
             "model": "sd = anchor * sqrt(1/n0 + 1/n90) / sqrt(1/195 + 1/51), n = fraction * bound"}
    write_json(Path(out_dir) / f"{stem}.json", table)
    return table


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split", help="six-part manifest and per-fold guard from a full census")
    s.add_argument("--census", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--allow-smoke", action="store_true", help="tests and smoke censuses only")
    m = sub.add_parser("smoke", help="six pinned bound files for the GPU smoke")
    m.add_argument("--inventory", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    f = sub.add_parser("plot-sd", help="projected paired SD against the test fraction")
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--census", type=Path, help="take truth-bound counts from a full census")
    args = p.parse_args(argv)
    if args.command == "split":
        manifest = build_manifest(args.census, args.output, allow_smoke=args.allow_smoke)
        print(json.dumps({k: manifest["folds"][k]["counts"] for k in manifest["folds"]}, indent=1))
    elif args.command == "smoke":
        print(json.dumps(smoke_split(args.inventory, args.output), indent=1))
    else:
        bound = None
        if args.census:
            census = load_census(args.census)
            bound = {t: v["counts"]["truth_bound_groups"] for t, v in census["tags"].items()}
        args.output.mkdir(parents=True, exist_ok=True)
        print(json.dumps(plot_projected_sd(args.output, bound), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
