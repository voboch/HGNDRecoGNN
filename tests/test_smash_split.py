"""Six-part rule: determinism, fold disjointness, exactly-once testing, guard."""

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.content_census import sha256  # noqa: E402
from HGNDRecoGNN.scripts.smash_split import (  # noqa: E402
    FOLDS,
    TAGS,
    assign_parts,
    build_manifest,
    fold_partitions,
    plot_projected_sd,
    projected_sd,
    rank_digest,
    smoke_split,
)


def digest(*parts):
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def write_census(root, groups_per_tag=(14, 13, 11), shared=None):
    """A compact census with one or two contents per group and random events."""
    root.mkdir(parents=True, exist_ok=True)
    census = {"status": "complete", "coverage": "full", "tags": {}}
    rng = np.random.default_rng(3)
    for tag, count in zip(TAGS, groups_per_tag):
        contents, groups, rows = {}, [], []
        for i in range(count):
            kept = f"10943245/{i + 1:04d}"
            members = [kept] + ([f"10943245/{i + 500:04d}"] if i % 4 == 0 else [])
            bound = i % 3 != 2
            for cid in members:
                contents[cid] = {"hits_sha256": digest(tag, cid, "h"), "vacs_sha256": digest(tag, cid, "v")}
                for row in range(int(rng.integers(2, 6))):
                    sig = digest(tag, kept, row)  # a dropped copy repeats kept events
                    rows.append(f"{tag},{cid},{row},3,{sig}\n")
            groups.append({"id": f"{tag}:{kept}", "kept": kept, "contents": members,
                           "truth_bound": bound, "primary": kept if bound else None, "events": 5})
        if shared and tag == shared[0]:
            rows.append(f"{tag},{shared[1]},99,1,{shared[2]}\n")
        path = root / f"{tag}_events.csv.gz"
        with gzip.open(path, "wt") as f:
            f.write("group,file,Row,records,sha256\n")
            f.writelines(rows)
        census["tags"][tag] = {"raw_root": f"/raw/{tag}", "contents": contents, "groups": groups,
                               "counts": {"truth_bound_groups": sum(g["truth_bound"] for g in groups)},
                               "indexes": {"events": {"path": path.name, "sha256": sha256(path),
                                                      "rows": len(rows)}}}
    (root / "census.json").write_text(json.dumps(census))
    return root / "census.json"


def test_split_is_deterministic_and_follows_the_rank_rule(tmp_path):
    census = write_census(tmp_path / "c")
    first = build_manifest(census, tmp_path / "a")
    second = build_manifest(census, tmp_path / "b")
    assert first == second
    assert (tmp_path / "a" / "split_manifest.json").read_bytes() == (tmp_path / "b" / "split_manifest.json").read_bytes()
    for tag in TAGS:
        groups = [g for g in first["groups"] if g["tag"] == tag]
        assert [g["rank"] for g in groups] == list(range(len(groups)))
        assert [g["rank_digest"] for g in groups] == sorted(g["rank_digest"] for g in groups)
        g = groups[0]
        expected = hashlib.sha256(
            ("smash-train-split-v1:" + tag + ":" + g["hits_sha256"] + g["vacs_sha256"]).encode()).hexdigest()
        assert g["rank_digest"] == expected == rank_digest(tag, g["hits_sha256"], g["vacs_sha256"])
        assert all(x["part"] == x["rank"] % 6 for x in groups)


def test_folds_are_disjoint_and_each_bound_group_is_tested_once(tmp_path):
    manifest = build_manifest(write_census(tmp_path / "c"), tmp_path / "m")
    by_id = {g["id"]: g for g in manifest["groups"]}
    tested = []
    for k in range(FOLDS):
        fold = manifest["folds"][str(k)]
        parts = fold["partitions"]
        ids = [i for p in parts.values() for i in p]
        assert len(ids) == len(set(ids)) == len(by_id)
        assert fold["test_parts"] == [2 * k, 2 * k + 1] and fold["val_part"] == (2 * k + 2) % 6
        assert all(by_id[i]["truth_bound"] and by_id[i]["part"] in fold["test_parts"] for i in parts["test"])
        assert all(not by_id[i]["truth_bound"] and by_id[i]["part"] in fold["test_parts"] for i in parts["unused"])
        assert all(by_id[i]["part"] == fold["val_part"] for i in parts["val"])
        assert all(by_id[i]["part"] in fold["train_parts"] for i in parts["train"])
        assert fold["guard_events_checked"] == manifest["guard_index"]["rows"]
        tested += parts["test"]
    bound = sorted(i for i, g in by_id.items() if g["truth_bound"])
    assert sorted(tested) == bound and len(tested) == len(set(tested))
    # Each part is test in exactly one fold and validation in at most one.
    roles = {p: [] for p in range(6)}
    for k in range(FOLDS):
        f = fold_partitions(manifest["groups"], k)
        for p in f["test_parts"]:
            roles[p].append("test")
        roles[f["val_part"]].append("val")
        for p in f["train_parts"]:
            roles[p].append("train")
    assert all(len(r) == 3 and r.count("test") == 1 and r.count("val") <= 1 for r in roles.values())


def test_guard_stops_on_event_content_crossing_groups(tmp_path):
    census_path = write_census(tmp_path / "c")
    census = json.loads(census_path.read_text())
    # Plant the signature of one zeroSpot group's event into the content of a
    # group drawn into another part, so some fold separates the two.
    groups = {g["id"]: g for g in assign_parts(census)}
    source = groups["zeroSpot:10943245/0001"]
    victim = next(g for g in groups.values() if g["tag"] == "zeroSpot" and g["part"] != source["part"])
    planted = digest("zeroSpot", source["kept"], 0)
    bad = write_census(tmp_path / "bad", shared=("zeroSpot", victim["kept"], planted))
    with pytest.raises(ValueError, match="crosses"):
        build_manifest(bad, tmp_path / "m")


def test_unbound_or_changed_census_is_refused(tmp_path):
    census_path = write_census(tmp_path / "c")
    data = json.loads(census_path.read_text())
    data["coverage"] = "subset"
    census_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="subset"):
        build_manifest(census_path, tmp_path / "m")
    census_path = write_census(tmp_path / "d")
    with gzip.open(tmp_path / "d" / "bigSpot_events.csv.gz", "at") as f:
        f.write("bigSpot,10943245/0001,77,1,abc\n")
    with pytest.raises(ValueError, match="index changed"):
        build_manifest(census_path, tmp_path / "m2")


def test_smoke_split_takes_two_pinned_bound_files_per_tag(tmp_path):
    inv = {"tags": {}}
    for tag in TAGS:
        names = [f"10943245/{i:04d}" for i in range(1, 9)]
        inv["tags"][tag] = {
            "raw_root": f"/raw/{tag}", "common_primary_pair_names": names,
            "primary_member_map": {m: (None if i == 2 else m) for i, m in enumerate(names)},
            "duplicate_exports": {names[3]: names[1]},
            "input_sha256": {m: {"hits": digest(tag, m, "h"), "vacs": digest(tag, m, "v")} for m in names},
            "primary_sha256": {m: digest(tag, m, "p") for m in names}}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inv))
    a = smoke_split(path, tmp_path / "smoke_split.json")
    b = smoke_split(path, tmp_path / "again.json")
    assert a == b and len(a["train"]) == len(a["val"]) == 3
    chosen = a["train"] + a["val"]
    assert sorted(x["tag"] for x in chosen) == sorted(TAGS * 2)
    for x in chosen:
        info = inv["tags"][x["tag"]]
        assert info["primary_member_map"][x["member"]] is not None
        assert x["member"] not in info["duplicate_exports"]
        assert x["hits_sha256"] == info["input_sha256"][x["member"]]["hits"]


def test_projected_sd_matches_the_anchor_and_plan_figures(tmp_path):
    assert projected_sd(195, 51, 1.5) == pytest.approx(1.5)
    assert projected_sd(65, 58, 1.5) == pytest.approx(1.72, abs=0.01)
    assert projected_sd(65, 58, 1.7) == pytest.approx(1.95, abs=0.01)
    assert projected_sd(65, 17, 1.5) == pytest.approx(2.60, abs=0.01)
    assert projected_sd(195, 173, 1.5) == pytest.approx(1.00, abs=0.01)
    table = plot_projected_sd(tmp_path, {"zeroSpot": 195, "bigSpot": 120})
    assert (tmp_path / "fig_split_projected_sd.pdf").stat().st_size > 0
    assert (tmp_path / "fig_split_projected_sd.png").stat().st_size > 0
    assert {r["n90_bound"] for r in table["rows"]} == {51, 120, 173}
