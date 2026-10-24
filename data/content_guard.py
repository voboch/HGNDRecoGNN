"""Fail closed when an exact-content census is missing or crosses partitions."""

from __future__ import annotations
import csv
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_disjoint(index_path, partition_source_runs):
    lookup = {
        str(run): part for part, runs in partition_source_runs.items() for run in runs
    }
    if len(lookup) != sum(map(len, partition_source_runs.values())):
        raise ValueError("Source occurs in several partitions")
    contents = {}
    n = 0
    with gzip.open(index_path, "rt") as stream:
        for row in csv.DictReader(stream):
            part = lookup[row["group"]]
            previous = contents.setdefault(row["sha256"], part)
            if previous != part:
                raise ValueError(
                    "Exact event content crosses training/validation/evaluation partitions"
                )
            n += 1
    if not n:
        raise ValueError("Empty content identity index")
    return n


def validate_census(path, raw_root, prepared, partition_source_runs):
    if path is None:
        raise ValueError(
            "A complete exact-content census is required before preparing or scoring new inputs"
        )
    path, raw_root = Path(path), Path(raw_root)
    x = json.loads(path.read_text())
    if x.get("status") != "complete" or x.get("smoke") is not False:
        raise ValueError("Content census is incomplete or only a smoke")
    identity = x["identities"]["dcm_retained_hits"]
    index = path.parent / "dcm_retained_hits.csv.gz"
    if digest(index) != identity["index_sha256"]:
        raise ValueError("Content census identity index changed")
    # Bind the result to every usable file of the frozen prepared population.
    with (Path(prepared) / "source_row_offsets.csv").open() as stream:
        expected = {
            str(raw_root / r["run"] / (r["stem"] + "_hits.csv"))
            for r in csv.DictReader(stream)
        }
    files = [v for v in x["files"] if v["kind"] == "dcm_hits"]
    if {v["path"] for v in files} != expected:
        raise ValueError("Content census does not cover the frozen prepared population")
    for item in files:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Raw DCM changed since its content census")
    if assert_disjoint(index, partition_source_runs) != identity["events"]:
        raise ValueError("Content census event count mismatch")
    return {
        "events": identity["events"],
        "unique_contents": identity["unique_contents"],
        "census_sha256": digest(path),
        "scope": "Exact retained-hit identity; generator collision lineage remains unverified",
    }
