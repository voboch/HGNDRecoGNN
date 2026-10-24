"""Cross-source content copies must block the real frozen-input preparation."""

import gzip
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytest
from HGNDRecoGNN.data.content_guard import assert_disjoint, validate_census


def test_content_overlap_blocks_even_distinct_source_directory_ids(tmp_path):
    p = tmp_path / "index.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write("group,file,Row,records,sha256\nA,f1,1,3,same\nB,f2,9,3,same\n")
    with pytest.raises(ValueError, match="crosses"):
        assert_disjoint(p, {"train": ["A"], "test": ["B"]})


def test_within_partition_copies_are_counted_and_missing_receipt_fails(tmp_path):
    p = tmp_path / "index.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write(
            "group,file,Row,records,sha256\nA,f1,1,3,same\nA,f2,9,3,same\nB,f3,1,3,different\n"
        )
    assert assert_disjoint(p, {"train": ["A"], "test": ["B"]}) == 3
    with pytest.raises(ValueError, match="census is required"):
        validate_census(None, tmp_path, tmp_path, {})


def test_frozen_inputs_really_calls_content_gate(monkeypatch, tmp_path):
    from argparse import Namespace
    from HGNDRecoGNN.scripts import student_pipeline as pipeline

    expected = {"train": ["988975"], "validation": ["988974"], "test": ["988976"]}
    monkeypatch.setattr(
        pipeline,
        "digest",
        lambda p: (
            pipeline.SPLIT_HASH if p.name == "manifest.json" else pipeline.FROZEN_HASH
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_prepared",
        lambda p: ({"partition_source_runs": expected, "raw_root": str(tmp_path)}, {}),
    )
    with pytest.raises(ValueError, match="census is required"):
        pipeline.frozen_inputs(
            Namespace(prepared=tmp_path, frozen_data=tmp_path / "frozen.json")
        )
