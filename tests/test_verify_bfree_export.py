"""Incomplete local downloads must fail before extraction or certification."""

from argparse import Namespace
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.verify_bfree_export import verify
from HGNDRecoGNN.scripts.content_census import sha256


def fixture(tmp):
    source = tmp / "source"
    source.mkdir()
    for name, value in [
        ("campaign.json", {}),
        ("primary_manifest.json", {"files": []}),
    ]:
        (source / name).write_text(json.dumps(value))
    (source / "SHA256SUMS").write_text(
        "".join(
            sha256(source / n) + "  " + n + "\n"
            for n in ["campaign.json", "primary_manifest.json"]
        )
    )
    archive = tmp / "partial.tar.gz"
    archive.write_bytes(b"not-yet-downloaded")
    slurm = tmp / "slurm.json"
    slurm.write_text(
        json.dumps(dict(state="COMPLETED", exit_code="0:0", job_key="fixture"))
    )
    return Namespace(
        source=source,
        slurm=slurm,
        archive=archive,
        extracted=tmp / "extract",
        receipt=tmp / "receipt.json",
        remote_archive="fixture",
        full_receipts=None,
        expected_archive_sha256="0" * 64,
    )


def test_wrong_download_hash_cannot_create_receipt_or_extraction(tmp_path):
    args = fixture(tmp_path)
    with pytest.raises(ValueError, match="completed job hash"):
        verify(args)
    assert not args.extracted.exists() and not args.receipt.exists()


def test_failed_slurm_cannot_start_extraction(tmp_path):
    args = fixture(tmp_path)
    args.slurm.write_text(json.dumps(dict(state="FAILED", exit_code="1:0")))
    with pytest.raises(ValueError, match="terminal successful"):
        verify(args)
    assert not args.extracted.exists() and not args.receipt.exists()
