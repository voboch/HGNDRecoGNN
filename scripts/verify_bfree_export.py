"""Verify downloaded compute exports against Slurm and the immutable package.

Run locally after the task is terminal. This verifies artifacts and receipts;
scientific aggregation stays in its separately gated CPU job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.analysis.truth_provenance import load_certified  # noqa: E402
from HGNDRecoGNN.scripts.bfree_truth_campaign import TAGS, source_digest, write_json  # noqa: E402
from HGNDRecoGNN.scripts.content_census import sha256  # noqa: E402


def verify(args):
    slurm = json.loads(args.slurm.read_text())
    if slurm["state"] != "COMPLETED" or slurm["exit_code"] != "0:0":
        raise ValueError("Only terminal successful Slurm tasks can be certified")
    source = source_digest(args.source)
    config = json.loads((args.source / "campaign.json").read_text())
    manifest_path = args.source / "primary_manifest.json"
    files = json.loads(manifest_path.read_text())["files"]
    with tarfile.open(args.archive) as tar:
        tar.extractall(args.extracted, filter="data")
    result = args.extracted / "result"
    summary = json.loads((result / "summary.json").read_text())
    if (
        summary["job_id"] != slurm["job_key"]
        or summary["status"] != "verified"
        or summary["source_digest"] != source
        or summary["input_manifest_sha256"] != sha256(manifest_path)
    ):
        raise ValueError("Export does not bind the declared Slurm/source/input")
    mode = summary["mode"]
    expected = TAGS if mode == "smoke" else (summary["tag"],)
    estimates = {}
    for tag in expected:
        members = config["smoke_members"][tag] if mode == "smoke" else None
        events, _, proof = load_certified(
            result / tag, "acceptance", smoke_members=members
        )
        tag_files = {
            str(Path(f["path"]).parent.name + "/" + Path(f["path"]).name): f
            for f in files
            if f["tag"] == tag
        }
        selected = set(members) if members is not None else set(tag_files)
        if set(events.index.get_level_values("member")) != selected:
            raise ValueError("Certified membership differs from frozen census")
        if proof["input_manifest_sha256"] != sha256(manifest_path):
            raise ValueError("Certificate used another manifest")
        if len(proof["raw_files"]) != len(selected):
            raise ValueError("Missing raw-file proof")
        if set(v["member"] for v in proof["raw_files"]) != selected:
            raise ValueError("Raw-file proof membership differs")
        for item in proof["raw_files"]:
            pinned = tag_files[item["member"]]
            if item["sha256"] != pinned["sha256"] or item["events"] != pinned["events"]:
                raise ValueError("Raw hash/event count differs from census")
        ref = config["references"][tag]
        if any(
            proof["reference_" + key + "_sha256"] != ref[key + "_sha256"]
            for key in ("events", "progress")
        ):
            raise ValueError("Historical reference hash changed")
        for name, h in proof["source_sha256"].items():
            candidates = list((args.source / "HGNDRecoGNN").rglob(name))
            if len(candidates) != 1 or sha256(candidates[0]) != h:
                raise ValueError("Counter certificate used another source")
        with np.load(result / tag / "spectra.npz", allow_pickle=False) as z:
            names = z["members"].tolist()
            grouped = events.groupby(level="member")
            if set(names) != selected or len(set(names)) != len(names):
                raise ValueError("Spectrum membership mismatch")
            if not np.array_equal(
                z["counts"].sum(axis=2), grouped[["acc_n", "acc_p"]].sum().loc[names]
            ):
                raise ValueError("Spectrum/counter integral mismatch")
            if not np.array_equal(z["events"], grouped.size().loc[names]):
                raise ValueError("Spectrum normalisation mismatch")
        if mode == "smoke":
            total_bytes = sum(f["bytes"] for f in tag_files.values())
            subset_bytes = sum(tag_files[m]["bytes"] for m in members)
            scale = max(len(tag_files) / len(members), total_bytes / subset_bytes)
            estimates[tag] = 1.5 * proof["seconds"] * scale + 120
    if mode == "smoke":
        if sha256(result / "spectra.npz") != summary["spectra_sha256"]:
            raise ValueError("Smoke aggregation spectra changed")
        if max(estimates.values()) > 2700:
            raise ValueError(
                f"Full run exceeds preregistered runtime bound: {estimates}"
            )
    receipt = dict(
        status="verified",
        mode=mode,
        job_id=slurm["job_key"],
        slurm_state=slurm["state"],
        exit_code=slurm["exit_code"],
        source_digest=source,
        environment=summary["environment"],
        archive=args.remote_archive,
        archive_sha256=sha256(args.archive),
        summary_sha256=sha256(result / "summary.json"),
        local_archive=str(args.archive.resolve()),
    )
    if mode == "smoke":
        receipt.update(
            estimated_full_seconds_with_margin=max(estimates.values()),
            estimates_seconds=estimates,
        )
    else:
        receipt["tag"] = summary["tag"]
    write_json(args.receipt, receipt)
    # Keep the standalone summary for inspection; the archive hash binds it too.
    shutil.copyfile(result / "summary.json", args.receipt.with_suffix(".summary.json"))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "slurm", "archive", "extracted", "receipt"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--remote-archive", required=True)
    verify(p.parse_args())
