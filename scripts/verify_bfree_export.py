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
    if mode not in ("smoke", "certify", "aggregate"):
        raise ValueError("Unknown export mode")
    expected = TAGS if mode in ("smoke", "aggregate") else (summary["tag"],)
    if mode == "aggregate" and args.full_receipts is None:
        raise ValueError("Aggregate requires all externally verified full receipts")
    full_receipts = (
        json.loads(args.full_receipts.read_text()) if mode == "aggregate" else {}
    )
    estimates = {}
    numerical_rows = []
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
                z["counts"].sum(axis=2),
                grouped[list(config.get("counter_columns", ["acc_n", "acc_p"]))[:2]]
                .sum()
                .loc[names],
            ):
                raise ValueError("Spectrum/counter integral mismatch")
            if not np.array_equal(z["events"], grouped.size().loc[names]):
                raise ValueError("Spectrum normalisation mismatch")
        if mode == "aggregate":
            columns = config.get(
                "counter_columns",
                ["acc_n", "acc_p", "acc_n_hi", "acc_n_lo", "acc_p_hi", "acc_p_lo"],
            )
            for member, frame in events.groupby(level="member"):
                matrix = frame[columns].to_numpy(dtype=np.int64)
                numerical_rows.append(
                    {
                        "tag": tag,
                        "member": member,
                        "events": len(frame),
                        "counts": matrix.sum(axis=0).tolist(),
                        "second_moments": (matrix.T @ matrix).tolist(),
                    }
                )
            bound = json.loads(Path(full_receipts[tag]).read_text())
            if (
                bound["slurm_state"] != "COMPLETED"
                or bound["exit_code"] != "0:0"
                or bound["tag"] != tag
                or bound["source_digest"] != source
            ):
                raise ValueError("Unverified full task in aggregate")
            archive = Path(bound["local_archive"])
            if sha256(archive) != bound["archive_sha256"]:
                raise ValueError("Full source archive changed")
            with tarfile.open(archive) as tar:
                original = tar.extractfile("result/" + tag + "/proof.json").read()
            if original != (result / tag / "proof.json").read_bytes():
                raise ValueError(
                    "Aggregate certificate differs from verified full task"
                )
        if mode == "smoke":
            total_bytes = sum(f["bytes"] for f in tag_files.values())
            subset_bytes = sum(tag_files[m]["bytes"] for m in members)
            scale = max(len(tag_files) / len(members), total_bytes / subset_bytes)
            estimates[tag] = 1.5 * proof["seconds"] * scale + 120
    if mode == "aggregate":
        from HGNDRecoGNN.analysis.truth_statistics import replay

        rebuilt = replay({"columns": columns, "rows": numerical_rows})
        for tag in TAGS:
            for key in ("counts", "observables", "bootstrap_sd"):
                old, new = summary["samples"][tag][key], rebuilt["samples"][tag][key]
                if set(old) != set(new) or not np.allclose(
                    list(old.values()), list(new.values()), rtol=1e-12, atol=1e-14
                ):
                    raise ValueError("Aggregate numerical replay mismatch: " + key)
        for key in (
            "endpoint_fraction_big_over_zero_minus_one",
            "endpoint_bootstrap_sd",
        ):
            if not np.allclose(summary[key], rebuilt[key], rtol=1e-12, atol=1e-14):
                raise ValueError("Aggregate numerical replay mismatch: " + key)
    if mode in ("smoke", "aggregate"):
        if (
            "per_file_counts_sha256" in summary
            and sha256(result / "per_file_counts.json")
            != summary["per_file_counts_sha256"]
        ):
            raise ValueError("Per-file counters changed")
        if sha256(result / "spectra.npz") != summary["spectra_sha256"]:
            raise ValueError("Smoke aggregation spectra changed")
        if estimates and max(estimates.values()) > 2700:
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
    elif mode == "certify":
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
    p.add_argument("--full-receipts", type=Path)
    verify(p.parse_args())
