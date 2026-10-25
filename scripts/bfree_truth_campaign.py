"""Finite B-free primary replay and whole-file bootstrap, with pinned inputs.

The inherited rectangle and energy thresholds stay fixed. All quantities are MC
truth, unweighted over the represented production sample. Missing B is ignored.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import platform
import sys
import tarfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.analysis.truth_provenance import load_certified  # noqa: E402
from HGNDRecoGNN.scripts.certify_primary_counters import certify  # noqa: E402
from HGNDRecoGNN.scripts.content_census import sha256  # noqa: E402
from HGNDRecoGNN.scripts.reduce_prim_full import E_EDGES  # noqa: E402

TAGS = ("zeroSpot", "defaultSpot", "bigSpot")
# Inherited high/low thresholds are evaluated by the reducer, never from bins.
COLUMNS = ("acc_n", "acc_p", "acc_n_hi", "acc_n_lo", "acc_p_hi", "acc_p_lo")
# 400 matches the earlier file-bootstrap convention. Seed is preregistered by
# campaign date; changing it changes Monte Carlo precision, not the estimator.
REPLICATES, SEED = 400, 20261025


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def runtime():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "executable_sha256": sha256(Path(sys.executable)),
    }


def source_digest(root):
    """Verify every archived source/config byte before deriving its binding."""
    lines = (root / "SHA256SUMS").read_text().splitlines()
    for line in lines:
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError(f"Changed campaign source: {name}")
    return sha256(root / "SHA256SUMS")


def validate_receipt(path, source, environment, *, mode):
    receipt = json.loads(path.read_text())
    if (
        receipt.get("status") != "verified"
        or receipt.get("slurm_state") != "COMPLETED"
        or receipt.get("exit_code") != "0:0"
        or receipt.get("mode") != mode
        or receipt.get("source_digest") != source
        or receipt.get("environment") != environment
    ):
        raise ValueError(f"Invalid external verification receipt: {path}")
    if sha256(Path(receipt["archive"])) != receipt["archive_sha256"]:
        raise ValueError("Verified export archive changed")
    if mode == "smoke" and receipt["estimated_full_seconds_with_margin"] > 2700:
        raise ValueError("Measured smoke does not support the 45-minute bound")
    return receipt


def observables(counts):
    if (counts[..., [1, 3, 5]] <= 0).any():
        raise ValueError("Empty denominator in fixed-window observable")
    rn = counts[..., 2] / counts[..., 3]
    rp = counts[..., 4] / counts[..., 5]
    if (rp <= 0).any():
        raise ValueError("Empty proton hardness denominator")
    return np.stack((rn, rp, rn / rp, counts[..., 0] / counts[..., 1]), axis=-1)


def summarize(root, output, *, smoke=False, config=None):
    rng = np.random.default_rng(SEED)
    results, draws, spectra = {}, {}, {}
    edges = None
    for tag in TAGS:
        members = config["smoke_members"][tag] if smoke else None
        events, _, proof = load_certified(
            root / tag, "acceptance", smoke_members=members
        )
        if "spectra.npz" not in proof["artifacts_sha256"]:
            raise ValueError("Certificate lacks fresh file spectra")
        manifest = json.loads((config["manifest_path"]).read_text())
        declared = [f for f in manifest["files"] if f["tag"] == tag]
        wanted = {
            str(Path(f["path"]).parent.name + "/" + Path(f["path"]).name): f
            for f in declared
        }
        selected = set(members) if smoke else set(wanted)
        if set(events.index.get_level_values("member")) != selected:
            raise ValueError("Certificate membership differs from campaign manifest")
        if proof["input_manifest_sha256"] != sha256(config["manifest_path"]):
            raise ValueError("Certificate was produced against another census")
        reference = config["references"][tag]
        if any(
            proof["reference_" + key + "_sha256"] != reference[key + "_sha256"]
            for key in ("events", "progress")
        ):
            raise ValueError("Reference binding differs from the pinned campaign")
        for item in proof["raw_files"]:
            expected = wanted[item["member"]]
            if (
                item["sha256"] != expected["sha256"]
                or item["events"] != expected["events"]
            ):
                raise ValueError("Certified file hash/event count differs from census")
        with np.load(root / tag / "spectra.npz", allow_pickle=False) as z:
            names = z["members"].tolist()
            spec, nevent = z["counts"], z["events"]
            if edges is None:
                edges = z["e_edges"].copy()
            if not np.array_equal(z["e_edges"], edges) or not np.array_equal(
                edges, E_EDGES
            ):
                raise ValueError("Energy binning differs from the fixed recipe")
        if len(set(names)) != len(names) or set(names) != selected:
            raise ValueError("Spectrum file membership mismatch")
        grouped = events.groupby(level="member")
        counts = grouped[list(COLUMNS)].sum().loc[names].to_numpy(dtype=np.int64)
        if (spec < 0).any() or not np.array_equal(spec.sum(axis=2), counts[:, :2]):
            raise ValueError("Spectrum integral does not equal selected particles")
        if not np.array_equal(nevent, grouped.size().loc[names]):
            raise ValueError(
                "Spectrum event normalisation differs from certified table"
            )
        # One whole-file multiplicity draw drives counts, yields and all bins.
        weights = rng.multinomial(
            len(names), np.full(len(names), 1 / len(names)), REPLICATES
        )
        boot_counts = weights @ counts
        boot_events = weights @ nevent
        boot_spectra = (
            np.einsum("rf,fse->rse", weights, spec) / boot_events[:, None, None]
        )
        central = observables(counts.sum(axis=0))
        boot = observables(boot_counts)
        draws[tag] = boot
        normalized = spec.sum(axis=0) / nevent.sum()
        spectra[tag + "_per_event"] = normalized
        spectra[tag + "_sd"] = boot_spectra.std(axis=0, ddof=1)
        # Empty energy bins are masked, never divided by a pseudocount.
        valid = (normalized[1] > 0) & (boot_spectra[:, 1, :] > 0).all(axis=0)
        ratio, ratio_sd = (
            np.full(len(edges) - 1, np.nan),
            np.full(len(edges) - 1, np.nan),
        )
        ratio[valid] = normalized[0, valid] / normalized[1, valid]
        ratio_sd[valid] = (boot_spectra[:, 0, valid] / boot_spectra[:, 1, valid]).std(
            axis=0, ddof=1
        )
        spectra[tag + "_np_ratio"] = ratio
        spectra[tag + "_np_ratio_sd"] = ratio_sd
        spectra[tag + "_ratio_valid"] = valid
        results[tag] = {
            "files": len(names),
            "events": int(nevent.sum()),
            "counts": dict(zip(COLUMNS, counts.sum(axis=0).tolist())),
            "observables": dict(zip(("R_n", "R_p", "D", "n_over_p"), central.tolist())),
            "bootstrap_sd": dict(
                zip(("R_n", "R_p", "D", "n_over_p"), boot.std(axis=0, ddof=1).tolist())
            ),
            "proof_sha256": sha256(root / tag / "proof.json"),
            "seconds": proof["seconds"],
            "bytes": sum(
                Path(v["path_read"]).stat().st_size for v in proof["raw_files"]
            ),
        }
    contrast = draws["bigSpot"] / draws["zeroSpot"] - 1
    point = (
        np.array(list(results["bigSpot"]["observables"].values()))
        / np.array(list(results["zeroSpot"]["observables"].values()))
        - 1
    )
    np.savez_compressed(output / "spectra.npz", e_edges=edges, **spectra)
    return {
        "schema": "bfree-truth-summary-v1",
        "status": "verified",
        "coverage": "smoke_subset" if smoke else "full",
        "samples": results,
        "endpoint_fraction_big_over_zero_minus_one": point.tolist(),
        "endpoint_bootstrap_sd": contrast.std(axis=0, ddof=1).tolist(),
        "observable_order": ["R_n", "R_p", "D", "n_over_p"],
        "bootstrap": {
            "replicates": REPLICATES,
            "seed": SEED,
            "unit": "whole production file",
            "error": "SD, ddof=1",
        },
        "scope": "MC truth; fixed inherited straight-ray rectangle; unweighted; B ignored. Proton magnetic response is not modelled. File independence and unknown centrality mixture limit interpretation.",
        "spectra_sha256": sha256(output / "spectra.npz"),
    }


def main(args):
    started = time.monotonic()
    source = source_digest(args.source)
    config = json.loads((args.source / "campaign.json").read_text())
    config["manifest_path"] = args.source / "primary_manifest.json"
    environment = runtime()
    if args.mode != "smoke":
        validate_receipt(args.smoke_receipt, source, environment, mode="smoke")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode in ("smoke", "certify"):
        tags = TAGS if args.mode == "smoke" else (TAGS[args.task],)
        for tag in tags:
            ref = config["references"][tag]
            for kind in ("events", "progress"):
                if sha256(Path(ref[kind])) != ref[kind + "_sha256"]:
                    raise ValueError("Historical reference changed before replay")
            certify(
                Namespace(
                    input_manifest=config["manifest_path"],
                    reference_events=Path(ref["events"]),
                    reference_progress=Path(ref["progress"]),
                    tag=tag,
                    role="acceptance",
                    all=args.mode == "certify",
                    member=config["smoke_members"][tag],
                    raw_copy=None,
                    save_spectra=True,
                    output=args.output / tag,
                )
            )
        if args.mode == "smoke":
            summary = summarize(args.output, args.output, smoke=True, config=config)
        else:
            tag = TAGS[args.task]
            proof = json.loads((args.output / tag / "proof.json").read_text())
            summary = {
                "schema": "bfree-certify-job-v1",
                "status": "verified",
                "tag": tag,
                "coverage": proof["coverage"],
                "events": proof["events"],
                "files": proof["files_checked"],
            }
    else:
        receipts = json.loads(args.full_receipts.read_text())
        if set(receipts) != set(TAGS):
            raise ValueError(
                "All three externally verified complete tasks are required"
            )
        for tag in TAGS:
            receipt = validate_receipt(
                Path(receipts[tag]), source, environment, mode="certify"
            )
            if receipt["tag"] != tag:
                raise ValueError("Full receipt tag mismatch")
            with tarfile.open(receipt["archive"]) as tar:
                tar.extractall(args.output / "inputs" / tag, filter="data")
            (args.output / "inputs" / tag / "result" / tag).rename(args.output / tag)
        summary = summarize(args.output, args.output, config=config)
    if source_digest(args.source) != source:
        raise ValueError("Campaign source changed during execution")
    summary.update(
        mode=args.mode,
        source_digest=source,
        environment=environment,
        job_id=args.job_id,
        seconds=time.monotonic() - started,
        input_manifest_sha256=sha256(config["manifest_path"]),
    )
    write_json(args.output / "summary.json", summary)
    print(
        json.dumps({k: summary[k] for k in ("mode", "status", "seconds", "job_id")}),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--mode", choices=["smoke", "certify", "aggregate"], required=True)
    p.add_argument("--task", type=int, choices=range(3), default=0)
    p.add_argument("--smoke-receipt", type=Path)
    p.add_argument("--full-receipts", type=Path)
    p.add_argument("--job-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
