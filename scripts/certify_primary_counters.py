"""Certify saved event counters against explicitly hash-pinned primary bytes.

The original reducer is replayed without changing any selection. A certificate
covering only selected members is marked subset and cannot pass the merge gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.analysis import truth_provenance as provenance  # noqa: E402
from HGNDRecoGNN.scripts import reduce_prim_full as reducer  # noqa: E402
from HGNDRecoGNN.scripts import hgnd_acceptance as geometry  # noqa: E402
from HGNDRecoGNN.scripts import content_census as identity_code  # noqa: E402
from HGNDRecoGNN.scripts.content_census import sha256  # noqa: E402
from HGNDRecoGNN.analysis import truth_reach  # noqa: E402


def certify(args):
    started = time.monotonic()
    source_files = [
        Path(__file__),
        Path(provenance.__file__),
        Path(reducer.__file__),
        Path(geometry.__file__),
        Path(identity_code.__file__),
        Path(truth_reach.__file__),
    ]
    source_hashes = {p.name: sha256(p) for p in source_files}
    manifest_hash = sha256(args.input_manifest)
    manifest = json.loads(args.input_manifest.read_text())
    declared = [
        v
        for v in manifest["files"]
        if v["kind"] == "smash_primary" and v["tag"] == args.tag
    ]
    if not declared:
        raise ValueError("No primary membership declared for sample")
    # A full member includes its generator run directory. The source inventory
    # produced by the census uses exactly this run/member layout.
    raw = {"/".join(Path(v["path"]).parts[-2:]): v for v in declared}
    if len(raw) != len(declared):
        raise ValueError("Ambiguous run/member primary names")
    progress_hash = sha256(args.reference_progress)
    events_hash = sha256(args.reference_events)
    progress = json.loads(args.reference_progress.read_text())
    if set(progress["done"]) != set(raw):
        raise ValueError("Reference and declared raw file membership differ")
    reference = provenance.keyed(pd.read_pickle(args.reference_events), progress)
    selected = sorted(raw) if args.all else sorted(set(args.member))
    if not selected or not set(selected).issubset(raw):
        raise ValueError("Unknown or empty member selection")
    if args.raw_copy and len(selected) != 1:
        raise ValueError("A raw copy override requires exactly one declared member")
    # Leave any existing result untouched, including failed/subset certificates.
    args.output.mkdir(parents=True, exist_ok=False)
    frames, identities, verified_files, spectra = [], [], [], []
    output_progress = {"done": {}}
    for i, member in enumerate(selected):
        item = raw[member]
        path = args.raw_copy or Path(item["path"])
        if getattr(args, "raw_root", None):
            path = args.raw_root.joinpath(*Path(item["path"]).parts[-3:])
        before = sha256(path)
        if before != item["sha256"]:
            raise ValueError(
                f"Primary input differs from the pinned manifest: {member}"
            )
        primary = pd.read_csv(path)
        identity = provenance.particle_identities(primary)
        if args.role == "impact" and (primary.B < 0).any():
            raise ValueError("Impact-parameter source still contains B < 0")
        fresh, _, _, _ = reducer.reduce_member(path, i, histograms=False)
        fresh = fresh.set_index("Row").sort_index()
        old = reference.xs(member, level="member").sort_index()
        if not fresh.index.equals(old.index) or not fresh.index.equals(identity.index):
            raise ValueError("Event keys differ from raw primary export")
        for name in old.columns:
            if name == "file_idx":
                continue
            if name not in fresh or not np.array_equal(old[name], fresh[name]):
                raise ValueError(f"Counter differs from raw replay: {member}/{name}")
        if not np.array_equal(fresh.nprim, identity.nprim):
            raise ValueError("Independent primary-row multiplicity mismatch")
        floor = getattr(args, "reach_floor", None)
        edges = (
            reducer.E_EDGES
            if floor is None
            else np.unique(np.r_[reducer.E_EDGES, floor])
        )
        if floor is not None:
            fresh = fresh.join(
                truth_reach.reach_counters(
                    primary,
                    floor,
                    independent=getattr(args, "independent_window", False),
                )
            )
        if getattr(args, "save_spectra", False):
            acc = geometry.in_acceptance(
                *(primary[c].to_numpy() for c in ("Px", "Py", "Pz", "vX", "vY", "vZ"))
            )
            counts = []
            for pdg, column in ((2112, "acc_n"), (2212, "acc_p")):
                energy = primary.loc[acc & (primary.PDG == pdg), "Ekin"].to_numpy()
                if floor is not None:
                    energy = energy[energy >= floor]
                    column += "_reach"
                if not np.isfinite(energy).all() or (energy < 0).any():
                    raise ValueError("Invalid selected kinetic energy")
                hist = np.histogram(energy, bins=edges)[0]
                if int(hist.sum()) != int(fresh[column].sum()):
                    raise ValueError("Spectrum overflow or selected-count mismatch")
                counts.append(hist)
            spectra.append(counts)
        if sha256(path) != before:
            raise ValueError("Raw input changed during certification")
        output_progress["done"][member] = len(fresh)
        frames.append(fresh.reset_index())
        identities.append(identity.reset_index().assign(member=member))
        verified_files.append(
            {
                "member": member,
                "path_read": str(path),
                "sha256": before,
                "events": len(fresh),
                "counter_columns_checked": sorted(set(old.columns) - {"file_idx"}),
            }
        )
        print(f"Certified {member}: {len(fresh)} events", flush=True)
    if (
        sha256(args.reference_events) != events_hash
        or sha256(args.reference_progress) != progress_hash
    ):
        raise ValueError("Reference inputs changed during certification")
    if (
        source_hashes != {p.name: sha256(p) for p in source_files}
        or sha256(args.input_manifest) != manifest_hash
    ):
        raise ValueError("Source or manifest changed during certification")
    pd.concat(frames, ignore_index=True).to_pickle(args.output / "events.pkl")
    pd.concat(identities, ignore_index=True).to_csv(
        args.output / "identity.csv.gz", index=False
    )
    (args.output / "progress.json").write_text(
        json.dumps(output_progress, indent=2) + "\n"
    )
    artifacts = ["events.pkl", "progress.json", "identity.csv.gz"]
    if getattr(args, "save_spectra", False):
        np.savez_compressed(
            args.output / "spectra.npz",
            counts=np.asarray(spectra, dtype=np.int64),
            members=np.asarray(selected),
            e_edges=edges,
            events=np.asarray(list(output_progress["done"].values()), dtype=np.int64),
        )
        artifacts.append("spectra.npz")
    proof = {
        "schema": "primary-counter-proof-v1",
        "status": "verified",
        "role": args.role,
        "tag": args.tag,
        "coverage": "full" if len(selected) == len(raw) else "subset",
        "files_expected": len(raw),
        "files_checked": len(selected),
        "events": sum(output_progress["done"].values()),
        "counter_mismatches": 0,
        "particle_fields": provenance.PARTICLE_FIELDS,
        "source_sha256": source_hashes,
        "effective_recipe": {
            "hardness_low_exclusive_GeV": reducer.ELO,
            "hardness_high_inclusive_GeV": reducer.EHI,
            "polar_angle_band_deg": [reducer.TH_LO, reducer.TH_HI],
            "rapidity_shift": reducer.YS,
            "rectangle_cm": [
                geometry.Z_FRONT,
                geometry.X_LO,
                geometry.X_HI,
                geometry.Y_LO,
                geometry.Y_HI,
            ],
            "selection_policy": "Unchanged inherited reducer conventions; no fitted selection",
            "B_policy": "Metadata only; no B histogram or weighting",
            "spectra_E_edges_GeV": edges.tolist() if spectra else None,
            "reach_floor_GeV": getattr(args, "reach_floor", None),
            "proxy_cm_rapidity": float(truth_reach.Y_CM)
            if getattr(args, "reach_floor", None)
            else None,
            "independent_window_check": getattr(args, "independent_window", False),
        },
        "input_manifest_sha256": manifest_hash,
        "reference_events_sha256": events_hash,
        "reference_progress_sha256": progress_hash,
        "raw_files": verified_files,
        "artifacts_sha256": {n: sha256(args.output / n) for n in artifacts},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "seconds": time.monotonic() - started,
        "scope": "Exact saved-counter replay on the declared bytes; generator provenance is not inferred.",
    }
    (args.output / "proof.json").write_text(
        json.dumps(proof, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Coverage: {proof['coverage']}; {proof['events']} verified events", flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-manifest", type=Path, required=True)
    p.add_argument("--reference-events", type=Path, required=True)
    p.add_argument("--reference-progress", type=Path, required=True)
    p.add_argument(
        "--tag", choices=["zeroSpot", "defaultSpot", "bigSpot"], required=True
    )
    p.add_argument("--role", choices=["acceptance", "impact"], required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--member", action="append")
    p.add_argument("--raw-copy", type=Path)
    p.add_argument("--save-spectra", action="store_true")
    p.add_argument("--reach-floor", type=float, choices=[0.3])
    p.add_argument("--independent-window", action="store_true")
    p.add_argument("--raw-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    certify(p.parse_args())
