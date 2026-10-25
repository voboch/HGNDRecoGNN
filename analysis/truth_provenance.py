"""Exact primary-content and counter provenance for joining recovered b.

Event multiplicity is necessary but does not identify a collision. Bind the
complete exported particle multiset, retaining run/member and local Row keys.
B and NPrim are excluded from particle identity because those are the metadata
being repaired; multiplicity is independently counted from particle rows.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from HGNDRecoGNN.scripts.content_census import event_signatures, sha256

PARTICLE_FIELDS = (
    "Id",
    "PDG",
    "E",
    "Ekin",
    "Mass",
    "Px",
    "Py",
    "Pz",
    "Pt",
    "P",
    "Rapid",
    "vX",
    "vY",
    "vZ",
)
KEYS = ["member", "Row"]


def member_map(progress):
    """Keep complete member names; basename collisions must never be merged."""
    done = progress["done"]
    for name, count in done.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not name.endswith("_prim.csv"):
            raise ValueError(f"Invalid primary member {name}")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Invalid represented-event count")
    if not done:
        raise ValueError("Empty primary membership")
    return dict(enumerate(done))


def keyed(events, progress):
    mapping = member_map(progress)
    if set(events.file_idx.unique()) != set(mapping):
        raise ValueError("Counter file indices do not cover the declared membership")
    for column in ("Row", "file_idx", "nprim"):
        values = events[column].to_numpy()
        if not np.isfinite(values).all() or (values != values.astype(np.int64)).any():
            raise ValueError(f"Noninteger counter identifier {column}")
    if (events.Row < 0).any() or (events.nprim <= 0).any():
        raise ValueError("Invalid counter row or multiplicity")
    result = (
        events.assign(member=events.file_idx.map(mapping)).set_index(KEYS).sort_index()
    )
    if result.index.has_duplicates:
        raise ValueError("Duplicate event key in counter table")
    counts = result.groupby(level="member").size().to_dict()
    if counts != progress["done"]:
        raise ValueError("Counter event counts do not match progress membership")
    return result


def particle_identities(frame):
    if frame.empty or not set(PARTICLE_FIELDS + ("Row", "B", "NPrim")).issubset(frame):
        raise ValueError("Incomplete primary export")
    for name in ("B", "NPrim"):
        if not np.isfinite(frame[name].to_numpy()).all():
            raise ValueError(f"Nonfinite metadata {name}")
        if (frame.groupby("Row")[name].nunique() != 1).any():
            raise ValueError(f"{name} is not constant within each event")
    return pd.DataFrame(
        event_signatures(frame, PARTICLE_FIELDS),
        columns=["Row", "nprim", "particle_sha256"],
    ).set_index("Row")


def load_certified(directory, role, *, smoke_members=None):
    directory = Path(directory)
    proof_path = directory / "proof.json"
    proof = json.loads(proof_path.read_text())
    if (
        proof.get("schema") != "primary-counter-proof-v1"
        or proof.get("role") != role
        or proof.get("status") != "verified"
        or proof.get("coverage") != ("subset" if smoke_members is not None else "full")
        or tuple(proof.get("particle_fields", [])) != PARTICLE_FIELDS
    ):
        raise ValueError(
            "A complete primary-counter certificate with the declared role is required"
        )
    if (
        smoke_members is None and proof["files_checked"] != proof["files_expected"]
    ) or proof["counter_mismatches"] != 0:
        raise ValueError("Incomplete or mismatched primary-counter certificate")
    required_artifacts = {
        "events.pkl",
        "progress.json",
        "identity.csv.gz",
    }
    if not required_artifacts.issubset(proof["artifacts_sha256"]) or (
        set(proof["artifacts_sha256"]) - required_artifacts - {"spectra.npz"}
    ):
        raise ValueError("Certificate does not pin every consumed artifact")
    for name, expected in proof["artifacts_sha256"].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError(f"Certified artifact changed: {name}")
    events = keyed(
        pd.read_pickle(directory / "events.pkl"),
        json.loads((directory / "progress.json").read_text()),
    )
    if smoke_members is not None and (
        set(events.index.get_level_values("member")) != set(smoke_members)
        or proof["files_checked"] != len(smoke_members)
    ):
        raise ValueError("Smoke membership differs from the declared subset")
    identities = (
        pd.read_csv(directory / "identity.csv.gz", dtype={"member": str})
        .set_index(KEYS)
        .sort_index()
    )
    if identities.index.has_duplicates or not events.index.equals(identities.index):
        raise ValueError("Certified identity/event membership mismatch")
    if (
        not np.array_equal(events.nprim, identities.nprim)
        or len(events) != proof["events"]
    ):
        raise ValueError("Certified multiplicity or event count mismatch")
    return events, identities, proof


def merge_certified(acc, bee, acc_id, b_id):
    """Require full one-to-one content equality before transferring b values."""
    if (
        not acc.index.equals(bee.index)
        or not acc.index.equals(acc_id.index)
        or not acc.index.equals(b_id.index)
    ):
        raise ValueError("Full event membership differs; intersections are forbidden")
    if not np.array_equal(acc.nprim, bee.nprim):
        raise ValueError("Primary multiplicity differs")
    if not np.array_equal(acc_id.particle_sha256, b_id.particle_sha256):
        raise ValueError(
            "Primary particle contents differ despite matching event keys/multiplicity"
        )
    b = bee.B.to_numpy()
    if not np.isfinite(b).all() or (b < 0).any():
        raise ValueError("Recovered impact parameter must be finite and nonnegative")
    result = acc.copy()
    result["B"] = b
    return result
