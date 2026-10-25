"""Wrong collisions with the same keys/counts must fail the b-transfer gate."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
import pytest

from HGNDRecoGNN.analysis.truth_provenance import (
    keyed,
    load_certified,
    merge_certified,
    particle_identities,
)
from HGNDRecoGNN.scripts.certify_primary_counters import certify
from HGNDRecoGNN.scripts.content_census import sha256
from HGNDRecoGNN.scripts.reduce_prim_full import reduce_member


def primary():
    return pd.DataFrame(
        [
            dict(
                Row=0,
                Instance=1,
                Id=1,
                PDG=2112,
                E=3.0,
                Ekin=2.1,
                Mass=0.939,
                Px=-0.5,
                Py=0.0,
                Pz=3.0,
                Pt=0.5,
                P=3.04,
                Rapid=2.0,
                vX=0.0,
                vY=0.0,
                vZ=0.0,
                B=-1.0,
                NPrim=-1,
            ),
            dict(
                Row=0,
                Instance=2,
                Id=2,
                PDG=2212,
                E=1.4,
                Ekin=0.46,
                Mass=0.938,
                Px=-0.17,
                Py=0.0,
                Pz=1.0,
                Pt=0.17,
                P=1.01,
                Rapid=1.0,
                vX=0.0,
                vY=0.0,
                vZ=0.0,
                B=-1.0,
                NPrim=-1,
            ),
        ]
    )


def test_particle_identity_ignores_repaired_metadata_and_record_order():
    a = primary()
    b = a.iloc[::-1].copy()
    b["B"] = 8.0
    b["NPrim"] = 2
    b["Instance"] = [99, 88]
    pd.testing.assert_frame_equal(particle_identities(a), particle_identities(b))
    b.loc[b.Id == 1, "Px"] = -0.51
    assert (
        particle_identities(a).particle_sha256.iloc[0]
        != particle_identities(b).particle_sha256.iloc[0]
    )
    assert particle_identities(a).nprim.iloc[0] == particle_identities(b).nprim.iloc[0]


def test_metadata_must_be_constant_within_an_event():
    a = primary()
    a.loc[1, "B"] = 4.0
    with pytest.raises(ValueError, match="constant"):
        particle_identities(a)


def test_same_basename_in_different_runs_remains_separate():
    d = pd.DataFrame(
        {"file_idx": [0, 1], "Row": [0, 0], "nprim": [2, 2], "B": [-1.0, -1.0]}
    )
    result = keyed(d, {"done": {"run1/0001_prim.csv": 1, "run2/0001_prim.csv": 1}})
    assert len(result) == 2 and not result.index.has_duplicates


def test_wrong_particles_fail_despite_matching_event_keys_and_multiplicity():
    a = keyed(
        pd.DataFrame({"file_idx": [0], "Row": [0], "nprim": [2], "B": [-1.0]}),
        {"done": {"run/0001_prim.csv": 1}},
    )
    b = a.copy()
    b["B"] = 7.0
    ai = pd.DataFrame({"nprim": [2], "particle_sha256": ["original"]}, index=a.index)
    bi = ai.copy()
    bi["particle_sha256"] = "different"
    with pytest.raises(ValueError, match="particle contents"):
        merge_certified(a, b, ai, bi)
    result = merge_certified(a, b, ai, ai)
    assert result.B.iloc[0] == 7.0 and a.B.iloc[0] == -1.0
    with pytest.raises(ValueError, match="intersections"):
        merge_certified(a, b.iloc[:0], ai, ai)
    with pytest.raises(ValueError, match="nonnegative"):
        merge_certified(a, a, ai, ai)


@pytest.mark.parametrize("mutation", ["none", "counter", "raw", "subset"])
def test_certificate_replays_counters_and_rejects_missing_or_changed_inputs(
    tmp_path, mutation
):
    from argparse import Namespace

    raw = tmp_path / "0001_prim.csv"
    primary().to_csv(raw, index=False)
    frame, _, _, _ = reduce_member(raw, 0)
    if mutation == "counter":
        frame.loc[0, "acc_n_hi"] += 1
    ref = tmp_path / "ref.pkl"
    frame.to_pickle(ref)
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"done": {"run/0001_prim.csv": 1}}))
    manifest = tmp_path / "manifest.json"
    files = [
        {
            "kind": "smash_primary",
            "tag": "zeroSpot",
            "path": "/declared/run/0001_prim.csv",
            "sha256": sha256(raw),
        }
    ]
    if mutation == "subset":
        files.append(dict(files[0], path="/declared/run/0002_prim.csv"))
        frame2 = frame.copy()
        frame2["file_idx"] = 1
        pd.concat([frame, frame2], ignore_index=True).to_pickle(ref)
        progress.write_text(
            json.dumps({"done": {"run/0001_prim.csv": 1, "run/0002_prim.csv": 1}})
        )
    manifest.write_text(json.dumps({"files": files}))
    if mutation == "raw":
        raw.write_text(raw.read_text().replace("2.1", "2.2"))
    args = Namespace(
        input_manifest=manifest,
        reference_events=ref,
        reference_progress=progress,
        tag="zeroSpot",
        role="acceptance",
        all=False,
        member=["run/0001_prim.csv"],
        raw_copy=raw,
        output=tmp_path / "certificate",
    )
    if mutation in ("counter", "raw"):
        with pytest.raises(ValueError):
            certify(args)
        assert not (args.output / "proof.json").exists()
        return
    certify(args)
    if mutation == "subset":
        with pytest.raises(ValueError, match="complete"):
            load_certified(args.output, "acceptance")
    else:
        events, _, proof = load_certified(args.output, "acceptance")
        assert len(events) == 1 and proof["counter_mismatches"] == 0
        (args.output / "identity.csv.gz").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="changed"):
            load_certified(args.output, "acceptance")


def test_complete_three_sample_join_preserves_certified_counters(tmp_path, monkeypatch):
    from argparse import Namespace
    from HGNDRecoGNN.scripts.merge_acc_with_b import main

    tags = ("zeroSpot", "defaultSpot", "bigSpot")
    memberships = {"acceptance": [], "impact": []}
    sources = {}
    for role in memberships:
        for tag in tags:
            folder = tmp_path / "raw" / role / tag / "run"
            folder.mkdir(parents=True)
            raw = folder / "0001_prim.csv"
            data = primary()
            if role == "impact":
                data["B"] = 7.0
                data["NPrim"] = 2
            data.to_csv(raw, index=False)
            frame, _, _, _ = reduce_member(raw, 0)
            reference = folder / "reference.pkl"
            frame.to_pickle(reference)
            progress = folder / "reference_progress.json"
            progress.write_text(json.dumps({"done": {"run/0001_prim.csv": 1}}))
            memberships[role].append(
                {
                    "kind": "smash_primary",
                    "tag": tag,
                    "path": str(raw),
                    "sha256": sha256(raw),
                }
            )
            sources[role, tag] = (raw, reference, progress)
    manifests = {}
    for role, files in memberships.items():
        manifests[role] = tmp_path / f"{role}_manifest.json"
        manifests[role].write_text(
            json.dumps({"files": files, "status": "complete", "smoke": False})
        )
        for tag in tags:
            _, reference, progress = sources[role, tag]
            certify(
                Namespace(
                    input_manifest=manifests[role],
                    reference_events=reference,
                    reference_progress=progress,
                    tag=tag,
                    role=role,
                    all=True,
                    member=None,
                    raw_copy=None,
                    output=tmp_path / role / tag,
                )
            )
    output = tmp_path / "joined"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_acc_with_b",
            "--acc-dir",
            str(tmp_path / "acceptance"),
            "--b-dir",
            str(tmp_path / "impact"),
            "--census",
            str(manifests["acceptance"]),
            "--out-dir",
            str(output),
        ],
    )
    main()
    receipt = json.loads((output / "merge_report.json").read_text())
    assert receipt["status"] == "complete" and len(receipt["samples"]) == 3
    for tag in tags:
        frame = pd.read_pickle(output / f"{tag}_events.pkl")
        assert len(frame) == 1 and frame.B.iloc[0] == 7.0
        assert frame.nprim.iloc[0] == 2
        assert frame.acc_n_hi.iloc[0] == frame.acc_p_lo.iloc[0] == 1
        assert receipt["samples"][tag]["particle_mismatches"] == 0
