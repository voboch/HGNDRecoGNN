"""Synthetic exports: header-only, byte copies, truncated copies, binding, stops."""

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.audit_smash_sources import HITS, VACS  # noqa: E402
from HGNDRecoGNN.scripts.content_census import sha256  # noqa: E402
from HGNDRecoGNN.scripts.smash_export_census import (  # noqa: E402
    census_tag,
    combine,
    project,
)

RUN = "10943245"
PRIM = ["Row", "Instance", "Id", "PDG", "E", "Ekin", "Mass", "Px", "Py", "Pz", "Pt", "P",
        "Rapid", "vX", "vY", "vZ", "B", "NPrim"]


def synthetic(seed, rows):
    """Two direct primary neutrons per event, two hits each, six-decimal CSV values."""
    rng = np.random.default_rng(seed)
    hits, vacs, prim = [], [], []
    for r in rows:
        vertex = rng.uniform(-1, 1, 3)
        instance = 0
        for t in (1, 2):
            p = rng.uniform(-0.5, 0.5, 3) + [0, 0, 2]
            ekin = rng.uniform(0.3, 3.0)
            track = {"Id": 100 * r + t, "PDG": 2112, "Ekin": ekin, "Px": p[0], "Py": p[1], "Pz": p[2],
                     "vX": vertex[0], "vY": vertex[1], "vZ": vertex[2]}
            prim.append({"Row": r, "Instance": t, "E": ekin + .94, "Mass": .94, "Pt": 0.1, "P": 2.0,
                         "Rapid": 1.0, "B": 3.0, "NPrim": 2, **track})
            for _ in range(2):
                instance += 1
                x = rng.uniform(-100, 100, 3)
                hits.append({"Row": r, "Instance": instance, "fX": x[0], "fY": x[1], "fZ": 700 + x[2],
                             "DetectorId": 1, "LayerId": rng.integers(0, 8), "RowId": rng.integers(0, 10),
                             "ColumnId": rng.integers(0, 10), "fTime": rng.uniform(20, 40),
                             "fELoss": rng.uniform(.001, .05)})
                vacs.append({"Row": r, "Instance": instance, "Rapid": 1.0, "fMotherId": -1,
                             "Weight": .01, "DetectorID": 1, "Side": 5, "X": x[0], "Y": x[1], "Z": x[2],
                             **track})
    return (pd.DataFrame(hits)[HITS], pd.DataFrame(vacs)[VACS], pd.DataFrame(prim)[PRIM])


def csv_text(frame):
    return frame.to_csv(index=False, float_format="%.6f", lineterminator="\n")


def write_pair(root, name, hits, vacs):
    (root / RUN).mkdir(parents=True, exist_ok=True)
    for kind, value in (("hits", hits), ("vacs", vacs)):
        text = value if isinstance(value, str) else csv_text(value)
        (root / RUN / f"{name}_{kind}.csv").write_text(text)


def header_only(root, name):
    write_pair(root, name, ",".join(HITS) + "\n", ",".join(VACS) + "\n")


def write_primary(root, name, prim):
    (root / RUN).mkdir(parents=True, exist_ok=True)
    (root / RUN / f"{name}_prim.csv").write_text(csv_text(prim))


def inventory(tmp, tag, pinned_primary=None, empty=(), aliases=None):
    raw, prim = tmp / "raw" / tag, tmp / "prim" / tag
    names = sorted(str(p.relative_to(raw)).removesuffix("_hits.csv") for p in raw.glob("*/*_hits.csv"))
    primaries = sorted(str(p.relative_to(prim)).removesuffix("_prim.csv") for p in prim.glob("*/*_prim.csv"))
    pinned_primary = pinned_primary or {}
    pinned = sorted(set(pinned_primary) | set(empty) | set(aliases or {}))
    return {"raw_root": str(raw), "primary_root": str(prim), "all_pair_names": names,
            "common_primary_pair_names": primaries,
            "primary_member_map": {n: pinned_primary.get(n) for n in pinned},
            "empty_exports": {n: {} for n in empty}, "duplicate_exports": aliases or {},
            "input_sha256": {n: {"hits": sha256(raw / f"{n}_hits.csv"), "vacs": sha256(raw / f"{n}_vacs.csv")}
                             for n in pinned},
            "primary_sha256": {p: sha256(prim / f"{p}_prim.csv") for p in primaries}}


def n(i):
    return f"{RUN}/{i:04d}"


@pytest.fixture
def population(tmp_path):
    tag = "zeroSpot"
    raw, prim = tmp_path / "raw" / tag, tmp_path / "prim" / tag
    a_hits, a_vacs, a_prim = synthetic(1, range(10))
    b_hits, b_vacs, b_prim = synthetic(2, range(10))
    c_hits, c_vacs, _ = synthetic(3, range(8))
    write_pair(raw, "0001", a_hits, a_vacs)
    header_only(raw, "0002")
    write_pair(raw, "0003", a_hits, a_vacs)  # byte alias
    # Truncated copy: whole lines, cut inside event 6 of the hits and after
    # event 5 of the vacs.
    cut = a_hits.index[a_hits.Row == 6][-1]
    write_pair(raw, "0004", a_hits.loc[:cut - 1], a_vacs.loc[a_vacs.Row <= 5])
    write_pair(raw, "0005", b_hits, b_vacs)  # unbound in the inventory, bindable
    write_pair(raw, "0006", c_hits, c_vacs)  # no primary file
    write_primary(prim, "0001", a_prim)
    write_primary(prim, "0002", b_prim)
    write_primary(prim, "0003", synthetic(4, range(5))[2])
    info = inventory(tmp_path, tag, {n(1): n(1), n(3): n(1)}, empty=[n(2)], aliases={n(3): n(1)})
    return tmp_path, tag, info


def test_header_bytes_alias_truncation_and_binding(population):
    tmp, tag, info = population
    s = census_tag(info, tag, tmp / "out", workers=2, expected_pinned_bound=1)
    assert s["status"] == "complete", s["stop_reasons"]
    assert s["header_only"] == [n(2)]
    assert s["pinned"] == {"hashes_verified": 3, "bound_members_verified": 1,
                           "header_only_declared": 1, "aliases_declared": 1}
    assert s["contents"][n(1)]["members"] == [n(1), n(3)]
    assert set(s["contents"]) == {n(1), n(4), n(5), n(6)}
    groups = {g["kept"]: g for g in s["groups"]}
    assert set(groups) == {n(1), n(5), n(6)}
    a = groups[n(1)]
    assert a["contents"] == [n(1), n(4)] and a["primary"] == n(1) and a["truth_bound"]
    assert a["dropped"] == [{"content": n(4), "relation": "event_subset_partial_final_event"}]
    assert a["pinned_member_kept"] is True
    assert {"event_signature", "primary_file", "primary_records"} <= set(a["evidence"])
    # The census binds the unpinned content exactly; the orphan stays unbound.
    assert groups[n(5)]["primary"] == n(2) and groups[n(5)]["truth_bound"]
    assert groups[n(6)]["primary"] is None and not groups[n(6)]["truth_bound"]
    assert s["contents"][n(6)]["binding"]["state"] == "unresolved"
    assert s["counts"]["kept_events"] == 10 + 10 + 8
    assert s["vertex_spread_mc_truth"]["events"] == 28
    events = pd.read_csv(tmp / "out" / "events.csv.gz")
    assert len(events) == s["indexes"]["events"]["rows"] == 10 + 7 + 10 + 8
    assert sha256(tmp / "out" / "events.csv.gz") == s["indexes"]["events"]["sha256"]


def test_header_only_detection_requires_exact_header_bytes(population):
    tmp, tag, info = population
    raw = Path(info["raw_root"]) / RUN
    # Same 69/88-byte sizes but a different header: not the pinned header-only content.
    text = (raw / "0002_hits.csv").read_text()
    (raw / "0002_hits.csv").write_text(text.replace("fELoss", "fEloss"))
    info["input_sha256"].pop(n(2))
    info["empty_exports"].pop(n(2))
    s = census_tag(info, tag, tmp / "out")
    assert s["status"] == "stopped"
    assert "header" in s["stop_reasons"][0]["detail"].lower() or "schema" in s["stop_reasons"][0]["detail"]


def test_byte_grouping_without_pins(tmp_path):
    tag = "defaultSpot"
    raw, prim = tmp_path / "raw" / tag, tmp_path / "prim" / tag
    h, v, p = synthetic(5, range(4))
    for i in (7, 3, 9):
        write_pair(raw, f"{i:04d}", h, v)
    write_primary(prim, "0001", p)
    s = census_tag(inventory(tmp_path, tag), tag, tmp_path / "out")
    assert s["status"] == "complete"
    assert list(s["contents"]) == [n(3)]
    assert s["contents"][n(3)]["members"] == [n(3), n(7), n(9)]
    assert len(s["groups"]) == 1 and s["groups"][0]["dropped"] == []


def test_non_subset_overlap_stops(population):
    tmp, tag, info = population
    raw = Path(info["raw_root"])
    a_hits, a_vacs, _ = synthetic(1, range(10))
    x_hits, x_vacs, _ = synthetic(9, [20])
    # Shares five events with 0001 and adds, first in the file, one that 0001
    # does not contain (so it cannot pass as a cut final event).
    write_pair(raw, "0007", pd.concat([x_hits, a_hits.loc[a_hits.Row < 5]]),
               pd.concat([x_vacs, a_vacs.loc[a_vacs.Row < 5]]))
    info["all_pair_names"].append(n(7))
    s = census_tag(info, tag, tmp / "out")
    assert s["status"] == "stopped"
    stop = [r for r in s["stop_reasons"] if r["kind"] == "dropped_not_subset"]
    assert stop == [{"kind": "dropped_not_subset", "group": f"{tag}:{n(1)}", "content": n(7),
                     "kept": n(1), "missing_events": 1}]


def test_other_detector_realisation_of_a_bound_source_stops(population):
    tmp, tag, info = population
    raw = Path(info["raw_root"])
    _, a_vacs, _ = synthetic(1, range(10))
    other_hits, _, _ = synthetic(11, range(10))
    write_pair(raw, "0008", other_hits, a_vacs)
    info["all_pair_names"].append(n(8))
    s = census_tag(info, tag, tmp / "out")
    assert s["status"] == "stopped"
    group = next(g for g in s["groups"] if g["kept"] == n(1))
    assert n(8) in group["contents"] and group["evidence"].get("primary_records")
    assert any(r["kind"] == "dropped_not_subset" and r["content"] == n(8) for r in s["stop_reasons"])


def test_ambiguous_binding_is_refused(population):
    tmp, tag, info = population
    prim = Path(info["primary_root"])
    # A second, identical primary file makes every record of 0005 ambiguous.
    (prim / RUN / "0009_prim.csv").write_bytes((prim / RUN / "0002_prim.csv").read_bytes())
    info["common_primary_pair_names"].append(n(9))
    info["primary_sha256"][n(9)] = sha256(prim / RUN / "0009_prim.csv")
    s = census_tag(info, tag, tmp / "out")
    assert s["status"] == "complete", s["stop_reasons"]
    b = s["contents"][n(5)]["binding"]
    assert b["state"] == "unresolved" and b["primary"] is None
    assert b["ambiguous_records"] == b["eligible_records"] == 20
    assert not next(g for g in s["groups"] if g["kept"] == n(5))["truth_bound"]


def test_pinned_binding_and_hash_failures_stop(population):
    tmp, tag, info = population
    info["primary_member_map"][n(1)] = n(2)
    s = census_tag(info, tag, tmp / "a")
    assert s["status"] == "stopped" and "Pinned binding" in s["stop_reasons"][0]["detail"]
    info["primary_member_map"][n(1)] = n(1)
    info["input_sha256"][n(1)]["hits"] = "0" * 64
    s = census_tag(info, tag, tmp / "b")
    assert s["status"] == "stopped" and "pinned hash" in s["stop_reasons"][0]["detail"]


def test_combine_detects_cross_potential_overlap(tmp_path):
    h, v, p = synthetic(21, range(6))
    dirs = {}
    for tag in ("zeroSpot", "bigSpot"):
        write_pair(tmp_path / "raw" / tag, "0001", h, v)
        write_primary(tmp_path / "prim" / tag, "0001", p)
        s = census_tag(inventory(tmp_path, tag), tag, tmp_path / tag, meta={"code_sha256": "x"})
        assert s["status"] == "complete"
        dirs[tag] = tmp_path / tag
    census = combine(dirs, tmp_path / "combined", coverage="smoke")
    assert census["status"] == "stopped"
    x = census["cross_potential"]
    assert x["shared_event_signatures"] == 6 and x["event_pairs_with_min_shared_records"] == 6
    assert x["shared_raw_files"] == 2 and x["shared_primary_files"] == 1
    # Distinct contents per potential pass, and the compact census is written.
    h2, v2, p2 = synthetic(22, range(6))
    write_pair(tmp_path / "raw2" / "bigSpot", "0001", h2, v2)
    write_primary(tmp_path / "prim2" / "bigSpot", "0001", p2)
    info = inventory(tmp_path, "bigSpot")
    info.update(raw_root=str(tmp_path / "raw2" / "bigSpot"), primary_root=str(tmp_path / "prim2" / "bigSpot"),
                primary_sha256={n(1): sha256(tmp_path / "prim2" / "bigSpot" / RUN / "0001_prim.csv")})
    census_tag(info, "bigSpot", tmp_path / "big2", meta={"code_sha256": "x"})
    census = combine({"zeroSpot": dirs["zeroSpot"], "bigSpot": tmp_path / "big2"},
                     tmp_path / "combined2", coverage="smoke")
    assert census["status"] == "complete", census["stop_reasons"]
    stored = json.loads((tmp_path / "combined2" / "census.json").read_text())
    assert stored["tags"]["bigSpot"]["groups"][0]["truth_bound"]
    path = tmp_path / "combined2" / stored["tags"]["bigSpot"]["indexes"]["events"]["path"]
    assert sha256(path) == stored["tags"]["bigSpot"]["indexes"]["events"]["sha256"]
    with gzip.open(path, "rt") as f:
        assert f.readline() == "group,file,Row,records,sha256\n"


def test_projection_scales_measured_rates():
    summary = {"timings": {"hash": {"seconds": 1.0, "bytes": 100}, "contents": {"seconds": 2.0, "contents": 1},
                           "binding": {"seconds": 3.0, "bytes": 30}, "groups": {"seconds": 0.0}}}
    out = project(summary, {"raw_bytes": 1000, "size_groups": 10, "primary_bytes": 300, "primary_files": 2})
    assert out["estimated_seconds"] == pytest.approx(10 + 20 + 30)
    assert out["estimated_seconds_with_margin"] == pytest.approx(1.5 * 60 + 300)


def test_record_hash_ignores_row_and_id_and_sign_of_zero():
    from HGNDRecoGNN.scripts.smash_export_census import record_hashes
    frame = pd.DataFrame({"Row": [0, 7], "Id": [1, 9], "PDG": [2112, 2112], "Ekin": [1.0, 1.0],
                          "Px": [0.0, -0.0], "Py": [0.1, 0.1], "Pz": [2.0, 2.0],
                          "vX": [0.0, 0.0], "vY": [0.0, 0.0], "vZ": [0.0, 0.0]})
    a, b = record_hashes(frame)
    assert a == b and len(a) == 64
    assert a != record_hashes(frame.assign(Ekin=np.nextafter(1.0, 2.0)))[0]
    assert hashlib.sha256(b"x").hexdigest() != a


def test_job_smoke_verify_and_gated_full_run(tmp_path, monkeypatch):
    """The cluster entry point end to end on synthetic roots: smoke, receipt, full task."""
    import tarfile
    from HGNDRecoGNN.scripts import smash_export_census as census

    inv = {"tags": {}}
    for i, tag in enumerate(census.TAGS):
        raw, prim = tmp_path / "raw" / tag, tmp_path / "prim" / tag
        h, v, p = synthetic(30 + i, range(5))
        write_pair(raw, "0001", h, v)
        header_only(raw, "0002")
        write_pair(raw, "0003", h, v)
        write_primary(prim, "0001", p)
        write_primary(prim, "0002", synthetic(40 + i, range(3))[2])
        inv["tags"][tag] = inventory(tmp_path, tag, {n(1): n(1), n(3): n(1)}, empty=[n(2)],
                                     aliases={n(3): n(1)})
    package = tmp_path / "package"
    package.mkdir()
    (package / census.INVENTORY).write_text(json.dumps(inv))
    (package / "SHA256SUMS").write_text(f"{sha256(package / census.INVENTORY)}  {census.INVENTORY}\n")
    monkeypatch.setattr(census, "SMOKE_MEMBERS", {t: (n(1), n(3), n(2)) for t in census.TAGS})
    out = tmp_path / "stage" / "result"
    assert census.main(["run", "--source", str(package), "--mode", "smoke", "--job-id", "7_single",
                        "--output", str(out), "--workers", "2"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["census"]["status"] == "complete"
    assert all(v["projection"]["estimated_seconds"] > 0 for v in summary["tags"].values())
    archive = tmp_path / "result.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out, arcname="result")
    slurm = tmp_path / "slurm.json"
    slurm.write_text(json.dumps({"job_key": "7_single", "state": "COMPLETED", "exit_code": "0:0"}))
    receipt = tmp_path / "smoke_verified.json"
    args = ["verify-export", "--archive", str(archive), "--slurm", str(slurm), "--source", str(package),
            "--mode", "smoke", "--extracted", str(tmp_path / "x"), "--receipt", str(receipt),
            "--cluster-archive", str(archive)]
    with pytest.raises(ValueError, match="differs from the hash"):
        census.main(args + ["--expected-sha256", "0" * 64])
    assert census.main(args + ["--expected-sha256", sha256(archive)]) == 0
    assert json.loads(receipt.read_text())["status"] == "verified"
    # The full task needs the receipt and checks the pinned count of its tag.
    monkeypatch.setattr(census, "PINNED_BOUND", {t: 1 for t in census.TAGS})
    full = ["run", "--source", str(package), "--mode", "full", "--task", "2", "--job-id", "8_2",
            "--smoke-receipt", str(receipt), "--output", str(tmp_path / "full" / "result")]
    assert census.main(full) == 0
    s = json.loads((tmp_path / "full" / "result" / "bigSpot" / "summary.json").read_text())
    assert s["coverage"] == "full" and s["counts"]["truth_bound_groups"] == 1
    receipt.write_text(receipt.read_text().replace('"verified"', '"pending"', 1))
    with pytest.raises(ValueError, match="Invalid smoke receipt"):
        census.main(full[:-1] + [str(tmp_path / "full2" / "result")])
