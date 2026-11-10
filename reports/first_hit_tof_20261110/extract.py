"""Recover the original and network-input time on the earliest retained hit."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from time import monotonic

import numpy as np
import pandas as pd
import torch


def sha(path):
    with Path(path).open("rb") as f:
        digest = hashlib.sha256()
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
        return digest.hexdigest()


def digest(keys):
    return hashlib.sha256(np.ascontiguousarray(keys, dtype="<i8").tobytes()).hexdigest()


def extract_packs(plan, work):
    from HGNDRecoGNN.scripts.smash_train_prepare import stage

    archive = work / "packs.tar"
    stage(Path(plan["archive"]), archive, plan["archive_sha256"])
    assert archive.stat().st_size == plan["archive_bytes"]
    expected = {r["pack"]["member"]: r["pack"] for r in plan["files"]}
    assert len(expected) == 286
    seen = set()
    with tarfile.open(archive, "r:") as source:
        for member in source:
            if member.name not in expected:
                continue
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and ".." not in name.parts
            assert member.name not in seen
            target = work / "packs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as inp, target.open("xb") as out:
                shutil.copyfileobj(inp, out)
            pin = expected[member.name]
            assert (
                target.stat().st_size == pin["bytes"] and sha(target) == pin["sha256"]
            )
            seen.add(member.name)
    assert seen == expected.keys()


def extract_hits(record, cluster_selection, work):
    from HGNDRecoGNN.scripts import audit_smash_sources as audit
    from HGNDRecoGNN.scripts.smash_train_prepare import stage
    from HGNDRecoGNN.data.smash_transfer import _ordered
    from HGNDRecoGNN.data.graph_dataset import _dfToF

    cid = record["content"]
    private = work / "files" / cid
    private.mkdir(parents=True)
    raw = {}
    for kind in ("hits", "vacs"):
        raw[kind] = private / (kind + ".csv")
        source = Path(record["raw_root"]) / (record["member"] + "_" + kind + ".csv")
        stage(source, raw[kind], record["sha256"][kind])
    hf = audit.read_csv(raw["hits"], audit.HITS)
    vf = audit.read_csv(raw["vacs"], audit.VACS)
    _, observed, _ = audit.observe(hf, vf, record["member"] + "_hits.csv")
    columns = [c for c in observed.columns if c != "_merge"]
    replay = private / (Path(record["member"]).name + ".pt")
    torch.save(
        {
            "columns": columns,
            "values": torch.tensor(observed[columns].to_numpy(np.float64)),
        },
        replay,
    )
    assert sha(replay) == record["observation"]["sha256"], cid
    observed, row, side = _ordered(observed)
    pack = torch.load(
        work / "packs" / record["pack"]["member"], map_location="cpu", weights_only=True
    )
    a = {k: v.numpy() for k, v in pack.items()}
    nodes = np.diff(a["node_ptr"])
    clusters = np.diff(a["cluster_ptr"])
    ncl = len(a["label"])
    assert np.array_equal(row, np.repeat(a["row"], nodes))
    assert np.array_equal(side, np.repeat(a["side"], nodes))
    cluster = a["cluster"] + np.repeat(a["cluster_ptr"][:-1], nodes)
    assert ((cluster >= 0) & (cluster < ncl)).all()
    keys = np.column_stack(
        [
            np.repeat(a["row"], clusters),
            np.arange(ncl) - np.repeat(a["cluster_ptr"][:-1], clusters),
            np.repeat(a["side"], clusters),
        ]
    )
    assert digest(keys) == record["keys_sha256"]
    assert digest(keys[cluster_selection]) == record["isolated_keys_sha256"]
    assert len(cluster_selection) == record["isolated_count"]
    raw_keys = pd.MultiIndex.from_frame(hf[["Row", "Instance"]])
    idx = raw_keys.get_indexer(pd.MultiIndex.from_frame(observed[["Row", "Instance"]]))
    assert (idx >= 0).all()
    raw_time = hf.fTime.to_numpy()[idx]
    observed_time = observed.fTime.to_numpy()
    length = np.linalg.norm(observed[["fX", "fY", "fZ"]].to_numpy() / 100, axis=1)
    energy = observed.Ekin.to_numpy()
    prompt = (
        (np.abs(observed.PDG.to_numpy()) == 2112)
        & observed.Side.isin([5, 6]).to_numpy()
        & (energy >= _dfToF(length, observed_time + 0.3))
        & (energy <= _dfToF(length, observed_time - 0.3))
    )
    native = prompt & (energy > 0.3)
    labels, targets = np.zeros(ncl, a["label"].dtype), np.zeros(ncl)
    np.maximum.at(labels, cluster, native.astype(labels.dtype))
    np.maximum.at(targets, cluster, np.where(native, energy, 0))
    assert labels.tobytes() == a["label"].tobytes()
    assert targets.astype(a["energy"].dtype).tobytes() == a["energy"].tobytes()
    # Hit selection uses detector observables only. Instance resolves time ties.
    order = np.lexsort((observed.Instance.to_numpy(), raw_time, cluster))
    unique, offsets = np.unique(cluster[order], return_index=True)
    assert np.array_equal(unique, np.arange(ncl))
    first = order[offsets][cluster_selection]
    values = np.column_stack([raw_time[first], observed_time[first], length[first]])
    assert np.isfinite(values).all()
    shutil.rmtree(private)
    return cid, values, len(observed), ncl


def main(args):
    started = monotonic()
    plan = json.loads(args.plan.read_text())
    assert (
        sha(Path(plan["old_source"]) / "SHA256SUMS") == plan["old_source_sums_sha256"]
    )
    sys.path.insert(0, plan["old_source"])
    args.work.mkdir()
    args.output.mkdir()
    extract_packs(plan, args.work)
    rows, nodes, clusters = {}, 0, 0
    with (
        np.load(args.selection) as selection,
        ProcessPoolExecutor(max_workers=2) as pool,
    ):
        assert set(selection.files) == {r["content"] for r in plan["files"]}
        tasks = iter(plan["files"])

        def launch(r):
            return pool.submit(extract_hits, r, selection[r["content"]], args.work)

        active = {launch(next(tasks)) for _ in range(2)}
        while active:
            done, active = wait(active, return_when=FIRST_COMPLETED)
            batch = [future.result() for future in done]
            for cid, data, n, k in batch:
                assert cid not in rows
                rows[cid] = data
                nodes += n
                clusters += k
                if len(rows) % 25 == 0:
                    print(f"Replayed {len(rows)}/286 files", flush=True)
            for _ in batch:
                record = next(tasks, None)
                if record is not None:
                    active.add(launch(record))
    assert (
        len(rows) == 286 and sum(map(len, rows.values())) == plan["expected_isolated"]
    )
    assert nodes == plan["expected_hits"] and clusters == plan["expected_clusters"]
    output = args.output / "first_hits.npz"
    np.savez_compressed(output, **rows)
    summary = {
        "state": "complete",
        "files": len(rows),
        "isolated": sum(map(len, rows.values())),
        "nodes": nodes,
        "clusters": clusters,
        "seconds": monotonic() - started,
        "columns": ["original_time_ns", "jittered_time_ns", "nominal_length_m"],
        "selection_sha256": sha(args.selection),
        "plan_sha256": sha(args.plan),
        "extractor_sha256": sha(__file__),
        "first_hits_sha256": sha(output),
        "model_calls": 0,
        "native_replay": "exact",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("plan", "selection", "work", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
