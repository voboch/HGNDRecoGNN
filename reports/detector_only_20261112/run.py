"""Compare truth-free reconstruction with the pinned SMASH legacy path."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from HGNDRecoGNN.data.smash_transfer import DETECTOR_COLUMNS, detector_observations, detector_pack
from HGNDRecoGNN.data.student_pipeline import DetectorPackDataset, NativePackDataset
from HGNDRecoGNN.scripts import audit_smash_sources as audit
from HGNDRecoGNN.scripts.smash_train import TASKS, build_model, epoch_checkpoint_name, load_epoch
from HGNDRecoGNN.scripts.smash_train_prepare import (
    IDENTITY_A, IDENTITY_B, SIDES, extract_members, sha, stage, write_json,
)
from HGNDRecoGNN.reports.rule_tof_benchmark_20261110.rule import reconstruct

STATE = {}


def checked_json(path, expected):
    assert sha(path) == expected, path
    return json.loads(path.read_text())


def detector_bytes(frame):
    return np.ascontiguousarray(frame.loc[:, list(DETECTOR_COLUMNS)], dtype="<f8").tobytes()


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def exact(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert left.contiguous().numpy().tobytes() == right.contiguous().numpy().tobytes()


def candidate_keys(pack):
    ptr = pack["cluster_ptr"].numpy()
    counts = np.diff(ptr)
    return np.column_stack([
        np.repeat(pack["row"].numpy(), counts),
        np.arange(ptr[-1]) - np.repeat(ptr[:-1], counts),
        np.repeat(pack["side"].numpy(), counts),
    ]).astype("<i8", copy=False)


@torch.no_grad()
def predict(model, dataset):
    output = [model(graph) for graph in DataLoader(dataset, batch_size=256, shuffle=False)]
    return tuple(torch.cat(values) for values in zip(*output, strict=True))


def witness_models(row, new, old):
    fold = row["fold"]
    checks = {}
    for seed_text, pin in STATE["plan"]["folds"][str(fold)]["runs"].items():
        seed = int(seed_text)
        assert TASKS[pin["task"]] == (fold, seed)
        receipt = checked_json(Path(pin["receipt"]), pin["receipt_sha256"])
        name = "train/complete.json"
        epochs = {key: pin["selection"][key] for key in ("e_cls", "e_en")}
        members = {key: "train/checkpoints/" + epoch_checkpoint_name(epoch)
                   for key, epoch in epochs.items()}
        paths = extract_members(receipt["bundle_dir"], receipt["manifest_sha256"],
                                [name, *set(members.values())],
                                STATE["work"] / "train" / str(pin["task"]))
        complete = checked_json(paths[name], pin["complete_sha256"])
        assert complete["state"] == "complete" and complete["evaluable"] and not complete["test_opened"]
        assert (complete["fold"], complete["seed"]) == (fold, seed)
        assert {key: complete["selection"][key] for key in pin["selection"]} == pin["selection"]
        assert complete["identity"]["split_sha256"] == STATE["plan"]["split_sha256"]
        assert complete["identity"]["prepare_manifest_sha256"] == STATE["plan"]["prepare"]["manifest_sha256"]
        detector, legacy = DetectorPackDataset(new), NativePackDataset(old)
        _, model, _, _ = build_model(seed, detector, "cpu")
        result = {"complete_sha256": pin["complete_sha256"], "checkpoints": {}, "cut": pin["cut"]}
        matched = {}
        for index, (key, epoch) in enumerate(epochs.items()):
            member = members[key]
            expected = complete["files"][member.removeprefix("train/")]
            load_epoch(model, paths[member].parent, epoch, expected)
            model.eval()
            a, b = predict(model, detector)[index], predict(model, legacy)[index]
            assert torch.isfinite(a).all() and a.numel() == int(new["cluster_ptr"][-1])
            exact(a, b)
            matched[key] = a
            result["checkpoints"][key] = {"epoch": epoch, "sha256": expected,
                                         "output_sha256": digest(a.numpy())}
        selected = matched["e_cls"].sigmoid() > pin["cut"]
        energy = matched["e_en"]
        result["selected"] = int(selected.sum())
        result["low_band"] = int((selected & (energy >= .3) & (energy < 1)).sum())
        result["high_band_open"] = int((selected & (energy >= 2)).sum())
        result["high_band_within_reach"] = int((selected & (energy >= 2) & (energy < 4)).sum())
        checks[seed_text] = result
    return checks


def witness_controls(raw, truth, detector, key):
    hit = detector.iloc[0]
    mask = (truth.Row == hit.Row) & (truth.Instance == hit.Instance)
    assert mask.sum() == 1
    control = truth.copy()
    control.loc[mask, "PDG"] = 0
    _, observed, dropped = audit.observe(raw, control, key)
    affected = (detector.Row == hit.Row) & (detector.fY.gt(0) == (hit.fY > 0))
    assert dropped == affected.sum() > 0
    assert detector_bytes(observed) == detector_bytes(detector.loc[~affected])
    contaminated = raw.assign(PDG=0, Id=-1, Ekin=np.nan)
    assert detector_bytes(detector_observations(contaminated, np.random.default_rng(audit.noise_seed(key)))) == detector_bytes(detector)
    return {"injected_pdg0_half_event_rejected_hits": dropped, "extra_truth_columns_ignored": True}


def analyze(row, witness=False):
    cid, fold = row["content"], row["fold"]
    private = STATE["work"] / "raw" / cid
    key = row["member"] + "_hits.csv"
    hits, vacs = private / "hits.csv", private / "vacs.csv"
    stage(Path(row["raw_root"]) / key, hits, row["sha256"]["hits"])
    raw = audit.read_csv(hits, audit.HITS)
    detector = detector_observations(raw, np.random.default_rng(audit.noise_seed(key)))
    new = detector_pack(detector, STATE["scalers"][fold], IDENTITY_A, IDENTITY_B, STATE["plan"]["geometry"])
    rules = reconstruct(detector)

    # The legacy oracle is only opened after producing detector-only observations and candidates.
    stage(Path(row["raw_root"]) / (row["member"] + "_vacs.csv"), vacs, row["sha256"]["vacs"])
    truth = audit.read_csv(vacs, audit.VACS)
    _, legacy, dropped = audit.observe(raw, truth, key)
    obs = row["observation"]
    assert dropped == obs["rejected_pdg0_hits"] == 0, cid
    assert len(raw) == obs["exported_hits"] and len(truth) == obs["exported_truth_records"], cid
    assert detector_bytes(detector) == detector_bytes(legacy), cid
    columns = [column for column in legacy if column != "_merge"]
    replay = private / (Path(row["member"]).name + ".pt")
    torch.save({"columns": columns, "values": torch.tensor(legacy[columns].to_numpy(np.float64))}, replay)
    assert sha(replay) == obs["sha256"], cid

    for left, right in zip(rules, reconstruct(legacy), strict=True):
        np.testing.assert_array_equal(left, right)
    pack = row["pack"]
    path = STATE["packs"][pack["member"]]
    assert path.stat().st_size == pack["bytes"] and sha(path) == pack["sha256"], cid
    old = torch.load(path, map_location="cpu", weights_only=True)
    assert set(new) == set(old) - {"label", "energy"}, cid
    for field in new:
        exact(new[field], old[field])
    assert digest(candidate_keys(new)) == row["keys_sha256"], cid
    events = int(detector.Row.nunique())
    span = row["processed_span"]
    assert detector.Row.min() >= 0 and detector.Row.max() + 1 == span
    counts = {"events_with_hits": events, "events_without_hits": span - events,
              "hits": len(detector), "graphs": len(new["row"]),
              "edges": new["edges"].shape[1], "clusters": int(new["cluster_ptr"][-1])}
    assert counts["hits"] == obs["observed_hits"]
    assert events == row["population"]["physical_events"]
    for field in ("graphs", "edges", "clusters"):
        assert counts[field] == row["population"][field], (cid, field)
    result = {"content": cid, "fold": fold, "processed_span": span, "counts": counts, "legacy_observation_sha256": obs["sha256"],
              "detector_fields_sha256": hashlib.sha256(detector_bytes(detector)).hexdigest(),
              "pack_sha256": pack["sha256"], "candidate_keys_sha256": row["keys_sha256"],
              "rules_counts": rules[-1].tolist(), "rules_output_sha256": [digest(values) for values in rules]}
    if witness:
        result["control"] = witness_controls(raw, truth, detector, key)
        result["models"] = witness_models(row, new, old)
    shutil.rmtree(private)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "work", "output", "source-sums"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.plan.read_text())
    prep = plan["prepare"]
    files = plan["files"]
    witnesses = {fold["witness"] for fold in plan["folds"].values()}
    assert len(files) == len({r["content"] for r in files}) == 286
    assert len(witnesses) == 3 and all(r["part"] in (0, 2, 4) for r in files)
    bundle = Path(prep["bundle_dir"])
    manifest = checked_json(bundle / "manifest.json", prep["manifest_sha256"])
    assert manifest["archive_sha256"] == prep["archive_sha256"]
    scalers = {int(i): fold["scaler"] for i, fold in plan["folds"].items()}
    members = [r["pack"]["member"] for r in files] + [r["member"] for r in scalers.values()]
    paths = extract_members(bundle, prep["manifest_sha256"], members, args.work / "packs")
    STATE.update(plan=plan, work=args.work, packs=paths, scalers={})
    for fold, pin in scalers.items():
        path = paths[pin["member"]]
        assert sha(path) == pin["sha256"]
        with path.open("rb") as stream:
            fitted = pickle.load(stream)
        STATE["scalers"][fold] = [fitted[side] for side in SIDES]
    args.output.mkdir(parents=True)
    rows = []
    for record in files:
        if record["content"] in witnesses:
            rows.append(analyze(record, witness=True))
            write_json(args.output / "witnesses.json", rows)
            print("Witness accepted", record["content"], flush=True)
    with ProcessPoolExecutor(max_workers=plan["caps"]["workers"],
                             mp_context=multiprocessing.get_context("fork")) as pool:
        for result in pool.map(analyze, (r for r in files if r["content"] not in witnesses)):
            rows.append(result)
            if len(rows) % 25 == 0:
                print("Accepted file count", len(rows), flush=True)
    rows.sort(key=lambda row: row["content"])
    totals = {field: sum(row["counts"][field] for row in rows) for field in rows[0]["counts"]}
    assert totals["hits"] == 5833940 and totals["clusters"] == 1798225
    write_json(args.output / "files.json", rows)
    write_json(args.output / "summary.json", {
        "state": "complete", "slurm_job": os.environ["SLURM_JOB_ID"],
        "plan_sha256": sha(args.plan), "source_sums_sha256": sha(args.source_sums),
        "prepare_manifest_sha256": prep["manifest_sha256"], "files": len(rows), "totals": totals,
        "witnesses": len(witnesses), "fold_seed_forward_comparisons": sum(len(r.get("models", {})) for r in rows),
        "legacy_scope": "Historical even SMASH files only",
    })


if __name__ == "__main__":
    main()
