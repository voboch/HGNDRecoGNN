"""Finite reproduction of the six deposited student UFC training recipes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.data.student_ufc import (  # noqa: E402
    UFCRecipe, build_ufc_event, prepare_observables, signal_hit_targets,
)
from HGNDRecoGNN.models.student_ufc import load_student_checkpoint  # noqa: E402
from HGNDRecoGNN.scripts.student_ufc import sha256  # noqa: E402
from HGNDRecoGNN.training.student_reproduce import (  # noqa: E402
    MODEL_ORDER, SEEDS, StudentTrainConfig, atomic_json, atomic_torch,
    pack_graphs, score_statistics, train, unpack_graphs,
)

RAW_HASH = "183e38f1d22829dc4ed0979b53d82a0db803aa6fd3b5795122fc1c5c664fa4df"
EXPECTED = {"train": (137918, 168108, 45740), "val": (29554, 36269, 9924),
            "test": (29555, 36343, 9857)}


def environment():
    names = ("torch", "torch-geometric", "torch-cluster", "torch-scatter",
             "numpy", "pandas", "scipy", "scikit-learn")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(),
            "packages": versions,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
            "hostname": platform.node()}


def verify_source(path):
    manifest = json.loads(path.read_text())
    root = path.parent
    for relative, expected in manifest["files"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    return sha256(path)


def prepare(args):
    """Rebuild the original population once; training never accesses raw truth."""
    if sha256(args.hits_pickle) != RAW_HASH:
        raise ValueError("Input differs from audited hits_Jan_26.pkl")
    args.output.mkdir(parents=True, exist_ok=False)
    raw = pd.read_pickle(args.hits_pickle)
    recipe = UFCRecipe(time_column="fTime_nonoise")
    observations, rows = prepare_observables(raw, recipe)
    signal, truth = signal_hit_targets(raw)
    energy = truth.Ekin.to_numpy()
    indices = np.arange(len(rows))
    np.random.default_rng(42).shuffle(indices)  # Deposited split, unchanged.
    cut1 = int(.70 * len(rows))
    # Match int(train*n)+int(val*n), which can differ from int(.85*n).
    cut2 = cut1 + int(.15 * len(rows))
    split_of_row = np.zeros(len(rows), dtype=np.int8)
    split_of_row[indices[cut1:cut2]] = 1
    split_of_row[indices[cut2:]] = 2
    names = ("train", "val", "test")
    buffers = {name: [] for name in names}
    parts = {name: [] for name in names}
    counts = {name: {"events": int((split_of_row == i).sum()), "events_with_candidates": 0,
                     "graphs": 0, "positives": 0} for i, name in enumerate(names)}
    started = time.monotonic()

    def flush(name):
        pack = pack_graphs(buffers[name])
        file = f"{name}_{len(parts[name]):04d}.pt"
        atomic_torch(args.output / file, pack)
        parts[name].append({"file": file, "sha256": sha256(args.output / file),
                            "graphs": len(buffers[name]), "nodes": len(pack["x"]),
                            "edges": int(pack["edge_index"].shape[1]),
                            "positives": int(pack["y_cls"].sum())})
        buffers[name].clear()
        del pack
        gc.collect()

    for processed, (row, hits) in enumerate(observations.groupby("Row", sort=True), 1):
        name = names[int(split_of_row[np.searchsorted(rows, row)])]
        graphs = build_ufc_event(hits, recipe, RAW_HASH, spectral=True)
        for graph in graphs:
            hit_ids = graph.hit_ids.numpy()
            mask = signal[hit_ids]
            graph.y_cls = torch.tensor([float(mask.any())])
            graph.y_E = torch.tensor([float(energy[hit_ids][mask].max()) if mask.any() else 0.])
        buffers[name].extend(graphs)
        counts[name]["events_with_candidates"] += int(bool(graphs))
        counts[name]["graphs"] += len(graphs)
        counts[name]["positives"] += sum(int(g.y_cls.item()) for g in graphs)
        # Notebook convention: flush only after a whole event, at >=100000 graphs.
        if len(buffers[name]) >= 100000:
            flush(name)
        if processed % 20000 == 0:
            print(f"Prepared {processed}/{len(rows)} events in {time.monotonic()-started:.1f}s", flush=True)
    for name in names:
        if buffers[name]:
            flush(name)
        value = counts[name]
        if (value["events"], value["graphs"], value["positives"]) != EXPECTED[name]:
            raise ValueError(f"Population mismatch: {name}: {value}")
    if [p["graphs"] for p in parts["train"]] != [100000, 68108]:
        raise ValueError("Original training shard boundary changed")
    split_hashes = {name: hashlib.sha256(rows[split_of_row == i].astype("<i8").tobytes()).hexdigest()
                    for i, name in enumerate(names)}
    manifest = {"schema": "student-ufc-prepared-v1", "raw_sha256": RAW_HASH,
                "recipe": asdict(recipe), "recipe_sha256": recipe.fingerprint(),
                "split_seed": 42, "split_row_sha256": split_hashes,
                "counts": counts, "parts": parts, "environment": environment(),
                "seconds": time.monotonic() - started,
                "scientific_parameter_policy": "Inherited notebook conventions; no tuning in this campaign",
                "test_policy": "No test scoring until all 18 train/validation runs complete"}
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def load_parts(directory, split, spectral, *, limit=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    result = []
    remaining = limit
    for item in manifest["parts"][split]:
        path = directory / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Prepared part changed: {path}")
        packed = torch.load(path, map_location="cpu", weights_only=True)
        if remaining is not None:
            # Limit engineering samples only, never the full campaign.
            count = min(remaining, len(packed["row"]))
            packed = {k: v[:int(packed["node_ptr"][count])] if k in ("x", "pe")
                      else v[:, :int(packed["edge_ptr"][count])] if k == "edge_index"
                      else v[:count+1] if k in ("node_ptr", "edge_ptr") else v[:count]
                      for k, v in packed.items()}
            remaining -= count
        result.append(unpack_graphs(packed, spectral))
        if remaining == 0:
            break
    return result


def run_name(model, seed):
    return f"{model}_seed{seed}"


def require_complete(campaign, data_hash, source_hash):
    """Test scoring is a separate phase, barred by any missing/incomplete seed."""
    complete = []
    for seed in SEEDS:
        for model in MODEL_ORDER:
            directory = campaign / "runs" / run_name(model, seed)
            receipt = json.loads((directory / "complete.json").read_text())
            identity = json.loads((directory / "identity.json").read_text())
            history = json.loads((directory / "history.json").read_text())
            status = json.loads((directory / "status.json").read_text())
            config = asdict(StudentTrainConfig(model, seed))
            if (receipt["state"] != "complete" or identity["config"] != config
                    or identity["data_sha256"] != data_hash or identity["source_sha256"] != source_hash
                    or receipt["epochs"] != len(history)
                    or [r["epoch"] for r in history] != list(range(1, len(history)+1))
                    or status["state"] != "complete" or status["epoch"] != len(history)
                    or not (len(history) == config["max_epochs"] or status["patience"] >= config["patience"])
                    or receipt["best_sha256"] != sha256(directory / "best.pt")
                    or receipt["resume_sha256"] != sha256(directory / "resume.pt")
                    or receipt["history_sha256"] != sha256(directory / "history.json")):
                raise ValueError(f"Incomplete/inconsistent seed: {directory}")
            complete.append((model, seed, directory))
    return complete


def training(args, source_hash):
    if not torch.cuda.is_available():
        raise RuntimeError("Campaign training requires the Slurm GPU allocation")
    seed, model = SEEDS[args.task // len(MODEL_ORDER)], MODEL_ORDER[args.task % len(MODEL_ORDER)]
    output = args.campaign / "runs" / run_name(model, seed)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Existing run: {output}; use coherent explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "environment.json", environment())
    spectral = model.startswith("spectral_")
    data_hash = sha256(args.prepared / "manifest.json")
    train_parts = load_parts(args.prepared, "train", spectral)
    val_parts = load_parts(args.prepared, "val", spectral)
    config = StudentTrainConfig(model, seed)
    try:
        _, history, finished = train(config, train_parts, val_parts, output,
                                     data_hash=data_hash, source_hash=source_hash,
                                     resume=args.resume)
        if not finished:
            raise RuntimeError("Trainer returned before terminal early stopping/max_epochs")
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        atomic_json(output / "complete.json", {"state": "complete", "model": model, "seed": seed,
                    "epochs": len(history), "best_epoch": best["epoch"], "validation": best["validation"],
                    "best_sha256": sha256(output / "best.pt"),
                    "resume_sha256": sha256(output / "resume.pt"),
                    "history_sha256": sha256(output / "history.json"),
                    "data_sha256": data_hash, "source_sha256": source_hash})
    except BaseException as exc:
        # Failure reporting cannot turn a failed Slurm task into success.
        try:
            atomic_json(output / "failure.json", {"state": "failed", "type": type(exc).__name__,
                        "error": str(exc), "traceback": traceback.format_exc()})
        except OSError as reporting:
            print(f"Failure receipt also failed: {reporting}", file=sys.stderr, flush=True)
        raise


def smoke(args, source_hash):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke received no CUDA device")
    args.output.mkdir(parents=True, exist_ok=False)
    env = environment()
    if not env["packages"]["torch"].startswith("2.4.0") or env["packages"]["torch-geometric"] != "2.7.0":
        raise ValueError("Smoke environment differs from original torch2.4.0/PyG2.7.0")
    data_hash = sha256(args.prepared / "manifest.json")
    result = {"state": "running", "environment": env, "source_sha256": source_hash,
              "data_sha256": data_hash, "scope": "engineering train/validation only, no quality claim",
              "models": {}}
    for model in MODEL_ORDER:
        spectral = model.startswith("spectral_")
        # Fixed small training subset tests the real optimiser/save/resume path.
        train_parts = load_parts(args.prepared, "train", spectral, limit=2048)
        val_parts = load_parts(args.prepared, "val", spectral, limit=512)
        config = StudentTrainConfig(model, 42)
        output = args.output / model
        _, history, finished = train(config, train_parts, val_parts, output,
                                    data_hash=data_hash, source_hash=source_hash, stop_after_epochs=1)
        assert not finished and len(history) == 1
        restored, _ = load_student_checkpoint(output / "best.pt", model, "cuda")
        batch = next(iter(DataLoader(val_parts[0], batch_size=256))).to("cuda")
        with torch.no_grad():
            assert all(torch.isfinite(t).all() for t in restored(batch))
        _, resumed, finished = train(config, train_parts, val_parts, output,
                                    data_hash=data_hash, source_hash=source_hash,
                                    resume=True, stop_after_epochs=1)
        assert not finished and len(resumed) == 2
        result["models"][model] = {"epochs": 2, "atomic_resume_verified": True,
                                    "checkpoint_reload_verified": True,
                                    "epoch_seconds": [r["seconds"] for r in resumed]}
        atomic_json(args.output / "smoke.json", result)
        del restored, batch, train_parts, val_parts
        gc.collect()
        torch.cuda.empty_cache()
    result["state"] = "passed"
    atomic_json(args.output / "smoke.json", result)


def evaluate(args, source_hash):
    data_hash = sha256(args.prepared / "manifest.json")
    completed = require_complete(args.campaign, data_hash, source_hash)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for model_key, seed, directory in completed:
        parts = load_parts(args.prepared, "test", model_key.startswith("spectral_"))
        model, checkpoint = load_student_checkpoint(directory / "best.pt", model_key, "cuda")
        scores, labels, energies, rows, clusters = [], [], [], [], []
        with torch.no_grad():
            for graphs in parts:
                for batch in DataLoader(graphs, batch_size=256, shuffle=False, num_workers=0):
                    gpu = batch.to("cuda")
                    logits, prediction = model(gpu)
                    scores.append(logits.sigmoid().cpu().numpy())
                    energies.append(prediction.cpu().numpy())
                    labels.append(gpu.y_cls.cpu().numpy().astype(int))
                    rows.append(gpu.row.cpu().numpy())
                    clusters.append(gpu.cluster.cpu().numpy())
        y, p = np.concatenate(labels), np.concatenate(scores)
        result = {"model": model_key, "seed": seed, "graphs": len(y), "positives": int(y.sum()),
                  "best_epoch": checkpoint["epoch"], **score_statistics(y, p),
                  "checkpoint_sha256": sha256(directory / "best.pt")}
        if (len(y), int(y.sum())) != EXPECTED["test"][1:]:
            raise ValueError("Test prediction population differs from fixed original split")
        frame = pd.DataFrame({"row": np.concatenate(rows), "cluster": np.concatenate(clusters),
                              "true_y": y, "score": p, "pred_E": np.concatenate(energies)})
        if frame.duplicated(["row", "cluster"]).any():
            raise ValueError("Duplicate test candidates")
        file = args.output / f"{run_name(model_key, seed)}.csv"
        frame.to_csv(file, index=False, float_format="%.9g")
        result["predictions_sha256"] = sha256(file)
        results.append(result)
        atomic_json(args.output / "per_seed.json", results)
        del parts, model
        gc.collect()
        torch.cuda.empty_cache()
    summary = {}
    for key in MODEL_ORDER:
        values = [r["roc_auc"] for r in results if r["model"] == key]
        summary[key] = {"seeds": list(SEEDS), "roc_auc_mean": float(np.mean(values)),
                        "roc_auc_sd": float(np.std(values, ddof=1)), "n_seeds": len(values)}
    atomic_json(args.output / "summary.json", {"state": "complete", "per_model": summary,
                "test_scope": "Original student event split in hits_Jan_26.pkl; not source-run holdout",
                "data_sha256": data_hash, "source_sha256": source_hash, "environment": environment()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--hits-pickle", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for command in ("smoke", "train", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--prepared", type=Path, required=True)
        p.add_argument("--source-manifest", type=Path, required=True)
        if command in ("train", "evaluate"):
            p.add_argument("--campaign", type=Path, required=True)
        if command in ("smoke", "evaluate"):
            p.add_argument("--output", type=Path, required=True)
        if command == "train":
            p.add_argument("--task", type=int, choices=range(18), required=True)
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "prepare":
        prepare(args)
    else:
        source_hash = verify_source(args.source_manifest)
        frozen = json.loads(args.source_manifest.read_text())
        if sha256(args.prepared / "manifest.json") != frozen["prepared_sha256"]:
            raise ValueError("Prepared manifest differs from the campaign's frozen source receipt")
        {"train": training, "smoke": smoke, "evaluate": evaluate}[args.command](args, source_hash)


if __name__ == "__main__":
    main()
