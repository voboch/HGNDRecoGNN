"""Run the separate student UFC inference branch or audit exported per-seed CSVs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from HGNDRecoGNN.data.student_ufc import (  # noqa: E402
    UFCRecipe, attach_targets, build_ufc_event, prepare_observables, truth_inventory,
)
from HGNDRecoGNN.models.student_ufc import MODEL_CLASSES, load_student_checkpoint  # noqa: E402
from HGNDRecoGNN.training.student_ufc import (  # noqa: E402
    candidate_metrics, evaluate_events, predict_candidates, seed_metrics,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def infer(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Original checkpoint is missing: {args.checkpoint}")
    if args.limit_events is not None and args.limit_events < 1:
        raise ValueError("limit-events must be positive")
    model, checkpoint = load_student_checkpoint(args.checkpoint, args.model, args.device)
    recipe = UFCRecipe(time_column=args.time_column)
    raw = pd.read_pickle(args.hits_pickle)
    observables, event_rows = prepare_observables(raw, recipe)
    # This is the student's inherited event shuffle, not a source-run holdout.
    indices = np.arange(len(event_rows))
    np.random.default_rng(args.split_seed).shuffle(indices)
    train_end = int(.70 * len(indices))
    val_end = train_end + int(.15 * len(indices))
    partitions = {"train": indices[:train_end], "val": indices[train_end:val_end],
                  "test": indices[val_end:], "all": indices}
    chosen = np.sort(event_rows[partitions[args.split]])
    if args.limit_events is not None:
        chosen = chosen[:args.limit_events]
    source_id = args.source_id or sha256(args.hits_pickle)
    observables = observables[observables.Row.isin(chosen)]
    event_keys = [(source_id, int(row)) for row in chosen]
    spectral = args.model.startswith("spectral_")

    def graphs():
        for _, hits in observables.groupby("Row", sort=True):
            group = build_ufc_event(hits, recipe, source_id, spectral=spectral)
            if args.with_truth:
                attach_targets(group, raw)
            yield from group

    predictions = predict_candidates(model, graphs(), batch_size=args.batch_size, device=args.device)
    predictions["model_key"] = args.model
    predictions["seed"] = args.seed
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    pd.DataFrame(event_keys, columns=["source_id", "row"]).to_csv(args.output / "events.csv", index=False)
    if args.limit_events is None:
        metrics = candidate_metrics(predictions)
    else:
        metrics = {"candidates": len(predictions), "roc_auc": None, "pr_auc": None,
                   "scope": "truncated engineering smoke; no quality claim"}
    metrics["scored_on"] = f"student {args.split} event partition of supplied input"
    if args.threshold is not None:
        if not args.with_truth:
            raise ValueError("Event metrics need --with-truth and its eligible-object inventory")
        truth = truth_inventory(observables, raw, source_id, args.truth_energy_min)
        events, event_metrics = evaluate_events(
            predictions, event_keys, truth_objects=truth, threshold=args.threshold,
            reco_energy_min=args.reco_energy_min, topk=args.topk)
        events.to_csv(args.output / "event_metrics.csv", index=False)
        metrics["eligible_object_event_metrics"] = event_metrics
    write_json(args.output / "metrics.json", metrics)
    config = {key: value for key, value in checkpoint.items()
              if isinstance(value, (str, int, float, bool, type(None)))}
    modules = [Path(__file__), Path(__file__).resolve().parents[1] / "data/student_ufc.py",
               Path(__file__).resolve().parents[1] / "models/student_ufc.py",
               Path(__file__).resolve().parents[1] / "training/student_ufc.py"]
    write_json(args.output / "manifest.json", {
        "model_key": args.model, "seed_label_supplied_by_user": args.seed,
        "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_config": config,
        "input_sha256": sha256(args.hits_pickle), "recipe": asdict(recipe),
        "recipe_sha256": recipe.fingerprint(), "source_id": source_id,
        "split": args.split, "split_seed": args.split_seed, "events": len(chosen),
        "event_inventory_sha256": sha256(args.output / "events.csv"),
        "predictions_sha256": sha256(args.output / "predictions.csv"),
        "partial": args.limit_events is not None, "trained_here": False,
        "execution_source_sha256": {str(path): sha256(path) for path in modules},
        "provenance_limit": "Original checkpoint/data/split lineage needs external receipts",
    })


def csv_metrics(args):
    frames = []
    for path in args.predictions:
        frame = pd.read_csv(path)
        if "source_id" not in frame:
            if not args.source_id:
                raise ValueError("CSV lacks source_id; supply the actual --source-id")
            frame["source_id"] = args.source_id
        frames.append(frame)
    result = seed_metrics(pd.concat(frames, ignore_index=True), expected_seeds=args.expected_seeds)
    result["inputs_sha256"] = {str(path): sha256(path) for path in args.predictions}
    result["provenance_limit"] = "CSV arithmetic verified; checkpoint/split lineage not established"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("infer")
    run.add_argument("--model", choices=sorted(MODEL_CLASSES), required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--hits-pickle", type=Path, required=True)
    run.add_argument("--time-column", choices=("fTime_nonoise", "fTime"), required=True)
    run.add_argument("--split", choices=("all", "train", "val", "test"), required=True)
    run.add_argument("--seed", type=int, required=True, help="Recorded training seed label")
    run.add_argument("--split-seed", type=int, default=42, help="Original student event split seed")
    run.add_argument("--source-id")
    run.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    run.add_argument("--batch-size", type=int, default=256, help="Inherited computational batch size")
    run.add_argument("--limit-events", type=int, help="Engineering smoke only; suppresses AUC")
    run.add_argument("--with-truth", action="store_true")
    run.add_argument("--threshold", type=float, help="Frozen operating point; never selected here")
    run.add_argument("--truth-energy-min", type=float, default=.1, help="Student convention, GeV")
    run.add_argument("--reco-energy-min", type=float, default=.1, help="Student convention, GeV")
    run.add_argument("--topk", type=int, default=3, help="Student multiplicity convention")
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(func=infer)
    csv = subs.add_parser("csv-metrics")
    csv.add_argument("--predictions", type=Path, nargs="+", required=True)
    csv.add_argument("--source-id")
    csv.add_argument("--expected-seeds", type=int, nargs="+", default=[42, 7, 21])
    csv.add_argument("--output", type=Path, required=True)
    csv.set_defaults(func=csv_metrics)
    args = parser.parse_args()
    args.func(args)
    print(args.output)


if __name__ == "__main__":
    main()
