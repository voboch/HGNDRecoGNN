"""Candidate export and event aggregation for the separate student UFC branch."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch_geometric.data import Batch


PREDICTION_COLUMNS = ["source_id", "row", "cluster", "score", "pred_E", "n_hits"]


@torch.no_grad()
def predict_candidates(model, graphs, *, batch_size, device="cpu"):
    """Stream candidate graphs, retaining event IDs outside minibatch arithmetic."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = model.to(device).eval()
    records = []

    def flush(group):
        batch = Batch.from_data_list(group).to(device)
        logits, energy = model(batch)
        scores = logits.sigmoid().detach().cpu().numpy()
        energies = energy.detach().cpu().numpy()
        if scores.shape != (len(group),) or energies.shape != (len(group),):
            raise ValueError("Expected one score and energy per candidate")
        for graph, score, predicted_energy in zip(group, scores, energies):
            record = {"source_id": graph.source_id, "row": int(graph.row.item()),
                      "cluster": int(graph.cluster.item()), "score": float(score),
                      "pred_E": float(predicted_energy), "n_hits": graph.num_nodes}
            if "y_cls" in graph:
                record.update(true_y=int(graph.y_cls.item()), true_E=float(graph.y_E.item()),
                              signal_ids=json.dumps(graph.signal_ids.tolist()))
            records.append(record)

    pending = []
    for graph in graphs:
        pending.append(graph)
        if len(pending) == batch_size:
            flush(pending)
            pending = []
    if pending:
        flush(pending)
    result = pd.DataFrame(records) if records else pd.DataFrame(columns=PREDICTION_COLUMNS)
    return validate_predictions(result)


def validate_predictions(frame):
    required = set(PREDICTION_COLUMNS) - {"n_hits"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    result = frame.copy()
    if result.source_id.isna().any() or result.source_id.astype(str).str.len().eq(0).any():
        raise ValueError("source_id is required")
    result["source_id"] = result.source_id.astype(str)
    for key in ("row", "cluster"):
        values = pd.to_numeric(result[key], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{key} must contain finite integers")
        result[key] = values.astype(np.int64)
    for key in ("score", "pred_E"):
        values = pd.to_numeric(result[key], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{key} must be finite")
        result[key] = values
    if not result.score.between(0, 1).all() or (result.pred_E < 0).any():
        raise ValueError("Scores must be probabilities and energy non-negative")
    if "true_y" in result:
        values = pd.to_numeric(result.true_y, errors="coerce")
        if not values.isin([0, 1]).all():
            raise ValueError("true_y must be binary")
        result["true_y"] = values.astype(int)
    keys = [key for key in ("model_key", "seed", "source_id", "row", "cluster") if key in result]
    if result.duplicated(keys).any():
        raise ValueError("Duplicate candidate keys within a model/seed population")
    return result


def candidate_metrics(predictions):
    """ROC/PR metrics for one model and seed, with no threshold selection."""
    frame = validate_predictions(predictions)
    for key in ("model_key", "seed"):
        if key in frame and frame[key].nunique() > 1:
            raise ValueError("Evaluate each model and seed separately")
    if "true_y" not in frame:
        return {"candidates": len(frame), "roc_auc": None, "pr_auc": None,
                "scope": "unlabelled candidate inference"}
    both = frame.true_y.nunique() == 2
    return {"candidates": len(frame), "positives": int(frame.true_y.sum()),
            "roc_auc": float(roc_auc_score(frame.true_y, frame.score)) if both else None,
            "pr_auc": float(average_precision_score(frame.true_y, frame.score)) if both else None,
            "scope": "supplied MC-labelled candidate population"}


def seed_metrics(predictions, expected_seeds=(42, 7, 21)):
    """Recompute complete-seed metrics; never substitute a pooled AUC."""
    frame = validate_predictions(predictions)
    if "model_key" not in frame or "seed" not in frame or "true_y" not in frame:
        raise ValueError("Per-seed evaluation needs model_key, seed and true_y")
    numeric = pd.to_numeric(frame.seed, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Seeds must be finite integers")
    frame["seed"] = numeric.astype(int)
    # Candidate identities and labels must agree across all compared weights.
    populations = []
    rows = []
    for (model, seed), group in frame.groupby(["model_key", "seed"], sort=True):
        population = group.set_index(["source_id", "row", "cluster"]).true_y.sort_index()
        if populations and not population.equals(populations[0]):
            raise ValueError("Different candidate identities or labels across model/seed populations")
        populations.append(population)
        rows.append({"model_key": model, "seed": int(seed), **candidate_metrics(group)})
    summary = []
    for model in sorted(frame.model_key.unique()):
        metrics = [r for r in rows if r["model_key"] == model]
        present = {r["seed"] for r in metrics}
        complete = present == set(expected_seeds) and all(r["roc_auc"] is not None for r in metrics)
        aucs = [r["roc_auc"] for r in metrics] if complete else []
        summary.append({"model_key": model, "complete_expected_seeds": complete,
                        "expected_seeds": list(expected_seeds), "present_seeds": sorted(present),
                        "auc_mean": float(np.mean(aucs)) if aucs else None,
                        "auc_seed_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None})
    return {"per_seed": rows, "summary": summary,
            "uncertainty": "sample SD over complete supplied training seeds; no source-generation uncertainty"}


def _signal_ids(value):
    values = json.loads(value) if isinstance(value, str) else value
    return {int(v) for v in values}


def match_unique_objects(selected, truth_ids):
    """Maximum-cardinality candidate/object matching, including merged candidates."""
    truth = sorted(set(truth_ids))
    if not len(selected) or not truth:
        return 0
    connections = np.array([[obj in _signal_ids(value) for obj in truth]
                            for value in selected.signal_ids], dtype=bool)
    rows, columns = linear_sum_assignment((~connections).astype(np.int8))
    return int(connections[rows, columns].sum())


def evaluate_events(predictions, event_keys, *, truth_objects, threshold,
                    reco_energy_min, topk):
    """Aggregate once per full event, including events with no candidate.

    Threshold/energy/top-k are explicit operating settings, not tuned here.
    truth_objects must be an independently retained event inventory, not labels
    reconstructed only from candidates that survived clustering.
    """
    if not 0 <= threshold <= 1 or reco_energy_min < 0 or topk < 1:
        raise ValueError("Invalid operating settings")
    frame = validate_predictions(predictions)
    for key in ("model_key", "seed"):
        if key in frame and frame[key].nunique() > 1:
            raise ValueError("Event aggregation needs one model and seed")
    keys = [(str(source), int(row)) for source, row in event_keys]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate event inventory keys")
    allowed = set(keys)
    supplied = set(zip(frame.source_id, frame.row))
    if not supplied <= allowed or not set(truth_objects) <= allowed:
        raise ValueError("Candidate/truth events absent from explicit event inventory")
    if len(frame) and "signal_ids" not in frame:
        raise ValueError("Unique-object metrics need candidate truth-object memberships")
    groups = dict(tuple(frame.groupby(["source_id", "row"], sort=False)))
    rows = []
    for key in keys:
        candidates = groups.get(key, frame.iloc[:0])
        selected = candidates[(candidates.score >= threshold) & (candidates.pred_E > reco_energy_min)]
        selected = selected.sort_values(["score", "cluster"], ascending=[False, True]).head(topk)
        truth = set(truth_objects.get(key, []))
        matched = match_unique_objects(selected, truth)
        true_candidates = sum(bool(_signal_ids(ids) & truth) for ids in selected.signal_ids) if len(selected) else 0
        rows.append({"source_id": key[0], "row": key[1], "n_candidates": len(candidates),
                     "n_true_objects": len(truth), "n_selected": len(selected),
                     "n_matched_unique": matched, "n_selected_fake": len(selected) - true_candidates,
                     "n_selected_duplicate": true_candidates - matched,
                     "event_has_true": bool(truth), "event_recalled": bool(matched)})
    events = pd.DataFrame(rows)
    totals = {column: int(events[column].sum()) if len(events) else 0 for column in (
        "n_true_objects", "n_selected", "n_matched_unique", "n_selected_fake", "n_selected_duplicate",
        "event_has_true", "event_recalled")}
    efficiency = totals["event_recalled"] / totals["event_has_true"] if totals["event_has_true"] else None
    object_efficiency = totals["n_matched_unique"] / totals["n_true_objects"] if totals["n_true_objects"] else None
    return events, {"events": len(keys), **totals, "event_efficiency": efficiency,
                    "eligible_object_efficiency": object_efficiency,
                    "truth_scope": "explicit supplied eligible objects; not all generated neutrons",
                    "threshold": threshold, "reco_energy_min_GeV": reco_energy_min, "topk": topk}
