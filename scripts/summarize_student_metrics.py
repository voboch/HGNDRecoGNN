"""Summarise held-out candidate metrics for the user-requested non-GAT models.

Uses the frozen test predictions. Energy errors use the original training
target on MC-positive candidates with target E>0, without a classifier cut.
No operating threshold, new event selection or pooled-seed AUC is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import sklearn
import torch

MODELS = {
    "baseline_graphsage": "GraphSAGE",
    "spectral_gated_graphsage": "Spectral-gated GraphSAGE",
    "baseline_dynamicedgeconv": "DynamicEdgeConv",
    "spectral_gated_dynamicedgeconv": "Spectral-gated DynamicEdgeConv",
}
SEEDS = (42, 7, 21)
METRICS = ("roc_auc", "pr_auc", "energy_mae_GeV", "energy_rmse_GeV", "energy_bias_GeV")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_metrics(frame, truth):
    keys = ["row", "cluster"]
    if frame.duplicated(keys).any() or truth.duplicated(keys).any():
        raise ValueError("Duplicate candidate identities")
    joined = truth.merge(frame, on=keys, how="outer", suffixes=("_target", "_csv"),
                         validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all() or not joined.true_y_target.equals(joined.true_y_csv):
        raise ValueError("Prediction candidate population or labels differ from frozen test targets")
    y = joined.true_y_target.to_numpy(dtype=int)
    # Nine significant CSV digits round-trip float32 model outputs exactly.
    score = joined.score.to_numpy(dtype=np.float32).astype(np.float64)
    prediction = joined.pred_E.to_numpy(dtype=np.float32).astype(np.float64)
    target = joined.true_E.to_numpy(dtype=np.float64)
    if (not np.isfinite(score).all() or not np.isfinite(prediction).all()
            or not np.isfinite(target).all() or not np.isin(y, (0, 1)).all()
            or not ((score >= 0) & (score <= 1)).all() or (prediction < 0).any()):
        raise ValueError("Invalid score, predicted energy or target")
    # Match the original regression-loss population, not a test-optimised cut.
    signal = (y == 1) & (target > 0)
    error = prediction[signal] - target[signal]
    if len(np.unique(y)) != 2 or not len(error):
        raise ValueError("Both classes and positive regression targets are required")
    return {"roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score)),
            "energy_mae_GeV": float(np.mean(np.abs(error))),
            "energy_rmse_GeV": float(np.sqrt(np.mean(error**2))),
            "energy_bias_GeV": float(np.mean(error)),
            "candidates": len(y), "positives": int(y.sum()),
            "regression_candidates": int(signal.sum())}


def summarise(rows):
    result = {}
    for model in MODELS:
        subset = [r for r in rows if r["model"] == model]
        if len(subset) != len(SEEDS) or {r["seed"] for r in subset} != set(SEEDS):
            raise ValueError(f"Incomplete or repeated seeds: {model}")
        result[model] = {metric: {"mean": float(np.mean([r[metric] for r in subset])),
                                 "seed_sd": float(np.std([r[metric] for r in subset], ddof=1))}
                         for metric in METRICS}
    return result


def render_report(result):
    def value(model, metric, scale=1, digits=5):
        stat = result["summary"][model][metric]
        return f"{stat['mean']*scale:.{digits}f} ± {stat['seed_sd']*scale:.{digits}f}"

    lines = ["# Тестовые метрики студенческих моделей без GATv2 — 2026-10-20", "",
             "<!-- table-provenance: Scored on held-out student test events; "
             "weights fitted on 137918 train events, checkpoints selected on 29554 validation events; "
             "same frozen Jan26 input and preprocessing for all rows. -->", "",
             "Обучение и оценка завершены. Продолжение 4423612 и оценка 4423613: "
             "Slurm COMPLETED 0:0. Все 18 runs имеют полные receipts; хеши "
             "best/resume/history и 18 CSV проверены. По запросу пользователя "
             "здесь исключены baseline и spectral-gated GATv2; остаются 12 обученных replicas.", "",
             "Scored on: held-out student event split of hits_Jan_26.pkl. "
             "Fitted on 137918 train events; checkpoint selected on 29554 validation events. "
             "Test: 29555 events, 24811 with UFC candidates, 36343 candidates, 9857 MC-positive. "
             "No source-run holdout is established; the input has no source-run manifest.", "",
             "Reconstructed candidate scores against MC truth labels. Higher is better. "
             "Each entry is mean ± sample SD over training seeds 42/7/21, n=3; "
             "SD is not SE, a confidence interval or between-source uncertainty. "
             "PR-AUC denotes average precision. No seed pooling or threshold selection.", "",
             "| Модель | Test ROC-AUC | Test PR-AUC (AP) |",
             "|---|---:|---:|"]
    lines += [f"| {label} | {value(key, 'roc_auc')} | {value(key, 'pr_auc')} |"
              for key, label in MODELS.items()]
    lines += ["", "Reconstructed energy against MC target on 9857 MC-positive test candidates "
              "with target E>0, without a classifier score cut. Lower is better for MAE/RMSE; "
              "absolute bias closer to zero is better. Same mean ± training-seed SD, n=3.", "",
              "| Модель | Energy MAE (MeV) | Energy RMSE (MeV) | Energy bias (MeV) |",
              "|---|---:|---:|---:|"]
    lines += [f"| {label} | {value(key, 'energy_mae_GeV', 1000, 1)} | "
              f"{value(key, 'energy_rmse_GeV', 1000, 1)} | {value(key, 'energy_bias_GeV', 1000, 1)} |"
              for key, label in MODELS.items()]
    lines += ["", "Энергетическая мишень — максимум истинной кинетической энергии "
              "подходящих neutron hits внутри UFC-компоненты, как при обучении. "
              "Эти ошибки не являются разрешением энергии уникального нейтрона или "
              "ошибкой восстановленного физического спектра. Эффективность обнаружения "
              "отдельных нейтронов здесь не измеряется.", "",
              "Параметры preprocessing унаследованы из тетрадок: fTime_nonoise без "
              "дополнительного размытия, прежние UFC и MC labels. Эти метрики не "
              "переносятся напрямую на другой HGND/DCM benchmark.", "",
              "Канонический источник чисел: "
              "results/student_ufc_reproduce_20261019/no_gatv2_metrics_20261020.json. "
              "Скрипт: scripts/summarize_student_metrics.py. Полный архив сохраняет "
              "результаты всех шести моделей; их обучение и исходные предсказания не изменены.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    upstream = args.export / "test_evaluation"
    evaluation = json.loads((upstream / "summary.json").read_text())
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    if evaluation["state"] != "complete" or sha256(args.prepared / "manifest.json") != evaluation["data_sha256"]:
        raise ValueError("Upstream evaluation or frozen preparation differs")
    pieces = []
    for item in manifest["parts"]["test"]:
        path = args.prepared / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError("Frozen test shard changed")
        packed = torch.load(path, map_location="cpu", weights_only=True)
        pieces.append(pd.DataFrame({"row": packed["row"].numpy(), "cluster": packed["cluster"].numpy(),
                                    "true_y": packed["y_cls"].numpy().astype(int),
                                    "true_E": packed["y_E"].numpy()}))
    truth = pd.concat(pieces, ignore_index=True)
    expected = manifest["counts"]["test"]
    if len(truth) != expected["graphs"] or int(truth.true_y.sum()) != expected["positives"]:
        raise ValueError("Frozen test population count mismatch")
    original = json.loads((upstream / "per_seed.json").read_text())
    rows, inputs = [], {}
    max_delta = {"roc_auc": 0., "pr_auc": 0.}
    for row in original:
        if row["model"] not in MODELS:
            continue
        name = f"{row['model']}_seed{row['seed']}"
        path = upstream / (name + ".csv")
        if sha256(path) != row["predictions_sha256"]:
            raise ValueError("Predictions changed")
        receipt_path = args.export / "runs" / name / "complete.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt["state"] != "complete" or receipt["best_sha256"] != row["checkpoint_sha256"]:
            raise ValueError("Incomplete checkpoint lineage")
        metrics = candidate_metrics(pd.read_csv(path), truth)
        for metric in max_delta:
            delta = abs(metrics[metric] - row[metric])
            max_delta[metric] = max(max_delta[metric], delta)
            # Numerical reconciliation only; not an acceptance cut on model quality.
            if delta > 1e-12:
                raise ValueError(f"Metric differs from original evaluation: {name}/{metric}")
        rows.append({"model": row["model"], "seed": row["seed"], "best_epoch": row["best_epoch"],
                     "checkpoint_sha256": row["checkpoint_sha256"], **metrics})
        inputs[name] = {"prediction_sha256": sha256(path), "complete_receipt_sha256": sha256(receipt_path)}
    result = {"state": "complete", "selected_models": list(MODELS),
              "excluded_models": ["baseline_gatv2", "spectral_gated_gatv2"],
              "selection_reason": "Explicit user request after training; no retraining or checkpoint reselection",
              "seeds": list(SEEDS), "test_population": expected, "per_seed": rows,
              "summary": summarise(rows), "classification_threshold": None,
              "uncertainty": "sample SD across three training seeds; fixed test population, no pooled scores",
              "energy_population": "MC-positive UFC candidates with original target E>0; no score cut",
              "energy_target": "maximum qualifying MC neutron-hit Ekin in each component",
              "max_recomputed_upstream_metric_delta": max_delta,
              "provenance": {"source_sha256": evaluation["source_sha256"],
                             "prepared_sha256": evaluation["data_sha256"],
                             "evaluation_summary_sha256": sha256(upstream / "summary.json"),
                             "evaluation_per_seed_sha256": sha256(upstream / "per_seed.json"),
                             "script_sha256": sha256(__file__), "inputs": inputs},
              "analysis_environment": {"numpy": np.__version__, "pandas": pd.__version__,
                                       "sklearn": sklearn.__version__, "torch": torch.__version__}}
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.report.write_text(render_report(result))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
