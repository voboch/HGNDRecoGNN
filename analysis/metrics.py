"""Reusable evaluation metrics.

Extracted from `notebooks/results_smash.ipynb`. Each function is pure —
takes DataFrames or arrays and returns arrays / DataFrames — so it can
be called from a notebook, a CLI, or a SLURM aggregate job identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)


# ── Classifier curves ───────────────────────────────────────────────────

@dataclass
class ClassifierCurves:
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    roc_auc: float
    pr_precision: np.ndarray
    pr_recall: np.ndarray
    pr_ap: float


def classifier_curves(scores: np.ndarray, labels: np.ndarray) -> ClassifierCurves:
    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)
    return ClassifierCurves(
        roc_fpr=fpr, roc_tpr=tpr, roc_auc=roc_auc_score(labels, scores),
        pr_precision=prec, pr_recall=rec,
        pr_ap=average_precision_score(labels, scores),
    )


# ── Efficiency / purity vs threshold ────────────────────────────────────

def efficiency_purity_vs_threshold(
    clusters_df: pd.DataFrame,
    thresholds: Iterable[float] = np.linspace(0.05, 0.95, 19),
    label_col: str = 'cl_label',
    score_col: str = 'cl_score',
) -> pd.DataFrame:
    """Global (dataset-wide) efficiency, purity, fake fraction vs threshold.

    Returns one row per threshold with columns
    `threshold, tp, fp, fn, purity, efficiency, fake_frac, f1`.
    """
    y = clusters_df[label_col].values.astype(int)
    s = clusters_df[score_col].values.astype(float)
    total_pos = int(y.sum())

    rows = []
    for t in thresholds:
        pred = s > t
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        purity     = tp / max(tp + fp, 1)
        efficiency = tp / max(total_pos, 1)
        fake_frac  = fp / max(tp + fp, 1)
        f1         = 2 * purity * efficiency / max(purity + efficiency, 1e-12)
        rows.append({
            'threshold': float(t), 'tp': tp, 'fp': fp, 'fn': fn,
            'purity': purity, 'efficiency': efficiency,
            'fake_frac': fake_frac, 'f1': f1,
        })
    return pd.DataFrame(rows)


# ── Per-event detection efficiency ──────────────────────────────────────

def per_event_detection_efficiency(
    clusters_df: pd.DataFrame,
    threshold: float = 0.3,
    label_col: str = 'cl_label',
    score_col: str = 'cl_score',
    row_col: str = 'Row',
) -> float:
    """Fraction of true-neutron events with ≥1 predicted cluster above threshold.

    Copies the definition from results_smash.ipynb cell 40.
    """
    true_events = clusters_df.loc[clusters_df[label_col] == 1, row_col].unique()
    if len(true_events) == 0:
        return 0.0
    hits = clusters_df[(clusters_df[score_col] > threshold)
                       & (clusters_df[row_col].isin(true_events))]
    detected = hits[row_col].nunique()
    return detected / len(true_events)


# ── Energy resolution ───────────────────────────────────────────────────

def energy_resolution(
    clusters_df: pd.DataFrame,
    ekin_bins: np.ndarray,
    score_threshold: float = 0.5,
    label_col: str = 'cl_label',
    score_col: str = 'cl_score',
    epred_col: str = 'e_pred',
    etrue_col: str = 'e_true',
) -> pd.DataFrame:
    """Per-Ekin-bin bias and σ of `(e_pred - e_true) / e_true`.

    Restricted to signal clusters (`cl_label == 1`) above `score_threshold`.
    """
    df = clusters_df[(clusters_df[label_col] == 1)
                     & (clusters_df[score_col] > score_threshold)
                     & (clusters_df[etrue_col] > 0)].copy()
    df['dErel'] = (df[epred_col] - df[etrue_col]) / df[etrue_col]
    df['ekin_bin'] = pd.cut(df[etrue_col], bins=ekin_bins, right=False)

    grouped = df.groupby('ekin_bin', observed=True)['dErel']
    out = grouped.agg(['count', 'mean', 'std']).reset_index()
    out['ekin_lo'] = np.array([float(b.left)  for b in out['ekin_bin']])
    out['ekin_hi'] = np.array([float(b.right) for b in out['ekin_bin']])
    out['ekin_mid'] = 0.5 * (out['ekin_lo'] + out['ekin_hi'])
    return out.drop(columns=['ekin_bin'])[
        ['ekin_lo', 'ekin_hi', 'ekin_mid', 'count', 'mean', 'std']
    ]


# ── Multiplicity confusion ──────────────────────────────────────────────

def multiplicity_confusion(
    clusters_df: pd.DataFrame,
    threshold: float,
    label_col: str = 'cl_label',
    score_col: str = 'cl_score',
    row_col: str = 'Row',
) -> pd.DataFrame:
    """N_reco vs N_true multiplicity per event.

    Returns a long-format DataFrame with `n_true`, `n_reco`, `count` columns
    that plots cleanly as a heatmap.
    """
    per_evt = (clusters_df.assign(pred=(clusters_df[score_col] > threshold))
               .groupby(row_col)
               .agg(n_true=(label_col, 'sum'), n_reco=('pred', 'sum'))
               .astype(int).reset_index())
    conf = (per_evt.groupby(['n_true', 'n_reco']).size()
            .rename('count').reset_index())
    return conf
