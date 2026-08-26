"""Reconstruction efficiency vs kinetic energy.

Estimates ε_n(Ekin, threshold) from a per-cluster prediction DataFrame.
Two binning bases are supported:

  * ``ekin_col='e_pred'`` (default) → ε binned by *predicted* energy,
        ε_n(E_pred, t) = P(cl_score > t | cl_label == 1, e_pred in bin)
    This basis is **required** by the closure identity
        N_true(E_pred) = N_reco(E_pred) / ε(E_pred)
    because N_reco is itself binned by e_pred: any resolution smearing
    then cancels in numerator and denominator instead of dropping the
    denominator's clusters into the wrong bin.

  * ``ekin_col='e_true'`` → ε binned by MC-truth energy,
        ε_n(E_true, t) = P(cl_score > t | cl_label == 1, e_true in bin)
    This is the "physical" efficiency curve used for detector-level
    physics interpretation. It is **not** consistent with a
    ``N_reco(E_pred) / ε`` correction and should not be used to solve
    the true yield; use it only for the eff-vs-Ekin performance plot.

The uncertainty is a Wilson binomial interval so the analysis remains
well-behaved even in low-statistics bins near the acceptance edge.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for binomial success rate. Returns (lo, hi)."""
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(k), where=(n > 0))
    z2 = z * z
    denom = 1.0 + z2 / np.maximum(n, 1)
    centre = (p + z2 / (2.0 * np.maximum(n, 1))) / denom
    half = z * np.sqrt(np.divide(p * (1.0 - p), np.maximum(n, 1),
                                 out=np.zeros_like(p), where=(n > 0))
                       + z2 / (4.0 * np.maximum(n, 1) ** 2)) / denom
    lo = np.clip(centre - half, 0.0, 1.0)
    hi = np.clip(centre + half, 0.0, 1.0)
    return lo, hi


def epsilon_neutron_vs_ekin(
    clusters_df: pd.DataFrame,
    ekin_bins: np.ndarray,
    threshold: float = 0.5,
    label_col: str = 'cl_label',
    score_col: str = 'cl_score',
    etrue_col: str = 'e_true',
    ekin_col: str = 'e_pred',
) -> pd.DataFrame:
    """ε_n(Ekin) at a fixed score threshold.

    Signal clusters (`cl_label == 1`, `e_true > 0`) are binned by
    ``ekin_col`` — default ``'e_pred'`` (the closure-consistent basis
    used by ``solve_true_yield``).  Pass ``ekin_col='e_true'`` when a
    physics-truth efficiency curve is desired instead.

    Returns a DataFrame with `ekin_lo, ekin_hi, ekin_mid, n_true, n_pass,
    epsilon, epsilon_lo, epsilon_hi` — one row per bin.
    """
    signal = clusters_df[(clusters_df[label_col] == 1)
                         & (clusters_df[etrue_col] > 0)
                         & (clusters_df[ekin_col] > 0)].copy()
    signal['ekin_bin'] = pd.cut(signal[ekin_col], bins=ekin_bins, right=False,
                                include_lowest=True)
    signal['pass'] = signal[score_col] > threshold

    grouped = signal.groupby('ekin_bin', observed=True)
    n_true = grouped.size().astype(int)
    n_pass = grouped['pass'].sum().astype(int)

    out = pd.DataFrame({'n_true': n_true, 'n_pass': n_pass}).reset_index()
    # pd.cut returns Categorical of Interval; extract endpoints as floats
    # to keep downstream arithmetic simple.
    out['ekin_lo'] = np.array([float(b.left)  for b in out['ekin_bin']])
    out['ekin_hi'] = np.array([float(b.right) for b in out['ekin_bin']])
    out['ekin_mid'] = 0.5 * (out['ekin_lo'] + out['ekin_hi'])

    denom = out['n_true'].to_numpy(dtype=float)
    out['epsilon'] = np.divide(out['n_pass'], denom,
                               out=np.zeros_like(denom), where=denom > 0)
    lo, hi = wilson_interval(out['n_pass'].to_numpy(), out['n_true'].to_numpy())
    out['epsilon_lo'] = lo
    out['epsilon_hi'] = hi
    return out.drop(columns=['ekin_bin'])[
        ['ekin_lo', 'ekin_hi', 'ekin_mid',
         'n_true', 'n_pass', 'epsilon', 'epsilon_lo', 'epsilon_hi']
    ]


def epsilon_vs_ekin_scan(
    clusters_df: pd.DataFrame,
    ekin_bins: np.ndarray,
    thresholds: Iterable[float] = (0.3, 0.5, 0.7),
    **kwargs,
) -> pd.DataFrame:
    """Return per-bin ε for a list of thresholds (long format)."""
    frames = []
    for t in thresholds:
        df = epsilon_neutron_vs_ekin(clusters_df, ekin_bins, threshold=t, **kwargs)
        df['threshold'] = float(t)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
