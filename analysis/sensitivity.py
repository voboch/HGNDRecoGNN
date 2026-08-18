"""Cross-dataset sensitivity study: neutron yield vs Ekin under different
symmetry potentials (zeroSpot / defaultSpot / bigSpot ↔ 0 / 18 / 90 MeV).

The observable the user cares about is `N_true_n(Ekin; dataset)`, obtained
by applying the per-dataset reconstruction efficiency correction to the
raw reco yield:

    N_true_n(E; d) = N_reco_n(E; d) / ε_n(E; d)

Comparing this curve across `d ∈ {zero, default, big}` shows whether the
pipeline is sensitive to the symmetry potential (curves separate) or
robust to it (curves overlap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .efficiency import epsilon_neutron_vs_ekin, wilson_interval


DATASET_TO_POTENTIAL_MEV = {
    'zeroSpot':    0.0,
    'defaultSpot': 18.0,
    'bigSpot':     90.0,
}


@dataclass
class DatasetRun:
    """One dataset's prediction bundle."""
    name: str
    clusters_df: pd.DataFrame
    label_col: str = 'cl_label'
    score_col: str = 'cl_score'
    epred_col: str = 'e_pred'
    etrue_col: str = 'e_true'

    @property
    def potential_mev(self) -> Optional[float]:
        return DATASET_TO_POTENTIAL_MEV.get(self.name)


def reco_yield_vs_ekin(
    run: DatasetRun,
    ekin_bins: np.ndarray,
    threshold: float = 0.5,
    bin_by: str = 'e_pred',
) -> pd.DataFrame:
    """N_reco_n(Ekin) — count of predicted-signal clusters per Ekin bin.

    `bin_by`: which energy column to bin on — `e_pred` (default, measured)
    or `e_true` (Monte-Carlo, ideal — useful for closure tests).
    """
    pred = run.clusters_df[run.clusters_df[run.score_col] > threshold].copy()
    energy = pred[bin_by]
    pred = pred[energy > 0]
    if len(pred) == 0:
        # Emit an empty frame with the expected columns.
        return pd.DataFrame(columns=['ekin_lo', 'ekin_hi', 'ekin_mid', 'n_reco'])
    pred['ekin_bin'] = pd.cut(pred[bin_by], bins=ekin_bins, right=False,
                              include_lowest=True)
    counts = pred.groupby('ekin_bin', observed=True).size().rename('n_reco')
    out = counts.reset_index()
    out['ekin_lo'] = np.array([float(b.left)  for b in out['ekin_bin']])
    out['ekin_hi'] = np.array([float(b.right) for b in out['ekin_bin']])
    out['ekin_mid'] = 0.5 * (out['ekin_lo'] + out['ekin_hi'])
    return out.drop(columns=['ekin_bin'])[
        ['ekin_lo', 'ekin_hi', 'ekin_mid', 'n_reco']
    ]


def solve_true_yield(
    run: DatasetRun,
    ekin_bins: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return the efficiency-corrected true neutron yield per Ekin bin.

    Merges `reco_yield_vs_ekin` (from measured `e_pred`) with
    `epsilon_neutron_vs_ekin` (from MC truth `e_true`). Bin edges are
    identical so the merge is safe on `ekin_lo, ekin_hi`.
    """
    reco = reco_yield_vs_ekin(run, ekin_bins, threshold=threshold, bin_by='e_pred')
    eff  = epsilon_neutron_vs_ekin(
        run.clusters_df, ekin_bins, threshold=threshold,
        label_col=run.label_col, score_col=run.score_col,
        etrue_col=run.etrue_col,
    )
    df = eff.merge(reco[['ekin_lo', 'ekin_hi', 'n_reco']],
                   on=['ekin_lo', 'ekin_hi'], how='left').fillna({'n_reco': 0})
    df['n_reco'] = df['n_reco'].astype(int)

    eps = df['epsilon'].to_numpy()
    df['n_true_solved'] = np.divide(df['n_reco'], eps,
                                    out=np.zeros_like(eps),
                                    where=eps > 0)

    # Propagate binomial-corrected uncertainty on ε to the solved yield.
    # δN_solved / N_solved ≈ δε / ε   (Poisson on N_reco is added in quadrature)
    n_reco = df['n_reco'].to_numpy(dtype=float)
    poisson_rel = np.divide(np.sqrt(n_reco), np.maximum(n_reco, 1e-12),
                            out=np.zeros_like(n_reco), where=n_reco > 0)
    eps_lo, eps_hi = df['epsilon_lo'].to_numpy(), df['epsilon_hi'].to_numpy()
    eps_rel = 0.5 * (eps_hi - eps_lo) / np.maximum(eps, 1e-12)
    combined_rel = np.sqrt(poisson_rel ** 2 + eps_rel ** 2)
    df['n_true_solved_err'] = df['n_true_solved'] * combined_rel
    return df


def compare_datasets(
    runs: list[DatasetRun],
    ekin_bins: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Compute solved neutron yield for every dataset. Returns a dict
    `{run.name: solved_df}` plus a `combined` frame with dataset id +
    potential appended for direct plotting.
    """
    out: dict[str, pd.DataFrame] = {}
    combined_rows = []
    for run in runs:
        df = solve_true_yield(run, ekin_bins, threshold=threshold)
        df['dataset'] = run.name
        df['potential_mev'] = run.potential_mev
        out[run.name] = df
        combined_rows.append(df)
    out['combined'] = pd.concat(combined_rows, ignore_index=True)
    return out


def default_ekin_bins() -> np.ndarray:
    """Standard log-spaced Ekin bins in GeV — from ~50 MeV to 3 GeV."""
    return np.geomspace(0.05, 3.0, 21)
