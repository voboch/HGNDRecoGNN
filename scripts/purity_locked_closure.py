"""Per-dataset purity-locked closure test.

Motivates: after the binning-basis fix, defaultSpot still has a
~35 pp closure deficit vs zeroSpot / bigSpot at t = 0.5. One
candidate is that a fixed classifier threshold picks *different*
physical purity across datasets because their cluster-signal
prevalence differs (defaultSpot 28.6% positive vs zero/big ~22%).

This script re-computes N_reco, N_true and closure at the per-dataset
threshold that yields a locked global signal purity pi (default 0.7,
matching the Morozov 2024 / Dombay 2026 convention), for each of the
three SMASH datasets. If closure equalises after the lock, the
deficit was operating-point conflation; if it does not, the deficit
is a real physical or model-training effect.

Usage
-----
    conda activate pyg
    python -m HGNDRecoGNN.scripts.purity_locked_closure \\
        --dataset zeroSpot=results/sensitivity_hpc_pooled_smoke/zeroSpot_hpc/pred_clusters_hpc_pooled.pkl \\
        --dataset defaultSpot=results/sensitivity_hpc_pooled_smoke/defaultSpot_hpc/pred_clusters_hpc_pooled.pkl \\
        --dataset bigSpot=results/sensitivity_hpc_pooled_smoke/bigSpot_hpc/pred_clusters_hpc_pooled.pkl \\
        --dataset-cache-root notebooks/cache \\
        --purity 0.7 \\
        --out-dir results/sensitivity_hpc_pooled_smoke_purityLocked
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def _add_package_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.abspath(os.path.join(here, '..', '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def _discover_parquet(cache_root: str, name: str) -> str | None:
    for suffix in ('_smoke', ''):
        pat = os.path.join(cache_root,
                           f'ndet_dataset_smash_{name}{suffix}',
                           'processed', '_hits_cache_*.parquet')
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def _mc_truth_per_neutron(parquet_path, row_filter, bins):
    cols = ['Row', 'Id', 'n0_label', 'Ekin']
    df = pd.read_parquet(parquet_path, columns=cols)
    if row_filter is not None:
        df = df[df['Row'].isin(row_filter)]
    n_events = int(df['Row'].nunique())
    mc_e = df[df['n0_label'] == 1].groupby(['Row', 'Id'])['Ekin'].mean()
    counts, _ = np.histogram(mc_e.to_numpy(), bins=bins)
    return counts, n_events


def _find_threshold_for_purity(clusters_df, target_purity: float,
                               n_scan: int = 401) -> tuple[float, float, float]:
    """Return (t*, actual_purity, cluster_efficiency) via a bisection-free
    fine scan on [0, 1]. Picks the smallest threshold that reaches
    ``target_purity`` — matching a permissive-then-tighten protocol."""
    y = clusters_df['cl_label'].to_numpy(dtype=int)
    s = clusters_df['cl_score'].to_numpy(dtype=float)
    total_pos = int(y.sum())
    if total_pos == 0:
        return 0.5, 0.0, 0.0
    thresholds = np.linspace(0.001, 0.999, n_scan)
    best_t, best_p, best_e = float('nan'), 0.0, 0.0
    for t in thresholds:
        pred = s > t
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        pur = tp / max(tp + fp, 1)
        eff = tp / total_pos
        if pur >= target_purity:
            best_t, best_p, best_e = float(t), float(pur), float(eff)
            break
    if np.isnan(best_t):
        # No threshold hits the target — return the max-purity operating point.
        # Useful diagnostic (means the classifier is too weak on this dataset).
        best_t = 0.999
        pred = s > best_t
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        best_p = tp / max(tp + fp, 1)
        best_e = tp / total_pos
    return best_t, best_p, best_e


def main() -> int:
    p = argparse.ArgumentParser(description='Per-dataset purity-locked closure.')
    p.add_argument('--dataset', action='append', required=True,
                   help='name=path — pickled cluster DataFrame. Repeat per dataset.')
    p.add_argument('--dataset-cache-root', default=None,
                   help='Parent of ndet_dataset_smash_<name>/ dirs — enables '
                        'per-neutron MC-truth from parquet.')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--purity', type=float, default=0.7,
                   help='Target signal purity to lock (default 0.7, '
                        'matching Morozov 2024 / Dombay 2026 convention).')
    p.add_argument('--reference-threshold', type=float, default=0.5,
                   help='Fixed threshold used as the comparison baseline.')
    args = p.parse_args()

    _add_package_to_path()
    from HGNDRecoGNN.analysis.sensitivity import (
        DatasetRun, compare_datasets, default_ekin_bins,
    )

    cluster_dfs: dict[str, pd.DataFrame] = {}
    for spec in args.dataset:
        name, path = spec.split('=', 1)
        cluster_dfs[name.strip()] = pd.read_pickle(path)

    bins = default_ekin_bins()
    os.makedirs(args.out_dir, exist_ok=True)

    # Pull MC-truth totals from parquet caches (if provided).
    mc_totals: dict[str, tuple[int, int]] = {}
    if args.dataset_cache_root:
        for name, df in cluster_dfs.items():
            parquet = _discover_parquet(args.dataset_cache_root, name)
            if parquet is None:
                print(f'  MC parquet: {name:15s}  NOT FOUND', file=sys.stderr)
                continue
            counts, ne = _mc_truth_per_neutron(parquet, set(df['Row'].unique()), bins)
            mc_totals[name] = (int(counts.sum()), int(ne))

    rows = []
    for name, df in cluster_dfs.items():
        # Reference: fixed t = --reference-threshold
        runs = [DatasetRun(name=name, clusters_df=df)]
        ref = compare_datasets(runs, bins, threshold=args.reference_threshold,
                               efficiency_basis='e_pred')[name]
        ref_n_reco = int(ref['n_reco'].sum())
        ref_n_true = float(ref['n_true_solved'].sum())

        # Locked: t = t*(purity)
        t_star, pi_actual, eff_at_t = _find_threshold_for_purity(df, args.purity)
        loc = compare_datasets(runs, bins, threshold=t_star,
                               efficiency_basis='e_pred')[name]
        loc_n_reco = int(loc['n_reco'].sum())
        loc_n_true = float(loc['n_true_solved'].sum())

        n_mc, n_ev = mc_totals.get(name, (int(df['cl_label'].sum()),
                                          int(df['Row'].nunique())))

        rows.append({
            'dataset':               name,
            'N_events':              n_ev,
            'N_MC_truth':            n_mc,
            'N_MC/ev':               round(n_mc / max(n_ev, 1), 3),
            'ref_threshold':         args.reference_threshold,
            'ref_N_reco':            ref_n_reco,
            'ref_N_reco/ev':         round(ref_n_reco / max(n_ev, 1), 3),
            'ref_N_true':            round(ref_n_true, 1),
            'ref_closure':           round(ref_n_true / n_mc, 3) if n_mc else float('nan'),
            'locked_purity_target':  args.purity,
            'locked_purity_actual':  round(pi_actual, 3),
            'locked_threshold':      round(t_star, 3),
            'locked_cluster_eff':    round(eff_at_t, 3),
            'locked_N_reco':         loc_n_reco,
            'locked_N_reco/ev':      round(loc_n_reco / max(n_ev, 1), 3),
            'locked_N_true':         round(loc_n_true, 1),
            'locked_closure':        round(loc_n_true / n_mc, 3) if n_mc else float('nan'),
        })

    out = pd.DataFrame(rows).sort_values('dataset').reset_index(drop=True)
    csv_path = os.path.join(args.out_dir, f'purity_locked_pi{int(args.purity*100)}.csv')
    out.to_csv(csv_path, index=False)
    print(out.to_string(index=False))
    print(f'\nwrote {csv_path}')

    # Closure spread — the key diagnostic for the reframing.
    if 'ref_closure' in out.columns and out['ref_closure'].notna().all():
        ref_spread = float(out['ref_closure'].max() - out['ref_closure'].min())
        loc_spread = float(out['locked_closure'].max() - out['locked_closure'].min())
        print(f'\nclosure spread (max-min): ref={ref_spread:.3f}  locked={loc_spread:.3f}')
        if loc_spread < ref_spread - 0.02:
            print('  → operating-point conflation was a real contributor.')
        elif loc_spread <= ref_spread + 0.02:
            print('  → purity-lock does not equalise closure; deficit is physical / '
                  'model-training.')
        else:
            print('  → purity-lock WORSENS closure spread — investigate.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
