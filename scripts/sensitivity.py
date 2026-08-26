#!/usr/bin/env python
"""Cross-dataset sensitivity comparison.

Reads per-dataset prediction pickles (as produced by `scripts/evaluate.py`)
and produces a solved-yield DataFrame + a comparison plot for the 3
symmetry-potential runs.

Usage:
  python -m HGNDRecoGNN.scripts.sensitivity \\
      --dataset zeroSpot=results/zeroSpot_net_default/pred_clusters_smash.pkl \\
      --dataset defaultSpot=results/defaultSpot_net_default/pred_clusters_smash.pkl \\
      --dataset bigSpot=results/bigSpot_net_default/pred_clusters_smash.pkl \\
      --dataset-cache-root notebooks/cache \\
      --out-dir results/sensitivity \\
      --threshold 0.5

If `--dataset-cache-root` points at the parent of the preprocessed
`ndet_dataset_smash_<name>[_smoke]/processed/_hits_cache_*.parquet` files,
the plot's MC-truth reference is computed per unique `(Row, Id)` neutron
track — matching the reference notebook and the check-grid plots in the
sensitivity notebooks. Without the flag we fall back to the cluster-level
`cl_label == 1` count from `analysis.sensitivity.mc_truth_yield_vs_ekin`.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def _add_package_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.abspath(os.path.join(here, '..', '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def _discover_parquet(cache_root: str, name: str) -> str | None:
    """Find the raw-hits parquet for a dataset under a cache root.

    Tries `_smoke`-suffixed dirs first (small subsets), then the full
    `ndet_dataset_smash_<name>` directory. Returns None if nothing found.
    """
    for suffix in ('_smoke', ''):
        pat = os.path.join(
            cache_root,
            f'ndet_dataset_smash_{name}{suffix}',
            'processed', '_hits_cache_*.parquet',
        )
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def _mc_truth_per_neutron(parquet_path, row_filter, bins):
    """Return (counts_per_bin, n_events) from raw hits.

    Uses the same recipe as `NeutronRecoGNN/notebooks/results_smash.ipynb`
    cell 23: one entry per unique `(Row, Id)` signal neutron
    (`n0_label == 1`), then histogram over `Ekin`.
    """
    import numpy as np
    import pandas as pd
    cols = ['Row', 'Id', 'n0_label', 'Ekin']
    df = pd.read_parquet(parquet_path, columns=cols)
    if row_filter is not None:
        df = df[df['Row'].isin(row_filter)]
    n_events = int(df['Row'].nunique())
    mc_e = df[df['n0_label'] == 1].groupby(['Row', 'Id'])['Ekin'].mean()
    counts, _ = np.histogram(mc_e.to_numpy(), bins=bins)
    return counts, n_events


def main() -> int:
    p = argparse.ArgumentParser(description='HGND cross-dataset sensitivity.')
    p.add_argument('--dataset', action='append', required=True,
                   help='name=path — path is a pickled per-cluster DataFrame. '
                        'Repeat for each dataset.')
    p.add_argument('--dataset-cache-root', default=None,
                   help='Directory containing ndet_dataset_smash_<name>/ dirs. '
                        'When present, MC-truth is computed per unique '
                        '(Row, Id) neutron from the raw-hits parquet and the '
                        'yield plot switches to per-event normalisation. '
                        'Falls back to per-cluster MC-truth otherwise.')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--no-plot', action='store_true',
                   help='Skip matplotlib figure (useful on headless nodes).')
    args = p.parse_args()

    _add_package_to_path()

    import numpy as np
    import pandas as pd
    from HGNDRecoGNN.analysis.sensitivity import (
        DatasetRun, compare_datasets, default_ekin_bins,
    )

    runs = []
    cluster_dfs: dict[str, pd.DataFrame] = {}
    for spec in args.dataset:
        if '=' not in spec:
            print(f'error: --dataset expects name=path, got {spec!r}',
                  file=sys.stderr)
            return 1
        name, path = spec.split('=', 1)
        name = name.strip()
        df = pd.read_pickle(path)
        cluster_dfs[name] = df
        runs.append(DatasetRun(name=name, clusters_df=df))
        print(f'  loaded {name:15s}  {len(df):,} clusters  ({path})')

    bins = default_ekin_bins()
    tables = compare_datasets(runs, bins, threshold=args.threshold)

    # ── Per-event N_events + optional per-neutron MC-truth from parquet ──
    n_events_by_ds = {n: int(df['Row'].nunique())
                      for n, df in cluster_dfs.items()}

    mc_from_parquet: dict[str, tuple] = {}
    if args.dataset_cache_root:
        for name in cluster_dfs:
            parquet = _discover_parquet(args.dataset_cache_root, name)
            if parquet is None:
                print(f'  MC parquet: {name:15s}  none found under '
                      f'{args.dataset_cache_root}')
                continue
            row_filter = set(cluster_dfs[name]['Row'].unique())
            counts, ne = _mc_truth_per_neutron(parquet, row_filter, bins)
            mc_from_parquet[name] = (counts, ne)
            print(f'  MC parquet: {name:15s}  {counts.sum():,} neutrons in '
                  f'{ne:,} events  ({parquet})')

    os.makedirs(args.out_dir, exist_ok=True)
    for name, df in tables.items():
        out = os.path.join(args.out_dir, f'sensitivity_{name}.csv')
        df.to_csv(out, index=False)
        print(f'  wrote {out}')

    # Summary CSV per dataset — carries both counts and per-event yields.
    summary_rows = []
    for run in runs:
        tab = tables[run.name]
        n_ev = n_events_by_ds.get(run.name, 1)
        n_reco = int(tab['n_reco'].sum())
        n_true = float(tab['n_true_solved'].sum())
        n_true_err = float(np.sqrt((tab['n_true_solved_err'] ** 2).sum()))
        if run.name in mc_from_parquet:
            counts, mc_ne = mc_from_parquet[run.name]
            n_mc = int(counts.sum())
            mc_source = 'parquet(per-neutron)'
        else:
            n_mc = int(tab['n_mc_truth'].sum())
            mc_source = 'cluster'
        summary_rows.append({
            'dataset':     run.name,
            'U_sym_MeV':   run.potential_mev,
            'threshold':   args.threshold,
            'N_events':    n_ev,
            'N_reco':      n_reco,
            'N_reco/ev':   round(n_reco / max(n_ev, 1), 3),
            'N_true':      round(n_true, 1),
            'N_true_err':  round(n_true_err, 1),
            'N_MC_truth':  n_mc,
            'N_MC/ev':     round(n_mc / max(n_ev, 1), 3),
            'closure':     round(n_true / n_mc, 3) if n_mc > 0 else float('nan'),
            'MC_source':   mc_source,
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        by='U_sym_MeV', na_position='last')
    summary_path = os.path.join(args.out_dir, 'sensitivity_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f'  wrote {summary_path}')

    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        mc_mid = 0.5 * (bins[:-1] + bins[1:])
        fig, ax = plt.subplots(figsize=(8, 5))
        for run in runs:
            tab = tables[run.name]
            n_ev = n_events_by_ds.get(run.name, 1)
            lbl = run.name
            if run.potential_mev is not None:
                lbl += f'  ({run.potential_mev:.0f} MeV, N_ev={n_ev:,})'
            line = ax.errorbar(tab['ekin_mid'],
                               tab['n_true_solved'] / n_ev,
                               yerr=tab['n_true_solved_err'] / n_ev,
                               marker='o', capsize=2, label=lbl)[0]
            # MC-truth: per-neutron from parquet when available, else
            # per-cluster from compare_datasets (dotted to distinguish).
            if run.name in mc_from_parquet:
                counts, mc_ne = mc_from_parquet[run.name]
                ax.plot(mc_mid, counts / max(mc_ne, 1),
                        ls='--', alpha=0.7, color=line.get_color(),
                        label=f'{run.name} MC truth / N_ev')
            else:
                ax.plot(tab['ekin_mid'], tab['n_mc_truth'] / n_ev,
                        ls=':', alpha=0.7, color=line.get_color(),
                        label=f'{run.name} MC (cluster) / N_ev')

        ax.set_xlim(0, float(bins[-1]))
        ax.set_xlabel(r'$E_{kin}$ [GeV]')
        ax.set_ylabel(r'Efficiency-corrected neutron yield / event')
        ax.set_title(f'HGND sensitivity — threshold t={args.threshold}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig_path = os.path.join(args.out_dir, 'sensitivity_yield.png')
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        print(f'  wrote {fig_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
