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
      --out-dir results/sensitivity \\
      --threshold 0.5
"""

from __future__ import annotations

import argparse
import os
import sys


def _add_package_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.abspath(os.path.join(here, '..', '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def main() -> int:
    p = argparse.ArgumentParser(description='HGND cross-dataset sensitivity.')
    p.add_argument('--dataset', action='append', required=True,
                   help='name=path — path is a pickled per-cluster DataFrame. '
                        'Repeat for each dataset.')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--no-plot', action='store_true',
                   help='Skip matplotlib figure (useful on headless nodes).')
    args = p.parse_args()

    _add_package_to_path()

    import pandas as pd
    from HGNDRecoGNN.analysis.sensitivity import (
        DatasetRun, compare_datasets, default_ekin_bins,
    )

    runs = []
    for spec in args.dataset:
        if '=' not in spec:
            print(f'error: --dataset expects name=path, got {spec!r}',
                  file=sys.stderr)
            return 1
        name, path = spec.split('=', 1)
        df = pd.read_pickle(path)
        runs.append(DatasetRun(name=name.strip(), clusters_df=df))
        print(f'  loaded {name.strip():15s}  {len(df):,} clusters  ({path})')

    bins = default_ekin_bins()
    tables = compare_datasets(runs, bins, threshold=args.threshold)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, df in tables.items():
        out = os.path.join(args.out_dir, f'sensitivity_{name}.csv')
        df.to_csv(out, index=False)
        print(f'  wrote {out}')

    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for run in runs:
            df = tables[run.name]
            lbl = run.name
            if run.potential_mev is not None:
                lbl += f' ({run.potential_mev:.0f} MeV)'
            ax.errorbar(df['ekin_mid'], df['n_true_solved'],
                        yerr=df['n_true_solved_err'],
                        marker='o', capsize=2, label=lbl)
        ax.set_xscale('log')
        ax.set_xlabel('Kinetic energy [GeV]')
        ax.set_ylabel(r'Efficiency-corrected neutron yield $N_n(E_k)$')
        ax.set_title(f'HGND sensitivity — threshold t={args.threshold}')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
        fig_path = os.path.join(args.out_dir, 'sensitivity_yield.png')
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        print(f'  wrote {fig_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
