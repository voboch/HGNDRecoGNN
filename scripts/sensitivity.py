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
    p.add_argument('--efficiency-basis', default='e_pred',
                   choices=['e_pred', 'e_true'],
                   help='Bin the efficiency correction by measured e_pred '
                        '(default, closure-consistent) or MC-truth e_true '
                        '(diagnostic only — mis-binned relative to N_reco).')
    p.add_argument('--unfold', action='store_true',
                   help='Additionally compute unfolded N_true via the '
                        'detector response matrix M[i,j] = P(e_pred in bin_j '
                        '& passed | e_true in bin_i, signal). Writes '
                        'sensitivity_unfold_<name>.csv per dataset and adds '
                        'N_true_unfold / closure_unfold to the summary.')
    p.add_argument('--unfold-method', default='tikhonov',
                   choices=['pinv', 'tikhonov'],
                   help='Response-matrix inversion method for --unfold.')
    p.add_argument('--unfold-lambda', type=float, default=1e-2,
                   help='Tikhonov regularisation strength for --unfold.')
    p.add_argument('--no-plot', action='store_true',
                   help='Skip matplotlib figure (useful on headless nodes).')
    args = p.parse_args()

    _add_package_to_path()

    import numpy as np
    import pandas as pd
    from HGNDRecoGNN.analysis.sensitivity import (
        DatasetRun, compare_datasets, default_ekin_bins,
        reco_yield_vs_ekin, response_matrix, unfold_yield,
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
    tables = compare_datasets(runs, bins, threshold=args.threshold,
                              efficiency_basis=args.efficiency_basis)
    print(f'  efficiency binned by: {args.efficiency_basis}')

    # ── Optional detector-matrix unfold (physics cross-check) ────────────
    unfold_by_ds: dict[str, np.ndarray] = {}
    if args.unfold:
        for run in runs:
            M = response_matrix(run, bins, threshold=args.threshold)
            reco = reco_yield_vs_ekin(run, bins, threshold=args.threshold,
                                      bin_by='e_pred')
            # Align reco to the full bin grid (some bins may be empty).
            n_reco_vec = np.zeros(len(bins) - 1)
            for _, row in reco.iterrows():
                idx = int(np.searchsorted(bins, row['ekin_lo'], side='right') - 1)
                if 0 <= idx < n_reco_vec.size:
                    n_reco_vec[idx] = row['n_reco']
            n_true_unfold = unfold_yield(n_reco_vec, M,
                                         method=args.unfold_method,
                                         tikhonov_lambda=args.unfold_lambda)
            unfold_by_ds[run.name] = n_true_unfold
            mid = 0.5 * (bins[:-1] + bins[1:])
            unfold_df = pd.DataFrame({
                'ekin_lo':   bins[:-1], 'ekin_hi': bins[1:], 'ekin_mid': mid,
                'n_reco':    n_reco_vec.astype(int),
                'n_true_unfold': n_true_unfold,
            })
            path = os.path.join(args.out_dir, f'sensitivity_unfold_{run.name}.csv')
            os.makedirs(args.out_dir, exist_ok=True)
            unfold_df.to_csv(path, index=False)
            print(f'  wrote {path}')

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

    # Summary CSV per dataset. Two closure definitions are quoted:
    #
    #   * closure          = N_true / N_MC_truth  (per-neutron denominator)
    #       Physics-quoted number. Sensitive to multi-neutron cluster
    #       merging: if two MC neutrons collapse into one cluster, the
    #       per-cluster reconstruction fundamentally cannot recover both,
    #       so closure < 1 even for a perfect classifier.
    #
    #   * closure_per_cluster = N_true / N_MC_clusters (cluster-signal denom)
    #       Pipeline-diagnostic closure. Cancels the cluster-merging term
    #       and measures only the classifier + regression stage. Should
    #       approach 1.0 for a well-calibrated model.
    #
    #   * clusters_per_neutron = N_MC_clusters / N_MC_truth
    #       The multiplicative bridge:  closure = clusters_per_neutron
    #       * closure_per_cluster. When clusters_per_neutron << 1 the
    #       cluster-merger term dominates the physics closure.
    summary_rows = []
    for run in runs:
        tab = tables[run.name]
        n_ev = n_events_by_ds.get(run.name, 1)
        n_reco = int(tab['n_reco'].sum())
        n_true = float(tab['n_true_solved'].sum())
        n_true_err = float(np.sqrt((tab['n_true_solved_err'] ** 2).sum()))
        n_mc_cluster = int((cluster_dfs[run.name][run.label_col] == 1).sum())
        if run.name in mc_from_parquet:
            counts, mc_ne = mc_from_parquet[run.name]
            n_mc = int(counts.sum())
            mc_source = 'parquet(per-neutron)'
        else:
            n_mc = int(tab['n_mc_truth'].sum())
            mc_source = 'cluster'
        row = {
            'dataset':               run.name,
            'U_sym_MeV':             run.potential_mev,
            'threshold':             args.threshold,
            'efficiency_basis':      args.efficiency_basis,
            'N_events':              n_ev,
            'N_reco':                n_reco,
            'N_reco/ev':             round(n_reco / max(n_ev, 1), 3),
            'N_true':                round(n_true, 1),
            'N_true_err':            round(n_true_err, 1),
            'N_MC_truth':            n_mc,
            'N_MC/ev':               round(n_mc / max(n_ev, 1), 3),
            'N_MC_clusters':         n_mc_cluster,
            'clusters_per_neutron':  (round(n_mc_cluster / n_mc, 3)
                                      if n_mc > 0 else float('nan')),
            'closure':               (round(n_true / n_mc, 3)
                                      if n_mc > 0 else float('nan')),
            'closure_per_cluster':   (round(n_true / n_mc_cluster, 3)
                                      if n_mc_cluster > 0 else float('nan')),
            'MC_source':             mc_source,
        }
        if args.unfold and run.name in unfold_by_ds:
            n_true_u = float(unfold_by_ds[run.name].sum())
            row['N_true_unfold']              = round(n_true_u, 1)
            row['closure_unfold']             = (round(n_true_u / n_mc, 3)
                                                  if n_mc > 0 else float('nan'))
            row['closure_unfold_per_cluster'] = (round(n_true_u / n_mc_cluster, 3)
                                                  if n_mc_cluster > 0 else float('nan'))
        summary_rows.append(row)
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
