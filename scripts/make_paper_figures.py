"""Emit the paper's figure PDFs from the local prediction pickles.

Fills the placeholder slots in ``paper/main.tex``:

  performance_roc            ← notebooks/results/hpc_defaultSpot_v2_seed42/roc_pr.png
  ereco_vs_etrue             ← notebooks/results/.../energy_regression.png
  multiplicity               ← notebooks/results/.../multiplicity_confusion.png
  efficiency_vs_ekin         ← notebooks/results/.../efficiency_vs_ekin_purity_locked.png
  mctruth_spectra            ← generated here from full-stats parquet
  sensitivity_yield          ← generated here from
                                results/sensitivity_full_hpc_4284506/sensitivity_e_pred_fixed/*.csv
  sensitivity_ratios         ← generated here
  sensitivity_threshold_scan ← generated here from local pooled-smoke pickles

The two manual figures (``detector_layout``, ``gnn_architecture``)
are left as placeholders — they need external draws.

Run from the repo root:

    conda activate pyg
    python scripts/make_paper_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NB_RESULTS = REPO / 'notebooks' / 'results' / 'hpc_defaultSpot_v2_seed42'
FULL_STATS = REPO / 'results' / 'sensitivity_full_hpc_4284506'
FIGS = REPO / 'paper' / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)


# ── Trivial copies from the eval notebook ──────────────────────────────────

DIRECT_COPIES = [
    ('roc_pr.png',                          'performance_roc.pdf'),
    ('energy_regression.png',               'ereco_vs_etrue.pdf'),
    ('multiplicity_confusion.png',          'multiplicity.pdf'),
    ('efficiency_vs_ekin_purity_locked.png', 'efficiency_vs_ekin.pdf'),
]


def _png_to_pdf(src: Path, dst: Path) -> None:
    """PDF-wrap the PNG so paper builds cleanly with either extension."""
    img = plt.imread(str(src))
    h, w = img.shape[:2]
    dpi = 140
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis('off')
    fig.savefig(str(dst), dpi=dpi)
    plt.close(fig)
    print(f'  {src.name}  →  {dst.name}')


def copy_direct() -> None:
    for src_name, dst_name in DIRECT_COPIES:
        src = NB_RESULTS / src_name
        dst = FIGS / dst_name
        if not src.exists():
            print(f'  SKIP {dst_name}: source {src} missing')
            continue
        _png_to_pdf(src, dst)


# ── MC-truth spectrum (per-neutron, three datasets on one axis) ────────────

def make_mctruth_spectra() -> None:
    """Overlay per-neutron MC-truth $E_k$ spectra for the three
    sensitivity samples, normalised per event."""
    e_pred_dir = FULL_STATS / 'sensitivity_e_pred_fixed'
    combined = pd.read_csv(e_pred_dir / 'sensitivity_combined.csv')
    summary = pd.read_csv(e_pred_dir / 'sensitivity_summary.csv')

    fig, ax = plt.subplots(figsize=(7, 4.4))
    colors = {'zeroSpot': 'C0', 'defaultSpot': 'C1', 'bigSpot': 'C2'}
    for ds, row in summary.set_index('dataset').iterrows():
        df = combined[combined.dataset == ds]
        n_ev = int(row['N_events'])
        y = df['n_mc_truth'] / max(n_ev, 1)
        yerr = df['n_mc_truth_err'] / max(n_ev, 1)
        ax.errorbar(df['ekin_mid'], y, yerr=yerr,
                    fmt='o-', capsize=2, color=colors.get(ds, 'k'),
                    label=f'{ds} (U_sym={int(row.U_sym_MeV)} MeV, N_ev={n_ev:,})')
    ax.set_xlabel(r'$E_{\rm kin}$ [GeV]')
    ax.set_ylabel(r'MC-truth neutrons per event')
    ax.set_xlim(0, 6)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(FIGS / 'mctruth_spectra.pdf'))
    plt.close(fig)
    print('  mctruth_spectra.pdf')


# ── Sensitivity yield (per-event N_reco and N_true) ────────────────────────

def make_sensitivity_yield() -> None:
    """Recreate the CLI's sensitivity_yield.png but from the _fixed
    directory (correct MC-truth), and save as PDF."""
    e_pred_dir = FULL_STATS / 'sensitivity_e_pred_fixed'
    summary = pd.read_csv(e_pred_dir / 'sensitivity_summary.csv')
    combined = pd.read_csv(e_pred_dir / 'sensitivity_combined.csv')

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = {'zeroSpot': 'C0', 'defaultSpot': 'C1', 'bigSpot': 'C2'}
    for ds, row in summary.set_index('dataset').iterrows():
        df = combined[combined.dataset == ds]
        n_ev = int(row['N_events'])
        y = df['n_true_solved'] / max(n_ev, 1)
        yerr = df['n_true_solved_err'] / max(n_ev, 1)
        c = colors.get(ds, 'k')
        line = ax.errorbar(df['ekin_mid'], y, yerr=yerr,
                           fmt='o-', capsize=2, color=c,
                           label=f'{ds} $N_{{\\rm true}}$/ev (U_sym={int(row.U_sym_MeV)} MeV)')[0]
        # MC-truth dashed
        y_mc = df['n_mc_truth'] / max(n_ev, 1)
        ax.plot(df['ekin_mid'], y_mc, '--', color=c, alpha=0.6,
                label=f'{ds} MC truth / ev')
    ax.set_xlabel(r'$E_{\rm kin}$ [GeV]')
    ax.set_ylabel(r'Neutrons per event')
    ax.set_xlim(0, 6)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.set_title(r'Efficiency-corrected yield $N_{\rm true}$ vs.\ MC truth ($t=0.5$)')
    fig.tight_layout()
    fig.savefig(str(FIGS / 'sensitivity_yield.pdf'))
    plt.close(fig)
    print('  sensitivity_yield.pdf')


# ── Cross-dataset ratios ───────────────────────────────────────────────────

def make_sensitivity_ratios() -> None:
    """N_true(dataset) / N_true(defaultSpot) and MC-truth analogue."""
    e_pred_dir = FULL_STATS / 'sensitivity_e_pred_fixed'
    combined = pd.read_csv(e_pred_dir / 'sensitivity_combined.csv')
    default = combined[combined.dataset == 'defaultSpot'].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = {'zeroSpot': 'C0', 'bigSpot': 'C2'}
    for ds, c in colors.items():
        df = combined[combined.dataset == ds].reset_index(drop=True)
        # protect against zero denominators
        mask = default['n_true_solved'] > 0
        ratio_reco = df.loc[mask, 'n_true_solved'].values \
                     / default.loc[mask, 'n_true_solved'].values
        ratio_mc = df.loc[mask, 'n_mc_truth'].values \
                   / np.maximum(default.loc[mask, 'n_mc_truth'].values, 1)
        # Poisson-propagated errors on the reco ratio
        r_err_num = df.loc[mask, 'n_true_solved_err'].values
        r_err_den = default.loc[mask, 'n_true_solved_err'].values
        num = df.loc[mask, 'n_true_solved'].values
        den = default.loc[mask, 'n_true_solved'].values
        rel = np.sqrt((r_err_num/np.maximum(num, 1e-9))**2
                      + (r_err_den/np.maximum(den, 1e-9))**2)
        ratio_err = ratio_reco * rel
        x = df.loc[mask, 'ekin_mid'].values
        ax.errorbar(x, ratio_reco, yerr=ratio_err, fmt='o-', capsize=2,
                    color=c, label=f'{ds} / defaultSpot (reco)')
        ax.plot(x, ratio_mc, '--', color=c, alpha=0.6,
                label=f'{ds} / defaultSpot (MC)')
    ax.axhline(1, color='k', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'$E_{\rm kin}$ [GeV]')
    ax.set_ylabel(r'yield ratio to defaultSpot')
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    ax.set_title(r'Cross-dataset per-event yield ratios ($t=0.5$)')
    fig.tight_layout()
    fig.savefig(str(FIGS / 'sensitivity_ratios.pdf'))
    plt.close(fig)
    print('  sensitivity_ratios.pdf')


# ── Threshold scan of the closure at t ∈ {0.3, 0.5, 0.7} ───────────────────

def make_threshold_scan() -> None:
    """Scan the per-cluster closure over three thresholds using the
    local pooled-smoke prediction pickles (cheap, no cluster round)."""
    import sys
    sys.path.insert(0, str(REPO.parent))
    from HGNDRecoGNN.analysis.sensitivity import (
        DatasetRun, compare_datasets, default_ekin_bins,
    )

    IN = REPO / 'results' / 'sensitivity_hpc_pooled_smoke'
    runs = []
    for ds in ('zeroSpot', 'defaultSpot', 'bigSpot'):
        pkl = IN / f'{ds}_hpc' / 'pred_clusters_hpc_pooled.pkl'
        runs.append(DatasetRun(name=ds, clusters_df=pd.read_pickle(pkl)))
    bins = default_ekin_bins()

    fig, ax = plt.subplots(figsize=(7, 4.4))
    markers = {0.3: 'o', 0.5: 's', 0.7: '^'}
    colors = {'zeroSpot': 'C0', 'defaultSpot': 'C1', 'bigSpot': 'C2'}
    for t in (0.3, 0.5, 0.7):
        tables = compare_datasets(runs, bins, threshold=t, efficiency_basis='e_pred')
        default_tab = tables['defaultSpot']
        default_sum = default_tab['n_true_solved'].sum()
        for ds in ('zeroSpot', 'bigSpot'):
            other_sum = tables[ds]['n_true_solved'].sum()
            ratio = other_sum / max(default_sum, 1e-9)
            ax.scatter([t], [ratio], marker=markers[t], s=60,
                       color=colors[ds],
                       label=(f'{ds}/default' if t == 0.3 else None))
    for ds in ('zeroSpot', 'bigSpot'):
        # Also connect with dotted line
        xs, ys = [], []
        for t in (0.3, 0.5, 0.7):
            tables = compare_datasets(runs, bins, threshold=t, efficiency_basis='e_pred')
            xs.append(t)
            ys.append(tables[ds]['n_true_solved'].sum()
                      / max(tables['defaultSpot']['n_true_solved'].sum(), 1e-9))
        ax.plot(xs, ys, ':', color=colors[ds], alpha=0.5)

    ax.axhline(1, color='k', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'classifier threshold $t$')
    ax.set_ylabel(r'integrated $N_{\rm true}$ ratio (smoke, pooled scaler)')
    ax.set_xlim(0.2, 0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(r'Threshold-scan robustness of cross-dataset ratio (smoke)')
    fig.tight_layout()
    fig.savefig(str(FIGS / 'sensitivity_threshold_scan.pdf'))
    plt.close(fig)
    print('  sensitivity_threshold_scan.pdf')


def main() -> int:
    print('== direct copies ==')
    copy_direct()
    print('== derived ==')
    make_mctruth_spectra()
    make_sensitivity_yield()
    make_sensitivity_ratios()
    make_threshold_scan()
    print('\nwrote figures to', FIGS)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
