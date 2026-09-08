"""Emit the paper's figure PDFs from local prediction pickles.

Fills the placeholder slots in ``paper/main.tex``:

  performance_roc            ← notebooks/results/hpc_defaultSpot_v2_seed42/roc_pr.png
  ereco_vs_etrue             ← notebooks/results/.../energy_regression.png
  multiplicity               ← two panels: per-event N_n distribution + confusion matrix
  efficiency_vs_ekin         ← three-SMASH ε_n(E_kin) from per-dataset sensitivity CSVs
  mctruth_spectra            ← lift-cap 4300982 sensitivity_combined.csv (838k defaultSpot)
  sensitivity_yield          ← two-panel: N_reco/ev (raw) + N_true/ev (corrected)
  sensitivity_ratios         ← reco ratios with propagated errors + MC-truth dashed
  sensitivity_threshold_scan ← full-stats scan using lift-cap prediction pickles

The two manual figures (``detector_layout``, ``gnn_architecture``) are
left as placeholders — they need external draws.

Run from the repo root:

    conda activate pyg
    python scripts/make_paper_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NB_RESULTS = REPO / 'notebooks' / 'results' / 'hpc_defaultSpot_v2_seed42'
FULL_STATS = REPO / 'results' / 'sensitivity_full_hpc_finish_4300982'
E_PRED_DIR = FULL_STATS / 'sensitivity_e_pred'
FIGS = REPO / 'paper' / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)

DATASETS = ('zeroSpot', 'defaultSpot', 'bigSpot')
COLORS = {'zeroSpot': 'C0', 'defaultSpot': 'C1', 'bigSpot': 'C2'}


# ── Direct copies from the eval notebook ──────────────────────────────────

DIRECT_COPIES = [
    ('roc_pr.png',            'performance_roc.pdf'),
    ('energy_regression.png', 'ereco_vs_etrue.pdf'),
]


def _png_to_pdf(src: Path, dst: Path) -> None:
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


# ── Multiplicity (two panels: per-event N_n dist + confusion matrix) ───────

def make_multiplicity() -> None:
    """(left) per-event N_n distribution, (right) confusion matrix at t=0.5.
    Uses the defaultSpot lift-cap cluster pickle."""
    pkl = FULL_STATS / 'defaultSpot_hpc' / 'pred_clusters_hpc_full.pkl'
    if not pkl.exists():
        print(f'  SKIP multiplicity.pdf: {pkl} missing')
        return
    df = pd.read_pickle(pkl)
    per_evt = (df.assign(pred=(df.cl_score > 0.5))
                 .groupby('Row')
                 .agg(n_true=('cl_label', 'sum'), n_reco=('pred', 'sum'))
                 .astype(int))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Left panel: event-level N_n distribution (MC-truth + reco)
    max_n = int(max(per_evt.n_true.max(), per_evt.n_reco.max())) + 1
    bins = np.arange(max_n + 2) - 0.5
    ax_l.hist(per_evt.n_true, bins=bins, alpha=0.55, color='C1',
              label=r'MC truth', edgecolor='black', linewidth=0.4)
    ax_l.hist(per_evt.n_reco, bins=bins, alpha=0.55, color='C0',
              label=r'reco ($t=0.5$)', edgecolor='black', linewidth=0.4)
    ax_l.set(xlabel=r'$N_n$ per event',
             ylabel='events',
             title=r'Per-event neutron multiplicity — defaultSpot')
    ax_l.set_yscale('log')
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(fontsize=10)

    # Right panel: confusion matrix at t=0.5
    conf = per_evt.groupby(['n_true', 'n_reco']).size().rename('count').reset_index()
    pivot = (conf.pivot(index='n_true', columns='n_reco', values='count')
                 .fillna(0).astype(int))
    im = ax_r.imshow(pivot.values, origin='lower', cmap='Blues', aspect='auto',
                     extent=[pivot.columns.min()-0.5, pivot.columns.max()+0.5,
                             pivot.index.min()-0.5,   pivot.index.max()+0.5])
    vmax = pivot.values.max()
    for i, n_true in enumerate(pivot.index):
        for j, n_reco in enumerate(pivot.columns):
            v = int(pivot.values[i, j])
            if v > 0:
                ax_r.text(n_reco, n_true, f'{v:,}', ha='center', va='center',
                          color='white' if v > 0.5 * vmax else 'black',
                          fontsize=7)
    ax_r.set(xlabel=r'$N_{\rm reco}$ ($t = 0.5$)',
             ylabel=r'$N_{\rm true}$',
             title=r'Multiplicity confusion at $t=0.5$')
    fig.colorbar(im, ax=ax_r, label='events')

    fig.tight_layout()
    fig.savefig(str(FIGS / 'multiplicity.pdf'))
    plt.close(fig)
    print('  multiplicity.pdf')


# ── ε_n(E_kin) for the three SMASH samples ─────────────────────────────────

def make_efficiency_vs_ekin() -> None:
    """Per-Ekin cluster efficiency at t=0.5 for the three SMASH datasets,
    using the per-dataset sensitivity CSVs (which carry epsilon + Wilson
    intervals per bin)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for ds in DATASETS:
        csv = E_PRED_DIR / f'sensitivity_{ds}.csv'
        if not csv.exists():
            print(f'  SKIP {csv}')
            continue
        df = pd.read_csv(csv)
        df = df[df.n_true > 0]
        lo_err = np.clip(df.epsilon - df.epsilon_lo, 0, None)
        hi_err = np.clip(df.epsilon_hi - df.epsilon, 0, None)
        ax.errorbar(df.ekin_mid, df.epsilon,
                    yerr=[lo_err, hi_err],
                    xerr=(df.ekin_hi - df.ekin_lo) / 2,
                    fmt='o-', capsize=3, color=COLORS[ds],
                    label=f'{ds}')
    ax.set(xlabel=r'$E_{\rm pred}$ [GeV]',
           ylabel=r'cluster efficiency $\varepsilon_{\rm cl}$',
           title=r'Per-dataset cluster efficiency at $t = 0.5$ '
                 r'($E_{\rm pred}$-binned)',
           xlim=(0, 6), ylim=(0, 1.02))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(str(FIGS / 'efficiency_vs_ekin.pdf'))
    plt.close(fig)
    print('  efficiency_vs_ekin.pdf')


# ── MC-truth spectrum (per-neutron, three datasets on one axis) ────────────

def make_mctruth_spectra() -> None:
    combined = pd.read_csv(E_PRED_DIR / 'sensitivity_combined.csv')
    summary = pd.read_csv(E_PRED_DIR / 'sensitivity_summary.csv')

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for ds, row in summary.set_index('dataset').iterrows():
        df = combined[combined.dataset == ds]
        n_ev = int(row['N_events'])
        y = df['n_mc_truth'] / max(n_ev, 1)
        yerr = df['n_mc_truth_err'] / max(n_ev, 1)
        ax.errorbar(df['ekin_mid'], y, yerr=yerr,
                    fmt='o-', capsize=2, color=COLORS.get(ds, 'k'),
                    label=(f'{ds} ($U_{{\\rm sym}}={int(row.U_sym_MeV)}$'
                           f' MeV, $N_{{\\rm ev}}={n_ev:,}$)'))
    ax.set(xlabel=r'$E_{\rm kin}$ [GeV]',
           ylabel=r'MC-truth neutrons per event',
           xlim=(0, 6))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(FIGS / 'mctruth_spectra.pdf'))
    plt.close(fig)
    print('  mctruth_spectra.pdf')


# ── Sensitivity yield (per-event N_reco AND N_true, two panels) ────────────

def make_sensitivity_yield() -> None:
    combined = pd.read_csv(E_PRED_DIR / 'sensitivity_combined.csv')
    summary = pd.read_csv(E_PRED_DIR / 'sensitivity_summary.csv')

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    markers = {'zeroSpot': 'o', 'defaultSpot': 's', 'bigSpot': '^'}
    for ds, row in summary.set_index('dataset').iterrows():
        df = combined[combined.dataset == ds]
        n_ev = int(row['N_events'])
        c = COLORS.get(ds, 'k')
        mk = markers.get(ds, 'o')
        # Left: raw N_reco/ev + MC-truth dashed
        y_reco = df['n_reco'] / max(n_ev, 1)
        yerr_reco = np.sqrt(df['n_reco']) / max(n_ev, 1)
        ax_l.errorbar(df['ekin_mid'], y_reco, yerr=yerr_reco,
                      fmt=mk+'-', capsize=2, color=c, alpha=0.85,
                      label=f'{ds} $N_{{\\rm reco}}$/ev')
        y_mc = df['n_mc_truth'] / max(n_ev, 1)
        ax_l.plot(df['ekin_mid'], y_mc, '--', color=c, alpha=0.55)
        # Right: N_true/ev (efficiency-corrected) + MC-truth dashed
        y_true = df['n_true_solved'] / max(n_ev, 1)
        yerr_true = df['n_true_solved_err'] / max(n_ev, 1)
        ax_r.errorbar(df['ekin_mid'], y_true, yerr=yerr_true,
                      fmt=mk+'-', capsize=2, color=c, alpha=0.85,
                      label=f'{ds} $N_{{\\rm true}}$/ev')
        ax_r.plot(df['ekin_mid'], y_mc, '--', color=c, alpha=0.55)

    for ax, title in ((ax_l, r'Raw reco $N_{\rm reco}$/ev at $t=0.5$'),
                      (ax_r, r'Efficiency-corrected $N_{\rm true}$/ev at $t=0.5$')):
        ax.set(xlabel=r'$E_{\rm kin}$ [GeV]',
               ylabel=r'neutrons per event',
               xlim=(0, 6))
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper right')
    ax_l.set_title(ax_l.get_title() + '\n(dashed: MC truth per event)')
    fig.tight_layout()
    fig.savefig(str(FIGS / 'sensitivity_yield.pdf'))
    plt.close(fig)
    print('  sensitivity_yield.pdf')


# ── Cross-dataset ratios ───────────────────────────────────────────────────

def make_sensitivity_ratios() -> None:
    combined = pd.read_csv(E_PRED_DIR / 'sensitivity_combined.csv')
    default = combined[combined.dataset == 'defaultSpot'].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for ds in ('zeroSpot', 'bigSpot'):
        c = COLORS[ds]
        df = combined[combined.dataset == ds].reset_index(drop=True)
        # Use bins where BOTH numerator and denominator are non-trivial.
        num = df['n_true_solved'].values
        den = default['n_true_solved'].values
        mc_num = df['n_mc_truth'].values
        mc_den = default['n_mc_truth'].values
        mask = (den > 0) & (num > 0) & (mc_den > 0)
        if mask.sum() == 0:
            print(f'  SKIP {ds} — no valid bins')
            continue
        r_reco = num[mask] / den[mask]
        r_mc   = mc_num[mask] / np.maximum(mc_den[mask], 1)
        # Propagate errors on the reco ratio.
        e_num = df['n_true_solved_err'].values[mask]
        e_den = default['n_true_solved_err'].values[mask]
        rel = np.sqrt((e_num / np.maximum(num[mask], 1e-9)) ** 2
                      + (e_den / np.maximum(den[mask], 1e-9)) ** 2)
        r_reco_err = r_reco * rel
        x = df['ekin_mid'].values[mask]
        ax.errorbar(x, r_reco, yerr=r_reco_err, fmt='o-', capsize=2,
                    color=c, label=f'{ds} / defaultSpot (reco $N_{{\\rm true}}$)')
        ax.plot(x, r_mc, '--', color=c, alpha=0.65,
                label=f'{ds} / defaultSpot (MC truth)')
    ax.axhline(1, color='k', lw=0.8, alpha=0.5)
    ax.set(xlabel=r'$E_{\rm kin}$ [GeV]',
           ylabel=r'yield ratio to defaultSpot',
           xlim=(0, 6),
           ylim=(0.4, 2.0),
           title=r'Cross-dataset per-event yield ratios ($t = 0.5$, full stats)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(str(FIGS / 'sensitivity_ratios.pdf'))
    plt.close(fig)
    print('  sensitivity_ratios.pdf')


# ── Threshold scan on the full-stats lift-cap predictions ──────────────────

def make_threshold_scan() -> None:
    import sys
    sys.path.insert(0, str(REPO.parent))
    from HGNDRecoGNN.analysis.sensitivity import (
        DatasetRun, compare_datasets, default_ekin_bins,
    )

    runs = []
    for ds in DATASETS:
        pkl = FULL_STATS / f'{ds}_hpc' / 'pred_clusters_hpc_full.pkl'
        if not pkl.exists():
            print(f'  SKIP threshold scan — missing {pkl}')
            return
        runs.append(DatasetRun(name=ds, clusters_df=pd.read_pickle(pkl)))
    bins = default_ekin_bins()

    thresholds = (0.3, 0.4, 0.5, 0.6, 0.7)
    ratios: dict[str, list[float]] = {'zeroSpot': [], 'bigSpot': []}
    for t in thresholds:
        tables = compare_datasets(runs, bins, threshold=t,
                                  efficiency_basis='e_pred')
        default_sum = tables['defaultSpot']['n_true_solved'].sum()
        for ds in ('zeroSpot', 'bigSpot'):
            other_sum = tables[ds]['n_true_solved'].sum()
            ratios[ds].append(other_sum / max(default_sum, 1e-9))

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for ds in ('zeroSpot', 'bigSpot'):
        ax.plot(thresholds, ratios[ds], 'o-', color=COLORS[ds],
                label=f'{ds} / defaultSpot')
    ax.axhline(1, color='k', lw=0.8, alpha=0.5)
    ax.set(xlabel=r'classifier threshold $t$',
           ylabel=r'integrated $N_{\rm true}$ ratio',
           title=r'Threshold-scan robustness (full statistics)',
           xlim=(0.25, 0.75),
           ylim=(0.95, 1.30))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(str(FIGS / 'sensitivity_threshold_scan.pdf'))
    plt.close(fig)
    print('  sensitivity_threshold_scan.pdf')


def main() -> int:
    print('== direct copies ==')
    copy_direct()
    print('== derived ==')
    make_multiplicity()
    make_efficiency_vs_ekin()
    make_mctruth_spectra()
    make_sensitivity_yield()
    make_sensitivity_ratios()
    make_threshold_scan()
    print('\nwrote figures to', FIGS)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
