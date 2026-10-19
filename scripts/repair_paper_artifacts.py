"""Export audited manuscript tables and vector diagnostics from frozen artifacts.

No inference or new fit is performed. Old detector-production diagnostics are
marked as such; primary truth and reused DCM validation remain distinct.
"""
from pathlib import Path
import ast
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from HGNDRecoGNN.scripts.plot_b_full import apply_style

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO.parent / 'HGNDPaper'
SOURCE = {}


def read(path):
    p = REPO / path
    SOURCE[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def save(fig, stem):
    fig.savefig(PAPER / 'figs' / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(PAPER / 'figs' / f'{stem}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def table_dcm():
    lines = [r'\begin{table*}[!t]', r'\centering',
        r'\caption{DCM single-seed reconstruction controls, scored on the same '
        r'241{,}259-cluster reused validation population from 283{,}398 arm graphs. '
        r'Whole-cache scaling; training seed 42; epoch selected by validation loss. '
        r'The threshold $t$ is chosen on validation by proximity to 0.87 '
        r'energy-weighted purity; displayed purity and efficiency are count-based. '
        r'AUC, purity and efficiency are dimensionless (higher is better at fixed '
        r'convention); $t$ is dimensionless. These are single-seed point estimates '
        r'without seed uncertainties; no resolved difference or independent-test '
        r'generalisation is claimed. The last three rows successively add a '
        r'cluster-size feature, mean/max/sum pooling and residual hit layers '
        r'to the three-epoch-warmup recipe. Parameter counts are descriptive '
        r'(millions); the pooling comparison changes capacity. Source: archived '
        r'per-run performance tables and loaded-checkpoint configurations.}',
        r'\label{tab:dcm_validation}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
        r'recipe & epochs & params (M) & AUC $\uparrow$ & $t$ & purity & efficiency \\', r'\midrule']
    recipes = [('ref', 'MultiStepLR', 25), ('epochs', 'MultiStepLR', 30),
               ('sched', 'Cosine, no warmup', 30), ('warmup', 'Cosine, 3-epoch warmup', 30),
               ('size', '+ cluster size', 30), ('pool', '+ mean/max/sum pooling', 30),
               ('residual', '+ residual hit layers', 30)]
    hashes = read('results/dcm_diagnostic_20261016/current/checkpoint_hashes.json')
    rows = []
    for i, (variant, label, epochs) in enumerate(recipes):
        base = f'results/dcm_diagnostic_20261016/current/{variant}'
        path = f'{base}/validation/performance.json'
        j = read(path)
        config = read(f'{base}/effective_run.json')
        assert read(f'{base}/status.json')['state'] == 'completed'
        read(f'{base}/source_manifest.json')
        log_path = f'results/dcm_diagnostic_20261016/current/logs/dcm_diag_4411340_{i}.out'
        log_bytes = (REPO/log_path).read_bytes()
        SOURCE[log_path] = hashlib.sha256(log_bytes).hexdigest()
        log = log_bytes.decode()
        loaded = re.search(r'ckpt: arch=(\w+)\s+kwargs=(\{.*?\})\s+epoch=(\d+)', log)
        assert loaded is not None and loaded.group(1) == 'net_default'
        arch = ast.literal_eval(loaded.group(2))
        params = int(re.search(r'Params:\s*([\d,]+)', log).group(1).replace(',', ''))
        assert config['seed'] == 42 and config['independent_test'] is False
        assert config['shards'] == 277 and f'epoch {epochs-1:03d}' in log
        assert arch['hidden_channels'] == 512 and arch['num_layers'] == 8
        assert bool(arch.get('cluster_size_feature', False)) == (i >= 4)
        assert (arch.get('cluster_pooling') == 'mean_max_sum') == (i >= 5)
        assert bool(arch.get('hit_residual', False)) == (i == 6)
        point = next(p for p in j['threshold_scan'] if p['threshold'] == j['working_threshold'])
        assert j['n_clusters'] == 241259
        if i == 4:
            lines.append(r'\midrule')
        lines.append(f"{label} & {epochs} & {params/1e6:.2f} & {j['roc_auc']:.4f} & "
                     f"{j['working_threshold']:.2f} & {point['purity_count']:.3f} & "
                     f"{point['efficiency_count']:.3f} " + r'\\')
        rows.append({'variant': variant, 'epochs': epochs, 'training_seed': 42,
                     'scored_on': 'reused graph-level validation',
                     'preprocessing_fit': 'whole delivered DCM cache',
                     'checkpoint_selection': 'validation loss', 'source': path,
                     'n_clusters': j['n_clusters'], 'roc_auc': j['roc_auc'],
                     'working_point': point, 'architecture': arch,
                     'n_parameters': params, 'selected_epoch': int(loaded.group(3)),
                     'checkpoint_sha256': hashes[variant]['sha256']})
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (PAPER / 'tables/dcm_validation_20261016.tex').write_text('\n'.join(lines) + '\n')
    return rows


def table_truth():
    j = read('results/acceptance_full/b_reweight.json')
    lines = [r'\begin{table*}[!t]', r'\centering',
        r'\caption{Generator-truth hardness and yield ratios, reweighted in impact '
        r'parameter, over 2{,}345{,}993 collision events in 568 generation files. '
        r'No detector model is fitted. Each uncertainty is one bootstrap standard '
        r'deviation from 400 whole-file resamples; the relative contrast is '
        r'$100[X(90)/X(0)-1]$ in percent. Ratios are dimensionless physical '
        r'observables, with no direction of better. $z$ is the absolute endpoint '
        r'contrast divided by its bootstrap SD, not a goodness-of-fit. The two '
        r'selections have different particle populations and are not interchangeable.}',
        r'\label{tab:truthscan}', r'\begin{tabular}{lccccc}', r'\toprule',
        r'observable & $U=0$ MeV & $U=18$ MeV & $U=90$ MeV & contrast (\%) & $z$ \\',
        r'\midrule']
    for selection, label in [('acc', 'front-face acceptance'), ('band', 'polar-angle band')]:
        lines.append(r'\multicolumn{6}{l}{\emph{' + label + r'}}\\')
        for key, name in [('Rn_over_Rp', r'$R_n/R_p$'), ('R_p', '$R_p$'),
                          ('R_n', '$R_n$'), ('np', '$n/p$')]:
            r = j['integrated'][f'{key}_{selection}']
            vals = [f"${r['central'][s]:.4f} \\pm {r['err'][s]:.4f}$"
                    for s in ('zeroSpot', 'defaultSpot', 'bigSpot')]
            contrast = f"${100*r['rel_90_over_0']:+.2f} \\pm {100*r['rel_err']:.2f}$"
            lines.append(' & '.join([name, *vals, contrast, f"{r['sigma']:.1f}"]) + r'\\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\end{table*}']
    (PAPER / 'tables/truth_scan.tex').write_text('\n'.join(lines) + '\n')
    return j['integrated']


def architecture():
    fig, ax = plt.subplots(figsize=(3.30, 3.35))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    labels = ['Hit features and radius graph', 'Hit message passing\nEdgeConv / SAGEConv / GraphConv',
              'Pooling into candidate clusters', 'Dynamic cluster graph']
    ys = [.86, .66, .46, .27]
    for y, label in zip(ys, labels):
        ax.add_patch(FancyBboxPatch((.06, y), .88, .12, boxstyle='round,pad=0.008',
                                   linewidth=.7, edgecolor='#1f77b4', facecolor='#edf3f8'))
        ax.text(.50, y+.06, label, ha='center', va='center', fontsize=8)
    for top, bot in zip(ys[:-1], ys[1:]):
        ax.annotate('', xy=(.5, bot+.13), xytext=(.5, top),
                    arrowprops=dict(arrowstyle='->', lw=.8))
    for x, label in [(.06, 'Neutron score'), (.55, 'Cluster energy')]:
        ax.add_patch(FancyBboxPatch((x, .045), .39, .11, boxstyle='round,pad=0.008',
                                   linewidth=.7, edgecolor='.2', facecolor='.96'))
        ax.text(x+.195, .100, label, fontsize=8, ha='center', va='center')
        ax.annotate('', xy=(x+.195, .16), xytext=(.5, .27),
                    arrowprops=dict(arrowstyle='->', lw=.8))
    save(fig, 'gnn_architecture')


def multiplicity():
    meta = read('results/update_note/multiplicity_efficiency.json')
    path = REPO / 'results/update_note/conditional_rank_efficiency.csv'
    SOURCE[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    ranks = pd.read_csv(path)
    colors = plt.cm.Blues([.45, .68, .95])
    fig, axes = plt.subplots(1, 2, figsize=(6.84, 3.10), sharey=True)
    for color, marker, (sample, entry) in zip(colors, ['o', 's', '^'], meta['samples'].items()):
        q = ranks[(ranks.dataset == sample) & (ranks.eligible_events > 0)]
        for ax, col in zip(axes, ['rank', 'cumulative']):
            field = f'{col}_efficiency'
            prefix = 'rank_eff' if col == 'rank' else 'cumulative_eff'
            y = q[field].to_numpy()
            err = np.maximum(0, np.vstack([y-q[prefix+'_lo'], q[prefix+'_hi']-y]))
            ax.errorbar(q['rank'], y, yerr=err, fmt=marker, ms=3,
                        color=color, capsize=2, label=rf"$U_{{\rm sym}}={entry['U_MeV']}$ MeV")
    for ax, title in zip(axes, ['Rank selection', 'At least k clusters selected']):
        ax.set(title=title, xlabel='MC-truth signal-cluster rank k', ylim=(-.02, 1.02))
        ax.set_xticks(sorted(ranks.loc[ranks.eligible_events > 0, 'rank'].unique()))
    axes[0].set_ylabel('Conditional selection probability')
    axes[0].legend(fontsize=7)
    fig.text(.5, .015, 'SMASH Xe+Cs, 2.87 AGeV; reconstructed selection at t=0.6\n'
             'MC-truth matched clusters, not neutron tracks; 68% Wilson intervals only',
             ha='center', fontsize=7)
    fig.tight_layout(rect=(0, .17, 1, 1))
    save(fig, 'conditional_neutron_efficiency')

    fig, axes = plt.subplots(1, 3, figsize=(6.84, 2.65), sharex=True, sharey=True,
                             layout='constrained')
    for ax, (sample, entry) in zip(axes, meta['samples'].items()):
        z = np.array(entry['response_probability'])
        n = z.shape[0]
        image = ax.imshow(np.ma.masked_where(z <= 0, z), origin='lower',
                          extent=(-.5, n-.5, -.5, n-.5), cmap='viridis',
                          norm=LogNorm(vmin=.001, vmax=1), aspect='equal')
        ax.plot([-.5, n-.5], [-.5, n-.5], color='white', ls='--', lw=.7)
        ax.set_title(rf"$U_{{\rm sym}}={entry['U_MeV']}$ MeV", fontsize=8)
        ax.set_xticks([0, 2, 4, 6]);ax.set_yticks([0, 2, 4, 6])
        ax.set_xlabel('Selected signal clusters')
    axes[0].set_ylabel('MC-truth signal clusters')
    fig.colorbar(image, ax=axes, shrink=.78, pad=.02, label=r'$P(C_{\rm sel}\mid C)$')
    fig.suptitle('SMASH Xe+Cs, 2.87 AGeV simulation; one evaluation model; fixed t=0.6\n'
                 'Cluster selection response, conditional on represented events', fontsize=8)
    save(fig, 'multiplicity_response')
    return meta['definition']


def ladder():
    j = read('results/reco_experiment/reco_experiment.json')['comparison_90_vs_0']
    paired = read('results/reco_experiment/response_diagnostics.json')['paired']
    # This separate paired diagnostic uses calibrated energy on the same
    # selected signal clusters. Raw E_pred and all-selected calibrated energy
    # are different estimators; neither can substitute for its paired arm.
    j['R_n_sig_trueE'] = paired['true_energy']
    j['R_n_sig_calibratedE'] = paired['reco_energy']
    keys = ['R_n_truth_primary', 'R_n_truth_reaching', 'R_n_sig_trueE',
            'R_n_sig_calibratedE', 'R_n_reco_calib']
    labels = ['Primary\nangular band', 'At detector\nMC truth', 'Selected signal\ntrue energy',
              'Same signal\ncalibrated energy', 'All selected\ncalibrated energy']
    fig, ax = plt.subplots(figsize=(6.84, 3.1))
    y = np.array([j[k]['rel']*100 for k in keys]); e = np.array([j[k]['err']*100 for k in keys])
    ax.errorbar(np.arange(len(keys)), y, yerr=e, fmt='o', color='#1f77b4', capsize=3)
    ax.axhline(0, color='.3', ls='--', lw=.7)
    ax.set_xticks(np.arange(len(keys)), labels)
    ax.set_ylabel(r'$100[R_n(90)/R_n(0)-1]$ (%)')
    lower, upper = min(y-e), max(y+e); margin=(upper-lower)*.25
    ax.set_ylim(lower-margin, upper+margin)
    ax.text(.02, .97, 'Archived superseded SMASH detector diagnostic\n'
            '20 generation files/sample; job-bootstrap SD; differing populations',
            transform=ax.transAxes, va='top', fontsize=7)
    fig.tight_layout();save(fig, 'reco_ladder')
    tex = [r'\begin{table*}[!t]', r'\centering',
        r'\caption{Archived superseded-production comparison: dimensionless '
        r'hardness contrasts, expressed in percent, for 20 generation files per '
        r'sample. Errors are one whole-file bootstrap SD. The first row uses an '
        r'angular band; subsequent rows use detector or cluster populations. '
        r'The third and fourth rows use the same selected MC-truth signal '
        r'clusters, with true and calibrated predicted energy; the fifth '
        r'includes selected background and the same energy '
        r'calibration fitted on the midpoint sample. Thresholds are calibrated '
        r'with each evaluated sample, not frozen on independent validation. '
        r'This is a methodological comparison, with no direction of better. '
        r'The paired third-to-fourth-row displacement is '
        r'$-8.66\pm2.21$ percentage points (same weights and selected clusters; '
        r'paired whole-file bootstrap, 3.91 bootstrap SD).}', r'\label{tab:ladder}',
        r'\begin{tabular}{lr}', r'\toprule',
        r'population and energy coordinate & $100[R_n(90)/R_n(0)-1]$ (\%) \\', r'\midrule']
    names = ['Primary neutrons, polar-angle band', 'Neutrons at detector, true energy',
             'Selected signal clusters, true energy', 'Same signal clusters, calibrated predicted energy',
             'All selected clusters, calibrated predicted energy']
    for key, name in zip(keys, names):
        tex.append(f"{name} & ${j[key]['rel']*100:+.2f} \\pm {j[key]['err']*100:.2f}$ " + r'\\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (PAPER/'tables/archived_ladder.tex').write_text('\n'.join(tex)+'\n')
    return {k: j[k] for k in keys}


def main():
    (PAPER/'tables').mkdir(exist_ok=True);(PAPER/'data').mkdir(exist_ok=True)
    apply_style()
    record = {'version': 'manuscript_repair_20261017', 'dcm': table_dcm(),
              'truth_scan': table_truth(), 'multiplicity_definition': multiplicity(),
              'archived_ladder': ladder()}
    grouped_path = REPO / 'results/dcm_grouped_20261017/summary.json'
    if grouped_path.exists():
        record['dcm_grouped'] = read('results/dcm_grouped_20261017/summary.json')
        SOURCE.update(record['dcm_grouped']['source_sha256'])
    architecture()
    record['sources_sha256'] = SOURCE
    record['limitations'] = ['Diagnostic DCM metrics reuse validation and one training seed',
        'generator-truth to corrected detector-production lineage not established',
        'multiplicity counts truth-matched clusters on an evaluation prefix',
        'four-epoch retraining is distinct from later twenty-epoch SMASH weights']
    (PAPER/'data/manuscript_provenance.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
