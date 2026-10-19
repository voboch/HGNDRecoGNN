"""Validate the six frozen evaluations and export their preregistered comparison.

No fitting, threshold optimisation or inference is performed here. The paired
training-seed SE is conditional on the one reserved test source run.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from HGNDRecoGNN.scripts.plot_b_full import apply_style

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/dcm_grouped_20261017'
CURRENT = ROOT / 'current'
PAPER = REPO.parent / 'HGNDPaper'
SOURCES = {}


def load_bytes(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path.relative_to(REPO))] = hashlib.sha256(data).hexdigest()
    return data


def read(path):
    return json.loads(load_bytes(path))


def mean_se(values):
    return statistics.mean(values), statistics.stdev(values) / math.sqrt(len(values))


def validate():
    manifest = read(CURRENT / 'prepared/manifest.json')
    submissions = read(CURRENT / 'submissions.json')
    status = read(CURRENT / 'launch_status.json')
    assert status['train_array_state'] == status['test_array_state'] == 'COMPLETED'
    assert len(status['completed_training_arms']) == len(status['completed_test_arms']) == 6
    accounting = status['slurm_accounting'].splitlines()[1:]
    for job in ('4412700', '4412701'):
        for task in range(6):
            row = next(x.split('|') for x in accounting if x.startswith(f'{job}_{task}|'))
            assert row[1] == 'COMPLETED' and row[3] == '0:0'
    assert manifest['training_seeds'] == [42, 123, 456]
    assert manifest['arms'] == ['multistep', 'cosine']
    assert manifest['partition_source_runs'] == {
        'train': ['988975'], 'validation': ['988974'], 'test': ['988976']}
    manifest_hash = hashlib.sha256((CURRENT / 'prepared/manifest.json').read_bytes()).hexdigest()
    rows, scans = [], {}
    for seed in manifest['training_seeds']:
        for arm in manifest['arms']:
            run = CURRENT / f'{arm}_{seed}'
            train = read(run / 'training_complete.json')
            test = read(run / 'test_complete.json')
            val = read(run / 'validation/performance.json')
            perf = read(run / 'test/performance.json')
            task = status['array_mapping'].index([arm, seed])
            log = load_bytes(CURRENT / f'logs/dcm_group_train_4412700_{task}.out').decode()
            assert [int(x) for x in re.findall(r'^epoch\s+(\d+)', log, re.M)] == list(range(30))
            test_log = load_bytes(CURRENT / f'logs/dcm_group_test_4412701_{task}.out').decode()
            assert '794395 hits  222251 clusters  3976683 edges' in test_log
            assert train['arm'] == test['arm'] == arm and train['seed'] == test['seed'] == seed
            assert train['epochs'] == 30 and 0 <= train['selected_epoch'] < 30
            assert train['test_opened'] is False
            assert train['split_manifest_sha256'] == manifest_hash
            assert train['checkpoint_sha256'] == test['checkpoint_sha256']
            assert test['threshold_source'] == 'validation only'
            assert test['n_test_graphs'] == manifest['graphs']['test'] == 131173
            assert test['test_source_runs'] == manifest['partition_source_runs']['test']
            assert val['n_clusters'] == 101107 and perf['n_clusters'] == 222251
            assert math.isfinite(perf['roc_auc']) and 0 < perf['roc_auc'] < 1
            assert test['roc_auc'] == perf['roc_auc']
            threshold = train['validation_threshold']
            assert val['working_threshold'] == perf['working_threshold'] == threshold
            scan_path = run / 'validation/threshold_scan.csv'
            scan = list(csv.DictReader(load_bytes(scan_path).decode().splitlines()))
            assert len(scan) == 401
            assert float(scan[0]['threshold']) == .001 and float(scan[-1]['threshold']) == .999
            # The inherited count-purity target trades efficiency for background
            # rejection. It is a convention, not an EoS-sensitivity optimum.
            first = next(x for x in scan if float(x['purity_count']) >= .70)
            assert math.isclose(float(first['threshold']), threshold, abs_tol=1e-12)
            for metrics in (val, perf):
                wp = metrics['working_point']
                assert wp['threshold'] == threshold
                assert math.isclose(wp['purity_count'], wp['n_selected_signal'] / wp['n_selected'])
            wp = perf['working_point']
            assert math.isclose(wp['n_selected_signal'] / wp['efficiency_count'], 36713)
            scans[(arm, seed)] = scan
            rows.append({'arm': arm, 'training_seed': seed,
                         'selected_epoch_zero_based': train['selected_epoch'],
                         'checkpoint_sha256': train['checkpoint_sha256'],
                         'test_roc_auc': perf['roc_auc'],
                         'validation_threshold': threshold,
                         'test_working_point': wp,
                         'test_energy_performance': perf['energy_performance'],
                         'n_test_clusters': perf['n_clusters'], 'n_test_signal_clusters': 36713})
    return manifest, submissions, rows, scans


def export_table(rows, paired, summary):
    lines = [r'\begin{table}[!t]', r'\centering',
        r'\caption{DCM reconstructed cluster-score AUC on the held-out source run '
        r'988976: 222{,}251 clusters in 131{,}173 detector-arm graphs from '
        r'91{,}081 physical events. Higher AUC is better; all entries are '
        r'dimensionless. Both schedules use 30 epochs, training-only scaling '
        r'on source run 988975 and validation-loss checkpoint selection on '
        r'988974. Rows pair different weights by training seed; the mean '
        r'$\Delta$ quotes SE over three training seeds. The paired '
        rf'$t={summary["t"]:.2f}$ on 2 df is below the two-sided 5\% critical '
        r'value 4.303. This SE conditions on one test source run and excludes '
        r'between-generation variance.}',
        r'\label{tab:dcm_grouped}', r'\begin{tabular}{rrrr}', r'\toprule',
        r'seed & MultiStepLR AUC & cosine AUC & $\Delta$ AUC \\', r'\midrule']
    for item in paired:
        lines.append(f"{item['training_seed']} & {item['multistep_auc']:.4f} & "
                     f"{item['cosine_auc']:.4f} & ${item['difference']:+.4f}$ " + r'\\')
    lines += [r'\midrule', r'mean & '
              f"${summary['multistep_mean']:.4f}$ & "
              f"${summary['cosine_mean']:.4f}$ & "
              f"${summary['difference_mean']:+.4f}\\pm{summary['difference_se']:.4f}$ " + r'\\',
              r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (PAPER / 'tables/dcm_grouped_20261017.tex').write_text('\n'.join(lines) + '\n')
    numbers = {'Delta': f"{summary['difference_mean']:+.4f}",
               'SE': f"{summary['difference_se']:.4f}", 'T': f"{summary['t']:.2f}",
               'CILow': f"{summary['ci95'][0]:+.4f}", 'CIHigh': f"{summary['ci95'][1]:+.4f}"}
    (PAPER / 'tables/dcm_grouped_numbers.tex').write_text('\n'.join(
        rf'\newcommand{{\DCMGrouped{k}}}{{{v}}}' for k, v in numbers.items()) + '\n')
    # Keep every frozen threshold and secondary count metric in a compact
    # generated appendix table; test purity is not constrained to equal 0.70.
    lines = [r'\begin{table}[!t]', r'\centering',
        r'\caption{DCM reconstructed operating points on the same held-out '
        r'test population and fitted models as Table~\ref{tab:dcm_grouped}. '
        r'Count purity $\pi_c$ and efficiency $\varepsilon_c$ are dimensionless; '
        r'at fixed selection convention, '
        r'higher purity and efficiency are better. Each row is one training '
        r'seed without a per-row uncertainty. The threshold $t$ is frozen '
        r'from validation count purity $\geq0.70$; test purity is measured '
        r'without retuning. Efficiency uses 36{,}713 MC-labelled signal clusters.}',
        r'\label{tab:dcm_grouped_points}', r'\begin{tabular}{llrrr}', r'\toprule',
        r'schedule & seed & $t$ & $\pi_c$ & $\varepsilon_c$ \\', r'\midrule']
    for r in rows:
        label = 'MultiStepLR' if r['arm'] == 'multistep' else 'cosine'
        wp = r['test_working_point']
        lines.append(f"{label} & {r['training_seed']} & {r['validation_threshold']:.3f} & "
                     f"{wp['purity_count']:.3f} & {wp['efficiency_count']:.3f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (PAPER / 'tables/dcm_grouped_points.tex').write_text('\n'.join(lines) + '\n')


def plot_scans(scans, rows):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.84, 2.75), sharex=True, sharey=True)
    colours = ['#2b2b2b', '#1f77b4', '#B22222']
    for ax, arm in zip(axes, ('multistep', 'cosine')):
        for seed, colour, marker in zip((42, 123, 456), colours, ('o', 's', '^')):
            scan = scans[(arm, seed)]
            x = [float(r['threshold']) for r in scan]
            ax.plot(x, [float(r['purity_count']) for r in scan], color=colour, label=str(seed))
            ax.plot(x, [float(r['efficiency_count']) for r in scan], color=colour, ls='--')
            row = next(r for r in rows if r['arm'] == arm and r['training_seed'] == seed)
            chosen = next(r for r in scan if math.isclose(float(r['threshold']), row['validation_threshold']))
            for quantity in ('purity_count', 'efficiency_count'):
                ax.plot(row['validation_threshold'], float(chosen[quantity]), marker=marker,
                        color=colour, ms=4, mew=.8, mfc='white')
        ax.axhline(.70, color='.5', ls=':', lw=.7)
        ax.set_title('MultiStepLR' if arm == 'multistep' else 'Cosine')
        ax.set_xlabel('Reconstructed score threshold (dimensionless)')
        ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.margins(x=.01, y=.03)
    axes[0].set_ylabel('Count purity / efficiency (dimensionless)')
    axes[0].legend(handles=[Line2D([], [], color=c, marker=m, mfc='white', label=str(s))
                           for s, c, m in zip((42, 123, 456), colours, ('o', 's', '^'))],
                   title='Training seed', loc='lower left', fontsize=7, title_fontsize=7)
    axes[1].legend(handles=[Line2D([], [], color='.2', label='Purity'),
                            Line2D([], [], color='.2', ls='--', label='Efficiency'),
                            Line2D([], [], color='.5', ls=':', label='Purity target 0.70')],
                   loc='lower left', fontsize=7)
    fig.text(.5, .985, 'BM@N HGND simulation | DCM Xe+CsI | reconstructed scores, MC labels',
             ha='center', va='top', fontsize=8)
    fig.text(.5, .925, 'Validation source run 988974 | 101,107 clusters | conventional count-purity target',
             ha='center', va='top', fontsize=7)
    fig.subplots_adjust(left=.085, right=.985, bottom=.17, top=.78, wspace=.09)
    # Disable the house style's tight cropping: the canvas is already at the
    # 6.84-inch printed width, with no LaTeX rescaling or font shrinkage.
    for ext in ('pdf', 'png'):
        with plt.rc_context({'savefig.bbox': None}):
            fig.savefig(PAPER / f'figs/dcm_grouped_thresholds.{ext}', dpi=200)
    plt.close(fig)


def main():
    manifest, submissions, rows, scans = validate()
    paired = []
    for seed in manifest['training_seeds']:
        m = next(r['test_roc_auc'] for r in rows if r['arm'] == 'multistep' and r['training_seed'] == seed)
        c = next(r['test_roc_auc'] for r in rows if r['arm'] == 'cosine' and r['training_seed'] == seed)
        paired.append({'training_seed': seed, 'multistep_auc': m, 'cosine_auc': c, 'difference': c-m})
    mean, se = mean_se([r['difference'] for r in paired])
    # Preregistered two-sided 5% paired t with n-1=2 df. The conventional
    # 4.303 critical value was fixed before test, not inferred from these AUCs.
    summary = {'difference_mean': mean, 'difference_se': se, 't': mean/se,
               'df': 2, 't_critical_95': 4.303, 'ci95': [mean-4.303*se, mean+4.303*se],
               'resolved_at_two_sided_005': abs(mean/se) > 4.303}
    for arm in manifest['arms']:
        summary[arm+'_mean'], summary[arm+'_se'] = mean_se(
            [r['test_roc_auc'] for r in rows if r['arm'] == arm])
    result = {'protocol': manifest, 'source_snapshot': submissions,
              'scored_on': 'held-out source run 988976; all six trainings completed before test',
              'rows': rows, 'paired_cosine_minus_multistep': paired, 'comparison': summary,
              'uncertainty': 'SE over three paired training seeds, conditional on one fixed test source run',
              'limits': 'No between-generation variance; no pure scaling ablation or event-level benchmark reproduction.',
              'source_sha256': SOURCES}
    export_table(rows, paired, summary); plot_scans(scans, rows)
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (PAPER / 'data/dcm_grouped_20261017.json').write_text(json.dumps(result, indent=2) + '\n')
    provenance_path = PAPER / 'data/manuscript_provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance['dcm_grouped'] = result
    provenance['sources_sha256'].update(SOURCES)
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
