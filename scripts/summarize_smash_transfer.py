"""Recompute first-file SMASH transfer metrics from saved predictions, without fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.diagnose_student_pipeline import measure  # noqa: E402
from HGNDRecoGNN.scripts.plot_b_full import apply_style, assert_contained  # noqa: E402

TAGS = {'zeroSpot': 0, 'defaultSpot': 18, 'bigSpot': 90}
SEEDS = (42, 123, 456)
METRICS = ['auc', 'ap', 'purity', 'efficiency', 'bias_GeV', 'mae_GeV',
           'selected_bias_GeV', 'selected_mae_GeV']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root):
    index_path = root/'transfer_summary_bundle.json'
    index = json.loads(index_path.read_text())
    if index['state'] != 'complete' or '4435668|COMPLETED|0:0' not in index['slurm_accounting']:
        raise ValueError('GPU job has not completed successfully')
    hashes, frames, rows = {str(index_path): sha(index_path)}, {}, {}
    for tag in TAGS:
        folder = root/'transfer_inputs/transfer_smoke'/tag
        summary_path = folder/'summary.json'
        if sha(summary_path) != index['summary_sha256'][tag]:
            raise ValueError('Exported summary changed')
        hashes[str(summary_path)] = sha(summary_path)
        data = index['summaries'][tag]
        if not data['smoke'] or data['state'] != 'complete' or data['model_device'] != 'cuda:0':
            raise ValueError('Incomplete or unexpected transfer configuration')
        if data['graph_recipe']['time_ns'] != 1.0 or len(data['files']) != 1:
            raise ValueError('Unexpected engineering transfer population/recipe')
        frames[tag], rows[tag] = {}, {}
        reference = None
        for seed in SEEDS:
            item = data['files'][0]['outputs'][str(seed)]
            path = folder/item['path']
            if sha(path) != item['sha256']:
                raise ValueError('Predictions changed')
            hashes[str(path)] = sha(path)
            frame = pd.read_csv(path)
            identity = frame[['Row', 'cl_istop', 'ClusterID', 'cl_label', 'e_true']]
            if reference is not None and not identity.equals(reference):
                raise ValueError('Different candidate population across training seeds')
            reference = identity
            metrics = measure(frame, item['threshold'])
            saved = data['per_seed'][str(seed)]['metrics']
            for key, value in metrics.items():
                # Nine-digit CSV output retains float32 score ordering. Energy
                # arithmetic after parsing may differ at roundoff, not physics.
                if value is None or saved[key] is None:
                    assert value is saved[key]
                elif not np.isclose(value, saved[key], rtol=1e-6, atol=1e-8):
                    raise ValueError(f'Metric mismatch: {tag}/{seed}/{key}')
            frames[tag][seed] = frame
            rows[tag][str(seed)] = {'threshold': item['threshold'], 'metrics': metrics,
                                    'checkpoint_sha256': item['checkpoint_sha256']}
    result = {'state': 'verified', 'scope': 'first-file retrospective SMASH engineering transfer only',
              'model': 'frozen native DCM GraphSAGE', 'seeds': list(SEEDS),
              'uncertainty': 'sample SD across three DCM training seeds on the same fixed SMASH candidates; not MC/source uncertainty',
              'no_smash_fitting': True, 'generated_neutron_efficiency_measured': False,
              'potential_sensitivity_measured': False, 'tags': {}, 'input_sha256': hashes}
    for tag, potential in TAGS.items():
        frame = frames[tag][SEEDS[0]]
        values = {k: {'seed_values': [rows[tag][str(s)]['metrics'][k] for s in SEEDS]} for k in METRICS}
        for value in values.values():
            value['mean'] = float(np.mean(value['seed_values']))
            value['sample_sd'] = float(np.std(value['seed_values'], ddof=1))
        result['tags'][tag] = {'U_sym_MeV': potential, 'clusters': len(frame),
                               'MC_positive_clusters': int(frame.cl_label.sum()),
                               'physical_events_with_candidates': int(frame.Row.nunique()),
                               'per_seed': rows[tag], 'summary': values}
    root.joinpath('metrics_verified.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    plot(root, frames)
    return result


def plot(root, frames, scope='first-file engineering probes'):
    apply_style()
    folder = root/'figures'
    folder.mkdir(exist_ok=True)
    styles = {42: ('#2b2b2b', '-', 'o'), 123: ('#0072B2', '--', 's'),
              456: ('#D55E00', ':', '^')}
    validation = {'scope': scope+'; PNG figures; no MC-statistical intervals estimated',
                  'plotting_source_sha256': sha(Path(__file__)), 'figures': []}
    for mode in ['roc', 'energy_residual_cdf']:
        fig, axes = plt.subplots(1, 3, figsize=(6.84, 3.1))
        fig.subplots_adjust(left=.065, right=.985, top=.90, bottom=.30, wspace=.28)
        for ax, (tag, potential) in zip(axes, TAGS.items()):
            base = frames[tag][42]
            ax.set_title(r'$U_{\rm sym}$ = '+str(potential)+' MeV')
            if mode == 'roc':
                # [0,1] is the probability domain, not a fitted display range.
                ax.plot([0, 1], [0, 1], color='#999999', ls='--', lw=.6)
                for seed in SEEDS:
                    f = frames[tag][seed]
                    x, y, _ = roc_curve(f.cl_label, f.cl_score)
                    color, line, _ = styles[seed]
                    ax.plot(x, y, color=color, ls=line, label=str(seed))
                ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='False positive rate')
                stamp = f'Reco score / MC labels\nn={len(base)}\nsignal={int(base.cl_label.sum())}'
                loc = (.97, .05, 'right', 'bottom')
            else:
                residuals = []
                for seed in SEEDS:
                    f = frames[tag][seed]
                    x = np.sort((f.e_pred-f.e_true).to_numpy()[f.cl_label.to_numpy() == 1])
                    y = np.arange(1, len(x)+1)/len(x)
                    residuals.extend(x)
                    color, line, _ = styles[seed]
                    ax.step(x, y, where='post', color=color, ls=line, label=str(seed))
                # Retain all tails; padding is display space, not a selection.
                lo, hi = min([0., *residuals]), max([0., *residuals])
                pad = .06*(hi-lo) or .1
                ax.set(xlim=(lo-pad, hi+pad), ylim=(0, 1), xlabel=r'$E_{\rm pred}-E_{\rm target}$ (GeV)')
                ax.axvline(0, color='#999999', ls='--', lw=.6)
                stamp = f'Reco - MC target\nMC-positive\nn={int(base.cl_label.sum())}'
                loc = (.03, .97, 'left', 'top')
            ax.minorticks_on()
            ax.tick_params(which='both', top=True, right=True, direction='in')
            ax.text(loc[0], loc[1], stamp,
                    transform=ax.transAxes, ha=loc[2], va=loc[3], fontsize=7)
            assert_contained(ax, mode+'/'+tag)
        axes[0].set_ylabel('True positive rate' if mode == 'roc' else 'Empirical cumulative fraction')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, ['Training seed '+x for x in labels], loc='lower center', ncol=3,
                   bbox_to_anchor=(.5, .005))
        fig.text(.5, .15, 'HGND simulation; Xe+Cs, 2.87 GeV/A; '+scope,
                 ha='center', fontsize=7)
        fig.text(.5, .095, 'Reconstructed scores/energies; native MC targets; no centrality cut; MC uncertainty not estimated',
                 ha='center', fontsize=7)
        path = folder/(mode+'.png')
        fig.savefig(path, dpi=200)
        fig.canvas.draw()
        for ax in axes:
            for text in ax.texts:
                box = text.get_window_extent(fig.canvas.get_renderer())
                if not ax.bbox.contains(box.x0, box.y0) or not ax.bbox.contains(box.x1, box.y1):
                    raise ValueError('Annotation outside panel: '+mode)
        validation['figures'].append({'path': str(path), 'sha256': sha(path),
                                       'data_bounds': 'passed', 'annotation_bounds': 'passed'})
        plt.close(fig)
    (root/'plot_validation.json').write_text(json.dumps(validation, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    result = run(parser.parse_args().root)
    print(json.dumps({k: {'clusters': v['clusters'], 'signal': v['MC_positive_clusters'],
                          'metrics': v['summary']} for k, v in result['tags'].items()}, indent=2))
