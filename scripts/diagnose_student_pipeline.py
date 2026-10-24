"""Post-training diagnostics of fixed DCM predictions; no fitting or inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.plot_b_full import apply_style  # noqa: E402
from HGNDRecoGNN.scripts.reco_performance import E_BINS  # noqa: E402

KEYS = ['Row', 'cl_istop', 'ClusterID']
# Inherit existing energy diagnostic bins, retaining the soft region and open
# tail explicitly. They display response; they do not select physics bands.
ENERGY_EDGES = np.r_[0., E_BINS[1:], np.inf]
# Powers-of-two display groups separate singletons from larger components.
# No threshold, checkpoint or model is selected using these test diagnostics.
SIZE_EDGES = [0, 1, 3, 7, np.inf]
NAMES = {'student_sage': 'GraphSAGE', 'student_dynedge': 'DynamicEdgeConv',
         'student_spectral_sage': 'Spectral GraphSAGE', 'student_spectral_dynedge': 'Spectral DynamicEdgeConv'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x


def metadata(pack):
    lengths = np.diff(pack['cluster_ptr'].numpy())
    rows = np.repeat(pack['row'].numpy(), lengths)
    sides = np.repeat(pack['side'].numpy(), lengths)
    start = np.repeat(pack['cluster_ptr'][:-1].numpy(), np.diff(pack['node_ptr'].numpy()))
    sizes = np.bincount(pack['cluster'].numpy().ravel() + start, minlength=int(lengths.sum()))
    cid = np.arange(lengths.sum()) - np.repeat(pack['cluster_ptr'][:-1].numpy(), lengths)
    f = pd.DataFrame(dict(Row=rows, cl_istop=sides, ClusterID=cid, size=sizes,
                          cl_label=pack['label'].numpy(), e_true=pack['energy'].numpy()))
    if f.duplicated(KEYS).any() or np.any(sizes < 1):
        raise ValueError('Invalid native candidate metadata')
    f['multiplicity'] = f.Row.map(f.groupby('Row').size())
    return f


def measure(f, threshold):
    y = f.cl_label.to_numpy() == 1
    sel = f.cl_score.to_numpy() > threshold
    residual = (f.e_pred-f.e_true).to_numpy()[y]
    chosen = residual[sel[y]]
    both = len(np.unique(y)) == 2
    return {'n': len(f), 'signal': int(y.sum()), 'selected': int(sel.sum()),
            'selected_signal': int((sel & y).sum()),
            'auc': roc_auc_score(y, f.cl_score) if both else None,
            'ap': average_precision_score(y, f.cl_score) if both else None,
            'purity': float(y[sel].mean()) if sel.any() else None,
            'efficiency': float(sel[y].mean()) if y.any() else None,
            'bias_GeV': float(residual.mean()) if len(residual) else None,
            'mae_GeV': float(np.abs(residual).mean()) if len(residual) else None,
            'selected_bias_GeV': float(chosen.mean()) if len(chosen) else None,
            'selected_mae_GeV': float(np.abs(chosen).mean()) if len(chosen) else None}


def summarize(per_seed):
    """Training variation on fixed inputs; never concatenate seed predictions."""
    fields = ['auc', 'ap', 'purity', 'efficiency', 'bias_GeV', 'mae_GeV',
              'selected_bias_GeV', 'selected_mae_GeV']

    def describe(rows):
        result = {'population_per_seed': [{k: r[k] for k in ['n', 'signal', 'selected', 'selected_signal']} for r in rows]}
        for field in fields:
            values = [r[field] for r in rows]
            full = len(values) == 3 and all(v is not None for v in values)
            result[field] = {'seed_values': values, 'mean': float(np.mean(values)) if full else None,
                             'sample_sd': float(np.std(values, ddof=1)) if full else None}
        return result

    result = {'uncertainty': 'sample SD across three training seeds; no source uncertainty', 'models': {}}
    for model in NAMES:
        runs = [r for r in per_seed if r['model'] == model]
        assert sorted(r['seed'] for r in runs) == [42, 123, 456]
        strata = []
        for i, s in enumerate(runs[0]['strata']):
            identity = {k: s[k] for k in ['axis', 'lo', 'hi', 'side'] if k in s}
            assert all(all(r['strata'][i][k] == v for k, v in identity.items()) for r in runs)
            strata.append(dict(identity, **describe([r['strata'][i] for r in runs])))
        result['models'][model] = {'total': describe([r['total'] for r in runs]), 'strata': strata}
    return result


def bounds(ax, xy):
    for axis in (0, 1):
        arrays = [np.asarray(pair[axis], dtype=float).ravel() for pair in xy]
        vals = np.concatenate(arrays)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            raise ValueError('Empty plot axis')
        lo, hi = vals.min(), vals.max()
        pad = .07*(hi-lo) if hi > lo else .07*max(abs(lo), 1.)
        (ax.set_xlim if axis == 0 else ax.set_ylim)(lo-pad, hi+pad)
    for x, y in xy:
        for vals, lim in ((x, ax.get_xlim()), (y, ax.get_ylim())):
            vals = np.asarray(vals, float)
            assert np.all((vals[np.isfinite(vals)] >= lim[0]) & (vals[np.isfinite(vals)] <= lim[1]))


def stamp(ax, text):
    ax.text(.02, .96, text, va='top', transform=ax.transAxes, fontsize=7,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.9), zorder=10)


def save(fig, path, qa):
    import matplotlib.pyplot as plt
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for text in ax.texts:
            box = text.get_window_extent(renderer)
            assert ax.get_window_extent().contains(*box.get_points()[0]), text.get_text()
            assert ax.get_window_extent().contains(*box.get_points()[1]), text.get_text()
        if ax.get_legend() is not None:
            box = ax.get_legend().get_window_extent(renderer)
            assert ax.get_window_extent().contains(*box.get_points()[0])
            assert ax.get_window_extent().contains(*box.get_points()[1])
    for ext in ('pdf', 'png'):
        p = path.with_suffix('.'+ext)
        fig.savefig(p, dpi=200)
        qa['outputs'][str(p.name)] = sha(p)
    plt.close(fig)


def plots(args, per_seed, qa):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    apply_style()
    figures = args.output/'figures'
    figures.mkdir()
    for r in per_seed:
        run = f"{r['model']}_{r['seed']}"
        h = json.loads((args.completed/run/'history.json').read_text())
        complete = json.loads((args.completed/run/'complete.json').read_text())
        epoch = np.array([v['epoch'] for v in h])
        fields = [('total', 'Composite loss (dimensionless)'), ('cluster_cls', 'Cluster BCE (dimensionless)'),
                  ('cluster_energy', 'Energy SmoothL1 (numeric loss)')]
        fig, axes = plt.subplots(1, 3, figsize=(6.84, 3.15), layout='constrained')
        for ax, (key, label) in zip(axes, fields):
            curves = []
            for part, ls, name in [('train', '-', 'train: batch mean'), ('validation', '--', 'validation: population')]:
                val = np.array([v[part]['total'] if key == 'total' else v[part]['components'][key] for v in h])
                ax.plot(epoch, val, ls=ls, label=name)
                curves.append((epoch, val))
            bounds(ax, curves)
            # Reserve upper space for provenance without hiding any curve.
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi+.45*(hi-lo))
            ax.axvline(complete['selected_epoch'], ls=':', color='.3')
            ax.set(xlabel='Epoch (zero-based)', ylabel=label)
        stamp(axes[0], f"{NAMES[r['model']]}\nSeed {r['seed']}; DCM / MC labels\nBM@N HGND simulation\nNo b cut (b unavailable)")
        axes[1].legend(loc='upper right')
        stamp(axes[2], f"Validation: 988974\nSelected epoch {complete['selected_epoch']}\nOptimization curves; no CI\nCollision lineage unverified")
        save(fig, figures/(run+'_history'), qa)
        scanpath = args.inputs/run/'validation/threshold_scan.csv'
        scan = pd.read_csv(scanpath)
        eligible = scan.loc[scan.purity_count >= .70]
        assert len(eligible) and np.isclose(float(eligible.iloc[0].threshold), r['threshold'], rtol=0, atol=1e-12)
        fig, ax = plt.subplots(figsize=(6.84, 3.6), layout='constrained')
        curves = []
        for col, ls, label in [('purity_count', '-', 'Count purity'), ('efficiency_count', '--', 'Count efficiency')]:
            ax.plot(scan.threshold, scan[col], ls=ls, label=label)
            curves.append((scan.threshold, scan[col]))
        ax.axhline(.70, ls=':', color='.3', label='Inherited target 0.70')
        curves.append(([scan.threshold.min(), scan.threshold.max()], [.70, .70]))
        bounds(ax, curves)
        ax.axvline(r['threshold'], ls='-.', color='.3', label=f"Frozen cut {r['threshold']:.6g}")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi+.27*(hi-lo))
        stamp(ax, f"{NAMES[r['model']]} / seed {r['seed']}\nReconstructed selection / MC truth; reused validation 988974\nBM@N HGND simulation; DCM; no b selection (b unavailable)\nUncertainty not estimated; full saved scan")
        ax.legend(loc='upper right')
        ax.set(xlabel='Cluster score cut (dimensionless)', ylabel='Fraction (dimensionless)')
        save(fig, figures/(run+'_threshold'), qa)
    for model in NAMES:
        rows = [r for r in per_seed if r['model'] == model]
        bins = [(lo, hi) for lo, hi in zip(ENERGY_EDGES[:-1], ENERGY_EDGES[1:]) if np.isfinite(hi) and hi > .3]
        fig, axes = plt.subplots(1, 2, figsize=(6.84, 3.6), layout='constrained')
        for ax, field, label in zip(axes, ['efficiency', 'bias_GeV'], ['Count efficiency (dimensionless)', 'Mean energy residual (GeV)']):
            mean, sd, edges = [], [], []
            for lo, hi in bins:
                v = [next(s[field] for s in r['strata'] if s['axis'] == 'signal_energy' and s['lo'] == lo and s['hi'] == hi) for r in rows]
                if any(x is None for x in v):
                    raise ValueError('Empty energy bin: display grouping must be revised')
                mean.append(np.mean(v))
                sd.append(np.std(v, ddof=1))
                edges.extend([lo, hi])
            x = np.array(edges)
            y, s = np.repeat(mean, 2), np.repeat(sd, 2)
            ax.plot(x, y, color='#1f77b4')
            ax.fill_between(x, y-s, y+s, color='#1f77b4', alpha=.2, label='Mean ± sample SD, 3 seeds')
            bounds(ax, [(x, y-s), (x, y+s)])
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi+.38*(hi-lo))
            ax.set(xlabel='Native MC cluster target (GeV)', ylabel=label)
            if field == 'bias_GeV':
                ax.axhline(0, color='.3', ls='--', linewidth=.6)
        stamp(axes[0], f"{NAMES[model]} / fixed validation cuts\nReconstructed selection / MC truth\nBM@N HGND simulation; DCM reused test988976\nNo b selection (b unavailable)")
        stamp(axes[1], 'Residuals on all MC-positive clusters\nNative max-hit energy; no track matching\nTraining variation only; no source uncertainty')
        axes[0].legend(loc='lower right')
        save(fig, figures/(model+'_energy_response'), qa)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--completed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifestpath = args.inputs/'diagnostic_export_manifest_20261021.json'
    manifest = json.loads(manifestpath.read_text())
    for rel, h in manifest.items():
        if sha(args.inputs/rel) != h:
            raise ValueError(f'Input changed: {rel}')
    packpath = args.inputs/'test_evaluation/test_pack.pt'
    pack = torch.load(packpath, map_location='cpu', weights_only=True)
    meta = metadata(pack)
    assert len(meta) == 222251 and int(meta.cl_label.sum()) == 36713 and meta.Row.nunique() == 91081
    expected_path = args.completed/'test_evaluation/per_seed.json'
    verified_path = args.completed/'validation_20261021.json'
    verified = json.loads(verified_path.read_text())
    assert sha(expected_path) == verified['per_seed_sha256']
    extra_inputs = {'completed/test_evaluation/per_seed.json': sha(expected_path),
                    'completed/validation_20261021.json': sha(verified_path)}
    expected = json.loads(expected_path.read_text())
    output = []
    for exp in expected:
        model, seed = exp['model'], exp['seed']
        run = f'{model}_{seed}'
        complete_path = args.completed/run/'complete.json'
        complete = json.loads(complete_path.read_text())
        history_path = args.completed/run/'history.json'
        assert sha(history_path) == complete['files']['history.json']
        assert complete['validation_threshold'] == exp['working_threshold']
        extra_inputs[f'completed/{run}/history.json'] = sha(history_path)
        extra_inputs[f'completed/{run}/complete.json'] = sha(complete_path)
        predpath = args.inputs/'test_evaluation'/run/'predictions.csv'
        f = pd.read_csv(predpath)
        if f.duplicated(KEYS).any() or len(f) != len(meta):
            raise ValueError('Invalid prediction population')
        joined = f.merge(meta, on=KEYS, suffixes=('', '_pack'), validate='one_to_one')
        assert len(joined) == len(f)
        assert np.array_equal(joined.cl_label, joined.cl_label_pack)
        assert np.allclose(joined.e_true, joined.e_true_pack, rtol=1e-8, atol=1e-8)
        t = exp['working_threshold']
        r = dict(model=model, seed=seed, threshold=t, total=measure(joined, t), strata=[])
        assert np.isclose(r['total']['auc'], exp['roc_auc'], atol=1e-7, rtol=0)
        assert np.isclose(r['total']['mae_GeV'], exp['energy_all_signal']['mae_GeV'], atol=1e-8, rtol=0)
        for axis in ('size', 'multiplicity'):
            for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:]):
                subset = joined.loc[(joined[axis] > lo) & (joined[axis] <= hi)]
                r['strata'].append(dict(axis=axis, lo=lo, hi=hi, **measure(subset, t)))
        for side in (0, 1):
            r['strata'].append(dict(axis='side', side=side, **measure(joined.loc[joined.cl_istop == side], t)))
        for lo, hi in zip(ENERGY_EDGES[:-1], ENERGY_EDGES[1:]):
            subset = joined.loc[(joined.cl_label == 1) & (joined.e_true >= lo) & (joined.e_true < hi)]
            metrics = measure(subset, t)
            # A truth-positive slice has no background denominator. Its
            # trivially unit purity must not appear as measured separation.
            metrics['purity'] = None
            r['strata'].append(dict(axis='signal_energy', lo=float(lo), hi=float(hi), **metrics))
            subset = joined.loc[(joined.e_pred >= lo) & (joined.e_pred < hi)]
            r['strata'].append(dict(axis='reconstructed_energy', lo=float(lo), hi=float(hi), **measure(subset, t)))
        output.append(r)
        print(run, 'verified and stratified', flush=True)
    manifest.update(extra_inputs)
    qa = {'outputs': {}, 'canonical_inputs_sha256': manifest, 'input_files_verified': len(manifest),
          'data_and_annotation_bounds_asserted': True, 'source_script_sha256': sha(Path(__file__)),
          'visual_inspection_complete': False}
    (args.output/'per_seed.json').write_text(json.dumps(clean(output), indent=2, allow_nan=False)+'\n')
    (args.output/'summary.json').write_text(json.dumps(clean(summarize(output)), indent=2, allow_nan=False)+'\n')
    qa['per_seed_sha256'] = sha(args.output/'per_seed.json')
    qa['summary_sha256'] = sha(args.output/'summary.json')
    plots(args, output, qa)
    qa.update(no_training=True, no_inference=True, no_test_tuning=True,
              scope='Reused test988976; 222251 native candidates /36713 positive /91081 events; no source uncertainty')
    (args.output/'validation.json').write_text(json.dumps(qa, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
