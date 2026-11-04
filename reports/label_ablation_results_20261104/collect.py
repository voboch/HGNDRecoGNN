"""Compare saved validation predictions on certified keys; no model or fit."""
from pathlib import Path
import hashlib
import json
import math
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'results/label_ablation_20261103/closure_20261104'
EXTRACTED = DATA / 'extracted'
ARCHIVE_SHA = '197198ca97f59f1cb21e1c2ad686a26300a47a746fa708d153111cb3a22adc3d'
MANIFEST_SHA = 'b9762d07680828edc22d8265ad26fa545c7f917d5ddadf5ef8bf597e6afefb80'
PATCH_SHA = 'd551fc41539742b54e9fe5b0d1036d1fe3909438f0cd0ee63f0248278d3c3bac'
PREPARE_SHA = '2b58f9ccce0efccb4c66afa96e93c4bbd124c1c61a22593baa05c26544de7133'
SEEDS = [42, 123, 456]
# Display grids, not new cut/window choices. Endpoints cover every score.
GRID = np.linspace(0., 1., 201)
# Common-positive response display follows existing report bins; the final
# boundary is extended to cover all targets. Soft bins use the registered
# census's 0.1-GeV intervals. Neither binning selects a label or working point.
ENERGY_EDGES = np.array([0., .1, .2, .3, .5, .7, 1., 1.4, 1.8, 2.2, 2.6, 3., 4., 6., 10., np.inf])


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def unpack():
    archive = DATA / 'closure_export_20261104_v1.tar.gz'
    assert sha(archive) == ARCHIVE_SHA
    if not EXTRACTED.exists():
        EXTRACTED.mkdir()
        with tarfile.open(archive) as t:
            seen = set()
            for item in t:
                p = Path(item.name)
                assert item.isfile() and not p.is_absolute() and '..' not in p.parts and item.name not in seen
                seen.add(item.name)
                target = EXTRACTED / p
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as out:
                    out.write(t.extractfile(item).read())
    assert sha(EXTRACTED / 'compact_manifest.json') == MANIFEST_SHA
    manifest = json.loads((EXTRACTED / 'compact_manifest.json').read_text())
    for name, item in manifest['files'].items():
        p = EXTRACTED / name
        assert sha(p) == item['sha256'] and p.stat().st_size == item['bytes']
    return manifest


def metrics(label, score, energy, target, native, cut):
    selected = score > cut
    out = {'native': {}, 'disabled': {}}
    for ref, mask in (('native', native), ('disabled', label)):
        tp, nsel, npos = int((selected & mask).sum()), int(selected.sum()), int(mask.sum())
        out[ref] = {'n': len(mask), 'positive': npos, 'auc': float(roc_auc_score(mask, score)),
                    'ap': float(average_precision_score(mask, score)), 'selected': nsel, 'selected_positive': tp,
                    'purity': tp/nsel if nsel else None, 'efficiency': tp/npos if npos else None}
    for pop, mask in (('common_positive', native), ('added_soft', label & ~native)):
        error = energy[mask].astype(float) - target[mask].astype(float)
        out[pop] = {'n': int(mask.sum()), 'zero_energy_targets': int((target[mask] == 0).sum()),
                    'raw_bias_GeV': float(error.mean()), 'raw_mae_GeV': float(abs(error).mean()),
                    'selected': int((selected & mask).sum()), 'selected_fraction': float(selected[mask].mean()),
                    'mean_score': float(score[mask].mean())}
    return out


def paired(values):
    a = np.array(values, float)
    mean, sd = a.mean(), a.std(ddof=1)
    se = sd / math.sqrt(3)
    stat = mean / se if se else None
    return {'by_seed_42_123_456': a.tolist(), 'mean': float(mean), 'seed_SD': float(sd),
            'SE_mean_3_seeds': float(se), 't': float(stat) if stat is not None else None, 'df': 2,
            'two_sided_p_conditional_initialization': float(2*student_t.sf(abs(stat), 2)) if stat is not None else None,
            'critical_t_two_sided_0.05': float(student_t.ppf(.975, 2))}


def collect():
    manifest = unpack()
    patch_path = ROOT / 'results/label_ablation_20261103/prepare_4482805/extracted/summary.json'
    prepare_path = ROOT / 'results/smash_train_20261028/prepare_4465983/summary.json'
    assert sha(patch_path) == PATCH_SHA and sha(prepare_path) == PREPARE_SHA
    patches, prepare = json.loads(patch_path.read_text())['files'], json.loads(prepare_path.read_text())['files']
    records, certified, plot, observed_cids = [], {}, {}, {}
    for task in range(9):
        d = EXTRACTED / f'task_{task}'
        new = json.loads((d / 'segment_2/complete.json').read_text())
        old = json.loads((d / 'baseline/train/complete.json').read_text())
        fold, seed = task // 3, SEEDS[task % 3]
        assert new['fold'] == old['fold'] == fold and new['seed'] == old['seed'] == seed
        assert sha(d / 'baseline/train/complete.json') == new['identity']['baseline_complete_sha256']
        prior = np.load(d / 'baseline/train/validation/predictions_B.npz')
        assert set(prior['content']) == set(new['predictions']) == set(old['identity']['validation_files'])
        expected = {cid for cid, item in prepare.items() if item['roles'].get(str(fold)) == 'validation'}
        assert set(new['predictions']) == expected
        observed_cids[fold] = sorted(expected)
        parts = []
        for cid, item in sorted(new['predictions'].items()):
            path = d / 'segment_2' / item['path']
            assert sha(path) == item['sha256']
            f = pd.read_csv(path)
            keys = f[['Row', 'cl_istop', 'ClusterID']].to_numpy(np.int64)
            header = json.dumps(['torch.int64', list(keys.shape)]).encode()
            digest = hashlib.sha256(header + np.ascontiguousarray(keys).tobytes()).hexdigest()
            assert digest == patches[cid]['candidate_keys_sha256']
            assert patches[cid]['native_pack_sha256'][f'B/fold_{fold}'] == prepare[cid]['packs'][f'B/fold_{fold}']['sha256']
            assert patches[cid]['observation_sha256'] == prepare[cid]['observation']['sha256']
            ix = np.flatnonzero(prior['content'] == cid)
            assert len(ix) == len(f) and np.array_equal(ix, np.arange(ix[0], ix[-1]+1))
            assert len(np.unique(keys, axis=0)) == len(f)
            assert np.array_equal(prior['cl_label'][ix], f.label_native.to_numpy(np.int8))
            assert np.array_equal(prior['e_true'][ix], f.energy_native.to_numpy(np.float32))
            assert np.array_equal(prior['cluster_size'][ix], f.cluster_size.to_numpy(np.int32))
            assert np.array_equal(f.label_disabled.to_numpy(np.int8), f.cl_label.to_numpy(np.int8))
            assert np.array_equal(f.energy_disabled.to_numpy(np.float32), f.e_true.to_numpy(np.float32))
            native = f.label_native.to_numpy() == 1
            disabled = f.label_disabled.to_numpy() == 1
            assert np.all(disabled >= native)
            assert np.array_equal(f.energy_native.to_numpy(np.float32)[native], f.energy_disabled.to_numpy(np.float32)[native])
            f['content'] = cid
            base = f[['content', 'Row', 'cl_istop', 'ClusterID']].copy()
            base['base_score'], base['base_energy'] = prior['cl_score'][ix], prior['e_pred'][ix]
            f = f.merge(base, on=['content', 'Row', 'cl_istop', 'ClusterID'], how='outer', validate='one_to_one', indicator=True)
            assert len(f) == len(ix) and (f['_merge'] == 'both').all()
            f.drop(columns='_merge', inplace=True)
            parts.append(f)
            certified.setdefault(cid, {'fold': fold, 'n': len(f), 'keys_sha256': digest,
                                      'native_pack_sha256': patches[cid]['native_pack_sha256'][f'B/fold_{fold}'],
                                      'truth_bound': prepare[cid]['truth_bound']})
        f = pd.concat(parts, ignore_index=True)
        native, disabled = (f[k].to_numpy() == 1 for k in ('label_native', 'label_disabled'))
        target = f.energy_disabled.to_numpy(np.float32)
        cut = old['cuts']['B0']
        assert cut == new['operating_point']['cut']
        m, curves = {}, {}
        for arm, score_col, energy_col in (('baseline', 'base_score', 'base_energy'), ('no_floor', 'cl_score', 'e_pred')):
            score, energy = f[score_col].to_numpy(np.float32), f[energy_col].to_numpy(np.float32)
            assert np.isfinite(score).all() and np.isfinite(energy).all()
            assert ((score >= 0) & (score <= 1)).all()
            m[arm] = metrics(disabled, score, energy, target, native, cut)
            for ref, label in (('native', native), ('disabled', disabled)):
                fp, tp, _ = roc_curve(label, score)
                precision, recall, _ = precision_recall_curve(label, score)
                plot[f'{task}__{arm}__{ref}__roc'] = np.interp(GRID, fp, tp)
                plot[f'{task}__{arm}__{ref}__pr'] = np.interp(GRID, recall[::-1], precision[::-1])
            for pop, mask in (('common_positive', native), ('added_soft', disabled & ~native)):
                x, y = target[mask].astype(float), energy[mask].astype(float)
                prefix = f'{task}__{arm}__{pop}'
                count = np.histogram(x, ENERGY_EDGES)[0]
                assert count.sum() == mask.sum()
                plot[prefix+'__n'] = count
                plot[prefix+'__target_sum'] = np.histogram(x, ENERGY_EDGES, weights=x)[0]
                plot[prefix+'__pred_sum'] = np.histogram(x, ENERGY_EDGES, weights=y)[0]
        # Float32 round-trip is exact for the new CSV. Runner energy metrics used
        # the same float32 values, then float64 residual arithmetic.
        for ref in ('native', 'disabled'):
            expected_new = new['validation_metrics']['pooled'][ref]
            for key, value in expected_new.items():
                assert np.isclose(m['no_floor'][ref][key], value, rtol=0, atol=1e-13), (task, ref, key)
        for key in ('auc', 'ap', 'selected', 'purity', 'efficiency'):
            assert np.isclose(m['baseline']['native'][key], old['validation_metrics']['B0']['pooled'][key], rtol=0, atol=1e-13)
        for pop in ('common_positive', 'added_soft'):
            for key in ('n', 'zero_energy_targets', 'raw_bias_GeV', 'raw_mae_GeV'):
                assert np.isclose(m['no_floor'][pop][key], new['validation_metrics']['pooled'][pop][key], rtol=0, atol=1e-12)
        resume = torch.load(d / 'segment_2/train/checkpoints/resume.pt', map_location='cpu', weights_only=False)
        history = json.loads((d / 'segment_2/train/history.json').read_text())
        assert resume['next_epoch'] == 30 and resume['history'] == history and len(history) == 30
        assert all(k in resume for k in ('model', 'optimizer', 'scheduler', 'rng'))
        assert all(k in resume['rng'] for k in ('torch', 'numpy', 'python', 'cuda'))
        assert resume['arch_name'] == 'student_sage' and resume['optimizer']['state']
        before = json.loads((d / 'baseline/train/effective_config.json').read_text())
        after = json.loads((d / 'segment_2/train/effective_config.json').read_text())
        same = ('model','seed','architecture','epochs','batch_size','optimizer','lr','weight_decay','scheduler','milestones','gamma','validation_aggregation','loss','parameters','train_graphs','validation_graphs')
        assert all(before[k] == after[k] for k in same)
        records.append({'task': task, 'fold': fold, 'seed': seed, 'cut': cut, 'n': len(f),
                        'groups': len(expected), 'metrics': m,
                        'selection_baseline': {k: old['selection'][k] for k in ('e_cls','e_en','e_comp')},
                        'selection_no_floor': {k: new['selection'][k] for k in ('e_cls','e_en','e_comp')},
                        'recipe_fields_equal': list(same), 'resume_components_verified': True})
        for arm, doc in (('baseline', old), ('no_floor', new)):
            for component in ('cluster_cls','cluster_energy'):
                key = 'validation_' + component
                plot[f'{task}__{arm}__{component}__history'] = np.array(doc['selection'][key])
        print(f'Certified validation join task {task}: {len(f)} candidates', flush=True)
    assert sum(len(v) for v in observed_cids.values()) == len(set().union(*map(set, observed_cids.values()))) == 290
    macro, comparison = {}, {}
    for population, keys in (('native', ('auc','ap','purity','efficiency')),
                             ('disabled', ('auc','ap','purity','efficiency')),
                             ('common_positive', ('raw_bias_GeV','raw_mae_GeV')),
                             ('added_soft', ('raw_bias_GeV','raw_mae_GeV','selected_fraction','mean_score'))):
        for metric in keys:
            name = population + '/' + metric
            macro[name] = {}
            for arm in ('baseline','no_floor'):
                values = [float(np.mean([r['metrics'][arm][population][metric] for r in records if r['seed']==seed])) for seed in SEEDS]
                macro[name][arm] = {'by_seed_42_123_456': values, 'mean': float(np.mean(values)), 'seed_SD': float(np.std(values,ddof=1))}
            comparison[name] = paired(np.array(macro[name]['no_floor']['by_seed_42_123_456'])-macro[name]['baseline']['by_seed_42_123_456'])
    result = {'date': '2026-11-04', 'population': 'original 290 fold-validation content groups, including four unbound; same validation used for epoch selection and baseline cut',
              'aggregation': 'equal macro of 3 folds within each seed, then 3 paired training seeds; no pooled cross-fold AUC',
              'uncertainty': 'seed SD/paired SE conditional on fixed validation production and recipe; no fresh holdout, domain or file-sampling uncertainty',
              'n_unique_candidates': sum(v['n'] for v in certified.values()), 'groups': len(certified),
              'unbound_groups': [k for k,v in certified.items() if not v['truth_bound']],
              'reference_counts': {p:sum(r['metrics']['baseline'][p]['positive'] for r in records if r['seed']==42) for p in ('native','disabled')},
              'complete_models': 9, 'additional_model_calls': 0, 'F_outcomes_evaluated': False,
              'cut_refitted': False, 'calibration_fitted': False, 'records': records, 'macro_by_seed': macro,
              'paired_differences': comparison, 'certified_candidates': certified}
    (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(HERE/'plot_data.npz', grid=GRID, energy_edges=ENERGY_EDGES, **plot)
    pins = {'compact_archive_sha256': ARCHIVE_SHA, 'compact_manifest_sha256': MANIFEST_SHA,
            'patch_summary_sha256': PATCH_SHA, 'prepare_summary_sha256': PREPARE_SHA,
            'compact_files_verified': len(manifest['files']), 'results_sha256': sha(HERE/'results.json'),
            'plot_data_sha256': sha(HERE/'plot_data.npz'), 'collector_sha256': sha(Path(__file__))}
    (HERE/'input_pins.json').write_text(json.dumps(pins,indent=2)+'\n')


if __name__ == '__main__':
    collect()
