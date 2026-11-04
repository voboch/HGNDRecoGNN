"""Fixed-policy validation reduction; reads saved predictions, never models."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.scripts import reco_rn_heldout as rn  # noqa: E402

spec = importlib.util.spec_from_file_location('accepted_closure', ROOT/'reports/label_ablation_results_20261104/collect.py')
closure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(closure)
SEEDS = [42, 123, 456]
TAGS = ['zeroSpot', 'defaultSpot', 'bigSpot']
RUNGS = ['T1', 'C1', 'C2', 'C3', 'C4']
PAIRS = {'18/0': (1, 0), '90/0': (2, 0), '90/18': (2, 1)}
OLD_COUNTS_SHA = '2f2f7ac1342651631d105701012776c539145166c4ae6826ad3522a239785218'
SPLIT_SHA = '848ebf492b254b75e09297b95f89d7c2773f7877afe5e629520d520904a59886'
BASELINE_COMMIT = 'd739c54'


def read(path):
    return json.loads(path.read_text())


def source_certificate(complete):
    identity = complete['identity']
    rows = []
    for path in identity['code_files']:
        rel = path.removeprefix('HGNDRecoGNN/')
        data = subprocess.run(['git', 'show', f'{BASELINE_COMMIT}:{rel}'], cwd=ROOT,
                              check=True, capture_output=True).stdout
        rows.append(f'{hashlib.sha256(data).hexdigest()}  {path}\n')
    digest = hashlib.sha256(''.join(rows).encode()).hexdigest()
    assert digest == identity['code_digest']
    # These functions prove the order of the actual saved baseline producer.
    paths = ['scripts/smash_train.py', 'scripts/smash_transfer_infer.py',
             'data/student_pipeline.py', 'training/eval.py']
    for rel in paths:
        data = subprocess.run(['git', 'show', f'{BASELINE_COMMIT}:{rel}'], cwd=ROOT,
                              check=True, capture_output=True).stdout
        assert data == (ROOT/rel).read_bytes(), rel
    return {'commit': BASELINE_COMMIT, 'code_digest': digest,
            'code_files': len(rows), 'unchanged_order_producer_files': paths}


def summarize(counts):
    """Shared draws, count before ratio; counts shape arm,seed,file,rung,hi/lo."""
    ids = [r['content'] for r in counts]
    array = np.asarray([r['counts'] for r in counts], np.int64).transpose(1, 2, 0, 3, 4)
    rng = np.random.default_rng(20261026)
    central, replicas, matrices = [], [], {}
    for tag in TAGS:
        ix = [i for i, cid in enumerate(ids) if cid.startswith(tag+'/')]
        x = array[:, :, ix]
        draw = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), 400)
        matrices[tag] = {'contents': [ids[i] for i in ix], 'draw_sha256': hashlib.sha256(draw.tobytes()).hexdigest()}
        total = x.sum(axis=2)
        sample = np.einsum('bf,asfrh->asbrh', draw, x)
        assert np.all(total[..., 1] > 0) and np.all(sample[..., 1] > 0)
        central.append(total[..., 0]/total[..., 1])
        replicas.append(sample[..., 0]/sample[..., 1])
    central = np.stack(central, axis=-1)  # arm,seed,rung,potential
    replicas = np.stack(replicas, axis=-1)  # arm,seed,draw,rung,potential
    assert np.isfinite(central).all() and np.isfinite(replicas).all()
    result = {}
    for pair, (hi, lo) in PAIRS.items():
        assert np.all(central[..., lo] > 0) and np.all(replicas[..., lo] > 0)
        c = 100*(central[..., hi]/central[..., lo]-1)
        b = 100*(replicas[..., hi]/replicas[..., lo]-1)
        delta, boot_delta = c[1]-c[0], b[1]-b[0]
        gap = c[:, :, -1]-c[:, :, 0]
        boot_gap = b[:, :, :, -1]-b[:, :, :, 0]
        assert np.array_equal(c[0, :, :2], c[1, :, :2])
        result[pair] = {'contrast_percent': c.tolist(), 'file_SD_percent': b.std(axis=2, ddof=1).tolist(),
                        'delta_no_floor_minus_baseline_pp': delta.tolist(),
                        'delta_file_SD_pp': boot_delta.std(axis=1, ddof=1).tolist(),
                        'C4_minus_T1_pp': gap.tolist(), 'C4_minus_T1_file_SD_pp': boot_gap.std(axis=2, ddof=1).tolist(),
                        'paired_training_seed_delta': {r: closure.paired(delta[:, j]) for j, r in enumerate(RUNGS)}}
    return result, matrices


def main():
    # Reuse the accepted archive verifier; every member hash is checked.
    closure.unpack()
    old_path = ROOT/'results/smash_train_20261028/heldout_rn/per_file_counts.json'
    assert closure.sha(old_path) == OLD_COUNTS_SHA
    previous = {r['content']: r for records in read(old_path)['primary']['B0'].values() for r in records}
    patch_path = ROOT/'results/label_ablation_20261103/prepare_4482805/extracted/summary.json'
    prepare_path = ROOT/'results/smash_train_20261028/prepare_4465983/summary.json'
    assert closure.sha(patch_path) == closure.PATCH_SHA and closure.sha(prepare_path) == closure.PREPARE_SHA
    patches, prepare = read(patch_path)['files'], read(prepare_path)['files']
    split_path = ROOT/'results/smash_train_20261028/split/split_manifest.json'
    assert closure.sha(split_path) == SPLIT_SHA
    events = rn.load_truth(ROOT, rn.TRUTH)
    cert, rows, coverage, soft = None, {}, {}, []
    for task in range(9):
        fold, si = divmod(task, 3)
        directory = closure.EXTRACTED/f'task_{task}'
        old = read(directory/'baseline/train/complete.json')
        new = read(directory/'segment_2/complete.json')
        if cert is None:
            cert = source_certificate(old)
        assert old['identity']['code_digest'] == cert['code_digest']
        assert old['identity']['split_sha256'] == new['identity']['split_sha256'] == SPLIT_SHA
        assert old['identity']['prepare_receipt_sha256'] == closure.PREPARE_SHA
        assert old['fold'] == new['fold'] == fold and old['seed'] == new['seed'] == SEEDS[si]
        assert closure.sha(directory/'baseline/train/complete.json') == new['identity']['baseline_complete_sha256']
        assert old['epochs'] == new['epochs'] == 30
        cut = old['cuts']['B0']
        assert np.isfinite(cut) and cut == new['operating_point']['cut']
        prior = np.load(directory/'baseline/train/validation/predictions_B.npz')
        expected = {cid for cid, item in prepare.items() if item['roles'].get(str(fold)) == 'validation'}
        assert set(new['predictions']) == set(prior['content']) == expected
        for cid, item in sorted(new['predictions'].items()):
            if not prepare[cid]['truth_bound']:
                continue
            path = directory/'segment_2'/item['path']
            assert closure.sha(path) == item['sha256']
            f = pd.read_csv(path)
            keys = f[['Row', 'cl_istop', 'ClusterID']].to_numpy(np.int64)
            header = json.dumps(['torch.int64', list(keys.shape)]).encode()
            assert hashlib.sha256(header+np.ascontiguousarray(keys).tobytes()).hexdigest() == patches[cid]['candidate_keys_sha256']
            assert len(np.unique(keys, axis=0)) == len(f)
            assert patches[cid]['native_pack_sha256'][f'B/fold_{fold}'] == prepare[cid]['packs'][f'B/fold_{fold}']['sha256']
            ix = np.flatnonzero(prior['content'] == cid)
            assert len(ix) == len(f) and np.array_equal(ix, np.arange(ix[0], ix[-1]+1))
            for source, col, dtype in [('cl_label', 'label_native', np.int8), ('e_true', 'energy_native', np.float32), ('cluster_size', 'cluster_size', np.int32)]:
                assert np.array_equal(prior[source][ix], f[col].to_numpy(dtype))
            tag = cid.split('/')[0]
            record = previous[cid]
            primary_member = prepare[cid]['primary_member'].removesuffix('_prim.csv')+'_prim.csv'
            assert primary_member == record['primary_member']
            primary = events[tag].xs(primary_member, level='member')
            candidate_rows = np.unique(keys[:, 0])
            span = int(candidate_rows.max())+1
            assert np.isin(candidate_rows, primary.index.to_numpy()).all()
            assert span == record['processed_span'] and len(candidate_rows) == record['events_with_candidates']
            truth = rn.truth_counts(primary, span, candidate_rows, rn.BANDS['primary'])
            assert np.array_equal(truth, np.asarray(record['truth_hi_lo']))
            coverage.setdefault(cid, {'fold': fold, 'primary_member': primary_member, 'processed_span': span,
                                     'candidate_events': len(candidate_rows), 'candidate_count': len(f),
                                     'keys_sha256': patches[cid]['candidate_keys_sha256'], 'truth_T0_T1_T2': truth.astype(int).tolist()})
            row = rows.setdefault(cid, {'content': cid, 'fold': fold, 'counts': [[None]*3 for _ in range(2)]})
            for ai in range(2):
                frame = f.copy()
                frame['cl_label'] = f.label_native.to_numpy(np.int8)
                frame['e_true'] = f.energy_native.to_numpy(np.float32)
                frame['cl_score'] = prior['cl_score'][ix] if ai == 0 else f.cl_score.to_numpy(np.float32)
                frame['e_pred'] = prior['e_pred'][ix] if ai == 0 else f.e_pred.to_numpy(np.float32)
                assert np.isfinite(frame[['cl_score', 'e_pred', 'e_true']].to_numpy()).all()
                cl = rn.cluster_counts(frame, cut, rn.BANDS['primary'])
                row['counts'][ai][si] = np.vstack((truth[1], cl)).astype(int).tolist()
                added = (f.label_disabled.to_numpy() == 1) & (f.label_native.to_numpy() != 1)
                selected = frame.cl_score.to_numpy() > cut
                bc = rn.band_counts(frame.e_pred.to_numpy()[added & selected], np.ones(int((added & selected).sum())), rn.BANDS['primary'])
                soft.append({'content': cid, 'seed': SEEDS[si], 'arm': ai, 'selected_added_soft_hi_lo': bc.astype(int).tolist()})
    assert len(rows) == len(coverage) == 286
    per_file = [rows[cid] for cid in sorted(rows)]
    result, matrices = summarize(per_file)
    out = {'schema': 'hgnd-rn-validation-v1', 'population': 'selected validation S286',
           'arms': ['native_B0_raw', 'no_floor_raw'], 'seeds': SEEDS, 'rungs': RUNGS,
           'groups': len(rows), 'candidates': sum(x['candidate_count'] for x in coverage.values()),
           'pairs': result, 'per_file_counts': per_file, 'coverage': coverage, 'draws': matrices,
           'selected_added_soft': soft, 'source_certificate': cert,
           'inputs': {'closure_manifest': closure.MANIFEST_SHA, 'previous_truth_counts': OLD_COUNTS_SHA,
                      'split': SPLIT_SHA, 'protocol': closure.sha(HERE/'protocol.md'), 'analysis': closure.sha(Path(__file__)),
                      'helper_sha256': {p: closure.sha(ROOT/p) for p in
                                       ['scripts/reco_rn_heldout.py', 'scripts/reco_rn_retrospective.py',
                                        'reports/label_ablation_results_20261104/collect.py',
                                        'analysis/truth_provenance.py']}},
           'new_model_calls': 0, 'new_jobs': [], 'F_scored': False, 'cut_or_calibration_fitted': False}
    (HERE/'results.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'groups': len(rows), 'candidates': out['candidates'],
                      'C4_delta_pp': {p: v['paired_training_seed_delta']['C4'] for p,v in result.items()}}, indent=2))


if __name__ == '__main__':
    main()
