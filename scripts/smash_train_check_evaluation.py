"""Check a downloaded SMASH evaluation export before any of its numbers are read.

The export directory holds receipt.json, manifest.json and the extracted bundle
(summary.json, evaluation/summary.json, evaluation/predictions/...). This verifies the
publication hashes, the fold's test files against the split, that each seed was scored by
the verified training run of this fold with the cuts frozen in its complete.json, and every
prediction CSV against its recorded hash and column layout. It prints no metric.

usage: python scripts/smash_train_check_evaluation.py <out.json> <export_dir> <training_record_dir>
  training_record_dir: results/smash_train_20261028/train_4466016 (task_<t>/complete.json)
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.scripts.smash_train_check_task import (  # noqa: E402
    CODE_DIGEST, PREPARE_MANIFEST_SHA, SPLIT_SHA, TASKS)
from HGNDRecoGNN.scripts.smash_train_prepare import load_split  # noqa: E402

ARMS = ('A0', 'A1', 'B0', 'B1', 'Bcomp')
SEEDS = ('42', '123', '456')
COLUMNS = 'Row,ClusterID,cl_score,e_pred,cl_istop,cl_label,e_true,cluster_size,e_pred_raw'
SPLIT = ROOT / 'results/smash_train_20261028/split/split_manifest.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    out, export, records = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    r = json.loads((export / 'receipt.json').read_text())
    m = json.loads((export / 'manifest.json').read_text())
    top = json.loads((export / 'summary.json').read_text())
    s = json.loads((export / 'evaluation/summary.json').read_text())
    listed = {x['member']: x['sha256'] for x in m['files']}
    fold = s['fold']
    test = sorted(load_split(SPLIT)['folds'][fold]['test'])
    checks = {
        'receipt_complete': r['state'] == 'complete' and top['state'] == 'complete' and s['state'] == 'complete',
        'manifest_sha': sha(export / 'manifest.json') == r['manifest_sha256'],
        'archive_sha_in_manifest': m['archive_sha256'] == r['archive_sha256'],
        'summary_sha': sha(export / 'summary.json') == listed['summary.json'] == r['summary_files']['summary.json'],
        'evaluation_summary_sha': sha(export / 'evaluation/summary.json') == top['evaluation_sha256']
        == listed['evaluation/summary.json'],
        'fold': top['fold'] == fold,
        'test_files_equal_split': s['test_files'] == test,
        'identity': (s['identity']['code_digest'] == CODE_DIGEST and s['identity']['split_sha256'] == SPLIT_SHA
                     and s['identity']['prepare_manifest_sha256'] == PREPARE_MANIFEST_SHA),
        'no_test_threshold_tuning': s['test_threshold_tuning'] is False,
    }
    training = {}
    for seed in SEEDS:
        t = int(s['training'][seed]['task'])
        complete_path = records / f'task_{t}' / 'complete.json'
        complete = json.loads(complete_path.read_text())
        training[seed] = {
            'task': t,
            'fold_seed': TASKS[t] == (fold, int(seed)),
            'complete_sha_equal_verified': s['training'][seed]['complete_sha256'] == sha(complete_path),
            'cuts_equal_frozen': s['training'][seed]['cuts'] == complete['cuts']
            and all(s['metrics'][arm][seed]['cut'] == complete['cuts'][arm] for arm in ARMS),
            'selection_equal': s['training'][seed]['selection'] == {
                h: complete['selection'][h] for h in ('e_cls', 'e_en', 'e_comp')}}
    checks['training_runs'] = all(all(v for k, v in x.items() if k != 'task') for x in training.values())
    counts, bad = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            items = s['predictions'][arm][seed]
            counts[f'{arm}/{seed}'] = len(items)
            for cid, record in items.items():
                path = export / 'evaluation' / record['path']
                member = 'evaluation/' + record['path']
                if cid not in test or listed.get(member) != record['sha256'] or sha(path) != record['sha256']:
                    bad.append(member)
                    continue
                with open(path) as stream:
                    if stream.readline().strip() != COLUMNS:
                        bad.append(member + ' (columns)')
    scored = set(s['predictions']['A0']['42'])
    checks['prediction_files'] = not bad and len(set(counts.values())) == 1 and all(
        set(s['predictions'][arm][seed]) == scored for arm in ARMS for seed in SEEDS)
    result = {'accepted': all(checks.values()), 'checks': checks, 'training': training,
              'prediction_counts': counts, 'bad_predictions': bad[:20],
              'test_files': len(test), 'scored_files': len(scored),
              'unscored_test_files': sorted(set(test) - scored), 'empty_after_cuts': s['empty_after_cuts'],
              'access_retries': {'evaluate': len(top.get('access_retries', [])),
                                 'publish': len(r.get('access_retries', []))},
              'bundle_dir': r['bundle_dir'], 'manifest_sha256': r['manifest_sha256']}
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: result[k] for k in ('accepted', 'checks', 'test_files', 'scored_files',
                                              'unscored_test_files', 'empty_after_cuts', 'access_retries')},
                     indent=1))


if __name__ == '__main__':
    main()
