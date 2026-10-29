"""Check downloaded SMASH training exports and the stop rules that gate each fold's evaluation.

For every task directory (receipt.json, manifest.json, complete.json and summary.json fetched from
the published bundle) this verifies the publication hashes, the run identity and the frozen
selection, then applies the protocol's stop rules per fold: e_cls = 0 in two or more seeds sends
the census back; a head minimum at the last epoch flags the run as budget-limited.

usage: python scripts/smash_train_check_task.py <out.json> <task_dir> [<task_dir> ...]
"""
import hashlib
import json
from pathlib import Path
import sys

TASKS = {t: (t // 3, (42, 123, 456)[t % 3]) for t in range(9)}
EPOCHS = 30
CODE_DIGEST = '4ee0d70fb9229b67cb28732431dc012b2e1e9a43089c48916951efdb20b57d3c'
SPLIT_SHA = '848ebf492b254b75e09297b95f89d7c2773f7877afe5e629520d520904a59886'
PREPARE_MANIFEST_SHA = 'c0156ec47c8ef9d7e5b4b714660b53a4e8e5fc3a10ad9a0c6421ecc7b1130501'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(task_dir):
    d = Path(task_dir)
    r = json.loads((d / 'receipt.json').read_text())
    m = json.loads((d / 'manifest.json').read_text())
    c = json.loads((d / 'complete.json').read_text())
    s = json.loads((d / 'summary.json').read_text())
    listed = {x['member']: x['sha256'] for x in m['files']}
    t = int(s['task'])
    checks = {
        'receipt_complete': r['state'] == 'complete' and s['state'] == 'complete' and c['state'] == 'complete',
        'manifest_sha': sha(d / 'manifest.json') == r['manifest_sha256'],
        'archive_sha_in_manifest': m['archive_sha256'] == r['archive_sha256'],
        'complete_sha': sha(d / 'complete.json') == listed['train/complete.json'] == s['complete_sha256'],
        'summary_sha': sha(d / 'summary.json') == listed['summary.json'] == r['summary_files']['summary.json'],
        'fold_seed': (c['fold'], c['seed']) == TASKS[t] == (s['fold'], s['seed']),
        'epochs': c['epochs'] == EPOCHS and len(c['selection']['validation_composite']) == EPOCHS,
        'test_not_opened': c['test_opened'] is False,
        'evaluable': c['evaluable'] is True,
        'identity': (c['identity']['code_digest'] == CODE_DIGEST and c['identity']['split_sha256'] == SPLIT_SHA
                     and c['identity']['prepare_manifest_sha256'] == PREPARE_MANIFEST_SHA
                     and c['identity']['task'] == t),
    }
    sel = c['selection']
    return t, {'checks': checks, 'ok': all(checks.values()),
               'selection': {k: sel[k] for k in ('e_cls', 'e_en', 'e_comp')},
               'budget_limited': sel['budget_limited'], 'e_cls_at_epoch_0': sel['e_cls_at_epoch_0'],
               'cuts': c['cuts'], 'calibration_errors': c['calibration_errors'],
               'access_retries': {'task': len(s.get('access_retries', [])),
                                  'publish': len(r.get('access_retries', []))},
               'bundle_dir': r['bundle_dir'], 'manifest_sha256': r['manifest_sha256']}


def main():
    out, dirs = Path(sys.argv[1]), sys.argv[2:]
    tasks = dict(check(d) for d in dirs)
    folds = {}
    for k in range(3):
        done = {t: v for t, v in tasks.items() if TASKS[t][0] == k}
        if len(done) != 3:
            continue
        e0 = sum(v['e_cls_at_epoch_0'] for v in done.values())
        folds[k] = {'tasks': sorted(done), 'all_ok': all(v['ok'] for v in done.values()),
                    'e_cls_at_epoch_0_seeds': e0,
                    'stop_census_check': e0 >= 2,
                    'budget_limited_tasks': sorted(t for t, v in done.items() if any(v['budget_limited'].values())),
                    'evaluate_allowed': all(v['ok'] for v in done.values()) and e0 < 2}
    result = {'tasks': {str(t): v for t, v in sorted(tasks.items())}, 'folds': {str(k): v for k, v in folds.items()}}
    out.write_text(json.dumps(result, indent=1) + '\n')
    for t, v in sorted(tasks.items()):
        print(t, 'ok' if v['ok'] else 'FAIL', {k: x for k, x in v['checks'].items() if not x} or '',
              v['selection'], 'budget_limited' if any(v['budget_limited'].values()) else '', v['access_retries'])
    for k, v in folds.items():
        print('fold', k, v)


if __name__ == '__main__':
    main()
