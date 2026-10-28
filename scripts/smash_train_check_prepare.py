"""Check a prepare export against the acceptance criteria in train_launch.json (written before the run).

usage: python scripts/smash_train_check_prepare.py <repo> <summary.json> <receipt.json> <published manifest sha256> <out.json>
"""
import json
from pathlib import Path
import sys

repo, summary_path, receipt_path, published_manifest_sha, out = sys.argv[1:6]
sys.path.insert(0, str(Path(repo).parent))
from HGNDRecoGNN.scripts.smash_train_prepare import load_split, roles_of, used_contents  # noqa: E402

split = load_split(Path(repo) / 'results/smash_train_20261028/split/split_manifest.json')
s = json.loads(Path(summary_path).read_text())
r = json.loads(Path(receipt_path).read_text())
ids = used_contents(split)
checks, notes = {}, {}
checks['state'] = s['state'] == 'complete' and r['state'] == 'complete'
checks['split_sha256'] = s['split_sha256'] == '848ebf492b254b75e09297b95f89d7c2773f7877afe5e629520d520904a59886'
checks['source_sums'] = s['source_sums_sha256'] == '4f82efeae9024a33fd7d181674da0b904a55a88f250fd3ebade9ac16152c033d'
checks['files_578'] = sorted(s['files']) == sorted(ids) and len(ids) == 578
empty = sorted(c for c, f in s['files'].items() if f['empty_after_cuts'])
a = sum('A' in f['packs'] for f in s['files'].values())
b = sum(k.startswith('B/') for f in s['files'].values() for k in f['packs'])
empty_b = sum(len(roles_of(split, c)) for c in empty)
checks['packs'] = a + len(empty) == 578 and b + empty_b == 1724
notes['packs'] = {'A': a, 'B': b, 'empty_after_cuts': empty}
fold_ok = True
for fold in split['folds']:
    k = fold['fold']
    sc = s['scalers'][f'fold_{k}']
    train = sorted(fold['train'])
    fold_ok &= sc['files'] == train and not (set(train) & (set(fold['validation']) | set(fold['test'])))
    fold_ok &= all(side['n_samples_seen'] > 0 for side in sc['sides'].values())
    notes[f'fold_{k}_train_files'] = len(train)
checks['scalers_train_only'] = bool(fold_ok)
ac = s['audit_comparison']
checks['audit_429_bit_identical'] = ac['files_compared'] == 429 and ac['all_bit_identical']
notes['audit_comparison'] = ac
checks['publication'] = r['manifest_sha256'] == published_manifest_sha
notes['access_retries'] = {'receipt_prepare': s['access_retries'], 'publish': r.get('access_retries')}
notes['phase_seconds'] = s['phase_seconds']
result = {'accepted': all(checks.values()), 'checks': checks, 'notes': notes,
          'manifest_sha256': r['manifest_sha256'], 'bundle_dir': r['bundle_dir'], 'archive_sha256': r['archive_sha256']}
Path(out).write_text(json.dumps(result, indent=1) + '\n')
print(json.dumps({k: result[k] for k in ('accepted', 'checks')}, indent=1))
print(json.dumps({k: v for k, v in notes.items() if k != 'access_retries'})[:800])
print('access retries:', len(s['access_retries']), 'publish:', len(r.get('access_retries') or []))
