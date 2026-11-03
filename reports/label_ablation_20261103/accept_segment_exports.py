"""Read-only acceptance of existing planned 15-epoch exports on cHARISMa.

Run through one-hop SSH with task IDs as arguments. This reads receipts,
archives and stable mirror generations; it never submits or changes a job.
The continuation trainer validates the serialized optimizer/RNG/history state.
"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path('/scratch/vbocharnikov/hgnd/experiments/label_ablation_20261103_921e04a4e6d1')
RECEIPTS = Path('/home/vbocharnikov/hgnd_experiments/label_ablation_20261103_921e04a4e6d1_receipts')
SOURCE_SHA = '921e04a4e6d1faec8e8b1a757e2a08df2b02f34d2a2f35e9d4e34c0ef750eb3c'
SPLIT_SHA = '848ebf492b254b75e09297b95f89d7c2773f7877afe5e629520d520904a59886'
PREPARE_SHA = '2b58f9ccce0efccb4c66afa96e93c4bbd124c1c61a22593baa05c26544de7133'
PATCH_SHA = 'd551fc41539742b54e9fe5b0d1036d1fe3909438f0cd0ee63f0248278d3c3bac'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


out = {'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'state': 'accepted_planned_boundaries', 'complete_30epoch_models': 0,
       'checks': ['receipt/manifest/archive/member SHA and sizes',
                  'same registered source/split/prepare/patch/fold/seed identity',
                  '15 ordered population-validation history entries',
                  'no scientific complete.json; continuation remains required',
                  'stable mirror state SHA and every named checkpoint/resume SHA',
                  'mirror epochs/resume identical to published archive'],
       'serialized_resume_contents': 'Not deserialized here; inherited continuation trainer checks next_epoch/history and restores model/optimizer/scheduler/RNG.',
       'models_or_tests_run': False, 'jobs_submitted': [], 'tasks': {}}
for task in map(int, sys.argv[1:]):
    assert 0 <= task < 9
    matches = sorted(RECEIPTS.glob(f'task_{task}_segment_1_*.json'))
    assert len(matches) == 1, (task, matches)
    receipt_path = matches[0]
    receipt = json.loads(receipt_path.read_text())
    assert receipt['state'] == 'complete'
    bundle = Path(receipt['bundle_dir'])
    assert bundle.parent == ROOT
    manifest_path, archive_path = bundle / 'manifest.json', bundle / 'inputs.tar'
    assert sha(manifest_path) == receipt['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['state'] == 'complete'
    assert sha(archive_path) == receipt['archive_sha256'] == manifest['archive_sha256']
    assert archive_path.stat().st_size == receipt['archive_bytes'] == manifest['archive_bytes']
    entries = {v['member']: v for v in manifest['files']}
    assert len(entries) == len(manifest['files']) == receipt['files']
    assert 'complete.json' not in entries
    documents, seen = {}, set()
    with tarfile.open(archive_path) as archive:
        for member in archive:
            assert member.isfile() and member.name in entries and member.name not in seen
            seen.add(member.name)
            data = archive.extractfile(member).read()
            item = entries[member.name]
            assert len(data) == item['bytes']
            assert hashlib.sha256(data).hexdigest() == item['sha256']
            if member.name.endswith('.json'):
                documents[member.name] = json.loads(data)
    assert seen == set(entries)
    summary, segment = documents['summary.json'], documents['segment.json']
    assert receipt['summary_files']['summary.json'] == entries['summary.json']['sha256']
    assert summary['state'] == 'complete' and summary['phase'] == 'planned_segment_1'
    assert summary['scientific_seed_complete'] is False and summary['requires_continuation'] is True
    assert summary['epochs_committed'] == segment['epochs_committed'] == 15
    assert segment['complete_seed'] is False and segment['state'] == 'planned_boundary'
    identity = summary['identity']
    assert identity == segment['identity'] == documents['train/effective_config.json']['identity']
    assert identity['source']['sums_sha256'] == SOURCE_SHA
    assert identity['split_sha256'] == SPLIT_SHA and identity['prepare_sha256'] == PREPARE_SHA
    assert identity['patch_manifest_sha256'] == PATCH_SHA and identity['epochs_total'] == 30
    assert identity['fold'] == task // 3 and identity['seed'] == [42, 123, 456][task % 3]
    execution = summary['execution']
    assert execution == segment['execution'] and execution['task'] == task and execution['segment'] == 1
    assert execution['test_queue_waived_by_human'] is True and execution['GPU_smoke_pass_claimed'] is False
    source = Path('/home/vbocharnikov/hgnd_experiments/label_ablation_20261103_921e04a4e6d1')
    plan = json.loads((source / 'HGNDRecoGNN/reports/label_ablation_20261103/launch_plan.json').read_text())
    assert identity['baseline_complete_sha256'] == plan['baseline'][str(task)]['sha256']
    history = documents['train/checkpoints/history.json']
    assert [h['epoch'] for h in history] == list(range(15))
    assert all(h['validation']['aggregation'] == 'population' for h in history)
    mirror = ROOT / 'mirror' / f'task_{task}'
    state_path = mirror / 'state.json'
    state_before = sha(state_path)
    assert state_before == segment['mirror_state_sha256']
    state = json.loads(state_path.read_text())
    assert state['next_epoch'] == 15 and state['resume'] == 'resume_014.pt'
    assert state['job'] == execution['job'] and int(state['array_task']) == task
    assert set(state['sha256']) == {f'epoch_{i:03d}.pt' for i in range(15)} | {'resume_014.pt'}
    for name, digest in state['sha256'].items():
        assert sha(mirror / name) == digest
        archived = 'train/checkpoints/' + ('resume.pt' if name == state['resume'] else name)
        assert entries[archived]['sha256'] == digest
    assert sha(state_path) == state_before
    out['tasks'][str(task)] = {'receipt_name': receipt_path.name, 'receipt_sha256': sha(receipt_path),
                              'receipt': receipt, 'manifest': manifest, 'identity': identity,
                              'execution': execution, 'epochs_committed': 15,
                              'history_sha256': entries['train/checkpoints/history.json']['sha256'],
                              'mirror_state_sha256': state_before, 'mirror_state': state,
                              'scientific_seed_complete': False}
out['squeue'] = subprocess.check_output(['squeue', '-j', '4482848,4482849', '-h', '-o', '%i|%T|%R|%S'], text=True)
out['sacct'] = subprocess.check_output(['sacct', '-j', '4482848,4482849', '--format=JobID,State,ExitCode,Elapsed,Start,End,NodeList', '-P', '-n'], text=True)
print(json.dumps(out, indent=2))
