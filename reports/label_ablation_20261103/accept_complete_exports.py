"""Read-only acceptance and compact export of the existing 15+15 campaign.

Prefix this file with a PINS dict from the local registered launch plan and
baseline receipts, then send it to python3 on charisma. No model is executed.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def audit_bundle(receipt, destination, keep):
    bundle = Path(receipt['bundle_dir'])
    assert receipt['state'] == 'complete'
    assert sha(bundle / 'manifest.json') == receipt['manifest_sha256']
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['state'] == 'complete'
    assert manifest['archive_sha256'] == receipt['archive_sha256'] == sha(bundle / 'inputs.tar')
    assert (bundle / 'inputs.tar').stat().st_size == receipt['archive_bytes'] == manifest['archive_bytes']
    expected = {v['member']: v for v in manifest['files']}
    assert len(expected) == len(manifest['files']) == receipt['files']
    seen, docs = set(), {}
    with tarfile.open(bundle / 'inputs.tar') as archive:
        for member in archive:
            assert member.isfile() and member.name in expected and member.name not in seen
            seen.add(member.name)
            stream = archive.extractfile(member)
            h, size, chunks = hashlib.sha256(), 0, []
            wanted = keep(member.name) or member.name.endswith('.json')
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
                size += len(block)
                if wanted:
                    chunks.append(block)
            assert size == expected[member.name]['bytes']
            assert h.hexdigest() == expected[member.name]['sha256']
            if wanted:
                data = b''.join(chunks)
                if member.name.endswith('.json'):
                    docs[member.name] = json.loads(data)
                if keep(member.name):
                    write(destination / member.name, data)
    assert seen == set(expected)
    for name, digest in receipt['summary_files'].items():
        assert expected[name]['sha256'] == digest
    write(destination / 'publication_manifest.json', (bundle / 'manifest.json').read_bytes())
    write(destination / 'publication_receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())
    return expected, docs


source = Path(PINS['source'])
root, receipts = Path(PINS['campaign']), Path(PINS['receipts'])
export = receipts / 'closure_export_20261104_v1'
export.mkdir(exist_ok=False)
assert sha(source / 'SHA256SUMS') == PINS['source_sums_sha256']
source_entries = []
for line in (source / 'SHA256SUMS').read_text().splitlines():
    digest, name = line.split(None, 1)
    p = source / name.lstrip('*')
    assert sha(p) == digest and not (p.stat().st_mode & 0o222)
    source_entries.append(name)
proof = {'observed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'state': 'accepted', 'source_files_readonly_verified': len(source_entries),
         'source_sums_sha256': PINS['source_sums_sha256'], 'tasks': {},
         'models_or_tests_run': False, 'new_jobs_submitted': []}
for task in range(9):
    target = export / f'task_{task}'
    records, docs, entries = {}, {}, {}
    for segment in (1, 2):
        matches = sorted(receipts.glob(f'task_{task}_segment_{segment}_*.json'))
        assert len(matches) == 1
        records[segment] = json.loads(matches[0].read_text())
        keep = lambda n, segment=segment: n.endswith('.json') or (segment == 2 and n.startswith('validation/')) or (segment == 2 and n == 'train/checkpoints/resume.pt')
        entries[segment], docs[segment] = audit_bundle(records[segment], target / f'segment_{segment}', keep)
    one, two = docs[1]['segment.json'], docs[2]['complete.json']
    summary = docs[2]['summary.json']
    assert one['state'] == 'planned_boundary' and one['epochs_committed'] == 15 and one['complete_seed'] is False
    assert two['state'] == 'complete' and two['complete_seed'] is True and two['epochs'] == 30
    assert summary['scientific_seed_complete'] is True and summary['epochs_committed'] == 30
    assert summary['complete_sha256'] == entries[2]['complete.json']['sha256']
    identity = two['identity']
    assert identity == one['identity'] == summary['identity']
    assert identity['source']['sums_sha256'] == PINS['source_sums_sha256']
    assert identity['split_sha256'] == PINS['split_sha256'] and identity['prepare_sha256'] == PINS['prepare_sha256']
    assert identity['patch_manifest_sha256'] == PINS['patch_summary_sha256']
    assert identity['baseline_complete_sha256'] == PINS['baseline'][str(task)]['complete_sha256']
    assert two['fold'] == task // 3 and two['seed'] == [42, 123, 456][task % 3]
    assert two['validation_only'] and not two['test_outcomes_opened'] and not two['soft_calibration_fitted']
    assert not two['operating_point']['cut_refitted'] and two['operating_point']['calibration'] is None
    assert not two['run']['mirror']['failures']
    history = docs[2]['train/history.json']
    assert [v['epoch'] for v in history] == list(range(30))
    assert history[:15] == docs[1]['train/checkpoints/history.json']
    assert all(v['validation']['aggregation'] == 'population' for v in history)
    effective = docs[2]['train/effective_config.json']
    assert effective['identity'] == identity and effective['epochs'] == 30
    assert effective['resumed']['next_epoch'] == 15
    assert effective['resumed']['state_sha256'] == one['mirror_state_sha256']
    for epoch in range(15):
        name = f'train/checkpoints/epoch_{epoch:03d}.pt'
        assert entries[1][name]['sha256'] == entries[2][name]['sha256']
    for name, digest in two['training_hashes'].items():
        assert entries[2]['train/' + name]['sha256'] == digest
    for item in two['predictions'].values():
        assert entries[2][item['path']]['sha256'] == item['sha256']
    for key, component in (('e_cls', 'cluster_cls'), ('e_en', 'cluster_energy')):
        values = [v['validation']['components'][component] for v in history]
        assert two['selection'][key] == min(range(30), key=lambda i: values[i])
    mirror = root / 'mirror' / f'task_{task}'
    state_sha = sha(mirror / 'state.json')
    state = json.loads((mirror / 'state.json').read_text())
    assert state['next_epoch'] == 30 and state['resume'] == 'resume_029.pt'
    for name, digest in state['sha256'].items():
        assert sha(mirror / name) == digest
        archived = 'train/checkpoints/' + ('resume.pt' if name == state['resume'] else name)
        assert entries[2][archived]['sha256'] == digest
    assert sha(mirror / 'state.json') == state_sha
    write(target / 'mirror_state.json', (mirror / 'state.json').read_bytes())
    base = PINS['baseline'][str(task)]
    base_entries, base_docs = audit_bundle(base['receipt'], target / 'baseline', lambda n: n.endswith('.json') or n in ('train/validation/predictions_B.npz', 'train/validation/threshold_scan_B.csv'))
    baseline = base_docs['train/complete.json']
    assert base_entries['train/complete.json']['sha256'] == base['complete_sha256']
    assert baseline['epochs'] == 30 and baseline['evaluable'] and baseline['state'] == 'complete'
    assert two['operating_point']['cut'] == baseline['cuts']['B0']
    proof['tasks'][str(task)] = {'fold': two['fold'], 'seed': two['seed'], 'complete_sha256': entries[2]['complete.json']['sha256'],
                               'complete_30epochs': True, 'segment1_receipt': records[1], 'segment2_receipt': records[2],
                               'baseline_complete_sha256': base['complete_sha256'], 'mirror_state_sha256': state_sha,
                               'prefix_15_epochs_identical': True, 'resume_state_bound': True,
                               'new_members_verified': len(entries[1]) + len(entries[2]), 'baseline_members_verified': len(base_entries),
                               'predictions_verified': len(two['predictions'])}
proof['squeue'] = subprocess.check_output(['squeue', '-j', '4482848,4482849', '-h', '-o', '%i|%T|%R|%S'], text=True)
proof['sacct'] = subprocess.check_output(['sacct', '-j', '4482848,4482849', '--format=JobID,State,ExitCode,Elapsed,Start,End,NodeList', '-P', '-n'], text=True)
assert not proof['squeue'].strip()
write(export / 'acceptance.json', (json.dumps(proof, indent=2) + '\n').encode())
files = {str(p.relative_to(export)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(export.rglob('*')) if p.is_file()}
write(export / 'compact_manifest.json', (json.dumps({'files': files}, indent=2) + '\n').encode())
archive = export.with_suffix('.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as t:
    for p in sorted(export.rglob('*')):
        if p.is_file():
            t.add(p, arcname=str(p.relative_to(export)))
print(json.dumps({'state': 'accepted', 'complete_models': 9, 'compact_archive': str(archive),
                  'archive_sha256': sha(archive), 'bytes': archive.stat().st_size,
                  'compact_manifest_sha256': sha(export / 'compact_manifest.json')}, indent=2))
