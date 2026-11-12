"""Accept both published GPU probes and project the registered fit segment caps."""
from argparse import ArgumentParser
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import tarfile

if not __debug__:
    raise RuntimeError('Acceptance checks require standard Python without -O')


def digest(data):
    return sha256(data).hexdigest()


parser = ArgumentParser(description=__doc__)
parser.add_argument('archive', type=Path, help='Tar of both published receipt/manifest/input triplets')
parser.add_argument('sacct', type=Path, help='Terminal sacct rows for array 4508255')
args = parser.parse_args()
archive_path, sacct_path = args.archive, args.sacct

root = Path(__file__).resolve().parent
submission = json.loads((root / 'probe_4508255_submission.json').read_bytes())
plan_bytes = (root / 'plan.json').read_bytes()
assert digest(plan_bytes) == submission['plan_sha256']
registered = json.loads(plan_bytes)
prepared_bytes = (root.parents[1] / 'results/smash_train_20261028/prepare_4465983/summary.json').read_bytes()
assert digest(prepared_bytes) == registered['prepare']['summary_sha256']
prepared = json.loads(prepared_bytes)

terminal = {}
for line in sacct_path.read_text().splitlines():
    if line.partition('|')[0] not in ('4508255_0', '4508255_9'):
        continue
    sid, raw, state, code, sec, start, end, node = line.split('|')
    task = int(sid.rsplit('_', 1)[-1])
    assert task not in terminal, f'Duplicate Slurm accounting row {sid}'
    assert state == 'COMPLETED' and code == '0:0', f'{sid}: {state} {code}'
    assert 0 < int(sec) <= submission['time_limit_seconds'], f'{sid}: invalid elapsed time'
    terminal[task] = dict(slurm_task=sid, job=raw, state=state, exit_code=code,
                          elapsed_seconds=int(sec), start_msk=start, end_msk=end, node=node)
assert set(terminal) == {0, 9}
accepted = {}
fold_zero = prepared['folds'][0]
with tarfile.open(archive_path) as outer:
    for task, model in ((0, 'student_dynedge'), (9, 'smash_hetero_sage')):
        case = terminal[task]
        key = f"probe_{task}_{case['job']}"
        receipt_bytes = outer.extractfile(key + '.json').read()
        receipt = json.loads(receipt_bytes)
        manifest_bytes = outer.extractfile(key + '/manifest.json').read()
        manifest = json.loads(manifest_bytes)
        archive = outer.extractfile(key + '/inputs.tar').read()
        assert receipt['state'] == manifest['state'] == 'complete'
        assert receipt['bundle_dir'] == submission['campaign'] + '/' + key
        assert digest(manifest_bytes) == receipt['manifest_sha256']
        assert digest(archive) == receipt['archive_sha256'] == manifest['archive_sha256']
        assert len(archive) == receipt['archive_bytes'] == manifest['archive_bytes']
        with tarfile.open(fileobj=io.BytesIO(archive)) as inner:
            assert inner.getnames() == ['summary.json']
            summary_bytes = inner.extractfile('summary.json').read()
        assert len(manifest['files']) == receipt['files'] == 1
        member = manifest['files'][0]
        assert member['member'] == 'summary.json' and member['bytes'] == len(summary_bytes)
        assert member['sha256'] == digest(summary_bytes) == receipt['summary_files']['summary.json']
        summary = json.loads(summary_bytes)
        assert summary['schema'] == 'smash-architecture-probe-v2' and summary['state'] == 'complete'
        assert (summary['model'], summary['fold'], summary['seed'], summary['job']) == (model, 0, 42, case['job'])
        for name in ('source_sums_sha256', 'plan_sha256'):
            assert summary[name] == submission[name]
        assert summary['split_sha256'] == registered['split_sha256']
        assert summary['prepare_manifest_sha256'] == registered['prepare']['manifest_sha256']
        assert summary['prepare_summary_sha256'] == registered['prepare']['summary_sha256']
        checks = summary['checks']
        assert checks['candidate_sizes'] == [1, 2, 7, 8, 9]
        assert all(checks[k] is True for k in ('truth_free_forward', 'candidate_isolation', 'resume_training', 'exact_live_restore'))
        assert 'V100' in summary['gpu'] and summary['rates']['max_memory_allocated_bytes'] > 0
        environment = summary['environment']
        assert environment['slurm_job_id'] == case['job'] and environment['slurm_array_task_id'] == str(task)
        assert environment['slurm_array_job_id'] == str(submission['array_job_id'])
        assert environment['hostname'].split('.')[0] == case['node'] and environment['cuda_runtime']
        for field in ('input_archive_check_seconds', 'sample_stage_seconds', 'epoch_seconds_projection'):
            assert math.isfinite(summary[field]) and summary[field] > 0, f'{key}: invalid {field}'
        for role in ('train', 'validation'):
            files = sorted((cid for cid in fold_zero[role] if not prepared['files'][cid]['empty_after_cuts']),
                           key=lambda cid: (prepared['files'][cid]['population']['edges'] / prepared['files'][cid]['population']['graphs'], cid))
            assert summary[f'{role}_files'] == sorted(fold_zero[role])
            assert summary['cohorts'][role]['sampled'] == [files[(len(files) - 1)*i//7] for i in range(8)]
            assert summary['cohorts'][role]['graphs_full_fold'] == sum(prepared['files'][cid]['population']['graphs'] for cid in files)
            rate = summary['rates'][role]
            assert rate['batches'] == 24 and rate['graphs'] == 24 * 256
            assert math.isfinite(rate['seconds_per_graph']) and rate['seconds_per_graph'] > 0
            assert math.isclose(rate['seconds_per_graph'], rate['seconds'] / rate['graphs'], rel_tol=1e-12)
        estimate = sum(cohort['graphs_full_fold'] * summary['rates'][role]['seconds_per_graph']
                       for role, cohort in summary['cohorts'].items())
        assert math.isclose(estimate, summary['epoch_seconds_projection'], rel_tol=1e-12)
        accepted[model] = {'slurm': case, 'receipt_sha256': digest(receipt_bytes), 'publication': receipt,
                           'summary_sha256': digest(summary_bytes), 'summary': summary}

def write(name, data):
    path = root / name
    payload = json.dumps(data, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == payload, f'Existing receipt differs: {path}'
        return
    with path.open('x') as out:
        out.write(payload)

acceptance = {'array_job_id': 4508255, 'state': 'accepted', 'terminal': terminal,
              'slurm_time_limit_seconds': submission['time_limit_seconds'],
              'source_sums_sha256': submission['source_sums_sha256'], 'models': accepted}
write('probe_4508255_acceptance.json', acceptance)

cost = {}
for model, result in accepted.items():
    summary = result['summary']
    rates = summary['rates']
    cost[model] = []
    for fold in prepared['folds']:
        graphs = {r: sum(prepared['files'][c]['population']['graphs'] for c in fold[r] if not prepared['files'][c]['empty_after_cuts'])
                  for r in ('train', 'validation')}
        times = {r: graphs[r] * rates[r]['seconds_per_graph'] for r in graphs}
        cost[model].append({'fold': fold['fold'], 'graphs': graphs, 'epoch_seconds': sum(times.values()),
                            'classifier_scoring_seconds': times['validation']})
rows = [f for model in cost.values() for f in model]
epoch_max = max(row['epoch_seconds'] for row in rows)
val_max = max(row['classifier_scoring_seconds'] for row in rows)
stage_measured = max(v['summary']['input_archive_check_seconds']+v['summary']['sample_stage_seconds'] for v in accepted.values())
stage_budget = math.ceil(max(900, 1.5 * stage_measured + 300) / 60) * 60
max_time = 9900
for n in range(2, 31):
    boundaries = [round(i*30/n) for i in range(n+1)]
    segments = []
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        restore = 180 if begin else 0
        publish = 300 + 2 * val_max if end == 30 else 90
        limit = 300 * math.ceil((stage_budget + restore + (end-begin)*(1.5*epoch_max+20) + publish) / 300)
        time = f'{limit // 3600:02d}:{limit % 3600 // 60:02d}:00'
        segments.append({'start_epoch': begin, 'end_epoch': end, 'time_limit_seconds': limit, 'time_limit': time,
                         'stage_budget_seconds': stage_budget, 'restore_and_republish_budget_seconds': restore,
                         'validation_and_publication_budget_seconds': publish})
    if max(seg['time_limit_seconds'] for seg in segments) <= max_time:
        break
else:
    raise ValueError('Observed resource estimates exceed the per-segment cap even at one epoch per job')
plan = {'schema': 'smash-architecture-fit-submission-v1',
        'source': {k: submission[k] for k in ('source','campaign','receipts','code_commit','source_sums_sha256','plan_sha256')},
        'probe_acceptance_sha256': digest((root/'probe_4508255_acceptance.json').read_bytes()),
        'prepare': registered['prepare'], 'split_sha256': registered['split_sha256'],
        'resources': {'partition':'rocky','account':'proj_1855','qos':'normal','constraint':'type_a','gpus':1,'cpus':4,
                      'mem':'0','requeue':False,'array':'0-17%2','max_active':2,'max_time_limit_seconds':max_time,
                      'exclude':['cn-007','cn-008','cn-034']},
        'epoch_budget':30, 'segments':segments, 'dependency':'afterok of whole predecessor array; kill-on-invalid-dep=yes',
        'fit_only':True, 'already_accepted_graphsage_runs_reused':9, 'new_models':18,
        'time_projections':cost, 'measured_max_archive_plus_sample_stage_seconds':stage_measured,
        'shared_max_projected_epoch_seconds':epoch_max, 'shared_max_projected_validation_seconds':val_max,
        'segment_cap_formula':'round up 5 min: stage + restore and republish + epochs*(1.5*max measured projected epoch + 20 s mirror I/O) + checkpoint checks, validation scan and publication',
        'stage_budget_formula':'round up minute: max(900 s, 1.5 * slowest observed archive SHA and sample stage + 300 s full-pack I/O margin)',
        'projected_training_gpu_hours':sum(row['epoch_seconds']*30*3 for row in rows)/3600,
        'maximum_allocated_gpu_hours':sum(seg['time_limit_seconds']*18 for seg in segments)/3600}
write('fit_launch_plan.json', plan)
print(json.dumps({m: {k: v['summary'][k] for k in ('epoch_seconds_projection','input_archive_check_seconds','sample_stage_seconds','parameters','gpu','rates','checks')} for m,v in accepted.items()}, indent=2))
print(json.dumps({k: plan[k] for k in ('segments','projected_training_gpu_hours','maximum_allocated_gpu_hours')}, indent=2))
