"""Verify a completed publication and extract only its registered files."""
from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def accept(receipt_path, tar_path, manifest_path, output):
    receipt, manifest = load(receipt_path), load(manifest_path)
    launch_path = HERE / 'launch.json'
    launch = load(launch_path)
    source_package = load(HERE / 'source_package.json')
    job_id = (HERE / 'submission.txt').read_text().strip()
    require(job_id.isdecimal() and receipt['bundle_dir'] == f"{source_package['campaign']}/result_{job_id}"
            and sha(launch_path) == source_package['launch_sha256']
            and receipt['publisher_sha256'] == sha(ROOT / 'scripts/publish_smash_directory.py'), 'Wrong job or publisher')
    plan = ROOT / launch['historical_plan']
    require(sha(plan) == launch['historical_plan_sha256'], 'Historical plan changed')
    records = {r['content']: r for r in load(plan)['files']}
    member_names = {'summary.json'} | {f'{cid}/{file}' for cid in records for file in ('kinematics.npz', 'receipt.json')}
    expected = {m['member']: m for m in manifest['files']}
    require(receipt['state'] == manifest['state'] == 'complete' and sha(manifest_path) == receipt['manifest_sha256']
            and sha(tar_path) == receipt['archive_sha256'] == manifest['archive_sha256']
            and tar_path.stat().st_size == receipt['archive_bytes'] == manifest['archive_bytes'], 'Archive receipt mismatch')
    require(set(expected) == member_names and len(expected) == len(manifest['files']) == receipt['files'], 'Archive inventory')
    output.mkdir(exist_ok=False)
    seen = set()
    with tarfile.open(tar_path, 'r:') as tar:
        for item in tar:
            name = item.name
            require(name in expected and item.isfile() and name not in seen, 'Unexpected archive member')
            pin = expected[name]
            require(item.size == pin['bytes'], 'Member size mismatch')
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(item) as stream, path.open('xb') as f:
                shutil.copyfileobj(stream, f)
            require(sha(path) == pin['sha256'], f'Member digest mismatch: {name}')
            seen.add(name)
    require(seen == member_names, 'Incomplete publication')
    path = output / 'summary.json'
    require(sha(path) == receipt['summary_files']['summary.json'], 'Summary pin mismatch')
    summary = load(path)
    require(summary['state'] == 'complete' and summary['schema'] == 'hgnd-rapidity-s286-v1'
            and summary['launch_sha256'] == sha(launch_path) and summary['checks'] == launch['expected_counts'],
            'Wrong launch or summary scope')
    require(summary['batch_sha256'] == sha(HERE / 'run_s286.py'), 'Batch source changed')
    require(len(summary['files']) == 286 and {r['content'] for r in summary['files']} == set(records), 'Summary inventory')
    script_sha = sha(ROOT / 'scripts/export_smash_rapidity.py')
    for entry in summary['files']:
        cid = entry['content']
        row = records[cid]
        directory = output / cid
        source = entry['source_sha256']
        candidates = entry['candidates']
        require(entry == load(directory / 'receipt.json') and entry['schema'] == 'smash-rapidity-kinematics-v1'
                and entry['state'] == 'complete' and entry['output_sha256'] == expected[f'{cid}/kinematics.npz']['sha256']
                and entry['script_sha256'] == script_sha and source['plan'] == launch['historical_plan_sha256']
                and entry['processed_span'] == row['processed_span']
                and source['primary'] == row['primary_sha256'] and source['hits'] == row['sha256']['hits']
                and source['vacs'] == row['sha256']['vacs'] and source['pack'] == row['pack']['sha256']
                and candidates['observation_sha256'] == row['observation']['sha256']
                and candidates['observation_serialization_exact'] and candidates['native_target_bytes_exact']
                and candidates['dynamic_scores']['acceptance_sha256'] == launch['score_acceptance_sha256']
                and set(candidates['dynamic_scores']['seeds']) == {'42', '123', '456'}, f'Proof differs: {cid}')
    result = {'state': 'accepted', 'job_id': job_id, 'source_sums_sha256': source_package['sums_sha256'],
              'observations_native_targets_and_frozen_scores_verified_files': len(summary['files']),
              'max_abs_y_lab_minus_csv': {species: max(r['primary'][species]['max_abs_y_minus_exported']
                                                      for r in summary['files']) for species in ('neutron', 'proton')},
              'checks': summary['checks'], 'source_receipt_sha256': sha(receipt_path),
              'archive_sha256': receipt['archive_sha256'], 'summary_sha256': sha(path), 'manifest_sha256': sha(manifest_path)}
    with (output / 'acceptance.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for name in ('receipt', 'archive', 'manifest', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    a = parser.parse_args()
    accept(a.receipt, a.archive, a.manifest, a.output)
