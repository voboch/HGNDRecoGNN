"""Run the accepted one-file kinematic exporter on the registered even inventory."""

from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import shutil
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.scripts.export_smash_rapidity import check, read_json, require, run, sha
from HGNDRecoGNN.scripts.smash_train_prepare import extract_members, stage


def export_one(task):
    row, scratch, output, scores = task
    cid = row['content']
    work = scratch / 'raw' / cid
    work.mkdir(parents=True, exist_ok=False)
    raw = {name: work / f'{name}.csv' for name in ('primary', 'hits', 'vacs')}
    stage(Path(row['primary_path']), raw['primary'], row['primary_sha256'])
    for name in ('hits', 'vacs'):
        src = Path(row['raw_root']) / f"{row['member']}_{name}.csv"
        stage(src, raw[name], row['sha256'][name])
    info = run(Namespace(content=cid, **raw, pack=scratch / 'packs' / row['pack']['member'],
                         observation=None, dynamic_scores=scores, output=output / cid), verbose=False)
    shutil.rmtree(work)
    return info


def main(args):
    start = monotonic()
    launch = read_json(args.launch)
    require(launch['schema'] == 'hgnd-rapidity-s286-launch-v1', 'Wrong launch schema')
    plan_path = check(ROOT / launch['historical_plan'], launch['historical_plan_sha256'])
    rows = read_json(plan_path)['files']
    require(len(rows) == len({r['content'] for r in rows}) == 286 and
            {r['part'] for r in rows} == {0, 2, 4}, 'S286 inventory changed')
    require(launch['workers'] == 2, 'Unexpected worker cap')
    scores = ROOT / launch['score_directory']
    check(scores / 'acceptance.json', launch['score_acceptance_sha256'])
    prep = launch['prepare']
    archive = Path(prep['bundle']) / 'inputs.tar'
    check(Path(prep['bundle']) / 'manifest.json', prep['manifest_sha256'])
    manifest = read_json(Path(prep['bundle']) / 'manifest.json')
    require(manifest['archive_sha256'] == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes'],
            'Native archive identity changed')
    args.output.mkdir(exist_ok=False)
    extract_members(archive.parent, prep['manifest_sha256'], [r['pack']['member'] for r in rows], args.work / 'packs')
    files = []
    with ProcessPoolExecutor(max_workers=launch['workers']) as pool:
        for info in pool.map(export_one, [(row, args.work, args.output, scores) for row in rows]):
            files.append(info)
            if len(files) == len(rows) or len(files) % 20 == 0:
                print(f'Exported and checked {len(files)}/286', flush=True)
    checks = {
        'files': len(files), 'processed_events': sum(r['processed_span'] for r in files),
        'candidate_keys': sum(r['candidates']['gnn_candidates'] for r in files),
        'first_hit_rows': sum(r['candidates']['isolated_first_hits_replayed'] for r in files),
    }
    require(checks == launch['expected_counts'], f'Population checks failed: {checks}')
    require(all(r['candidates']['observation_serialization_exact'] and r['candidates']['native_target_bytes_exact']
                and len(r['candidates']['dynamic_scores']['seeds']) == 3 for r in files), 'Incomplete parity or seed proof')
    summary = {'schema': 'hgnd-rapidity-s286-v1', 'state': 'complete', 'checks': checks,
               'elapsed_seconds': monotonic() - start, 'files': files, 'launch_sha256': sha(args.launch),
               'batch_sha256': sha(Path(__file__))}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for flag in ('launch', 'work', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    main(parser.parse_args())
