"""Read-only, exact exported-event census; no training or fitted selections.

Identity excludes bookkeeping IDs and added time noise, never rounds floats,
and retains multiplicity. The result is an export-content identity, not proof
of a generator seed, collision lineage, or events absent from the export.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

HIT_FIELDS = ('fX', 'fY', 'fZ', 'DetectorId', 'LayerId', 'RowId',
              'ColumnId', 'fTime', 'fELoss')
PRIMARY_FIELDS = ('PDG', 'E', 'Ekin', 'Mass', 'Px', 'Py', 'Pz', 'Pt',
                  'P', 'Rapid', 'vX', 'vY', 'vZ', 'B', 'NPrim')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(values, fields):
    """Canonical finite numeric multiset, including field schema and row count."""
    a = np.array(values, dtype='<f8', order='C', copy=True)
    if a.ndim != 2 or a.shape[1] != len(fields) or not np.isfinite(a).all():
        raise ValueError('Non-finite or malformed identity input')
    a[a == 0] = 0  # Equal physical zero irrespective of sign bit.
    if len(a):
        a = a[np.lexsort(a.T[::-1])]
    h = hashlib.sha256(('content-v1:' + ','.join(fields)).encode())
    h.update(np.array(a.shape, dtype='<i8').tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def event_signatures(frame, fields):
    """Yield sorted event IDs and hashes without mutating the caller's frame."""
    ordered = frame.sort_values('Row', kind='stable')
    rows = ordered.Row.to_numpy()
    if not len(rows):
        raise ValueError('Empty event export')
    if not np.isfinite(rows).all() or (rows != rows.astype(np.int64)).any():
        raise ValueError('Invalid event identifiers')
    a = ordered[list(fields)].to_numpy(dtype='<f8')
    offsets = np.r_[0, np.flatnonzero(rows[1:] != rows[:-1]) + 1, len(rows)]
    for start, stop in zip(offsets[:-1], offsets[1:]):
        yield int(rows[start]), stop - start, signature(a[start:stop], fields)


def summarise(records):
    """Separate event copies, within-file copies and cross-file/group overlap."""
    by_hash = defaultdict(list)
    for group, file, row, count, h in records:
        by_hash[h].append((group, file, row, count))
    multiplicity = Counter(map(len, by_hash.values()))
    cross_files = [v for v in by_hash.values() if len({(a, b) for a, b, _, _ in v}) > 1]
    cross_groups = [v for v in by_hash.values() if len({a for a, _, _, _ in v}) > 1]
    within = Counter((group, file, h) for group, file, _, _, h in records)
    counts = {}
    for group in sorted({r[0] for r in records}):
        selected = [r for r in records if r[0] == group]
        counts[group] = {'events': len(selected), 'unique_contents': len({r[-1] for r in selected}),
                         'files': len({r[1] for r in selected})}
    return {'events': len(records), 'unique_contents': len(by_hash),
            'events_in_repeated_groups': sum(k * v for k, v in multiplicity.items() if k > 1),
            'maximum_multiplicity': max(multiplicity, default=0),
            'multiplicity_histogram': dict(sorted(multiplicity.items())),
            'within_file_repeated_groups': sum(v > 1 for v in within.values()),
            'cross_file_groups': len(cross_files),
            'events_in_cross_file_groups': sum(map(len, cross_files)),
            'cross_directory_groups': len(cross_groups),
            'events_in_cross_directory_groups': sum(map(len, cross_groups)),
            'groups': counts,
            'cross_directory_examples': [v[:5] for v in cross_groups[:5]]}


def write_index(path, records):
    with gzip.open(path, 'wt') as f:
        f.write('group,file,Row,records,sha256\n')
        for item in records:
            f.write(','.join(map(str, item)) + '\n')


def validate_smoke(receipt, inventory_path, dcm_root, dcm_inventory):
    """A passed smoke is usable only for the exact current code and inputs."""
    verified = json.loads(Path(receipt).read_text())
    if verified['slurm_state'] != 'COMPLETED' or verified['exit_code'] != '0:0':
        raise ValueError('Smoke Slurm job did not complete successfully')
    summary = Path(verified['summary_path'])
    if sha256(summary) != verified['summary_sha256'] or sha256(verified['archive_path']) != verified['archive_sha256']:
        raise ValueError('Smoke export verification failed')
    x = json.loads(summary.read_text())
    inventory = json.loads(Path(inventory_path).read_text())
    if (x.get('status') != 'complete' or x.get('smoke') is not True
            or x.get('code_sha256') != sha256(__file__)
            or x.get('smash_inventory_sha256') != sha256(inventory_path)
            or x.get('dcm_inventory_sha256') != sha256(dcm_inventory)):
        raise ValueError('Smoke receipt is incomplete or belongs to different code/inventory')
    membership = json.loads(Path(dcm_inventory).read_text())['files']
    expected = {str(Path(dcm_root) / next(v['relative'] for v in membership if v['relative'].startswith(run + '/')))
                for run in ('988974', '988975', '988976')}
    expected.update(str(Path(info['primary_root']) / f'{info["common_primary_pair_names"][0]}_prim.csv')
                    for info in inventory['tags'].values())
    if len(x['files']) != 6 or {v['path'] for v in x['files']} != expected:
        raise ValueError('Smoke did not cover the declared input roots')
    for item in x['files']:
        if item['events'] <= 0 or item['records'] <= 0 or sha256(item['path']) != item['sha256']:
            raise ValueError('Smoke input is empty or changed')
    required = {'dcm_hits', 'dcm_retained_hits', 'smash_full', 'smash_without_vertex'}
    if set(x['identities']) != required:
        raise ValueError('Smoke skipped an identity definition')
    for name in required:
        if sha256(summary.parent / f'{name}.csv.gz') != x['identities'][name]['index_sha256']:
            raise ValueError('Smoke identity index missing or corrupt')


def run(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'schema': 'export-content-census-v1', 'code_sha256': sha256(__file__),
               'smoke': args.smoke, 'files': [], 'identities': {},
               'limits': 'Exact represented export contents only; absent collisions and generator lineage unknown.'}
    collections = defaultdict(list)
    if args.pickle:
        frame = pd.read_pickle(args.pickle).rename(columns={'fTime': 'noisy_time', 'fTime_nonoise': 'fTime'})
        # Every available physical/truth field except row, ordering, noisy and derived times.
        full = HIT_FIELDS + ('Id', 'fMotherId', 'PDG', 'Ekin', 'Side', 'X', 'Y', 'Z')
        for identity, fields in [('student_hits', HIT_FIELDS), ('student_hits_truth', full)]:
            collections[identity] = [('student', args.pickle.name, row, n, h)
                                     for row, n, h in event_signatures(frame, fields)]
        summary['files'].append({'kind': 'pickle', 'path': str(args.pickle), 'sha256': sha256(args.pickle),
                                 'records': len(frame), 'events': frame.Row.nunique()})
    if args.dcm_root:
        membership = json.loads(args.dcm_inventory.read_text())
        summary['dcm_inventory_sha256'] = sha256(args.dcm_inventory)
        declared = {v['relative'] for v in membership['files']}
        actual = {str(p.relative_to(args.dcm_root)) for p in args.dcm_root.glob('*/*_hits.csv')}
        if declared != actual:
            raise ValueError('DCM file membership differs from frozen inventory')
        for source in ('988974', '988975', '988976'):
            files = [v for v in membership['files'] if v['relative'].startswith(source + '/')]
            if not files:
                raise ValueError(f'Missing DCM source {source}')
            if args.smoke:
                files = files[:1]
            for item in files:
                file = args.dcm_root / item['relative']
                before = sha256(file)
                if before != item['sha256']:
                    raise ValueError(f'DCM input differs from frozen inventory: {file}')
                frame = pd.read_csv(file)
                if frame.empty:
                    if item['included_in_prepared'] or tuple(frame.columns[2:]) != HIT_FIELDS:
                        raise ValueError('Prepared file is empty or header is malformed')
                    summary['files'].append({'kind': 'dcm_header_only', 'path': str(file), 'sha256': before,
                                             'records': 0, 'events': 0})
                    continue
                collections['dcm_hits'].extend((source, file.name, row, n, h)
                                              for row, n, h in event_signatures(frame, HIT_FIELDS))
                # 3 MeV is the existing loader convention, not a fitted cut.
                retained = frame[frame.fELoss > .003]
                collections['dcm_retained_hits'].extend((source, file.name, row, n, h)
                                                       for row, n, h in event_signatures(retained, HIT_FIELDS))
                if sha256(file) != before:
                    raise ValueError(f'Input changed during census: {file}')
                summary['files'].append({'kind': 'dcm_hits', 'path': str(file), 'sha256': before,
                                         'records': len(frame), 'events': frame.Row.nunique()})
                print(f'DCM {source}/{file.name}: {frame.Row.nunique()} events', flush=True)
    if args.smash_inventory:
        inventory = json.loads(args.smash_inventory.read_text())
        summary['smash_inventory_sha256'] = sha256(args.smash_inventory)
        for tag, info in inventory['tags'].items():
            # Name membership is fixed by the pre-existing inventory, not by glob outcomes.
            files = [Path(info['primary_root']) / f'{name}_prim.csv'
                     for name in info['common_primary_pair_names']]
            expected = {'zeroSpot': 195, 'defaultSpot': 200, 'bigSpot': 173}[tag]
            if len(files) != expected or len(set(files)) != expected:
                raise ValueError(f'Incorrect declared primary membership: {tag}')
            if args.smoke:
                files = files[:1]
            for file in files:
                before = sha256(file)
                frame = pd.read_csv(file)
                full = PRIMARY_FIELDS
                if set(frame.columns) != set(full) | {'Row', 'Instance', 'Id'}:
                    raise ValueError(f'Unexpected primary schema: {file}')
                momentum = tuple(c for c in full if c not in ('vX', 'vY', 'vZ'))
                for name, fields in [('smash_full', full), ('smash_without_vertex', momentum)]:
                    collections[name].extend((tag, str(file.relative_to(info['primary_root'])), row, n, h)
                                             for row, n, h in event_signatures(frame, fields))
                if sha256(file) != before:
                    raise ValueError(f'Input changed during census: {file}')
                summary['files'].append({'kind': 'smash_primary', 'tag': tag, 'path': str(file),
                                         'sha256': before, 'records': len(frame), 'events': frame.Row.nunique(),
                                         'full_fields': full, 'without_vertex_fields': momentum})
                print(f'SMASH {tag}/{file.name}: {frame.Row.nunique()} events', flush=True)
    if not collections:
        raise ValueError('No input specified')
    for name, records in collections.items():
        summary['identities'][name] = summarise(records)
        path = args.output / f'{name}.csv.gz'
        write_index(path, records)
        summary['identities'][name]['index_sha256'] = sha256(path)
    summary['seconds'] = time.monotonic() - started
    summary['status'] = 'complete'
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(f'Complete in {summary["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pickle', type=Path)
    p.add_argument('--dcm-root', type=Path)
    p.add_argument('--dcm-inventory', type=Path)
    p.add_argument('--smash-inventory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
