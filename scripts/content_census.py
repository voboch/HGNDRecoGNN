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
        for source in ('988974', '988975', '988976'):
            files = sorted((args.dcm_root / source).glob('*_hits.csv'))
            if not files:
                raise ValueError(f'Missing DCM source {source}')
            if args.smoke:
                files = files[:1]
            for file in files:
                before = sha256(file)
                frame = pd.read_csv(file)
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
                full = tuple(c for c in frame if c not in ('Row', 'Instance', 'Id'))
                if not {'PDG', 'Px', 'Py', 'Pz', 'vX', 'vY', 'vZ'}.issubset(full):
                    raise ValueError(f'Missing primary schema: {file}')
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
    p.add_argument('--smash-inventory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
