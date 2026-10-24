"""Locate exact primary sources for frozen detector files, without scoring models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

# These are serialized initial MCTrack fields in both converter outputs.
# Vertices disambiguate repeated kinematics; no tolerance or fitted offset is used.
KEYS = ['PDG', 'Ekin', 'Px', 'Py', 'Pz', 'vX', 'vY', 'vZ']
COLS = ['Row', 'Id', *KEYS]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def targets_from_vacs(frame, member):
    selected = frame.loc[(frame.fMotherId == -1) & frame.Side.isin([5, 6])
                         & frame.PDG.abs().isin([2112, 2212]), COLS].drop_duplicates()
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite primary MCTrack attributes in ' + member)
    if selected.duplicated(['Row', 'Id']).any():
        raise ValueError('Conflicting primary MCTrack attributes in ' + member)
    selected = selected.copy()
    selected['detector_member'] = member
    return selected


def binding_candidates(targets, matches, members):
    """Retain missing and ambiguous records; never choose an approximate match."""
    result = []
    keys = ['detector_member', 'Row_det', 'Id_det']
    if len(matches):
        matches = matches.drop_duplicates(keys + ['primary_member', 'Row_prim', 'Id_prim'])
    for member in members:
        expected = targets.loc[targets.detector_member == member]
        got = matches.loc[matches.detector_member == member] if len(matches) else matches
        n = len(expected)
        counts = got.groupby(keys).size() if len(got) else pd.Series(dtype=int)
        files = sorted(got.primary_member.unique().tolist()) if len(got) else []
        same_ids = bool(len(got) and (got.Row_det == got.Row_prim).all()
                        and (got.Id_det == got.Id_prim).all())
        exact = bool(n > 0 and len(counts) == n and (counts == 1).all()
                     and len(files) == 1 and same_ids)
        result.append({'detector_member': member, 'eligible_records': n,
                       'matched_records': len(counts),
                       'ambiguous_records': int((counts > 1).sum()),
                       'primary_members': files, 'all_row_id_equal': same_ids,
                       'state': 'exact_unique_file_candidate' if exact else 'unresolved'})
    return result


def run(inventory, output, tag, staged_root=None, staged_primary_root=None):
    data = json.loads(inventory.read_text())['tags'][tag]
    members = data['common_primary_pair_names']
    output.mkdir(parents=True, exist_ok=False)
    targets, detector_files = [], []
    for member in members:
        raw_root = Path(data['raw_root']) if staged_root is None else staged_root / 'raw' / tag
        path = raw_root / (member + '_vacs.csv')
        frame = pd.read_csv(path, usecols=[*COLS, 'fMotherId', 'Side'])
        targets.append(targets_from_vacs(frame, member))
        detector_files.append({'member': member, 'sha256': sha(path)})
    targets = pd.concat(targets, ignore_index=True)
    if not len(targets):
        raise ValueError('No primary nucleons to bind')
    result = {'state': 'running', 'tag': tag,
              'scope': 'exact initial MCTrack source mapping on frozen candidate files; no fitting or model evaluation',
              'inventory_sha256': sha(inventory), 'source_script_sha256': sha(Path(__file__)),
              'keys': KEYS, 'files_expected': len(members), 'staged_detector_inputs': staged_root is not None,
              'staged_primary_inputs': staged_primary_root is not None,
              'target_records': len(targets), 'detector_files': detector_files,
              'primary_files': [], 'matches': []}
    started = time.monotonic()
    for i, member in enumerate(members):
        primary_root = (Path(data['primary_root']) if staged_primary_root is None
                        else staged_primary_root / 'primary' / tag)
        path = primary_root / (member + '_prim.csv')
        count = 0
        # 100k rows limits parsing memory, and is not a physics selection.
        for chunk in pd.read_csv(path, usecols=COLS, chunksize=100000):
            if not np.isfinite(chunk.to_numpy(dtype=float)).all():
                raise ValueError('Nonfinite primary inventory attributes in ' + member)
            joined = targets.merge(chunk, on=KEYS, suffixes=('_det', '_prim'))
            if len(joined):
                joined['primary_member'] = member
                result['matches'].extend(joined.to_dict('records'))
                count += len(joined)
        result['primary_files'].append({'member': member, 'sha256': sha(path), 'matches': count})
        result['elapsed_seconds'] = time.monotonic() - started
        if i % 5 == 0 or i + 1 == len(members):
            write(output / 'progress.json', result)
            print(i + 1, '/', len(members), 'matches', len(result['matches']),
                  'seconds', round(result['elapsed_seconds'], 1), flush=True)
    result['binding_candidates'] = binding_candidates(targets, pd.DataFrame(result['matches']), members)
    result['exact_unique_file_candidates'] = sum(x['state'] == 'exact_unique_file_candidate'
                                               for x in result['binding_candidates'])
    result['state'] = 'complete'
    result['downstream_audit_authorized_by_this_result'] = False
    write(output / 'summary.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', choices=['zeroSpot', 'defaultSpot', 'bigSpot'], required=True)
    parser.add_argument('--staged-root', type=Path, help='Hash-verified node-local input bundle')
    parser.add_argument('--staged-primary-root', type=Path,
                        help='Separate hash-verified node-local primary bundle; identity keys unchanged')
    args = parser.parse_args()
    if args.staged_root and not (args.staged_root / 'bundle_verified.json').is_file():
        raise ValueError('Staged bundle has not been verified')
    if args.staged_primary_root and not (args.staged_primary_root / 'bundle_verified.json').is_file():
        raise ValueError('Staged primary bundle has not been verified')
    run(args.inventory, args.output, args.tag, args.staged_root, args.staged_primary_root)


if __name__ == '__main__':
    main()
