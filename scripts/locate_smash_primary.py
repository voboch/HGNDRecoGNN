"""Exact MC fingerprint search for a failed detector/primary file association."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

KEYS = ['PDG', 'Ekin', 'Px', 'Py', 'Pz']
COLS = ['Row', 'Id', *KEYS, 'vX', 'vY', 'vZ']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, result):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    d = json.loads(args.inventory.read_text())['tags']['defaultSpot']
    member = d['common_primary_pair_names'][0]
    detector = Path(d['raw_root'])/(member+'_vacs.csv')
    v = pd.read_csv(detector)
    targets = v.loc[(v.fMotherId == -1) & v.Side.isin([5, 6])
                    & v.PDG.abs().isin([2112, 2212]), COLS].drop_duplicates(['Row','Id'])
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'state': 'running', 'scope': 'exact source lookup for first failed defaultSpot file; no model evaluation',
              'detector_file': str(detector), 'detector_sha256': sha(detector),
              'inventory_sha256': sha(args.inventory), 'target_nucleons': len(targets),
              'files_expected': len(d['common_primary_pair_names']), 'files_checked': [], 'matches': []}
    started = time.monotonic()
    for i, name in enumerate(d['common_primary_pair_names']):
        path = Path(d['primary_root'])/(name+'_prim.csv')
        file_matches = []
        # 100k rows bound CSV parsing memory; this is an I/O chunk, not a cut.
        for chunk in pd.read_csv(path, usecols=COLS, chunksize=100000):
            m = targets.merge(chunk, on=KEYS, suffixes=('_det', '_prim'))
            if len(m):
                m['primary_file'] = name
                file_matches.extend(m.to_dict('records'))
        result['files_checked'].append({'member': name, 'sha256': sha(path), 'matching_records': len(file_matches)})
        result['matches'].extend(file_matches)
        result['elapsed_seconds'] = time.monotonic()-started
        if i % 5 == 0 or i+1 == result['files_expected']:
            write(args.output/'progress.json', result)
            print(i+1, '/', result['files_expected'], 'matches', len(result['matches']),
                  'seconds', round(result['elapsed_seconds'], 1), flush=True)
    result['state'] = 'complete'
    if result['matches']:
        m = pd.DataFrame(result['matches'])
        result['matched_detector_records'] = len(m[['Row_det','Id_det']].drop_duplicates())
        exact_vertex = (m[['vX_det','vY_det','vZ_det']].to_numpy() == m[['vX_prim','vY_prim','vZ_prim']].to_numpy()).all(axis=1)
        result['matched_detector_records_with_exact_vertex'] = len(m.loc[exact_vertex, ['Row_det','Id_det']].drop_duplicates())
    else:
        result['matched_detector_records'] = 0
        result['matched_detector_records_with_exact_vertex'] = 0
    result['source_script_sha256'] = sha(Path(__file__))
    write(args.output/'summary.json', result)


if __name__ == '__main__':
    main()
