#!/usr/bin/env python
"""Regression test for cross-file event-ID collisions in load_hits().

Row is only unique within a single CSV. load_hits() maps it into a global
space; the original code did that with a hardcoded stride of 500 while real
files hold ~4150 events, so ~8 consecutive files aliased onto the same ids and
the hits<->MC merge (on ['Row', 'Instance']) matched hits to particles from
the wrong file: ~23 % duplicate truth keys, ~31 % of hits with more than one
candidate particle.

Run:  python tests/test_loader_row_keys.py
"""
import os
import shutil
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from HGNDRecoGNN.data.graph_dataset import load_hits, LOADER_V   # noqa: E402

# Consecutive file numbers are the case that collided; a strided sample hides it.
N_FILES = 6


def _find_csv_tree() -> str | None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for ds in ('zeroSpot', 'defaultSpot', 'bigSpot'):
        for base in ('data', 'data_fixed'):
            root = os.path.join(here, base,
                                f'smash_xecs_2.87gev_hardSkyrme_{ds}')
            if not os.path.isdir(root):
                continue
            for run in sorted(os.listdir(root)):
                d = os.path.join(root, run)
                if os.path.isdir(d) and any(f.endswith('hits.csv')
                                            for f in os.listdir(d)):
                    return d
    return None


def test_row_keys_unique_across_files():
    src = _find_csv_tree()
    if src is None:
        print('SKIP: no raw CSV tree available')
        return

    stems = sorted({f.rsplit('_', 1)[0] for f in os.listdir(src)
                    if f.endswith('hits.csv')})[:N_FILES]
    tmp = tempfile.mkdtemp()
    run = os.path.join(tmp, '10943245')
    os.makedirs(run)
    try:
        for st in stems:
            for kind in ('hits', 'vacs'):
                s = os.path.join(src, f'{st}_{kind}.csv')
                if os.path.exists(s):
                    shutil.copy(s, os.path.join(run, f'{st}_{kind}.csv'))

        df = load_hits(tmp, cache_dir=tmp)

        dup = int(df.duplicated(subset=['Row', 'Instance']).sum())
        assert dup == 0, f'{dup} duplicate (Row, Instance) pairs after merge'

        multi = int((df.groupby(['Row', 'Instance']).size() > 1).sum())
        assert multi == 0, f'{multi} hits matched more than one MC particle'

        # Each file must occupy its own Row band: the number of distinct events
        # cannot exceed the per-file total (the ELoss cut may drop some).
        per_file = sum(
            pd.read_csv(os.path.join(run, f'{st}_hits.csv'),
                        usecols=['Row']).Row.nunique()
            for st in stems
            if os.path.exists(os.path.join(run, f'{st}_hits.csv'))
        )
        got = int(df.Row.nunique())
        assert got <= per_file, (
            f'{got} distinct events > {per_file} counted per file — '
            f'Row blocks are overlapping')
        # A stride-500 collision would collapse ~8 files into one band; require
        # that we retain most of the per-file events.
        assert got > 0.5 * per_file, (
            f'only {got} of {per_file} events survived — suspicious collapse')

        print(f'  OK: {len(stems)} consecutive files, {got:,} distinct events, '
              f'0 duplicate keys, 0 multi-match hits (LOADER_V={LOADER_V})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    print('=== test_row_keys_unique_across_files ===')
    test_row_keys_unique_across_files()
    print('\nall tests passed.')
