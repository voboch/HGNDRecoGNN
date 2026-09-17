#!/usr/bin/env python
"""Resuming a half-built dataset must not duplicate graphs.

The checkpoint used to record only a "last completed phase" string, which was
'' while a phase was in flight. On resume nothing was skipped, the phase
replayed from its first event, and every rebuilt graph was appended again
under a continuing graph_idx -- job 4333774_0 was on course to duplicate
~2.3 M of zeroSpot's 5.3 M graphs.

Builds a small dataset twice: once straight through, once interrupted partway
and resumed. Both must yield the same number of graphs, and the resumed run
must not re-emit events the first pass already wrote.

Run:  python tests/test_resume_no_duplicates.py
"""
import glob
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch  # noqa: E402
from HGNDRecoGNN.data import graph_dataset as gd  # noqa: E402

N_FILES, MAX_EVENTS = 4, 600


def _tree() -> str | None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for ds in ('zeroSpot', 'defaultSpot', 'bigSpot'):
        for base in ('data', 'data_fixed'):
            root = os.path.join(here, base, f'smash_xecs_2.87gev_hardSkyrme_{ds}')
            if not os.path.isdir(root):
                continue
            for run in sorted(os.listdir(root)):
                d = os.path.join(root, run)
                if os.path.isdir(d) and any(f.endswith('hits.csv')
                                            for f in os.listdir(d)):
                    return d
    return None


def _stage(src: str, dst_root: str) -> None:
    run = os.path.join(dst_root, '10943245')
    os.makedirs(run, exist_ok=True)
    stems = sorted({f.rsplit('_', 1)[0] for f in os.listdir(src)
                    if f.endswith('hits.csv')
                    and os.stat(os.path.join(src, f)).st_size > 100})[:N_FILES]
    for st in stems:
        for kind in ('hits', 'vacs'):
            s = os.path.join(src, f'{st}_{kind}.csv')
            if os.path.exists(s):
                shutil.copy(s, os.path.join(run, f'{st}_{kind}.csv'))


def _count_graphs(processed: str) -> int:
    return sum(len(torch.load(p, weights_only=False))
               for p in sorted(glob.glob(os.path.join(processed, 'shard_*.pt'))))


def test_resume_does_not_duplicate():
    src = _tree()
    if src is None:
        print('SKIP: no raw CSV tree available')
        return
    base = tempfile.mkdtemp()
    try:
        csv_dir = os.path.join(base, 'csv')
        _stage(src, csv_dir)

        # 1) straight through
        root_a = os.path.join(base, 'a')
        gd.HGNDGraphDataset(root=root_a, hits_csv_dir=csv_dir,
                            num_workers=0, shard_size=128,
                            max_events=MAX_EVENTS)
        proc_a = os.path.join(root_a, 'processed')
        full = _count_graphs(proc_a)
        meta = json.load(open(os.path.join(proc_a, 'meta.json')))
        assert meta['num_graphs'] == full, (
            f"meta says {meta['num_graphs']} but shards hold {full}")

        # 2) interrupted partway, then resumed
        root_b = os.path.join(base, 'b')
        proc_b = os.path.join(root_b, 'processed')
        os.makedirs(proc_b, exist_ok=True)
        # reuse the parsed parquet so the interrupted build is cheap
        for pq in glob.glob(os.path.join(proc_a, '_hits_cache_*.parquet')):
            shutil.copy(pq, proc_b)

        real_save = gd.HGNDGraphDataset._save_shard
        state = {'n': 0}

        def _die_after_two(self, buf, path):
            if state['n'] >= 2:
                raise RuntimeError('simulated crash')
            state['n'] += 1
            return real_save(self, buf, path)

        gd.HGNDGraphDataset._save_shard = _die_after_two
        try:
            gd.HGNDGraphDataset(root=root_b, hits_csv_dir=csv_dir,
                                num_workers=0, shard_size=128,
                                max_events=MAX_EVENTS)
        except Exception:
            pass
        finally:
            gd.HGNDGraphDataset._save_shard = real_save

        partial = _count_graphs(proc_b)
        assert partial > 0, 'interrupted run wrote nothing to resume from'
        prog = json.load(open(os.path.join(proc_b, '_progress.json')))
        assert 'done_phases' in prog and 'cur_done' in prog, \
            f'checkpoint lacks event offset: {prog}'

        gd.HGNDGraphDataset(root=root_b, hits_csv_dir=csv_dir,
                            num_workers=0, shard_size=128,
                            max_events=MAX_EVENTS)
        resumed = _count_graphs(proc_b)

        assert resumed == full, (
            f'resume produced {resumed} graphs, straight-through gave {full} '
            f'(difference {resumed - full:+d} — duplicated work)')
        print(f'  OK: straight-through {full} graphs; interrupted at {partial} '
              f'then resumed to {resumed} — no duplication')
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == '__main__':
    print('=== test_resume_does_not_duplicate ===')
    test_resume_does_not_duplicate()
    print('\nall tests passed.')
