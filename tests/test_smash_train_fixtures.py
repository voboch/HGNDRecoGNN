"""Synthetic corrected-SMASH exports, audit bundles and splits for the SMASH-training tests."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from sklearn.preprocessing import StandardScaler

from HGNDRecoGNN.data.graph_dataset import FEATURES, _dfToF
from HGNDRecoGNN.scripts import audit_smash_sources as audit
from HGNDRecoGNN.scripts.smash_train_prepare import TAGS, sha

GEOMETRY = {'LayerId': [0, 9], 'RowId': [0, 10], 'ColumnId': [0, 10]}
MEMBERS = ('10943245/0001', '10943245/0002', '10943245/0003')


def write_export(root, member, seed, events=30, pdg0_row=None):
    """One hits/vacs pair with neutron and photon candidates on both detector sides."""
    rng = np.random.default_rng(seed)
    hits, vacs = [], []
    for row in range(events):
        instance = 0
        for side in (-1, 1):
            for _ in range(int(rng.integers(1, 4))):
                neutron = rng.random() < .5
                layer, cell_row, col = (int(v) for v in rng.integers(0, 8, 3))
                fx, fy, fz = rng.uniform(-100, 100), side * rng.uniform(10, 70), 730. + 10 * layer
                # Photons arrive early with larger deposits, so the features carry the label.
                t0 = rng.uniform(28, 40) if neutron else rng.uniform(24, 25.5)
                distance = np.linalg.norm([fx, fy, fz]) / 100.
                ek = float(_dfToF(np.array([distance]), np.array([t0]))[0]) if neutron else rng.uniform(.05, 2)
                pdg = 2112 if neutron else 22
                if pdg0_row is not None and row == pdg0_row and side == 1:
                    pdg = 0
                track = int(rng.integers(1, 10 ** 6))
                for h in range(int(rng.integers(1, 5))):
                    hits.append([row, instance, fx + h, fy, fz + h, 1, layer + h % 2, cell_row, col + h // 2,
                                 t0 + .05 * h, rng.uniform(.001, .02) if neutron else rng.uniform(.02, .06)])
                    vacs.append([row, instance, track, pdg, ek, .5, -1, 1., 1, 5, fx, fy, fz, 0., 0., 0., .1, .1, 1.])
                    instance += 1
    base = Path(root) / member
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, names, rows in (('_hits.csv', audit.HITS, hits), ('_vacs.csv', audit.VACS, vacs)):
        lines = [','.join(names)] + [','.join(repr(float(v)) for v in r) for r in rows]
        Path(str(base) + suffix).write_text('\n'.join(lines) + '\n')
    return {k: sha(str(base) + f'_{k}.csv') for k in ('hits', 'vacs')}


def write_raw(tmp_path, events=30):
    """Three exports per potential under the production directory layout."""
    raw = Path(tmp_path) / 'raw'
    hashes = {}
    for i, tag in enumerate(TAGS):
        root = raw / ('smash_xecs_2.87gev_hardSkyrme_' + tag)
        for j, member in enumerate(MEMBERS):
            hashes[f'{tag}/{member}'] = write_export(root, member, 100 * i + j, events, pdg0_row=3 if j == 0 else None)
    return raw, hashes


def write_audit_bundles(tmp_path, raw):
    """Run the audit with saved observations and publish one bundle per potential."""
    from HGNDRecoGNN.scripts.publish_smash_directory import publish
    index = {}
    for tag in TAGS:
        source = Path(tmp_path) / 'audit_source' / tag
        args = argparse.Namespace(tag=tag, raw_root=raw, primary_root=Path(tmp_path) / 'no_primary',
                                  output=source / 'audit' / tag, inventory=None, save_observations=True,
                                  staged_root=None, limit_files=None)
        audit.run(args)
        bundle = Path(tmp_path) / 'audit_bundle_work' / tag
        bundle.parent.mkdir(parents=True, exist_ok=True)
        destination = Path(tmp_path) / 'audit_bundles' / tag
        publish(source, bundle, destination, Path(tmp_path) / f'audit_receipt_{tag}.json')
        index[tag] = {'bundle_dir': str(destination), 'manifest_sha256': sha(destination / 'manifest.json')}
    path = Path(tmp_path) / 'audit_index.json'
    path.write_text(json.dumps(index))
    return path


def write_split(tmp_path, raw, hashes, members=MEMBERS[:2], test=(), name='split.json'):
    """A scripts/smash_split.py manifest: one fold, first member trains, second validates."""
    groups = []
    for cid, h in sorted(hashes.items()):
        tag, member = cid.split('/', 1)
        if member not in members + tuple(test):
            continue
        part = 2 if member == members[1] else (0 if member in test else 3)
        groups.append({'id': f'{tag}:{member}', 'tag': tag, 'kept': member, 'contents': [member],
                       'hits_sha256': h['hits'], 'vacs_sha256': h['vacs'], 'truth_bound': True,
                       'primary': member, 'events': 1, 'hits_path': member + '_hits.csv',
                       'vacs_path': member + '_vacs.csv', 'rank': 0, 'part': part})
    partitions = {'train': [g['id'] for g in groups if g['part'] == 3],
                  'val': [g['id'] for g in groups if g['part'] == 2],
                  'test': [g['id'] for g in groups if g['part'] == 0], 'unused': []}
    manifest = {'schema': 'smash-split-v1', 'folds_count': 1, 'parts': 6,
                'raw_roots': {t: str(Path(raw) / ('smash_xecs_2.87gev_hardSkyrme_' + t)) for t in TAGS},
                'groups': groups, 'folds': {'0': {'partitions': partitions}}}
    path = Path(tmp_path) / name
    path.write_text(json.dumps(manifest, indent=1))
    return path


def write_smoke_split(tmp_path, raw, hashes):
    """A scripts/smash_split.py smoke split: two files per potential."""
    out = {'schema': 'smash-smoke-split-v1', 'train': [], 'val': []}
    for tag in TAGS:
        for member, part in zip(MEMBERS[:2], ('train', 'val')):
            h = hashes[f'{tag}/{member}']
            out[part].append({'tag': tag, 'member': member,
                              'raw_root': str(Path(raw) / ('smash_xecs_2.87gev_hardSkyrme_' + tag)),
                              'hits_path': member + '_hits.csv', 'vacs_path': member + '_vacs.csv',
                              'hits_sha256': h['hits'], 'vacs_sha256': h['vacs'], 'primary': member})
    path = Path(tmp_path) / 'smoke_split.json'
    path.write_text(json.dumps(out, indent=1))
    return path


def fake_dcm():
    """Stand-in for the frozen DCM transform: train-like scalers and a non-trivial remap."""
    rng = np.random.default_rng(7)
    import pandas as pd
    scalers = []
    for side in (0, 1):
        x = pd.DataFrame(rng.normal(size=(200, 9)) * [50, 30, 30, 3, 3, 3, 5, .01, .3] + [0, 40 - 80 * (1 - side), 760, 4, 5, 5, 32, .02, .5],
                         columns=FEATURES[:-1])
        scalers.append(StandardScaler().fit(x))
    a = np.full((2, 9), 1.05)
    b = np.full((2, 9), -.1)
    return scalers, a, b, GEOMETRY, {'geometry_bounds': GEOMETRY, 'fake': True}


def test_fixture_export_is_readable_and_observable(tmp_path):
    raw, hashes = write_raw(tmp_path, events=5)
    member = MEMBERS[0]
    root = raw / ('smash_xecs_2.87gev_hardSkyrme_' + TAGS[0])
    hf = audit.read_csv(root / (member + '_hits.csv'), audit.HITS)
    vf = audit.read_csv(root / (member + '_vacs.csv'), audit.VACS)
    obs, kept, dropped = audit.observe(hf, vf, member + '_hits.csv')
    assert len(kept) and dropped > 0  # the PDG==0 half-event is rejected
    assert (kept.fY != 0).all() and (kept.fELoss > .003).all()
