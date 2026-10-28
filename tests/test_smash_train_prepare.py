"""Per-file SMASH packs: split gates, verified extraction, audit identity, train-only scalers."""
from pathlib import Path
import json
import pickle
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import torch

from HGNDRecoGNN.data.smash_transfer import observed_features
from HGNDRecoGNN.scripts import smash_train_prepare as prep
from HGNDRecoGNN.scripts import smash_transfer_infer
from HGNDRecoGNN.scripts.audit_smash_sources import HITS
from HGNDRecoGNN.tests.test_smash_train_fixtures import (
    GEOMETRY, MEMBERS, fake_dcm, write_audit_bundles, write_raw, write_smoke_split, write_split)


def prepare(tmp_path, split, audit_index=None, workers=1, name='out', keep=True):
    args = type('Args', (), dict(split=split, output=Path(tmp_path) / name, work=Path(tmp_path) / (name + '_work'),
                                 dcm_prepared=None, native_campaign=None, audit_index=audit_index,
                                 source_sums=None, keep_observations=keep, workers=workers))
    return prep.prepare(args)


@pytest.fixture
def dcm(monkeypatch):
    monkeypatch.setattr(smash_transfer_infer, 'frozen_dcm', lambda args: fake_dcm())


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('world')
    raw, hashes = write_raw(tmp, events=25)
    return tmp, raw, hashes, write_audit_bundles(tmp, raw)


def test_split_loader_normalises_both_manifests_and_rejects_leaks(tmp_path, world):
    _, raw, hashes, _ = world
    full = prep.load_split(write_split(tmp_path, raw, hashes, test=(MEMBERS[2],)))
    assert full['folds'][0]['train'] == [f'{t}/{MEMBERS[0]}' for t in sorted(prep.TAGS)]
    assert sorted(full['folds'][0]['test']) == sorted(f'{t}/{MEMBERS[2]}' for t in prep.TAGS)
    smoke = prep.load_split(write_smoke_split(tmp_path, raw, hashes))
    assert smoke['folds'][0]['test'] == [] and len(smoke['contents']) == 6
    path = write_split(tmp_path, raw, hashes, test=(MEMBERS[2],), name='bad.json')
    manifest = json.loads(path.read_text())
    cases = {
        'several partitions': lambda m: m['folds']['0']['partitions']['val'].append(m['folds']['0']['partitions']['train'][0]),
        'Unbound content in the test': lambda m: next(g for g in m['groups'] if g['part'] == 0).update(truth_bound=False),
        'Byte-identical contents': lambda m: m['groups'][1].update(hits_sha256=m['groups'][0]['hits_sha256'],
                                                                   vacs_sha256=m['groups'][0]['vacs_sha256']),
        'paths differ': lambda m: m['groups'][0].update(hits_path='other_hits.csv'),
    }
    for message, corrupt in cases.items():
        bad = json.loads(json.dumps(manifest))
        corrupt(bad)
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=message):
            prep.load_split(path)


def test_selective_extraction_hashes_first_and_writes_only_requested_members(tmp_path, world):
    _, _, _, index = world
    item = json.loads(index.read_text())['zeroSpot']
    bundle = Path(item['bundle_dir'])
    member = 'audit/zeroSpot/summary.json'
    out = prep.extract_members(bundle, item['manifest_sha256'], [member], tmp_path / 'x')
    assert [p.relative_to(tmp_path / 'x').as_posix() for p in (tmp_path / 'x').rglob('*') if p.is_file()] == [member]
    assert json.loads(out[member].read_text())['state'] == 'complete'
    with pytest.raises(ValueError, match='absent from bundle'):
        prep.extract_members(bundle, item['manifest_sha256'], ['audit/zeroSpot/missing.pt'], tmp_path / 'y')
    with pytest.raises(ValueError, match='manifest differs'):
        prep.extract_members(bundle, '0' * 64, [member], tmp_path / 'z')
    tampered = tmp_path / 'tampered'
    shutil.copytree(bundle, tampered)
    with tarfile.open(tampered / 'inputs.tar', 'a') as tar:
        tar.add(index, arcname='extra.json')
    with pytest.raises(ValueError, match='archive hash'):
        prep.extract_members(tampered, item['manifest_sha256'], [member], tmp_path / 'w')
    assert not (tmp_path / 'w').exists()


def test_prepare_matches_audit_tensors_and_keeps_a_b_candidates_identical(tmp_path, world, dcm):
    _, raw, hashes, index = world
    split_path = write_split(tmp_path, raw, hashes, test=(MEMBERS[2],))
    receipt = prepare(tmp_path, split_path, index)
    assert receipt['state'] == 'complete'
    assert receipt['audit_comparison'] == {'files_compared': 9, 'all_bit_identical': True,
                                           'file_sha256_identical': 9}
    split = prep.load_split(split_path)
    out = tmp_path / 'out'
    for cid, item in receipt['files'].items():
        a = torch.load(out / item['packs']['A']['path'], weights_only=True)
        b = torch.load(out / item['packs']['B/fold_0']['path'], weights_only=True)
        assert prep.same_candidates(a, b) and not torch.equal(a['x'], b['x'])
        assert item['population']['graphs'] == len(a['row'])
        role = item['roles']['0']
        assert cid in split['folds'][0][role]
    # The scaler is the train-only partial_fit in sorted order and nothing else.
    record = receipt['scalers']['fold_0']
    assert record['files'] == sorted(split['folds'][0]['train'])
    manual = [StandardScaler(), StandardScaler()]
    for cid in sorted(split['folds'][0]['train']):
        frame = prep.load_observation(out / receipt['files'][cid]['observation']['path'])
        for side, f in observed_features(frame, GEOMETRY).items():
            manual[side].partial_fit(f)
    with (out / record['pickle']['path']).open('rb') as stream:
        stored = pickle.load(stream)
    for name, m in zip(prep.SIDES, manual):
        assert np.array_equal(stored[name].mean_, m.mean_) and np.array_equal(stored[name].var_, m.var_)
        assert record['sides'][name]['n_samples_seen'] == m.n_samples_seen_


def poison(raw, cid, root_of):
    """Replace a raw hits file with physically absurd feature values; cells stay in the geometry."""
    tag, member = cid.split('/', 1)
    path = Path(root_of[tag]) / (member + '_hits.csv')
    lines = path.read_text().splitlines()
    rows = [line.split(',') for line in lines[1:]]
    for r in rows:
        r[HITS.index('fX')] = repr(float(r[HITS.index('fX')]) * 1e4)
        r[HITS.index('fTime')] = repr(float(r[HITS.index('fTime')]) + 1e4)
        r[HITS.index('fELoss')] = repr(float(r[HITS.index('fELoss')]) * 1e3)
    path.write_text('\n'.join([lines[0]] + [','.join(r) for r in rows]) + '\n')
    return prep.sha(path)


def test_poisoned_validation_and_test_files_never_reach_the_scaler(tmp_path, dcm):
    raw, hashes = write_raw(tmp_path, events=25)
    clean = write_split(tmp_path, raw, hashes, test=(MEMBERS[2],), name='clean.json')
    first = prepare(tmp_path, clean, name='clean')
    roots = {t: raw / ('smash_xecs_2.87gev_hardSkyrme_' + t) for t in prep.TAGS}
    split = prep.load_split(clean)
    for cid in split['folds'][0]['validation'] + split['folds'][0]['test']:
        hashes[cid] = {**hashes[cid], 'hits': poison(raw, cid, roots)}
    poisoned = write_split(tmp_path, raw, hashes, test=(MEMBERS[2],), name='poisoned.json')
    second = prepare(tmp_path, poisoned, name='poisoned')
    assert first['scalers']['fold_0']['sides'] == second['scalers']['fold_0']['sides']
    for cid in split['folds'][0]['train']:
        a = torch.load(tmp_path / 'clean' / first['files'][cid]['packs']['B/fold_0']['path'], weights_only=True)
        b = torch.load(tmp_path / 'poisoned' / second['files'][cid]['packs']['B/fold_0']['path'], weights_only=True)
        assert all(torch.equal(a[k], b[k]) for k in a)
    cid = split['folds'][0]['validation'][0]
    a = torch.load(tmp_path / 'clean' / first['files'][cid]['packs']['B/fold_0']['path'], weights_only=True)
    b = torch.load(tmp_path / 'poisoned' / second['files'][cid]['packs']['B/fold_0']['path'], weights_only=True)
    assert not torch.equal(a['x'], b['x'])  # the poison is visible where it lives
    # Moving one poisoned file into training changes the scaler: the test has teeth.
    moved = json.loads(poisoned.read_text())
    parts = moved['folds']['0']['partitions']
    gid = parts['val'].pop(0)
    parts['train'].append(gid)
    poisoned.write_text(json.dumps(moved))
    third = prepare(tmp_path, poisoned, name='moved')
    assert third['scalers']['fold_0']['sides'] != first['scalers']['fold_0']['sides']


def test_parallel_workers_write_the_same_packs(tmp_path, world, dcm):
    _, raw, hashes, _ = world
    split_path = write_split(tmp_path, raw, hashes)
    one = prepare(tmp_path, split_path, workers=1, name='one')
    two = prepare(tmp_path, split_path, workers=2, name='two')
    for cid, item in one['files'].items():
        for key, entry in item['packs'].items():
            a = torch.load(tmp_path / 'one' / entry['path'], weights_only=True)
            b = torch.load(tmp_path / 'two' / two['files'][cid]['packs'][key]['path'], weights_only=True)
            assert all(torch.equal(a[k], b[k]) for k in a)
    assert one['scalers']['fold_0']['sides'] == two['scalers']['fold_0']['sides']


def test_observation_that_differs_from_the_audit_stops_prepare(tmp_path, world, dcm):
    _, raw, hashes, index = world
    split_path = write_split(tmp_path, raw, hashes)
    refs = prep.audit_references(index, set(prep.load_split(split_path)['contents']), tmp_path / 'refs')
    cid = sorted(refs)[0]
    saved = torch.load(refs[cid]['path'], weights_only=True)
    saved['values'][0, 0] += 1
    torch.save(saved, refs[cid]['path'])
    content = prep.load_split(split_path)['contents'][cid]
    task = (cid, content, str(raw / ('smash_xecs_2.87gev_hardSkyrme_' + content['tag'])), str(tmp_path / 'w'), refs[cid])
    with pytest.raises(ValueError, match='differs from the saved audit tensor'):
        prep.observe_file(task)
