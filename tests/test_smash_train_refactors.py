"""Byte identity of the two refactors used by SMASH training.

`legacy_observation` and `legacy_transfer_pack` are verbatim copies of
scripts/audit_smash_sources.py (lines 260-284) and data/smash_transfer.py
(transfer_pack) at commit 00411bc, before observe() and observed_features()
were factored out. They are frozen here so the comparison survives later commits.
"""
from pathlib import Path
import argparse
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.nn import radius_graph

from HGNDRecoGNN.data.graph_dataset import FEATURES, _dfToF
from HGNDRecoGNN.data.smash_transfer import RADIUS_CELLS, TIME_NS, observed_features, transfer_pack
from HGNDRecoGNN.scripts import audit_smash_sources as audit
from HGNDRecoGNN.scripts.audit_smash_sources import HITS
from HGNDRecoGNN.tests.test_smash_train_fixtures import GEOMETRY, MEMBERS, fake_dcm, write_export, write_raw
from HGNDRecoGNN.tests.test_smash_transfer import fixture as transfer_fixture


def legacy_observation(hf, vf, root, original_hp, hp, vp):
    if vf.duplicated(['Row', 'Instance']).any() or hf.duplicated(['Row', 'Instance']).any():
        raise ValueError(f'Duplicate local Row/Instance in {hp}')
    inventory = np.union1d(hf.Row.unique(), vf.Row.unique())  # noqa: F841
    arrival = vf.loc[(vf.PDG.abs() == 2112) & vf.Side.isin([5, 6])]
    # The same transported object may have multiple hit associations.
    # This is detector-export coverage, not the generated inventory.
    arrival_ids = arrival[['Row', 'Id']].drop_duplicates()  # noqa: F841
    hard_arrival_ids = arrival.loc[arrival.Ekin > .3, ['Row', 'Id']].drop_duplicates()  # noqa: F841
    obs = hf.loc[(hf.fELoss > .003) & np.isfinite(hf[HITS]).all(axis=1)].copy()
    truth_cols = ['Row', 'Instance', 'Id', 'PDG', 'Ekin', 'Side', 'fMotherId', 'vX', 'vY', 'vZ']
    merged = obs.merge(vf[truth_cols], on=['Row', 'Instance'], how='left', validate='one_to_one', indicator=True)
    missing = merged._merge != 'both'
    if missing.any():
        raise ValueError(f'{int(missing.sum())} observed hits lack MC association in {hp}')
    if not np.isfinite(merged[['Id', 'PDG', 'Ekin', 'Side']]).all(axis=None):
        raise ValueError(f'Invalid truth fields in {vp}')
    seed = int.from_bytes(hashlib.sha256((str(original_hp.relative_to(root))+':audit-noise-v1:42').encode()).digest()[:8], 'little')
    merged['fTime'] += np.random.default_rng(seed).normal(0, .15, len(merged))
    merged = merged.loc[merged.fY != 0].copy()
    # Preserve the production half-event PDG==0 rejection, and record it.
    keys = pd.MultiIndex.from_arrays([merged.Row, merged.fY > 0])
    bad_keys = keys[merged.PDG.to_numpy() == 0].unique()
    rejected = keys.isin(bad_keys)
    dropped = int(rejected.sum())
    kept = merged.loc[~rejected].copy()
    return obs, kept, dropped


def legacy_transfer_pack(frame, scalers, affine_a, affine_b, geometry_bounds):
    """Keep per-event/arm namespaces, local cluster/edge IDs, and hit order."""
    if not len(frame):
        return None
    rows = frame.Row.to_numpy(np.int64)
    sides = (frame.fY.to_numpy() > 0).astype(np.int64)
    order = np.lexsort((np.arange(len(rows)), rows, sides))
    frame = frame.iloc[order].copy()
    rows, sides = rows[order], sides[order]
    keys = pd.MultiIndex.from_arrays([sides, rows])
    batch, uniques = pd.factorize(keys, sort=False)
    node_lengths = np.bincount(batch)
    node_ptr = np.r_[0, np.cumsum(node_lengths)]
    pos = torch.tensor(frame[['LayerId', 'RowId', 'ColumnId']].to_numpy(np.float32))
    times = torch.tensor(frame[['fTime']].to_numpy(np.float32))
    batch_t = torch.tensor(batch)
    spatial = radius_graph(pos, r=RADIUS_CELLS, batch=batch_t, loop=True, max_num_neighbors=32)
    temporal = radius_graph(times, r=TIME_NS, batch=batch_t, loop=True, max_num_neighbors=32)
    stride = len(frame)+1
    edges = spatial[:, torch.isin(spatial[0]*stride+spatial[1], temporal[0]*stride+temporal[1])].numpy()
    if edges.size and not np.array_equal(batch[edges[0]], batch[edges[1]]):
        raise ValueError('Inter-event edges in transfer view')
    edge_batch = batch[edges[0]]
    edge_order = np.argsort(edge_batch, kind='stable')
    edges, edge_batch = edges[:, edge_order], edge_batch[edge_order]
    edge_lengths = np.bincount(edge_batch, minlength=len(node_lengths))
    edge_ptr = np.r_[0, np.cumsum(edge_lengths)]
    adj = csr_matrix((np.ones(edges.shape[1]), edges), shape=(len(frame), len(frame)))
    nc, labels = connected_components(adj, directed=False)
    cluster_batch = np.full(nc, -1, dtype=np.int64)
    cluster_batch[labels] = batch
    cluster_lengths = np.bincount(cluster_batch, minlength=len(node_lengths))
    cluster_ptr = np.r_[0, np.cumsum(cluster_lengths)]
    local_cluster = labels - cluster_ptr[batch]
    local_edges = edges - node_ptr[edge_batch]
    distance = np.linalg.norm(frame[['fX', 'fY', 'fZ']].to_numpy()/100., axis=1)
    frame['eToF'] = _dfToF(distance, frame.fTime.to_numpy())
    frame['SurfaceHit'] = False
    for col, (lo, hi) in geometry_bounds.items():
        if not frame[col].between(lo, hi).all():
            raise ValueError('SMASH cell outside verified DCM geometry')
        frame['SurfaceHit'] |= (frame[col] == lo) | (frame[col] == hi)
    x = np.empty((len(frame), len(FEATURES)), dtype=np.float32)
    for side in (0, 1):
        mask = sides == side
        # Match DCM cache float32 storage followed by its frozen train-only
        # affine remap, rather than silently fitting a new SMASH scaler.
        if mask.any():
            standardized = scalers[side].transform(frame.loc[mask, FEATURES[:-1]]).astype(np.float32)
            x[mask, :-1] = standardized * np.asarray(affine_a[side], np.float32) + np.asarray(affine_b[side], np.float32)
    x[:, -1] = frame.SurfaceHit.to_numpy(np.float32)
    ek = frame.Ekin.to_numpy()
    prompt = ((frame.PDG.abs().to_numpy() == 2112) & frame.Side.isin([5, 6]).to_numpy()
              & (ek > .3) & (ek >= _dfToF(distance, frame.fTime.to_numpy()+.3))
              & (ek <= _dfToF(distance, frame.fTime.to_numpy()-.3)))
    cl_label = np.zeros(nc, dtype=np.float32)
    cl_energy = np.zeros(nc, dtype=np.float32)
    np.maximum.at(cl_label, labels, prompt)
    np.maximum.at(cl_energy, labels, np.where(prompt, ek, 0))
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transfer features')
    return {'x': torch.tensor(x), 'pe': torch.zeros((len(x), 4)),
            'edges': torch.tensor(local_edges), 'cluster': torch.tensor(local_cluster),
            'label': torch.tensor(cl_label), 'energy': torch.tensor(cl_energy),
            'row': torch.tensor([int(key[1]) for key in uniques]),
            'side': torch.tensor([int(key[0]) for key in uniques]),
            'index': torch.arange(len(uniques)), 'node_ptr': torch.tensor(node_ptr),
            'edge_ptr': torch.tensor(edge_ptr), 'cluster_ptr': torch.tensor(cluster_ptr)}


def read_pair(root, member):
    hp, vp = Path(root) / (member + '_hits.csv'), Path(root) / (member + '_vacs.csv')
    return audit.read_csv(hp, audit.HITS), audit.read_csv(vp, audit.VACS), hp, vp


def edge_case_export(tmp_path):
    """Low Edep, fY == 0, a nonfinite hit field and a PDG==0 half-event in one file."""
    root = Path(tmp_path) / 'edge'
    write_export(root, MEMBERS[0], 5, events=12, pdg0_row=4)
    hp = root / (MEMBERS[0] + '_hits.csv')
    lines = hp.read_text().splitlines()
    cells = [line.split(',') for line in lines[1:]]
    cells[0][HITS.index('fELoss')] = '0.001'
    cells[1][HITS.index('fY')] = '0.0'
    cells[2][HITS.index('fX')] = 'nan'
    hp.write_text('\n'.join([lines[0]] + [','.join(c) for c in cells]) + '\n')
    return root


@pytest.mark.parametrize('case', ['fixture', 'edge'])
def test_observe_reproduces_the_legacy_observation_block_bit_for_bit(tmp_path, case):
    if case == 'fixture':
        raw, _ = write_raw(tmp_path, events=25)
        root = raw / 'smash_xecs_2.87gev_hardSkyrme_bigSpot'
    else:
        root = edge_case_export(tmp_path)
    hf, vf, hp, vp = read_pair(root, MEMBERS[0])
    obs_old, kept_old, dropped_old = legacy_observation(hf.copy(), vf.copy(), root, hp, hp, vp)
    obs_new, kept_new, dropped_new = audit.observe(hf.copy(), vf.copy(), str(hp.relative_to(root)), hp, vp)
    assert dropped_new == dropped_old and dropped_old > 0
    assert obs_new.equals(obs_old)
    assert list(kept_new.columns) == list(kept_old.columns)
    assert kept_new.index.equals(kept_old.index)
    cols = [c for c in kept_old.columns if c != '_merge']
    assert kept_new[cols].to_numpy(np.float64).tobytes() == kept_old[cols].to_numpy(np.float64).tobytes()
    assert (kept_new._merge == kept_old._merge).all()


def test_observe_refuses_duplicates_and_unassociated_hits_as_before(tmp_path):
    root = edge_case_export(tmp_path)
    hf, vf, hp, vp = read_pair(root, MEMBERS[0])
    with pytest.raises(ValueError, match='Duplicate local Row/Instance'):
        audit.observe(pd.concat([hf, hf.iloc[:1]]), vf, 'x_hits.csv', hp, vp)
    with pytest.raises(ValueError, match='lack MC association'):
        audit.observe(hf, vf.drop(index=[3, 4]), 'x_hits.csv', hp, vp)


def test_audit_run_saves_exactly_the_legacy_observations(tmp_path):
    raw, _ = write_raw(tmp_path, events=20)
    out = tmp_path / 'audit'
    audit.run(argparse.Namespace(tag='zeroSpot', raw_root=raw, primary_root=tmp_path / 'none', output=out,
                                 inventory=None, save_observations=True, staged_root=None, limit_files=None))
    root = raw / 'smash_xecs_2.87gev_hardSkyrme_zeroSpot'
    for member in MEMBERS:
        hf, vf, hp, vp = read_pair(root, member)
        _, kept, _ = legacy_observation(hf, vf, root, hp, hp, vp)
        cols = [c for c in kept.columns if c != '_merge']
        saved = torch.load(out / 'observations' / (member + '.pt'), weights_only=True)
        assert saved['columns'] == cols
        assert saved['values'].numpy().tobytes() == kept[cols].to_numpy(np.float64).tobytes()


def observation_frames(tmp_path):
    raw, _ = write_raw(tmp_path, events=25)
    frames = []
    for tag in ('zeroSpot', 'bigSpot'):
        root = raw / ('smash_xecs_2.87gev_hardSkyrme_' + tag)
        hf, vf, hp, vp = read_pair(root, MEMBERS[1])
        _, kept, _ = audit.observe(hf, vf, MEMBERS[1] + '_hits.csv', hp, vp)
        cols = [c for c in kept.columns if c != '_merge']
        frames.append(pd.DataFrame(kept[cols].to_numpy(np.float64), columns=cols))
    return frames


def test_transfer_pack_is_byte_identical_to_the_legacy_function(tmp_path):
    scalers, a, b, geometry, _ = fake_dcm()
    for frame in observation_frames(tmp_path):
        old = legacy_transfer_pack(frame.copy(), scalers, a, b, geometry)
        new = transfer_pack(frame.copy(), scalers, a, b, geometry)
        assert set(old) == set(new)
        for key in old:
            assert old[key].dtype == new[key].dtype and torch.equal(old[key], new[key]), key
            assert old[key].numpy().tobytes() == new[key].numpy().tobytes(), key
    f = transfer_fixture()
    small = [StandardScaler().fit(pd.DataFrame(np.arange(90).reshape(10, 9) + s, columns=FEATURES[:-1])) for s in (0, 3)]
    bounds = dict(LayerId=[0, 8], RowId=[0, 6], ColumnId=[0, 11])
    old = legacy_transfer_pack(f.copy(), small, a, b, bounds)
    new = transfer_pack(f.copy(), small, a, b, bounds)
    assert all(torch.equal(old[k], new[k]) for k in old)
    assert transfer_pack(f.iloc[:0], small, a, b, bounds) is None
    outside = dict(bounds, ColumnId=[0, 5])
    for function in (legacy_transfer_pack, transfer_pack):
        with pytest.raises(ValueError, match='outside verified DCM geometry'):
            function(f.copy(), small, a, b, outside)


def test_observed_features_are_the_numbers_transfer_pack_standardises(tmp_path):
    for frame in observation_frames(tmp_path):
        features = observed_features(frame, GEOMETRY)
        scalers = [StandardScaler().fit(features[side]) for side in (0, 1)]
        pack = transfer_pack(frame, scalers, np.ones((2, 9)), np.zeros((2, 9)), GEOMETRY)
        sides = np.repeat(pack['side'].numpy(), np.diff(pack['node_ptr'].numpy()))
        for side in (0, 1):
            expected = scalers[side].transform(features[side]).astype(np.float32)
            assert np.array_equal(pack['x'][sides == side, :-1].numpy(), expected)
        assert sum(len(v) for v in features.values()) == len(frame)
        assert list(features[0].columns) == FEATURES[:-1]
    empty = observed_features(frame.iloc[:0], GEOMETRY)
    assert all(len(v) == 0 and list(v.columns) == FEATURES[:-1] for v in empty.values())


def test_noise_seed_matches_the_inherited_path_key():
    key = '10943245/0001_hits.csv'
    expected = int.from_bytes(hashlib.sha256((key + ':audit-noise-v1:42').encode()).digest()[:8], 'little')
    assert audit.noise_seed(key) == expected
    assert RADIUS_CELLS == 3.6 and TIME_NS == 1.0
    assert radius_graph is not None and csr_matrix is not None and connected_components is not None
    assert _dfToF is not None
