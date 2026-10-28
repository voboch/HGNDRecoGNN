"""Per-head selection, validation cut, monotone calibration, pack concatenation and the mirror."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import pytest
import torch

from HGNDRecoGNN.data.student_pipeline import NativePackDataset
from HGNDRecoGNN.scripts import smash_train as st
from HGNDRecoGNN.training.checkpoint import save_checkpoint
from HGNDRecoGNN.training.train import RESUME_NAME, epoch_checkpoint_name


def history(cls, energy, aggregation='population'):
    return [{'epoch': i, 'validation': {'aggregation': aggregation, 'total': c + .5 * e,
                                        'components': {'cluster_cls': c, 'cluster_energy': e}}}
            for i, (c, e) in enumerate(zip(cls, energy))]


def test_heads_are_selected_separately_and_ties_go_to_the_earliest_epoch():
    s = st.select_epochs(history([.5, .3, .3, .4], [.9, .8, .2, .2]))
    assert (s['e_cls'], s['e_en']) == (1, 2)
    assert s['e_comp'] == 2  # .3+.1 beats .3+.4
    assert not any(s['budget_limited'].values()) and not s['e_cls_at_epoch_0']
    s = st.select_epochs(history([.1, .3, .2], [.9, .8, .1]))
    assert s['e_cls_at_epoch_0'] and s['budget_limited']['e_en']
    with pytest.raises(ValueError, match='Population'):
        st.select_epochs(history([.1], [.1], aggregation='batch_mean'))
    with pytest.raises(ValueError, match='Nonfinite'):
        st.select_epochs(history([.1, float('nan')], [.1, .1]))


def test_cut_is_the_first_of_401_reaching_purity_070():
    rng = np.random.default_rng(1)
    y = rng.random(4000) < .3
    score = np.clip(np.where(y, rng.normal(.7, .2, 4000), rng.normal(.3, .2, 4000)), 0, 1)
    frame = pd.DataFrame({'cl_score': score, 'cl_label': y.astype(int), 'e_pred': 1., 'e_true': y * 1.})
    cut, scan = st.choose_cut(frame)
    assert len(scan) == 401 and np.isclose(scan.threshold.iloc[0], .001) and np.isclose(scan.threshold.iloc[-1], .999)
    i = int(np.flatnonzero(np.isclose(scan.threshold, cut))[0])
    assert scan.purity_count.iloc[i] >= .70 and (scan.purity_count.iloc[:i] < .70).all()
    hopeless = frame.assign(cl_label=0)
    assert st.choose_cut(hopeless)[0] is None


def test_calibration_inverts_the_median_response_and_stays_monotone():
    rng = np.random.default_rng(2)
    e_true = rng.uniform(.3, 6.5, 50000)
    e_pred = .8 * e_true * np.exp(rng.normal(0, .1, len(e_true)))
    cal = st.fit_calibration(e_true, e_pred)
    calibrated = st.apply_calibration(cal, e_pred)
    for lo, hi in zip(st.CAL_BINS[:-1], st.CAL_BINS[1:]):
        m = (e_true >= lo) & (e_true < hi)
        assert abs(np.median(calibrated[m] / e_true[m]) - 1) < .03
    grid = np.linspace(-1, 12, 2000)
    assert np.all(np.diff(st.apply_calibration(cal, grid)) > 0)
    # A saturating response is pooled into one point rather than inverted twice.
    sat = np.where(e_true > 3, 2.4 + .01 * (e_true - 3), .8 * e_true)
    sat[(e_true > 3.5) & (e_true < 4)] = 2.3
    cal = st.fit_calibration(e_true, sat)
    assert cal['pooled_bins'] >= 1 and np.all(np.diff(cal['x_e_pred']) > 0)
    with pytest.raises(ValueError, match='Fewer than two'):
        st.fit_calibration(e_true[:30], e_pred[:30])


def pack(n_graphs, seed):
    g = torch.Generator().manual_seed(seed)
    sizes = torch.randint(1, 5, (n_graphs,), generator=g)
    node_ptr = torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)])
    clusters = [torch.arange(int(s)) % 2 for s in sizes]
    nc = [int(c.max()) + 1 for c in clusters]
    edges = [torch.stack([torch.arange(int(s)), torch.arange(int(s))]) for s in sizes]
    return {'x': torch.randn(int(node_ptr[-1]), 10, generator=g), 'pe': torch.zeros(int(node_ptr[-1]), 4),
            'edges': torch.cat(edges, 1), 'cluster': torch.cat(clusters), 'label': torch.ones(sum(nc)),
            'energy': torch.rand(sum(nc), generator=g), 'row': torch.arange(n_graphs), 'side': torch.zeros(n_graphs, dtype=torch.long),
            'index': torch.arange(n_graphs), 'node_ptr': node_ptr,
            'edge_ptr': torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)]),
            'cluster_ptr': torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor(nc).cumsum(0)])}


def test_concatenated_packs_yield_the_same_graphs_in_order():
    parts = [pack(5, 1), pack(3, 2), pack(4, 3)]
    joined = NativePackDataset(st.concat_packs(parts))
    expected = [NativePackDataset(p)[i] for p in parts for i in range(len(p['row']))]
    assert len(joined) == len(expected)
    for a, b in zip((joined[i] for i in range(len(joined))), expected):
        for key in ('cluster', 'cllabel', 'clenergy', 'Row'):
            assert torch.equal(getattr(a, key), getattr(b, key))
        assert torch.equal(a['hits'].x, b['hits'].x)
        assert torch.equal(a['hits', 'hits'].edge_index, b['hits', 'hits'].edge_index)


def test_combine_refuses_checkpoints_that_scored_different_candidates():
    f = pd.DataFrame({'Row': [1, 1], 'ClusterID': [0, 1], 'cl_istop': [0, 0], 'cl_label': [1, 0], 'e_true': [1., 0.],
                      'cluster_size': [2, 1], 'cl_score': [.9, .1], 'e_pred': [1.1, .2]})
    g = f.assign(e_pred=[2., 3.], cl_score=[0., 0.])
    out = st.combine({'c': f}, {'c': g})['c']
    assert out.cl_score.tolist() == [.9, .1] and out.e_pred.tolist() == [2., 3.]
    with pytest.raises(ValueError, match='different candidates'):
        st.combine({'c': f}, {'c': g.assign(cl_label=[0, 0])})


def write_epoch(directory, epoch, value):
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(value)
    save_checkpoint(str(directory / epoch_checkpoint_name(epoch)), model, arch_name=st.MODEL, epoch=epoch)
    torch.save({'model': model.state_dict(), 'next_epoch': epoch + 1, 'best_epoch': 0,
                'history': [{}] * (epoch + 1)}, directory / RESUME_NAME)


def test_mirror_is_versioned_non_fatal_and_restores_the_newest_complete_generation(tmp_path):
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir()
    mirror = st.Mirror(local, remote)
    for epoch in range(3):
        write_epoch(local, epoch, epoch)
        mirror(epoch, [])
    state = json.loads((remote / 'state.json').read_text())
    assert state['next_epoch'] == 3 and set(state['sha256']) == {'epoch_000.pt', 'epoch_001.pt', 'epoch_002.pt', 'resume_002.pt'}
    n, _, _ = st.restore_from_mirror(remote, tmp_path / 'restored')
    assert n == 3 and (tmp_path / 'restored/model.pt').exists()
    # A failing copy is recorded and training goes on; the state keeps the last good generation.
    broken = st.Mirror(local, tmp_path / 'file_not_dir')
    (tmp_path / 'file_not_dir').rmdir() if (tmp_path / 'file_not_dir').is_dir() else None
    (tmp_path / 'file_not_dir').write_text('x')
    broken(0, [])
    assert broken.failures and broken.failures[-1]['epoch'] == 0
    (remote / 'epoch_001.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='differs from its state'):
        st.restore_from_mirror(remote, tmp_path / 'again')


def test_source_identity_verifies_every_file_and_ignores_inputs_for_the_code_digest(tmp_path, monkeypatch):
    root = tmp_path / 'pkg'
    (root / 'HGNDRecoGNN/scripts').mkdir(parents=True)
    (root / 'inputs').mkdir()
    for name in st.CODE_EXTRAS:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text('code ' + name)
    (root / 'inputs/split.json').write_text('{}')

    def sums():
        from hashlib import sha256
        lines = [f'{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}'
                 for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS']
        (root / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    sums()
    monkeypatch.setattr(st, '__file__', str(root / 'HGNDRecoGNN/scripts/smash_train.py'))
    first = st.source_identity(root / 'SHA256SUMS')
    (root / 'inputs/split.json').write_text('{"other": 1}')
    sums()
    second = st.source_identity(root / 'SHA256SUMS')
    assert first['code_digest'] == second['code_digest'] and first['sums_sha256'] != second['sums_sha256']
    assert first['code_files'] == sorted(st.CODE_EXTRAS)
    (root / 'HGNDRecoGNN/scripts/publish_smash_directory.py').write_text('changed')
    sums()
    assert st.source_identity(root / 'SHA256SUMS')['code_digest'] != first['code_digest']
    (root / 'HGNDRecoGNN/scripts/publish_smash_directory.py').unlink()
    sums()
    with pytest.raises(ValueError, match='outside the verified package'):
        st.source_identity(root / 'SHA256SUMS')
    (root / 'HGNDRecoGNN/scripts/publish_smash_directory.py').write_text('code')
    sums()
    (root / 'HGNDRecoGNN/scripts/smash_train.py').write_text('changed')
    with pytest.raises(ValueError, match='Source package file changed'):
        st.source_identity(root / 'SHA256SUMS')
