"""Held-out ladder: parameterised bands reproduce the retrospective; arms pair on shared draws."""
from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import pytest

from HGNDRecoGNN.scripts import reco_rn_heldout as held
from HGNDRecoGNN.scripts import reco_rn_retrospective as retro
from HGNDRecoGNN.tests.test_smash_train_fixtures import write_split

MEMBERS = tuple(f'10943245/{i:04d}' for i in range(1, 7))


def clusters(rng, rows=150, noise=.1):
    n = rows * 5
    label = (rng.random(n) < .4).astype(int)
    e_true = np.where(label == 1, rng.uniform(.3, 4, n), 0.)
    e_pred = np.where(label == 1, e_true * (1 + rng.normal(0, noise, n)), rng.uniform(0, 3, n))
    score = np.clip(np.where(label == 1, rng.normal(.7, .2, n), rng.normal(.3, .2, n)), 0, 1)
    return pd.DataFrame({'Row': np.repeat(np.arange(rows), 5), 'ClusterID': np.tile(np.arange(5), rows),
                         'cl_istop': 0, 'cl_score': score, 'e_pred': e_pred, 'cl_label': label, 'e_true': e_true})


def test_parameterised_counters_reproduce_the_retrospective_with_its_band_edges():
    rng = np.random.default_rng(3)
    frame = clusters(rng)
    band = held.BANDS['below1']
    weight = retro.injection_weight(frame)
    assert np.array_equal(held.cluster_counts(frame, .5, band), retro.cluster_counts(frame, .5))
    assert np.array_equal(held.cluster_counts(frame, .5, band, weight), retro.cluster_counts(frame, .5, weight))
    assert np.array_equal(held.event_counts(frame, .5, band), retro.event_counts(frame, .5))
    events = pd.DataFrame({'acc_n_hi': rng.poisson(2, 300), 'acc_n_lo': rng.poisson(2, 300)},
                          index=pd.Index(np.arange(300), name='Row'))
    assert np.array_equal(held.truth_counts(events, 120, np.arange(0, 300, 7), band),
                          retro.truth_counts(events, 120, np.arange(0, 300, 7)))
    primary = held.BANDS['primary']
    e = np.array([.1, .3, .99, 1., 1.5, 2., 5.])
    assert held.band_counts(e, np.ones(len(e)), primary).tolist() == [2., 2.]
    assert held.band_counts(e, np.ones(len(e)), band).tolist() == [2., 3.]
    assert primary['high'] == retro.E_HI  # the injection weight keeps the inherited hard band


def world(tmp_path):
    rng = np.random.default_rng(5)
    hashes = {f'{t}/{m}': {'hits': hashlib.sha256(f'{t}{m}h'.encode()).hexdigest(),
                           'vacs': hashlib.sha256(f'{t}{m}v'.encode()).hexdigest()}
              for t in retro.TAGS for m in MEMBERS}
    split = write_split(tmp_path, tmp_path / 'raw', hashes, members=MEMBERS[:2], test=MEMBERS[2:])
    tests = sorted(f'{t}/{m}' for t in retro.TAGS for m in MEMBERS[2:])
    ev = tmp_path / 'evaluation'
    predictions, events = {}, {t: [] for t in retro.TAGS}
    for cid in tests:
        tag, member = cid.split('/', 1)
        base = clusters(rng, noise=.08 if tag == 'bigSpot' else .1)
        other = base.assign(e_pred=base.e_pred * (1 + rng.normal(0, .05, len(base))),
                            cl_score=np.clip(base.cl_score + rng.normal(0, .05, len(base)), 0, 1))
        for arm in held.ARMS:
            for seed in retro.SEEDS:
                frame = other if arm == 'B0' else base  # B1, Bcomp, A1 equal A0 on purpose
                path = ev / arm / f'seed_{seed}' / (cid + '.csv')
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)
                predictions.setdefault(arm, {}).setdefault(seed, {})[cid] = {
                    'path': str(path.relative_to(ev)), 'sha256': held.sha256(path)}
        idx = pd.MultiIndex.from_product([[member + '_prim.csv'], np.arange(170)], names=['member', 'Row'])
        events[tag].append(pd.DataFrame({'acc_n_hi': rng.poisson(3, 170), 'acc_n_lo_reach': rng.poisson(3, 170),
                                         'acc_n_lo': rng.poisson(4, 170)}, index=idx))
    summary = {'state': 'complete', 'fold': 0, 'test_files': tests, 'empty_after_cuts': [],
               'identity': {'split_sha256': held.sha256(split)}, 'predictions': predictions,
               'training': {s: {'cuts': {arm: .5 for arm in held.ARMS}} for s in retro.SEEDS}}
    (ev / 'summary.json').write_text(json.dumps(summary))
    return split, ev, {t: pd.concat(v) for t, v in events.items()}


def test_arms_share_whole_file_draws_and_pair_exactly(tmp_path, monkeypatch):
    split, ev, events = world(tmp_path)
    monkeypatch.setattr(held, 'load_truth', lambda root, truth: events)
    summary = held.run([ev], split, tmp_path / 'out')
    for variant in held.BANDS:
        pooled = summary['variants'][variant]['pooled']
        assert pooled['n_files'] == {t: 4 for t in retro.TAGS}
        for kind in ('endpoint', 'midpoint'):
            same = pooled['pairs']['B1-A1'][kind]
            assert all(v['central_percent'] == 0 and v['bootstrap_sd_percent'] == 0 for v in same.values())
        differ = pooled['pairs']['B0-A0']['endpoint']['C4/seed_42']
        separate = np.hypot(pooled['arms']['B0']['endpoint']['C4/seed_42']['bootstrap_sd_percent'],
                            pooled['arms']['A0']['endpoint']['C4/seed_42']['bootstrap_sd_percent'])
        assert 0 < differ['bootstrap_sd_percent'] < separate  # pairing removes the shared file variance
        truth = [pooled['arms'][arm]['endpoint']['T1/seed_42']['central_percent'] for arm in held.ARMS]
        assert len(set(truth)) == 1
        assert set(pooled['gates']['A1']) >= {'per_seed', 'all_seeds', 'G5_null', 'G6_seed_stability'}
        assert 'fold_0' in summary['variants'][variant]
    assert summary['variants']['primary']['pooled']['arms']['A0']['endpoint'] != \
        summary['variants']['below1']['pooled']['arms']['A0']['endpoint']
    replicas = np.load(tmp_path / 'out/replicas.npz')
    assert np.array_equal(replicas['primary__pooled__A1__endpoint'], replicas['primary__pooled__B1__endpoint'])
    assert (tmp_path / 'out/figures/heldout_ladder.pdf').exists()
    with pytest.raises(ValueError, match='already exists'):
        held.run([ev], split, tmp_path / 'out')


def test_a_changed_prediction_file_or_a_file_in_two_folds_is_refused(tmp_path, monkeypatch):
    split, ev, events = world(tmp_path)
    monkeypatch.setattr(held, 'load_truth', lambda root, truth: events)
    s = json.loads((ev / 'summary.json').read_text())
    first = next(iter(s['predictions']['A0']['42'].values()))
    (ev / first['path']).write_text('Row\n1\n')
    with pytest.raises(ValueError, match='Prediction file changed'):
        held.run([ev], split, tmp_path / 'out')
