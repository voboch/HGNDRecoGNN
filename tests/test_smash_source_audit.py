"""Truth relabelling and batched candidate topology on small native fixtures."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from HGNDRecoGNN.scripts import audit_smash_sources as audit
from HGNDRecoGNN.scripts.audit_smash_sources import HITS, native_components, ledger, matching, read_csv
from HGNDRecoGNN.data.graph_dataset import _build_single_graph, _dfToF


def frame():
    # Ekin is set to exactly the ToF estimate. The first two hits form one
    # soft candidate, and the next two form a hard candidate; the last is fake.
    t = np.array([50., 50.1, 29., 29.1, 30.])
    df = pd.DataFrame({'Row': [0]*5, 'fY': [45.]*5, 'fX': [-120.]*5, 'fZ': [730.]*5,
                       'LayerId': [0, 0, 8, 8, 4], 'RowId': [0]*5, 'ColumnId': [0, 1, 10, 11, 5],
                       'fTime': t, 'PDG': [2112]*4+[22], 'Side': [5]*5, 'Id': [1, 1, 2, 2, 3]})
    d = np.linalg.norm(df[['fX','fY','fZ']].to_numpy().T/100., axis=0)
    df['Ekin'] = _dfToF(d, t)
    return df


def test_relaxing_floor_adds_only_soft_cluster_and_preserves_existing_target():
    f = frame()
    c = native_components(f)
    x = ledger(f,c)
    assert x['native_signal_clusters'] == 1
    assert x['relaxed_signal_clusters'] == 2
    assert x['added_soft_hits'] == 2 and x['added_soft_clusters'] == 1
    assert x['native_target_changes_above_floor'] == 0


def test_batched_radius_components_match_individual_native_builder():
    f = frame()
    g = f.copy()
    g.Row = 1
    g.Id += 100
    both = pd.concat([f,g], ignore_index=True)
    c = native_components(both)
    for row, part in both.groupby('Row',sort=False):
        x = torch.zeros((len(part),10)).numpy()
        meta={'n0_label':np.zeros(len(part)), 'Ekin':part.Ekin.to_numpy(), 'Id':part.Id.to_numpy(),
              'Row':int(row), 'istop':True, 'pos_local':part[['LayerId','RowId','ColumnId']].to_numpy(copy=True),
              'pos_time':part[['fTime']].to_numpy(copy=True)}
        native = _build_single_graph((x,meta,3.6,1.5)).cluster.reshape(-1).numpy()
        subset = c[part.index]
        assert np.array_equal(native[:,None]==native[None,:],subset[:,None]==subset[None,:])
    assert not set(c[:5]).intersection(c[5:])


def test_matching_counts_merged_and_split_objects_once():
    # Cluster0 has objects1/2, cluster1 also contains1. Maximum matching is2.
    x = matching(np.array([0,0,1]),np.array([7,7,7]),np.array([1,2,1]),np.array([True]*3),2)
    assert x == {'eligible_objects':2,'signal_clusters':2,'matched_objects':2,'merged_clusters':1,'split_objects':1}


def test_header_only_export_has_numeric_empty_columns(tmp_path):
    path = tmp_path/'empty_hits.csv'
    path.write_text(','.join(HITS)+'\n')
    f = read_csv(path, HITS)
    assert len(f) == 0
    assert np.isfinite(f).all(axis=None)
    assert native_components(f).size == 0


def test_nucleon_primary_inventory_does_not_need_mesons(tmp_path):
    # Primary exports contain n/p only; a meson is not a missing neutron.
    v = pd.DataFrame(dict(Row=[0, 0], Id=[1, 2], PDG=[2112, -211],
                          fMotherId=[-1, -1], Side=[5, 5], vX=[0., 0.], vY=[0., 0.], vZ=[0., 0.]))
    path = tmp_path/'prim.csv'
    v.loc[:0, ['Row','Id','PDG','vX','vY','vZ']].to_csv(path, index=False)
    r = audit.primary_probe(v, path)
    assert r['eligible_records'] == r['matched_ids'] == 1
    assert r['eligible_neutrons'] == r['neutron_pdg_and_vertex_agree'] == 1


def test_failure_reporting_keeps_original_when_output_is_inaccessible(tmp_path, monkeypatch, capsys):
    def denied(*args):
        raise PermissionError('output inaccessible')
    monkeypatch.setattr(audit, 'write', denied)
    audit.record_failure(tmp_path, PermissionError('input inaccessible'))
    message = capsys.readouterr().err
    assert 'output inaccessible' in message and 'input inaccessible' in message


def test_frozen_common_inventory_excludes_unrelated_pairs_and_saves_observables(tmp_path):
    import argparse
    import json
    name = 'smash_xecs_2.87gev_hardSkyrme_zeroSpot'
    raw, primary = tmp_path/'raw', tmp_path/'primary'
    folder = raw/name/'run'
    prim_folder = primary/name/'run'
    folder.mkdir(parents=True)
    prim_folder.mkdir(parents=True)
    f = frame()
    h = f.reindex(columns=audit.HITS).fillna(0)
    h.Instance = np.arange(len(f))
    h.fELoss = .01
    v = f.reindex(columns=audit.VACS).fillna(0)
    v.Instance = h.Instance
    v.fMotherId = -1
    h.to_csv(folder/'one_hits.csv', index=False)
    v.to_csv(folder/'one_vacs.csv', index=False)
    v.loc[v.PDG == 2112, ['Row','Id','PDG','vX','vY','vZ']].drop_duplicates(['Row','Id']).to_csv(prim_folder/'one_prim.csv', index=False)
    # This unrelated pair has invalid content and must never be read.
    for suffix in ['hits', 'vacs']:
        (folder/f'unrelated_{suffix}.csv').write_text('invalid\n')
    manifest = tmp_path/'inventory.json'
    manifest.write_text(json.dumps({'tags': {'zeroSpot': {
        'all_pair_names': ['run/one', 'run/unrelated'], 'common_primary_pair_names': ['run/one']}}}))
    args = argparse.Namespace(tag='zeroSpot', raw_root=raw, primary_root=primary,
                              output=tmp_path/'out', inventory=manifest,
                              limit_files=None, save_observations=True)
    audit.run(args)
    summary = json.loads((args.output/'summary.json').read_text())
    assert summary['files_processed'] == 1
    entry = summary['files'][0]
    p = args.output/entry['observations']['path']
    assert audit.sha(p) == entry['observations']['sha256']
    saved = torch.load(p, weights_only=True)
    assert 'fTime' in saved['columns'] and torch.isfinite(saved['values']).all()
    assert summary['primary_binding_probe']['eligible_neutrons'] == 2
    # A node-local copy with an explicit primary filename association keeps
    # the original detector-member jitter and observed-hit ordering.
    stage = tmp_path/'stage'
    raw_stage, prim_stage = stage/'raw/zeroSpot/run', stage/'primary/zeroSpot/run'
    raw_stage.mkdir(parents=True)
    prim_stage.mkdir(parents=True)
    for suffix in ['hits', 'vacs']:
        (raw_stage/f'one_{suffix}.csv').write_bytes((folder/f'one_{suffix}.csv').read_bytes())
    (prim_stage/'two_prim.csv').write_bytes((prim_folder/'one_prim.csv').read_bytes())
    (stage/'bundle_verified.json').write_text('{"state":"complete"}')
    mapped = json.loads(manifest.read_text())
    mapped['tags']['zeroSpot']['primary_member_map'] = {'run/one': 'run/two'}
    manifest.write_text(json.dumps(mapped))
    args.staged_root, args.output = stage, tmp_path/'mapped_out'
    # The frozen copy remains usable when the original directory is unavailable.
    hidden = tmp_path/'original_hidden'
    folder.rename(hidden)
    try:
        audit.run(args)
    finally:
        hidden.rename(folder)
    mapped_summary = json.loads((args.output/'summary.json').read_text())
    assert mapped_summary['files'][0]['primary_source_member'] == 'run/two'
    mapped_saved = torch.load(args.output/mapped_summary['files'][0]['observations']['path'], weights_only=True)
    assert saved['columns'] == mapped_saved['columns']
    assert torch.equal(saved['values'], mapped_saved['values'])
    # A declared exact alias retains input coverage without doubling counts.
    for suffix in ['hits', 'vacs']:
        (folder/f'two_{suffix}.csv').write_bytes((folder/f'one_{suffix}.csv').read_bytes())
    item = mapped['tags']['zeroSpot']
    item['all_pair_names'] = ['run/one', 'run/two', 'run/unrelated']
    item['common_primary_pair_names'] = ['run/one', 'run/two']
    item['primary_member_map'] = {'run/one':'run/one','run/two':'run/one'}
    item['duplicate_exports'] = {'run/two':'run/one'}
    manifest.write_text(json.dumps(mapped))
    args.staged_root, args.output = None, tmp_path/'alias_out'
    audit.run(args)
    alias_summary = json.loads((args.output/'summary.json').read_text())
    assert alias_summary['totals'] == summary['totals']
    assert alias_summary['file_accounting']['duplicate_byte_aliases'] == 1
    assert alias_summary['primary_probe_files'] == 1
    assert alias_summary['files'][1]['duplicate_of'] == 'run/one'
    assert alias_summary['files'][1]['observations'] is None


def test_full_audit_keeps_declared_header_only_exports_without_binding(tmp_path):
    import argparse
    import json
    name = 'smash_xecs_2.87gev_hardSkyrme_defaultSpot'
    raw = tmp_path/'raw'
    folder = raw/name/'run'
    folder.mkdir(parents=True)
    for suffix, columns in [('hits', audit.HITS), ('vacs', audit.VACS)]:
        (folder/f'empty_{suffix}.csv').write_text(','.join(columns)+'\n')
    hashes = {s: audit.sha(folder/f'empty_{s}.csv') for s in ['hits', 'vacs']}
    item = {'all_pair_names': ['run/empty'], 'common_primary_pair_names': ['run/empty'],
            'primary_member_map': {'run/empty': None},
            'input_sha256': {'run/empty': hashes},
            'empty_exports': {'run/empty': {'sha256': hashes, 'exported_hits': 0, 'exported_truth_records': 0}}}
    inventory = tmp_path/'inventory.json'
    inventory.write_text(json.dumps({'tags': {'defaultSpot': item}}))
    args = argparse.Namespace(tag='defaultSpot', raw_root=raw, primary_root=tmp_path/'primary',
                              output=tmp_path/'out', inventory=inventory,
                              limit_files=None, save_observations=True)
    audit.run(args)
    result = json.loads((args.output/'summary.json').read_text())
    assert result['totals']['clusters'] == 0
    assert result['file_accounting']['header_only_exports'] == 1
    assert result['file_accounting']['processed_root_events'] is None
    assert result['primary_probe_files'] == 0
    assert result['files'][0]['primary_source_member'] is None
    assert result['files'][0]['primary_binding_state'] == 'unresolved_header_only'
    # A declaration cannot turn real observed hits into a verified empty export.
    h = frame().reindex(columns=audit.HITS).fillna(0)
    h.Instance = np.arange(len(h))
    h.fELoss = .01
    h.to_csv(folder/'empty_hits.csv', index=False)
    item['input_sha256']['run/empty']['hits'] = audit.sha(folder/'empty_hits.csv')
    inventory.write_text(json.dumps({'tags': {'defaultSpot': item}}))
    args.output = tmp_path/'invalid'
    import pytest
    with pytest.raises(ValueError, match='Declared empty export contains rows'):
        audit.run(args)


def test_full_binding_rejects_changed_momentum_even_when_ids_vertices_agree(tmp_path):
    v = pd.DataFrame(dict(Row=[0], Id=[1], PDG=[2112], fMotherId=[-1], Side=[5],
                          Ekin=[1.], Px=[.1], Py=[.2], Pz=[1.], vX=[0.], vY=[0.], vZ=[0.]))
    path = tmp_path/'primary.csv'
    v.assign(Px=.11).to_csv(path, index=False)
    result = audit.primary_probe(v, path, exact_kinematics=True)
    assert result['eligible_records'] == result['pdg_and_vertex_agree'] == 1
    assert result['pdg_vertex_kinematics_agree'] == 0
