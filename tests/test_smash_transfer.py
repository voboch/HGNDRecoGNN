"""Pin native transfer topology, feature arithmetic and physical event keys."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from HGNDRecoGNN.data.graph_dataset import _build_single_graph, _dfToF, FEATURES
from HGNDRecoGNN.data.smash_transfer import transfer_pack
from HGNDRecoGNN.data.student_pipeline import NativePackDataset


def fixture():
    f = pd.DataFrame(dict(Row=[2, 1, 2, 1, 2, 1], fX=[-120.]*6,
         fY=[45., -45., 45., -45., -45., 45.], fZ=[730.]*6,
         LayerId=[0, 0, 0, 8, 8, 8], RowId=[1]*6,
         ColumnId=[0, 0, 1, 10, 11, 10], fTime=[29., 29., 29.1, 30., 30., 30.],
         fELoss=[.01]*6, PDG=[2112, 22, 2112, 2112, 22, 2112], Side=[5]*6))
    distance = np.linalg.norm(f[['fX','fY','fZ']].to_numpy()/100., axis=1)
    f['Ekin'] = _dfToF(distance, f.fTime.to_numpy())
    return f


def test_transfer_matches_native_builder_with_frozen_affine_remap():
    f = fixture()
    scalers = [StandardScaler().fit(pd.DataFrame(np.arange(90).reshape(10,9)+s, columns=FEATURES[:-1])) for s in (0, 3)]
    a = np.full((2, 9), 1.1)
    b = np.full((2, 9), -.2)
    bounds = dict(LayerId=[0, 8], RowId=[0, 6], ColumnId=[0, 11])
    pack = transfer_pack(f, scalers, a, b, bounds)
    ds = NativePackDataset(pack)
    assert len(ds) == 4  # Both arms of each physical event survive.
    for i in range(len(ds)):
        graph = ds[i]
        side, row = int(graph.istop.item()), int(graph.Row.item())
        part = f.loc[(f.Row == row) & ((f.fY > 0) == bool(side))].copy()
        distance = np.linalg.norm(part[['fX','fY','fZ']].to_numpy()/100., axis=1)
        part['eToF'] = _dfToF(distance, part.fTime.to_numpy())
        expected = scalers[side].transform(part[FEATURES[:-1]]).astype(np.float32)
        expected = expected*np.asarray(a[side], np.float32)+np.asarray(b[side], np.float32)
        assert np.array_equal(graph['hits'].x[:, :-1].numpy(), expected)
        prompt = ((part.PDG == 2112) & (part.Ekin > .3)).to_numpy(np.float32)
        meta = dict(n0_label=prompt, Ekin=part.Ekin.to_numpy(), Id=np.arange(len(part)),
                    Row=row, istop=bool(side),
                    pos_local=part[['LayerId','RowId','ColumnId']].to_numpy(copy=True),
                    pos_time=part[['fTime']].to_numpy(copy=True))
        native = _build_single_graph((graph['hits'].x.numpy(), meta, 3.6, 1.0))
        assert torch.equal(native['hits','hits'].edge_index, graph['hits','hits'].edge_index)
        assert torch.equal(native.cluster.ravel(), graph.cluster.ravel())
        assert torch.equal(native.cllabel, graph.cllabel)
        assert torch.allclose(native.clenergy, graph.clenergy)
    f2 = f.copy()
    f2['Ekin'] = 0.
    f2['PDG'] = 22
    other = transfer_pack(f2, scalers, a, b, bounds)
    assert torch.equal(pack['x'], other['x'])
    assert torch.equal(pack['edges'], other['edges'])
    assert not other['label'].any()


def test_transfer_gate_counts_declared_empty_files_but_rejects_fictitious_binding():
    import copy
    import pytest
    from HGNDRecoGNN.scripts.smash_transfer_infer import validate_audit_summary
    hashes = dict(hits='h', vacs='v')
    f = dict(run='run', stem='one', sha256=dict(hits='nonempty_h', vacs='nonempty_v'),
             primary_probe=dict(eligible_records=1, matched_ids=1, pdg_and_vertex_agree=1))
    empty = dict(run='run', stem='two', sha256=hashes, header_only_export=True,
                 exported_hits=0, exported_truth_records=0, observed_export_events=0,
                 primary_source_member=None, primary_probe=None)
    other_empty = {**empty, 'stem': 'three'}
    members = ['run/one', 'run/two', 'run/three']
    frozen = {'empty_exports': {m: dict(sha256=hashes, exported_hits=0, exported_truth_records=0)
                                for m in members[1:]}}
    summary = dict(state='complete', inventory_sha256='fixed', files=[f, empty, other_empty],
                   primary_probe_files=1, primary_binding_probe=f['primary_probe'],
                   file_accounting=dict(header_only_exports=2, bound_nonempty_files=1,
                                        processed_root_events=None, generated_event_count=None))
    validate_audit_summary(summary, frozen, members, 'fixed')
    bad = copy.deepcopy(summary)
    bad['files'][1]['primary_source_member'] = 'run/two'
    with pytest.raises(ValueError, match='header-only'):
        validate_audit_summary(bad, frozen, members, 'fixed')
    bad = copy.deepcopy(summary)
    bad['file_accounting']['generated_event_count'] = 0
    with pytest.raises(ValueError, match='Unknown processed-event'):
        validate_audit_summary(bad, frozen, members, 'fixed')
    bad = copy.deepcopy(summary)
    bad['files'][0]['primary_probe'] = None
    with pytest.raises(ValueError, match='lacks binding'):
        validate_audit_summary(bad, frozen, members, 'fixed')


def test_transfer_gate_excludes_only_frozen_byte_identical_source_alias():
    import copy
    import pytest
    from HGNDRecoGNN.scripts.smash_transfer_infer import validate_audit_summary
    first = dict(run='run', stem='one', sha256=dict(hits='h',vacs='v'),
                 primary_source_member='source/one',
                 observations=dict(path='one.pt',sha256='obs'),
                 primary_probe=dict(eligible_records=1,matched_ids=1,pdg_and_vertex_agree=1))
    alias = copy.deepcopy(first)
    alias.update(stem='two',duplicate_of='run/one',primary_probe=None,observations=None)
    frozen = dict(duplicate_exports={'run/two':'run/one'})
    summary = dict(state='complete',inventory_sha256='fixed',files=[first,alias],primary_probe_files=1,
                   primary_binding_probe=first['primary_probe'],
                   file_accounting=dict(header_only_exports=0,bound_nonempty_files=1,duplicate_byte_aliases=1,
                                        processed_root_events=None,generated_event_count=None))
    validate_audit_summary(summary,frozen,['run/one','run/two'],'fixed')
    bad = copy.deepcopy(summary)
    bad['files'][1]['sha256']['hits'] = 'different_transport'
    with pytest.raises(ValueError,match='exact-byte alias'):
        validate_audit_summary(bad,frozen,['run/one','run/two'],'fixed')
