"""Native membership, truth isolation and source-split barriers for student models."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader

from HGNDRecoGNN import models
from HGNDRecoGNN.data.student_pipeline import NativePackDataset, pack_native
from HGNDRecoGNN.models.student_pipeline import PIPELINE_MODELS, candidate_view
from HGNDRecoGNN.training.eval import predict


def graph(row=1):
    g = HeteroData()
    # Interleaved clusters: kNN needs an explicitly sorted batch view.
    g['hits'].x = torch.arange(50, dtype=torch.float32).reshape(5, 10) / 50
    g['hits'].student_pe = torch.zeros(5, 4)
    g['hits', 'hits'].edge_index = torch.tensor([[0, 2, 1, 3, 4], [2, 0, 3, 1, 4]])
    g['clusters'].x = torch.zeros(3, 1)
    g.cluster = torch.tensor([[0], [1], [0], [1], [2]])
    g.cllabel = torch.tensor([1., 0., 1.])
    g.clenergy = torch.tensor([1., 0., 2.])
    g.Row = torch.tensor([row])
    g.istop = torch.tensor([row % 2])
    return g


@pytest.mark.parametrize('name', PIPELINE_MODELS)
def test_native_models_isolate_candidates_and_events_and_do_not_read_truth(name):
    torch.manual_seed(42)
    args = {'hidden': 16, 'num_layers': 1, 'dropout': 0.}
    if 'spectral' in name:
        args.update(pe_dim=4, pe_hidden=8)
    if 'dynedge' in name:
        args['k'] = 8
    first, second = graph(), graph(2)
    model, _ = models.get(name, [first], **args)
    model.eval()
    with torch.no_grad():
        single = model(Batch.from_data_list([first]))
        both = model(Batch.from_data_list([first, second]))
        for a, b in zip(single, both):
            torch.testing.assert_close(a, b[:3], atol=1e-5, rtol=1e-4)
            assert len(a) == 3  # Retains the singleton candidate.
        other = first.clone()
        other['hits'].x[other.cluster.reshape(-1) == 1] += 100
        changed = model(Batch.from_data_list([other]))
        for a, b in zip(single, changed):
            torch.testing.assert_close(a[[0, 2]], b[[0, 2]], atol=1e-5, rtol=1e-4)
        batch = Batch.from_data_list([first])
        del batch.cllabel
        del batch.clenergy
        del batch.Row
        del batch.istop
        again = model(batch)
        for a, b in zip(single, again):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_pack_exactly_preserves_native_targets_topology_and_affine_features():
    original = graph()
    before = original.clone()
    a = np.array([[1.] * 9, [2.] * 9])
    b = np.array([[0.] * 9, [3.] * 9])
    pack = pack_native({17: original}, [17], a, b)
    reconstructed = NativePackDataset(pack)[0]
    expected = before['hits'].x.clone()
    expected[:, :9] = expected[:, :9] * 2 + 3
    torch.testing.assert_close(reconstructed['hits'].x, expected, atol=0, rtol=0)
    for key in ('cluster', 'cllabel', 'clenergy', 'Row', 'istop'):
        assert torch.equal(getattr(reconstructed, key), getattr(before, key))
    assert torch.equal(reconstructed['hits', 'hits'].edge_index, before['hits', 'hits'].edge_index)
    assert torch.equal(original['hits'].x, before['hits'].x)
    batch = Batch.from_data_list([reconstructed])
    view = candidate_view(batch, spectral=True)
    assert view.batch.tolist() == [0, 0, 1, 1, 2]
    assert view.n_hits.tolist() == [2., 2., 1.]
    assert view.pe.shape == (5, 4)


def test_native_predict_preserves_every_cluster_key_and_empty_unsupported_heads():
    gs = [graph(), graph(2)]
    model, spec = models.get('student_sage', gs, hidden=16, num_layers=1, dropout=0.)
    out = predict(model, DataLoader(gs, batch_size=2), forward_fn=spec.forward_fn, verbose=False)
    assert out['hits'].empty and out['edges'].empty
    cl = out['clusters']
    assert len(cl) == 6
    assert not cl.duplicated(['Row', 'cl_istop', 'ClusterID']).any()
    assert cl.cl_label.tolist() == [1, 0, 1, 1, 0, 1]
    assert cl.e_true.tolist() == [1., 0., 2., 1., 0., 2.]


def test_predict_applies_selected_device_plan(monkeypatch):
    from HGNDRecoGNN import device
    gs = [graph()]
    model, spec = models.get('student_sage', gs, hidden=16, num_layers=1, dropout=0.)
    applied = []
    original = device.to_device
    def record(model_arg, plan):
        applied.append(plan)
        return original(model_arg, plan)
    monkeypatch.setattr(device, 'to_device', record)
    plan = device.DeviceMap(torch.device('cpu'))
    result = predict(model, DataLoader(gs, batch_size=1), plan=plan,
                     forward_fn=spec.forward_fn, verbose=False)
    assert applied == [plan]
    assert len(result['clusters']) == 3


def test_test_barrier_requires_all_training_receipts_before_opening_test(tmp_path):
    from argparse import Namespace
    from HGNDRecoGNN.scripts.student_pipeline import test as evaluate_test
    args = Namespace(campaign=tmp_path)
    with pytest.raises(FileNotFoundError):
        evaluate_test(args, 'frozen-source')
    assert not (tmp_path / 'test_evaluation').exists()
