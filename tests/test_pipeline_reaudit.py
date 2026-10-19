"""Mechanism tests from the second pipeline audit; no cluster data needed."""
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from HGNDRecoGNN import device
from HGNDRecoGNN.training.metrics import LossAccumulator
from HGNDRecoGNN.training import train
from HGNDRecoGNN.training.losses import LossWeights
from HGNDRecoGNN.training.eval import predict


def test_population_loss_does_not_give_a_short_batch_half_the_weight():
    def loss_fn():
        pass
    loss_fn.normalizers = lambda g: {'cluster_cls': g.n, 'cluster_energy': g.signal}
    def aggregate(chunks, mode):
        a = LossAccumulator(loss_fn, LossWeights())
        for n, signal, cls, energy in chunks:
            a.update({'cluster_cls': torch.tensor(cls), 'cluster_energy': torch.tensor(energy),
                      'total': torch.tensor(cls+energy)}, SimpleNamespace(n=n, signal=signal))
        return a.result(mode)
    chunks = [(2, 0, 1., 0.), (18, 3, 9., 4.)]
    split = aggregate(chunks, 'population')
    merged = aggregate([(20, 3, 8.2, 4.)], 'population')
    assert split['total'] == pytest.approx(12.2)
    assert merged['total'] == pytest.approx(split['total'])
    assert aggregate(chunks, 'batch_mean')['total'] == 7.
    assert split['counts'] == {'cluster_cls': 20, 'cluster_energy': 3}


def test_resume_restores_python_numpy_torch_and_all_cuda_streams(monkeypatch):
    train.seed_everything(123)
    rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
           'torch': torch.get_rng_state(), 'cuda': [torch.tensor([1, 2], dtype=torch.uint8)]}
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    captured = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda states: captured.extend(states))
    train._restore_rng_state(rng)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    assert len(captured) == 1 and torch.equal(captured[0], rng['cuda'][0])


def test_atomic_resume_rejects_a_caller_epoch_from_a_different_generation(tmp_path):
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters())
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10])
    train._write_resume_state(str(tmp_path/'resume.pt'), model, opt, sch,
                             next_epoch=2, min_loss=.5, best_epoch=1,
                             arch_name='net_default', arch_kwargs={})
    cfg = train.TrainConfig(epochs=3, start_epoch=1, checkpoint_dir=str(tmp_path), verbose=False)
    with pytest.raises(ValueError, match='next_epoch'):
        train.fit(model, [], [], cfg, plan=device.DeviceMap(main=torch.device('cpu')))


def test_cluster_only_inference_preserves_truth_and_has_no_fake_hit_predictions():
    graphs = []
    for row, size in [(17, 2), (18, 1)]:
        g = HeteroData()
        g['hits'].x = torch.ones(size, 1)
        g['clusters'].x = torch.zeros(size, 1)
        g.cllabel = torch.tensor([1., 0.][:size]);g.clenergy = torch.ones(size)
        g.Row = torch.tensor(row);g.istop = torch.tensor([1])
        graphs.append(g)
    loader = DataLoader(graphs, batch_size=2)
    def forward(model, graph):
        return torch.zeros(graph.cllabel.numel()), torch.full_like(graph.clenergy, 2.)
    out = predict(torch.nn.Identity(), loader, plan=device.DeviceMap(main=torch.device('cpu')),
                  forward_fn=forward, verbose=False)
    assert out['clusters'].Row.tolist() == [17, 17, 18]
    assert out['clusters'].cl_score.tolist() == [.5, .5, .5]
    assert out['clusters'].cl_label.tolist() == [1, 0, 1]
    assert out['hits'].empty and out['edges'].empty
