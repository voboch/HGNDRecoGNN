"""The development campaign changes one declared factor per comparison."""
import hashlib
import sys
from types import SimpleNamespace

import pytest
import torch
from HGNDRecoGNN.scripts.dcm_development import recipe, load_selected, inventory_selected, selected_shards


@pytest.mark.parametrize('name,field', [('layernorm', 'architecture'),
                                      ('pool', 'architecture'), ('low_lr', 'lr')])
def test_each_recipe_changes_only_one_factor(name, field):
    base = recipe('reference', 42);candidate = recipe(name, 42)
    base.pop('recipe');candidate.pop('recipe')
    assert [key for key in base if base[key] != candidate[key]] == [field]
    assert candidate['test_evaluation'] is False


def test_selected_loader_rejects_content_changes_and_omits_other_indices(tmp_path):
    (tmp_path/'processed').mkdir()
    p=tmp_path/'processed/shard_0.pt';torch.save(['train', 'old-test', 'validation'], p)
    expected={'0': hashlib.sha256(p.read_bytes()).hexdigest()}
    assert load_selected(tmp_path, [0, 2], [3], expected) == {0: 'train', 2: 'validation'}
    p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ValueError, match='Cache content changed'):
        load_selected(tmp_path, [0, 2], [3], expected)


def test_short_intermediate_shards_preserve_flat_preload_indices(tmp_path):
    import json
    (tmp_path/'processed').mkdir()
    shards = [['train0', 'old-test1'], ['old-test2'], ['validation3', 'old-test4', 'train5']]
    meta = {'num_shards': 3, 'num_graphs': 6, 'shard_size': 3}
    (tmp_path/'processed/meta.json').write_text(json.dumps(meta))
    for sid, shard in enumerate(shards):torch.save(shard, tmp_path/f'processed/shard_{sid}.pt')
    indices = [0, 3, 5]
    graphs, frozen = inventory_selected(tmp_path, indices, meta)
    flat = sum(shards, [])  # HGNDGraphDataset.preload's original index semantics.
    expected = {i: flat[i] for i in indices}
    assert graphs == expected == {0: 'train0', 3: 'validation3', 5: 'train5'}
    assert frozen['shard_lengths'] == [2, 1, 3]
    assert selected_shards(indices, frozen['shard_lengths']) == [0, 2]
    # An unneeded shard is not read by a training task; its length still fixes offsets.
    (tmp_path/'processed/shard_1.pt').unlink()
    assert load_selected(tmp_path, indices, frozen['shard_lengths'], frozen['shards']) == expected
    with pytest.raises(ValueError, match='outside'):
        load_selected(tmp_path, [6], frozen['shard_lengths'], frozen['shards'])
    meta['num_graphs'] = 7
    torch.save(shards[1], tmp_path/'processed/shard_1.pt')
    with pytest.raises(ValueError, match='population disagrees'):
        inventory_selected(tmp_path, indices, meta)


def test_duplicate_active_task_cannot_enter_the_run(tmp_path):
    import fcntl
    from HGNDRecoGNN.scripts.dcm_development import run
    out=tmp_path/'reference_42';out.mkdir()
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run(SimpleNamespace(phase='train', campaign=tmp_path), 'reference', 42, None, None, None)


def test_layernorm_reaches_real_registry_and_preserves_parameter_count(monkeypatch):
    from torch_geometric.data import HeteroData
    from HGNDRecoGNN import models
    from HGNDRecoGNN.models import default_net
    class SelfPair(torch.nn.Module):
        def __init__(self, nn, **kwargs):super().__init__();self.nn=nn
        def forward(self, x, batch):return self.nn(torch.cat([x, x], dim=-1))
    monkeypatch.setattr(default_net, 'DynamicEdgeConv', SelfPair)
    g=HeteroData();g['hits'].x=torch.randn(3, 10)
    g['hits','hits'].edge_index=torch.tensor([[0,1,2],[0,1,2]])
    g['clusters','clusters'].edge_index=torch.tensor([[0,1,2],[0,1,2]])
    g.cluster=torch.arange(3).view(-1,1)
    counts=[]
    for name in ('batch', 'layer'):
        model,_=models.get('net_default', [g], hidden_channels=32, num_layers=8,
                           hit_normalization=name)
        assert model.hit_normalization==name
        assert all(isinstance(n, torch.nn.LayerNorm) for n in model.batch_norms) == (name=='layer')
        counts.append(sum(p.numel() for p in model.parameters()))
    assert counts[0]==counts[1]


def test_energy_flatten_cli_fits_only_the_training_partition(monkeypatch):
    from HGNDRecoGNN.scripts import train as cli
    from HGNDRecoGNN import models, device
    from HGNDRecoGNN.data import graph_dataset
    from HGNDRecoGNN.training import train as training
    from HGNDRecoGNN.training.losses import LossWeights
    class Dataset:
        def __init__(self, **kwargs):pass
        def preload(self, **kwargs):pass
        def __len__(self):return 10
    weights=LossWeights();train_subset=object()
    model=torch.nn.Linear(1,1)
    monkeypatch.setattr(graph_dataset, 'HGNDGraphDataset', Dataset)
    monkeypatch.setattr(models, 'get', lambda *a, **k:(model, SimpleNamespace(
        default_cpu_pinned=(), description='test', default_loss_weights=weights)))
    monkeypatch.setattr(device, 'to_device', lambda *a:None)
    monkeypatch.setattr(training, 'build_loaders', lambda *a:(SimpleNamespace(dataset=train_subset), None))
    class ReachedTrainingFit(Exception):pass
    def check(weights_arg, dataset, bins, sample):
        assert weights_arg is weights and dataset is train_subset
        raise ReachedTrainingFit
    monkeypatch.setattr(cli, '_with_energy_flattening', check)
    monkeypatch.setattr(sys,'argv',['train','--root','/unused','--device','cpu','--energy-flatten'])
    with pytest.raises(ReachedTrainingFit):cli.main()
