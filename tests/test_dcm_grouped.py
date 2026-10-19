"""Leakage guards for the source-run protocol; GPU behavior uses Slurm smoke."""
import numpy as np
import pytest
from HGNDRecoGNN.scripts.dcm_grouped import group_partition, split_indices


def test_both_detector_arms_and_all_files_share_source_run_partition():
    groups = ['r1', 'r1', 'r2', 'r3', 'r2', 'r3']
    partition = group_partition(groups)
    indices = split_indices(groups, partition)
    owner = {i: k for k, ix in indices.items() for i in ix}
    assert owner[0] == owner[1]
    assert owner[2] == owner[4]
    assert owner[3] == owner[5]
    assert len(set(owner.values())) == 3
    assert sorted(np.concatenate(list(indices.values())).tolist()) == list(range(6))


def test_split_is_independent_of_graph_order_and_training_seed():
    assert group_partition(['c', 'a', 'b']) == group_partition(['b', 'c', 'a', 'a'])


def test_refuse_ambiguous_or_partial_source_coverage():
    with pytest.raises(ValueError):
        group_partition(['one', 'two'])
    with pytest.raises(ValueError):
        split_indices(['a', 'b', 'c'], {'train': ['a'], 'validation': ['b'], 'test': ['b']})
    with pytest.raises(ValueError):
        split_indices(['a', 'b'], {'train': ['a'], 'validation': ['b'], 'test': ['c']})


def test_affine_remap_cannot_mutate_the_shared_frozen_graph():
    import torch
    from torch_geometric.data import HeteroData
    from HGNDRecoGNN.scripts.dcm_grouped import make_subset
    graph = HeteroData()
    graph['hits'].x = torch.tensor([[1., 2., 1.]])
    graph.istop = torch.tensor([1])
    before = graph['hits'].x.clone()
    a = np.array([[1., 1.], [2., 3.]])
    b = np.array([[0., 0.], [4., 5.]])
    subset = make_subset([graph], [0], a, b)
    first, second = subset[0], subset[0]
    assert torch.equal(graph['hits'].x, before)
    assert torch.equal(first['hits'].x, torch.tensor([[6., 11., 1.]]))
    assert torch.equal(first['hits'].x, second['hits'].x)
