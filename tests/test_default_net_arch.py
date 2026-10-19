"""The opt-in architecture flags must not disturb the v1 model.

Every published number comes from the default configuration, so the defaults
have to stay bit-compatible with the checkpoints on disk. These tests pin that
down, then check each flag does what it claims.


`models/default_net.py` is imported by file rather than through the package,
matching tests/test_energy_flatten.py: the package `__init__` pulls in
`data.graph_dataset` and the loss registration, neither of which this test
needs.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "_default_net", os.path.join(ROOT, "models", "default_net.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class _StubDynamicEdgeConv(torch.nn.Module):
    """Stand-in for DynamicEdgeConv, which needs pyg-lib for its knn kernel.

    pyg-lib is installed on the cluster but not here, and without it the real
    layer cannot even be constructed. The stub keeps the `nn` MLP and applies
    it to a self-pair, so the input/output widths these tests assert on are
    exactly the real layer's. It does no neighbour search: the knn itself is
    not what is under test.
    """

    def __init__(self, nn, k, aggr='max', **kw):
        super().__init__()
        self.nn, self.k, self.aggr = nn, k, aggr

    def forward(self, x, batch=None):
        return self.nn(torch.cat([x, x], dim=-1))


_mod.DynamicEdgeConv = _StubDynamicEdgeConv
Net = _mod.Net

CACHE = "notebooks/cache/ndet_dataset_smash_defaultSpot/processed"
CKPT = "checkpoints_hpc/defaultSpot_v2_seed42_20ep_after4278192/model.pt"


def _one_graph():
    files = sorted(glob.glob(os.path.join(CACHE, "shard_*.pt")))
    if not files:
        pytest.skip("defaultSpot cache not present")
    for g in torch.load(files[0], weights_only=False):
        if g['hits'].x.size(0) >= 4 and g.cllabel.numel() >= 2:
            return g
    pytest.skip("no suitable graph in the first shard")


def _run(model, g):
    cl = g.cluster.squeeze(-1) + 1
    b = torch.zeros(g['hits'].x.size(0), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        return model(g['hits'].x, g['hits', 'hits'].edge_index,
                     g['clusters', 'clusters'].edge_index, cl, b)


def _build(g, **kw):
    m = Net(hidden_channels=32, num_layers=8,
            num_features=g['hits'].x.size(1), **kw)
    _run(m, g)          # materialise the lazy SAGEConv shapes
    return m


def test_defaults_keep_the_v1_cluster_width():
    """With both flags off the cluster branch sees exactly hidden_channels."""
    g = _one_graph()
    m = _build(g)
    assert m.cluster_in_channels == 32
    assert m.cluster_conv_cpu.nn[0].in_features == 64, \
        "DynamicEdgeConv concatenates a pair, so 2 x hidden_channels"


def test_augment_is_identity_with_defaults():
    g = _one_graph()
    m = _build(g)
    cl_mean = torch.randn(5, 32)
    out = m._augment_cluster_features(cl_mean, torch.zeros(3, dtype=torch.long),
                                      torch.randn(3, 32))
    assert out is cl_mean, "defaults must not copy or alter the pooled tensor"


def test_real_checkpoint_still_loads_into_the_patched_model():
    """The strongest regression test available: a real trained state_dict."""
    if not os.path.exists(CKPT):
        pytest.skip("trained checkpoint not present")
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = blob['state_dict']
    kwargs = blob.get('arch_kwargs', {}) or {}
    g = _one_graph()
    m = Net(num_features=g['hits'].x.size(1),
            hidden_channels=kwargs.get('hidden_channels', 512),
            num_layers=kwargs.get('num_layers', 8))
    _run(m, g)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, f"patched model rejects checkpoint keys: {unexpected}"
    assert not missing, f"patched model expects keys the checkpoint lacks: {missing}"


def test_pooling_widens_the_cluster_vector_threefold():
    g = _one_graph()
    m = _build(g, cluster_pooling='mean_max_sum')
    assert m.cluster_in_channels == 96
    assert m.cluster_conv_cpu.nn[0].in_features == 192


def test_size_feature_adds_exactly_one_channel():
    g = _one_graph()
    m = _build(g, cluster_size_feature=True)
    assert m.cluster_in_channels == 33
    m2 = _build(g, cluster_pooling='mean_max_sum', cluster_size_feature=True)
    assert m2.cluster_in_channels == 97


def test_size_feature_is_log1p_of_the_hit_count():
    """The channel must carry the cluster's hit count, not something else."""
    g = _one_graph()
    m = _build(g, cluster_size_feature=True)
    # three clusters with 1, 3 and 2 hits
    cl_idx = torch.tensor([0, 1, 1, 1, 2, 2])
    cl_feat = torch.zeros(6, 32)
    out = m._augment_cluster_features(torch.zeros(3, 32), cl_idx, cl_feat)
    got = out[:, -1]
    assert torch.allclose(got, torch.log1p(torch.tensor([1.0, 3.0, 2.0]))), got


def test_max_and_sum_pooling_are_the_real_reductions():
    g = _one_graph()
    m = _build(g, cluster_pooling='mean_max_sum')
    cl_idx = torch.tensor([0, 0, 1])
    cl_feat = torch.tensor([[1.0, -2.0], [3.0, 5.0], [7.0, 7.0]])
    # pad to the model's width so the concat is well formed
    cl_feat = torch.cat([cl_feat, torch.zeros(3, 30)], dim=1)
    out = m._augment_cluster_features(torch.zeros(2, 32), cl_idx, cl_feat)
    mx, sm = out[:, 32:64], out[:, 64:96]
    assert mx[0, 0] == 3.0 and mx[0, 1] == 5.0, "max over cluster 0"
    assert sm[0, 0] == 4.0 and sm[0, 1] == 3.0, "sum over cluster 0"


def test_residual_changes_the_output_but_adds_no_parameters():
    g = _one_graph()
    plain = _build(g)
    res = _build(g, hit_residual=True)
    assert sum(p.numel() for p in plain.parameters()) == \
           sum(p.numel() for p in res.parameters()), \
           "a residual connection is parameter-free"
    res.load_state_dict(plain.state_dict())
    a = _run(plain, g)
    b = _run(res, g)
    assert not torch.allclose(a[1], b[1]), \
        "with identical weights the residual path must still alter hit scores"


def test_bad_pooling_name_is_refused():
    g = _one_graph()
    with pytest.raises(ValueError, match="cluster_pooling"):
        Net(hidden_channels=32, num_layers=8,
            num_features=g['hits'].x.size(1), cluster_pooling='sum')


# ── near-fixed-capacity pooling ──────────────────────────────────────────
# The 2026-10-17 `pool` recipe could not separate a pooling gain from a
# capacity gain: 'mean_max_sum' triples the DynamicEdgeConv input and so adds
# 4,194,304 parameters at H=512, which is 34 % more than the whole v1 model
# (reference 12,479,173 -> pool 16,673,477). 'mean_max_sum_proj' projects the
# concatenation back to hidden_channels first, leaving that layer's shape
# identical to the 'mean' baseline.

def test_proj_pooling_restores_the_baseline_cluster_width():
    g = _one_graph()
    base = _build(g)
    proj = _build(g, cluster_pooling='mean_max_sum_proj')
    assert proj.cluster_in_channels == base.cluster_in_channels == 32
    assert proj.cluster_cat_channels == 96, 'the concat is still 3 x hidden'
    assert proj.cluster_conv_cpu.nn[0].in_features == \
        base.cluster_conv_cpu.nn[0].in_features, \
        'DynamicEdgeConv must see the same width as the mean baseline'


def test_proj_pooling_costs_far_less_than_plain_concatenation():
    g = _one_graph()
    n = lambda m: sum(p.numel() for p in m.parameters())
    base = n(_build(g))
    wide = n(_build(g, cluster_pooling='mean_max_sum'))
    proj = n(_build(g, cluster_pooling='mean_max_sum_proj'))
    assert wide > proj > base, (base, proj, wide)
    # the projection must be the cheaper way to get max+sum information
    assert (proj - base) < 0.5 * (wide - base), \
        f'projection adds {proj - base}, concatenation adds {wide - base}'


def test_proj_pooling_still_sees_max_and_sum():
    """The projection must consume the wide concat, not discard it."""
    g = _one_graph()
    m = _build(g, cluster_pooling='mean_max_sum_proj')
    cl_idx = torch.tensor([0, 0, 1])
    feat_a = torch.zeros(3, 32)
    feat_b = feat_a.clone()
    feat_b[1, 0] = 5.0            # changes max and sum of cluster 0, not its mean
    mean = torch.zeros(2, 32)
    out_a = m._augment_cluster_features(mean, cl_idx, feat_a)
    out_b = m._augment_cluster_features(mean, cl_idx, feat_b)
    assert out_a.shape == (2, 32), out_a.shape
    assert not torch.allclose(out_a, out_b), \
        'the projected vector ignores max/sum information'


def test_proj_pooling_with_size_feature_keeps_the_baseline_width():
    g = _one_graph()
    m = _build(g, cluster_pooling='mean_max_sum_proj', cluster_size_feature=True)
    assert m.cluster_cat_channels == 97
    assert m.cluster_in_channels == 32
