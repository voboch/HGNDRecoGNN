"""Thesis-baseline models (no spectral gate).

Ports the "baseline" family from `GNN_for_neutron_reconstruction`:
  - ``BaselineClusterGNN_GraphSAGE_UFC`` → registered as ``thesis_sage``
  - ``BaselineClusterGNN_GATv2_UFC``     → registered as ``thesis_gat``
  - ``BaselineClusterGNN_DynamicEdgeConv_UFC`` → registered as ``thesis_dynedge``

Simplifications from the thesis source (documented for future faithfulness work):
  1. Aggregates per-cluster via `scatter_mean` instead of AttentionalAggregation.
  2. Single classification head (no regression) — energy regression is a
     Phase 2b extension since our composite loss already handles it via
     `hetero_cluster_loss` if the model returns `(cls_logits, e_pred)`.
  3. Operates on HGNDRecoGNN hit graphs (from `_build_single_graph`) —
     the thesis code assumed 6-dim features + per-cluster subgraphs. We use
     the 10-dim `hits.x` and group by `graph.cluster` at pool time.

Emits `(cls_logits, e_pred)` per cluster so it plugs into
`hetero_cluster_loss`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, GATv2Conv, SAGEConv
from torch_geometric.utils import scatter

from . import ModelSpec, register
from ..training.losses import HeteroClusterWeights, hetero_cluster_loss
from .adapters import batched_cluster_index


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class _ClusterHead(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.cls = _mlp(hidden, hidden // 2, 1, dropout)
        self.reg = _mlp(hidden, hidden // 2, 1, dropout)

    def forward(self, h_cluster):
        return self.cls(h_cluster).squeeze(-1), self.reg(h_cluster).squeeze(-1)


class _BaseClusterGNN(nn.Module):
    """Shared skeleton for thesis baselines. Subclass sets `self.blocks`."""

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = _ClusterHead(hidden, dropout)

    def _run_backbone(self, x, edge_index):
        raise NotImplementedError

    def forward(self, graph):
        x = graph['hits'].x
        edge_index = graph['hits', 'hits'].edge_index
        cluster_ids, n_clusters = batched_cluster_index(graph)
        cluster_ids = cluster_ids.to(x.device)

        h = self.input_proj(x)
        h = self._run_backbone(h, edge_index)

        # Per-cluster pooling (mean) → per-cluster prediction.
        h_cl = scatter(h, cluster_ids, dim=0, dim_size=n_clusters, reduce='mean')
        return self.head(h_cl)


class BaselineClusterGNN_GraphSAGE(_BaseClusterGNN):
    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__(in_dim, hidden, dropout)
        self.blocks = nn.ModuleList([
            SAGEConv(hidden, hidden) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = dropout

    def _run_backbone(self, x, edge_index):
        for conv, norm in zip(self.blocks, self.norms):
            x = F.relu(norm(x + conv(x, edge_index)))
            x = F.dropout(x, self.dropout, self.training)
        return x


class BaselineClusterGNN_GATv2(_BaseClusterGNN):
    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 4,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__(in_dim, hidden, dropout)
        self.blocks = nn.ModuleList([
            GATv2Conv(hidden, hidden, heads=heads, concat=False,
                      add_self_loops=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = dropout

    def _run_backbone(self, x, edge_index):
        for conv, norm in zip(self.blocks, self.norms):
            x = F.relu(norm(x + conv(x, edge_index)))
            x = F.dropout(x, self.dropout, self.training)
        return x


class BaselineClusterGNN_DynamicEdgeConv(_BaseClusterGNN):
    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 4,
                 k: int = 8, dropout: float = 0.1):
        super().__init__(in_dim, hidden, dropout)
        self.blocks = nn.ModuleList([
            DynamicEdgeConv(
                nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                              nn.LayerNorm(hidden), nn.Dropout(dropout),
                              nn.Linear(hidden, hidden), nn.ReLU()),
                k=k, aggr='max')
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = dropout

    def _run_backbone(self, x, edge_index):
        # DynamicEdgeConv is edge-agnostic (kNN in feature space) but we still
        # thread a batch to keep neighbours within the same event.
        for conv, norm in zip(self.blocks, self.norms):
            # DEC needs a batch tensor to keep neighbours within-graph.
            # We derive it from edge_index: all nodes referenced share a batch.
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = F.relu(norm(x + conv(x, batch)))
            x = F.dropout(x, self.dropout, self.training)
        return x


def _forward_cluster_model(model, graph):
    """forward_fn for thesis-family models — they take the whole graph."""
    return model(graph)


def _register_baseline(name: str, cls, tags: tuple[str, ...],
                       description: str, cpu_pinned: tuple[str, ...] = ()):
    @register(name)
    def _f(dataset, hidden: int = 128, num_layers: int = 4, dropout: float = 0.1,
           **kwargs):
        in_dim = int(dataset[0]['hits'].x.size(-1))
        model = cls(in_dim, hidden=hidden, num_layers=num_layers,
                    dropout=dropout, **kwargs)
        # lazy-materialise nothing here — layers are non-lazy.
        return model, ModelSpec(
            forward_fn=_forward_cluster_model,
            loss_fn=hetero_cluster_loss,
            default_loss_weights=HeteroClusterWeights(),
            default_cpu_pinned=cpu_pinned,
            description=description,
            tags=tags,
        )
    return _f


_register_baseline(
    'thesis_sage',
    BaselineClusterGNN_GraphSAGE,
    tags=('thesis', 'cluster', 'baseline'),
    description='Thesis baseline: per-cluster GraphSAGE + scatter-mean pool',
)
_register_baseline(
    'thesis_gat',
    BaselineClusterGNN_GATv2,
    tags=('thesis', 'cluster', 'baseline', 'attention'),
    description='Thesis baseline: per-cluster GATv2 + scatter-mean pool',
)
_register_baseline(
    'thesis_dynedge',
    BaselineClusterGNN_DynamicEdgeConv,
    tags=('thesis', 'cluster', 'baseline', 'dynedge'),
    description='Thesis baseline: per-cluster DynamicEdgeConv + scatter-mean pool',
    cpu_pinned=('blocks',),   # DEC has no MPS kernel
)
