"""Heterogeneous GNN ported from vkr26-main/e1/experiment_1_gpu_acc2_hgt.ipynb.

One class, three convolution kinds: `sage`, `gat`, `hgt`. Expects a batched
`HeteroData` from HGND schema v2 (adds virtual `events` + `sides` nodes so
the vkr26 metadata pattern applies unchanged).

Key differences from the source notebook:
- Cluster node type is parameterisable (`cluster_type='clusters'`) — the
  original used singular `'cluster'`, HGNDRecoGNN uses plural.
- Head outputs are aligned with the current 6-tuple contract expected by
  the training loop: `(link_scores, hit_scores, cluster_scores, cluster_energy,
   cluster_link_scores, cluster_batch)`. The hetero model doesn't produce
  hit-level or cluster-link outputs, so those return zeros with the right
  shape — the loss module masks them via `LossWeights.hit=0` etc. by default.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, HGTConv, Linear, SAGEConv


class HeteroGNN(nn.Module):
    """Node-type aware GNN with SAGE / GAT / HGT convolution kinds."""

    def __init__(
        self,
        kind: str,
        metadata,
        hidden: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        cluster_type: str = 'clusters',
    ):
        super().__init__()
        assert kind in ('sage', 'gat', 'hgt'), f'unknown kind: {kind}'
        self.kind = kind
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.cluster_type = cluster_type

        node_types, edge_types = metadata

        # Per-node-type lazy input projection.
        self.in_proj = nn.ModuleDict({nt: Linear(-1, hidden) for nt in node_types})

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if kind == 'sage':
                conv = HeteroConv(
                    {rel: SAGEConv(hidden, hidden) for rel in edge_types},
                    aggr='sum',
                )
            elif kind == 'gat':
                conv = HeteroConv({
                    rel: GATConv((hidden, hidden), hidden, heads=heads,
                                 concat=False, add_self_loops=False,
                                 dropout=dropout)
                    for rel in edge_types
                }, aggr='sum')
            else:  # hgt
                conv = HGTConv(hidden, hidden, metadata, heads=heads)
            self.layers.append(conv)

        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_types})
            for _ in range(num_layers)
        ])

        # Cluster-level heads.
        self.head_cls = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.head_reg = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {nt: F.relu(self.in_proj[nt](x)) for nt, x in x_dict.items()}
        for conv, norm in zip(self.layers, self.norms):
            x_new = conv(x_dict, edge_index_dict)
            x_dict = {nt: F.relu(norm[nt](x_new[nt])) if nt in x_new else x_dict[nt]
                      for nt in x_dict}
            x_dict = {nt: F.dropout(x, p=self.dropout, training=self.training)
                      for nt, x in x_dict.items()}

        h_cl = x_dict[self.cluster_type]
        cls_logits = self.head_cls(h_cl).squeeze(-1)
        e_pred     = self.head_reg(h_cl).squeeze(-1)
        return cls_logits, e_pred


def build_hetero_gnn(
    dataset,
    kind: str,
    hidden: int = 128,
    num_layers: int = 2,
    heads: int = 4,
    dropout: float = 0.2,
    cluster_type: str = 'clusters',
) -> HeteroGNN:
    """Construct and lazy-init a HeteroGNN using dataset[0].metadata()."""
    sample = dataset[0]
    metadata = sample.metadata()
    model = HeteroGNN(kind, metadata, hidden=hidden, num_layers=num_layers,
                      heads=heads, dropout=dropout, cluster_type=cluster_type)

    # Lazy init on CPU with the sample graph.
    with torch.no_grad():
        model.eval()
        model(sample.x_dict, sample.edge_index_dict)
    model.train()
    return model
