"""Spectral-gated variants of the thesis models.

Ports `SpectralMultiplicativeGate` verbatim from
`GNN_for_neutron_reconstruction/Challenger_ufc_DynamicEdgeConv.ipynb`
and combines it with the baseline architectures in `thesis_baselines.py`.

The gate takes per-hit Laplacian positional encoding (LapPE) computed on
each cluster's hit-hit subgraph and produces a bounded multiplicative
correction to the projected features:

    z = |pe|;  g = tanh(MLP_gate(z));  h_out = h_in * (1 + g)

LapPE is computed on-the-fly by `adapters.compute_lap_pe_per_cluster`.
For batches with hundreds of clusters this dominates runtime — Phase 2b
work should either cache LapPE at preprocess time (SCHEMA_V bump) or
port to a batched scipy-sparse variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from . import ModelSpec, register
from ..training.losses import HeteroClusterWeights, hetero_cluster_loss
from .adapters import batched_cluster_index, compute_lap_pe_per_cluster
from .thesis_baselines import (
    BaselineClusterGNN_DynamicEdgeConv,
    BaselineClusterGNN_GATv2,
    BaselineClusterGNN_GraphSAGE,
)


class SpectralMultiplicativeGate(nn.Module):
    """Sign-invariant LapPE gate. Verbatim from the thesis repo."""

    def __init__(self, pe_dim: int, hidden_dim: int, pe_hidden: int = 64,
                 dropout: float = 0.10, gate_init_std: float = 1e-3):
        super().__init__()
        assert pe_dim > 0
        self.pe_dim = int(pe_dim)
        self.gate = nn.Sequential(
            nn.Linear(pe_dim, pe_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(pe_hidden, hidden_dim),
        )
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=gate_init_std)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, h_base: torch.Tensor, pe: torch.Tensor):
        z = pe.abs()
        g = torch.tanh(self.gate(z))
        return h_base * (1.0 + g), g


class _SpectralGatedWrapper(nn.Module):
    """Wraps a baseline cluster-level GNN and injects spectral gating.

    The wrapped `backbone` is one of the `BaselineClusterGNN_*` classes.
    We reuse its `input_proj`, blocks, norms, and head, and insert the
    gate right after `input_proj`.
    """

    def __init__(self, backbone, pe_dim: int, pe_hidden: int = 64,
                 dropout: float = 0.10, k_pe: int = 4,
                 gate_init_std: float = 1e-3):
        super().__init__()
        self.backbone = backbone
        self.k_pe = k_pe
        hidden = backbone.input_proj[0].out_features   # nn.Linear(in, hidden)
        self.spectral_gate = SpectralMultiplicativeGate(
            pe_dim=pe_dim, hidden_dim=hidden, pe_hidden=pe_hidden,
            dropout=dropout, gate_init_std=gate_init_std,
        )

    def forward(self, graph):
        x = graph['hits'].x
        edge_index = graph['hits', 'hits'].edge_index
        cluster_ids, n_clusters = batched_cluster_index(graph)
        cluster_ids = cluster_ids.to(x.device)

        pe = compute_lap_pe_per_cluster(
            edge_index=edge_index,
            cluster_ids=cluster_ids,
            n_hits=x.size(0),
            k_pe=self.k_pe,
        ).to(x.device)

        h = self.backbone.input_proj(x)
        h, _gate = self.spectral_gate(h, pe)
        h = self.backbone._run_backbone(h, edge_index)

        h_cl = scatter(h, cluster_ids, dim=0, dim_size=n_clusters, reduce='mean')
        return self.backbone.head(h_cl)


def _register_spectral(name: str, base_cls, tags: tuple[str, ...],
                       description: str, cpu_pinned: tuple[str, ...] = ()):
    @register(name)
    def _f(dataset, hidden: int = 128, num_layers: int = 4, dropout: float = 0.1,
           k_pe: int = 4, pe_hidden: int = 64, **kwargs):
        in_dim = int(dataset[0]['hits'].x.size(-1))
        backbone = base_cls(in_dim, hidden=hidden, num_layers=num_layers,
                            dropout=dropout, **kwargs)
        model = _SpectralGatedWrapper(
            backbone, pe_dim=k_pe, pe_hidden=pe_hidden,
            dropout=dropout, k_pe=k_pe,
        )
        return model, ModelSpec(
            forward_fn=(lambda m, g: m(g)),
            loss_fn=hetero_cluster_loss,
            default_loss_weights=HeteroClusterWeights(),
            default_cpu_pinned=cpu_pinned,
            description=description,
            tags=tags,
        )
    return _f


_register_spectral(
    'spectral_sage',
    BaselineClusterGNN_GraphSAGE,
    tags=('thesis', 'cluster', 'spectral'),
    description='Spectral-Gated GraphSAGE (thesis Challenger)',
)
_register_spectral(
    'spectral_gat',
    BaselineClusterGNN_GATv2,
    tags=('thesis', 'cluster', 'spectral', 'attention'),
    description='Spectral-Gated GATv2 (thesis Challenger)',
)
_register_spectral(
    'spectral_dynedge',
    BaselineClusterGNN_DynamicEdgeConv,
    tags=('thesis', 'cluster', 'spectral', 'dynedge'),
    description='Spectral-Gated DynamicEdgeConv (thesis top-performing model)',
    cpu_pinned=('backbone.blocks',),
)
