"""Register `hetero_gat` — HeteroGNN with GATConv (heads=4) per relation."""

from __future__ import annotations

from . import ModelSpec, register
from .hetero_gnn import build_hetero_gnn
from ..training.losses import HeteroClusterWeights, hetero_cluster_loss
from ..training.train import forward_hetero


@register('hetero_gat')
def _factory(dataset, hidden: int = 128, num_layers: int = 2, heads: int = 4,
             dropout: float = 0.2, cluster_type: str = 'clusters'):
    model = build_hetero_gnn(dataset, kind='gat', hidden=hidden,
                             num_layers=num_layers, heads=heads,
                             dropout=dropout, cluster_type=cluster_type)
    spec = ModelSpec(
        forward_fn=forward_hetero,
        loss_fn=hetero_cluster_loss,
        default_loss_weights=HeteroClusterWeights(),
        default_cpu_pinned=(),
        description='vkr26 HeteroGNN with GATConv (heads=4) per relation',
        tags=('hetero', 'vkr26', 'attention'),
    )
    return model, spec
