"""Register `hetero_sage` — HeteroGNN with SAGEConv per relation."""

from __future__ import annotations

from . import ModelSpec, register
from .hetero_gnn import build_hetero_gnn
from ..training.losses import HeteroClusterWeights, hetero_cluster_loss
from ..training.train import forward_hetero


@register('hetero_sage')
def _factory(dataset, hidden: int = 128, num_layers: int = 2, dropout: float = 0.2,
             cluster_type: str = 'clusters'):
    model = build_hetero_gnn(dataset, kind='sage', hidden=hidden,
                             num_layers=num_layers, dropout=dropout,
                             cluster_type=cluster_type)
    spec = ModelSpec(
        forward_fn=forward_hetero,
        loss_fn=hetero_cluster_loss,
        default_loss_weights=HeteroClusterWeights(),
        default_cpu_pinned=(),   # SAGEConv is fine on MPS + CUDA
        description='vkr26 HeteroGNN with SAGEConv per relation',
        tags=('hetero', 'vkr26'),
    )
    return model, spec
