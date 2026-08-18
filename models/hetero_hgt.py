"""Register `hetero_hgt` — Heterogeneous Graph Transformer.

Source: vkr26-main/e1/experiment_1_gpu_acc2_hgt.ipynb cell 16.
"""

from __future__ import annotations

from . import ModelSpec, register
from .hetero_gnn import build_hetero_gnn
from ..training.losses import HeteroClusterWeights, hetero_cluster_loss
from ..training.train import forward_hetero


@register('hetero_hgt')
def _factory(dataset, hidden: int = 128, num_layers: int = 2, heads: int = 4,
             dropout: float = 0.2, cluster_type: str = 'clusters'):
    model = build_hetero_gnn(dataset, kind='hgt', hidden=hidden,
                             num_layers=num_layers, heads=heads,
                             dropout=dropout, cluster_type=cluster_type)
    spec = ModelSpec(
        forward_fn=forward_hetero,
        loss_fn=hetero_cluster_loss,
        default_loss_weights=HeteroClusterWeights(),
        default_cpu_pinned=(),   # HGTConv works on CUDA & CPU; on MPS see note
        description='vkr26 HGT — Heterogeneous Graph Transformer over '
                    'hits/clusters/events/sides node types',
        tags=('hetero', 'vkr26', 'transformer'),
    )
    return model, spec
