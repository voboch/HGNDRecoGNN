"""Register `net_default` in the model registry.

Kept in a separate module so `default_net.py` stays import-free of the
registry (avoids a circular import: registry → default_net → registry).
"""

from __future__ import annotations

from . import ModelSpec, register
from .default_net import build_default_net
from ..training.losses import LossWeights, default_net_loss
from ..training.train import _forward_default_net


@register('net_default')
def _factory(dataset, hidden_channels: int = 512, num_layers: int = 8,
             **arch_kwargs):
    """`**arch_kwargs` is forwarded to `build_default_net` verbatim.

    Without it this factory silently caps which architecture options can ever
    reach the model: `registry.get()` passes every kwarg straight through, so
    an option the model accepts but this signature does not raises TypeError
    at construction. That is how the opt-in cluster-pooling, cluster-size and
    residual flags were unreachable when first added -- they mapped correctly
    into arch_kwargs and matched Net.__init__, but died on this hop.
    """
    model = build_default_net(dataset,
                              hidden_channels=hidden_channels,
                              num_layers=num_layers,
                              **arch_kwargs)
    spec = ModelSpec(
        forward_fn=_forward_default_net,
        loss_fn=default_net_loss,
        default_loss_weights=LossWeights(),
        default_cpu_pinned=(
            'cluster_conv_cpu', 'clclass_out_cpu',
            'clenergy_out_cpu', 'cl_edge_out_cpu',
        ),
        description='v1 hit-branch (EdgeConv+SAGE+GraphConv) + '
                    'cluster-branch DynamicEdgeConv',
        tags=('hetero', 'baseline'),
    )
    return model, spec
