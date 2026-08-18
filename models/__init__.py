"""Model registry for the HGND GNN pipeline.

Each model is a factory callable returning `(model, ModelSpec)` where
`ModelSpec` tells the training loop:
  - `forward_fn(model, graph)` — how to run one forward pass on a batched HeteroData
  - `loss_fn(preds, graph, weights)` — how to compute the composite loss
  - `default_loss_weights` — reasonable defaults for that model's loss terms
  - `default_cpu_pinned` — submodule names to pin to CPU on MPS (see device.py)

Registration is decorator-based:

    from HGNDRecoGNN.models import register, ModelSpec

    @register('my_model')
    def build_my_model(dataset, **kwargs):
        model = MyModel(...)
        spec = ModelSpec(forward_fn=..., loss_fn=..., default_loss_weights=...)
        return model, spec

Lookup:

    from HGNDRecoGNN.models import get, available
    model, spec = get('net_default', dataset, hidden_channels=512, num_layers=8)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch


# ── Model spec ──────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Per-model metadata the training loop needs to stay model-agnostic."""
    forward_fn: Callable[[torch.nn.Module, Any], Any]
    loss_fn: Callable[..., dict[str, torch.Tensor]]
    default_loss_weights: Any = None
    default_cpu_pinned: tuple[str, ...] = ()
    description: str = ''
    tags: tuple[str, ...] = ()


# ── Registry ────────────────────────────────────────────────────────────

# name → factory(dataset, **kwargs) -> (model, ModelSpec)
REGISTRY: dict[str, Callable[..., tuple[torch.nn.Module, ModelSpec]]] = {}


def register(name: str) -> Callable[[Callable], Callable]:
    """Decorator: register a model factory under `name`."""
    def _deco(fn: Callable) -> Callable:
        if name in REGISTRY:
            raise ValueError(f'model {name!r} already registered')
        REGISTRY[name] = fn
        return fn
    return _deco


def get(name: str, dataset, **kwargs) -> tuple[torch.nn.Module, ModelSpec]:
    """Construct a registered model. `kwargs` are forwarded to the factory."""
    if name not in REGISTRY:
        raise KeyError(f'unknown model {name!r}. available: {available()}')
    return REGISTRY[name](dataset, **kwargs)


def available() -> list[str]:
    return sorted(REGISTRY.keys())


# ── Auto-import model modules so their @register decorators run ─────────

# Import order matters only for readability; there are no cross-model deps.
from .default_net import Net, build_default_net       # noqa: E402,F401
from . import default_net_registration                # noqa: E402,F401
from . import hetero_sage                              # noqa: E402,F401
from . import hetero_gat                               # noqa: E402,F401
from . import hetero_hgt                               # noqa: E402,F401
from . import thesis_baselines                        # noqa: E402,F401
from . import spectral                                 # noqa: E402,F401

__all__ = [
    'ModelSpec', 'REGISTRY', 'register', 'get', 'available',
    'Net', 'build_default_net',
]
