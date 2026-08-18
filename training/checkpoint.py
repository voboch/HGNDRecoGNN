"""Checkpoint save/load with explicit metadata.

The v1 notebook used `torch.save(model, path)` (full pickle), which breaks
as soon as the model class moves modules or renames files. This module
saves `{state_dict, arch_name, arch_kwargs, schema_version, epoch, metrics}`
so checkpoints survive refactors and carry provenance.

Use `scripts/convert_checkpoint.py` to migrate v1 pickles.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ..data.graph_dataset import SCHEMA_V

CHECKPOINT_FORMAT_V = 1


@dataclass
class Checkpoint:
    state_dict: dict[str, torch.Tensor]
    arch_name: str
    arch_kwargs: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_V
    epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    format_version: int = CHECKPOINT_FORMAT_V
    timestamp: float = field(default_factory=time.time)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    arch_name: str,
    arch_kwargs: dict[str, Any] | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save `model.state_dict()` plus provenance to `path`.

    Submodules on CPU (via DeviceMap pinning) are handled correctly by
    `state_dict()` — tensors are stored with their current device attribute
    but restored to whatever device the calling code moves the model to.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    ckpt = Checkpoint(
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        arch_name=arch_name,
        arch_kwargs=dict(arch_kwargs or {}),
        epoch=epoch,
        metrics=dict(metrics or {}),
    )
    torch.save(ckpt.__dict__, path)


def load_checkpoint(path: str) -> Checkpoint:
    """Load a checkpoint. Raises if format_version is unknown."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or 'state_dict' not in payload:
        raise ValueError(
            f'{path} does not look like a v{CHECKPOINT_FORMAT_V} checkpoint. '
            f'If it is a v0 full-pickle model, run scripts/convert_checkpoint.py.'
        )
    fmt = payload.get('format_version', 0)
    if fmt != CHECKPOINT_FORMAT_V:
        raise ValueError(
            f'{path} has format_version={fmt}, expected {CHECKPOINT_FORMAT_V}.'
        )
    return Checkpoint(**payload)
