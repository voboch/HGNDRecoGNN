"""Device detection and per-submodule dispatch for HGND models.

Priority: CUDA → MPS → CPU. Some PyG ops (DynamicEdgeConv, knn) are broken
on MPS but fine on CUDA — models declare which submodules need CPU pinning
on MPS via a `DeviceMap` returned from `plan_for(model, device)`, and
`to_device` walks named submodules to dispatch each to the right device.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn


# Fall back to CPU for MPS-unsupported ops instead of raising. Without this,
# hitting a missing MPS kernel (e.g. certain scatter reductions, torch_cluster
# ops) kills the Python process outright — the notebook kernel then just dies
# with no traceback. Setting the env var early is safe on all platforms and
# only has an effect on the MPS backend.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


def detect() -> str:
    """Return the best available accelerator: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return 'mps'
    return 'cpu'


def resolve(device: str | torch.device | None) -> torch.device:
    """Materialise a device spec, defaulting to `detect()` when None/'auto'."""
    if device is None or device == 'auto':
        return torch.device(detect())
    return torch.device(device)


# ── Ops known to be MPS-hostile ──────────────────────────────────────────
# DynamicEdgeConv uses torch_cluster.knn which lacks an MPS kernel; running
# it on MPS raises or silently produces wrong shapes. Anything relying on
# scatter-max style reductions with unusual dtypes tends to break too.
_MPS_HOSTILE_TYPES: tuple[type, ...] = ()  # populated lazily below


def _mps_hostile_types() -> tuple[type, ...]:
    global _MPS_HOSTILE_TYPES
    if _MPS_HOSTILE_TYPES:
        return _MPS_HOSTILE_TYPES
    try:
        from torch_geometric.nn import DynamicEdgeConv
        _MPS_HOSTILE_TYPES = (DynamicEdgeConv,)
    except Exception:
        _MPS_HOSTILE_TYPES = ()
    return _MPS_HOSTILE_TYPES


@dataclass
class DeviceMap:
    """Per-submodule device placement.

    - `main`: device for the bulk of the model (usually the accelerator).
    - `cpu_pinned`: named submodules that must stay on CPU regardless.
      A submodule name matches if it equals or is a prefix (dotted) of
      the module's registered name (e.g. `cluster_conv` matches `cluster_conv.lin`).
    """
    main: torch.device
    cpu_pinned: tuple[str, ...] = field(default_factory=tuple)

    def device_for(self, name: str) -> torch.device:
        if any(name == p or name.startswith(p + '.') for p in self.cpu_pinned):
            return torch.device('cpu')
        return self.main


def plan_for(model: nn.Module, device: str | torch.device | None = None,
             cpu_pinned: Iterable[str] | None = None) -> DeviceMap:
    """Build a `DeviceMap` for `model` on `device`.

    If `cpu_pinned` is None and `device` resolves to MPS, submodules whose
    class is in the MPS-hostile list are auto-pinned to CPU. On CUDA/CPU
    no auto-pinning is applied.
    """
    dev = resolve(device)
    if cpu_pinned is not None:
        return DeviceMap(main=dev, cpu_pinned=tuple(cpu_pinned))

    if dev.type != 'mps':
        return DeviceMap(main=dev, cpu_pinned=())

    hostile = _mps_hostile_types()
    if not hostile:
        return DeviceMap(main=dev, cpu_pinned=())

    pinned: list[str] = []
    for name, mod in model.named_modules():
        if name and isinstance(mod, hostile):
            pinned.append(name)
    return DeviceMap(main=dev, cpu_pinned=tuple(pinned))


def to_device(model: nn.Module, plan: DeviceMap) -> nn.Module:
    """Move `model` to `plan.main`, then move CPU-pinned submodules back.

    Returns `model` (in-place, matching `nn.Module.to` convention).
    """
    model.to(plan.main)
    for name in plan.cpu_pinned:
        try:
            sub = model.get_submodule(name)
        except AttributeError:
            continue
        sub.cpu()
    return model


def summarize(plan: DeviceMap) -> str:
    """Human-readable one-liner for logs."""
    if not plan.cpu_pinned:
        return f'device={plan.main}'
    return f'device={plan.main}  cpu_pinned=[{",".join(plan.cpu_pinned)}]'


def empty_cache(device: str | torch.device | None = None) -> None:
    """Release cached allocator memory on the current accelerator.

    No-op on CPU. Called between epochs to prevent MPS fragmentation from
    growing without bound during long training runs on unified memory.
    """
    dev = resolve(device)
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    elif dev.type == 'mps':
        # torch.mps.empty_cache exists since torch 2.0.
        cache = getattr(torch, 'mps', None)
        if cache is not None and hasattr(cache, 'empty_cache'):
            cache.empty_cache()


def available_memory_gb(device: str | torch.device | None = None) -> float:
    """Best-effort available memory (GB) for the target device.

    On MPS/CPU (unified memory on Apple silicon) returns the system's
    available RAM. On CUDA returns free device memory.
    """
    dev = resolve(device)
    if dev.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(dev)
        return free / 1024 ** 3
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return float('nan')


@dataclass
class SafeDefaults:
    """Device-aware knobs the notebook & scripts can pull to avoid OOM.

    - MPS on M-series has unified memory (~16-24 GB typical). Big batches
      or preloading many shards silently kills the Python kernel. These
      defaults were chosen from measurements on a 19 GB M2/M3.
    - CUDA on cluster GPUs has plenty of VRAM. Defaults match the
      pre-refactor notebook.
    - CPU falls in between; batches are the bottleneck, not memory.
    """
    batch_size: int
    hidden_channels: int
    num_layers: int
    max_preload_shards: int
    num_workers_loader: int
    note: str


def safe_defaults(device: str | torch.device | None = None) -> SafeDefaults:
    # Note: default_net.py hard-codes the second EdgeConv to 256-dim output
    # and only inserts (num_layers - 4) SAGEConv reducer layers, so a valid
    # (hidden_channels, num_layers) pair is either:
    #   • num_layers == 4  and hidden_channels == 256
    #   • num_layers >= 5  and hidden_channels arbitrary
    # These defaults obey that constraint.
    dev = resolve(device)
    if dev.type == 'cuda':
        return SafeDefaults(
            batch_size=512, hidden_channels=512, num_layers=8,
            max_preload_shards=1000, num_workers_loader=4,
            note='cuda: full-size defaults',
        )
    if dev.type == 'mps':
        # Empirically: batch=128 + hidden=256 + 4 layers fits in ~4 GB of
        # MPS activation memory on top of a ~4 GB preloaded dataset on a
        # 19 GB M-series. Larger settings crash the kernel mid-epoch.
        return SafeDefaults(
            batch_size=128, hidden_channels=256, num_layers=4,
            max_preload_shards=40, num_workers_loader=0,
            note='mps: reduced for unified-memory safety',
        )
    # CPU — small hidden but need num_layers>=5 to insert one SAGEConv
    # reducer (256 → hidden_channels) or edge_out shape mismatch.
    return SafeDefaults(
        batch_size=64, hidden_channels=128, num_layers=5,
        max_preload_shards=40, num_workers_loader=0,
        note='cpu: reduced for wall-clock',
    )
