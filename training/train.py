"""Training orchestration for the HGND GNN pipeline.

Extracted from the pre-refactor notebook cells 9 & 21-22. The `fit()` entry
point drives everything and is called identically by the notebook, by
`scripts/train.py`, and by SLURM jobs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

from .. import device as device_mod
from .checkpoint import save_checkpoint
from .losses import LossWeights, default_net_loss


@dataclass
class TrainConfig:
    """Full training configuration. Shared by notebook and CLI."""

    # ── Data / loader ────────────────────────────────────────────────────
    batch_size: int = 512
    num_workers_loader: int = 0
    train_frac: float = 0.5
    seed: int = 42

    # ── Optimizer / schedule ─────────────────────────────────────────────
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_milestones: tuple[int, ...] = (10, 15, 25)
    scheduler_gamma: float = 0.1
    grad_clip_norm: Optional[float] = None

    # ── Device ───────────────────────────────────────────────────────────
    device: Optional[str] = None
    cpu_pinned: Optional[tuple[str, ...]] = None

    # ── Loss ─────────────────────────────────────────────────────────────
    loss_weights: LossWeights = field(default_factory=LossWeights)

    # ── Provenance / I/O ─────────────────────────────────────────────────
    arch_name: str = 'net_default'
    arch_kwargs: dict = field(default_factory=dict)
    checkpoint_dir: str = 'checkpoints'
    checkpoint_name: str = 'model.pt'
    save_best_on: str = 'train'   # 'train' | 'val'
    verbose: bool = True


# ── Batch helpers ────────────────────────────────────────────────────────

def reindex_clusters(graph) -> torch.Tensor:
    """Re-index per-graph cluster ids into a contiguous batch-global range.

    Each graph inside a batched HeteroData has its own local cluster ids
    starting at 0. The default Net expects one flat id space for
    `avg_pool_x`, so we offset each graph's ids by the max seen so far.

    Vectorized: no per-graph Python loop with `.item()` syncs (the old
    version issued ~batch_size CPU/GPU syncs per training step, which
    hangs or crashes MPS). Computed on CPU because MPS lacks int64
    `scatter_reduce`; the result is moved back to the input device.
    Input tensors are index metadata (int64, one entry per hit) so the
    roundtrip is a few hundred KB per batch.
    """
    hit_batch = graph.batch_dict['hits']            # [N_hits], graph id per hit
    cluster_flat = graph.cluster.squeeze(-1)         # [N_hits], local cluster id
    if cluster_flat.numel() == 0:
        return cluster_flat.clone()

    device = cluster_flat.device
    hb = hit_batch.cpu()
    cf = cluster_flat.cpu()

    n_graphs = int(hb.max().item()) + 1
    per_graph_max = torch.full((n_graphs,), -1, dtype=cf.dtype)
    per_graph_max.scatter_reduce_(0, hb, cf, reduce='amax', include_self=True)
    # sizes[i] = max_id_i + 1 (0 for graphs with no hits — clamp -1 → 0)
    sizes = per_graph_max.clamp_min(-1) + 1
    # Prefix-sum offsets: offset[0] = 0, offset[i] = sum(sizes[:i]).
    # Add +1 so ids are 1-based (avg_pool_x expects 1-based labels here).
    offsets = torch.zeros_like(sizes)
    offsets[1:] = sizes.cumsum(0)[:-1]
    offsets = offsets + 1
    result = cf + offsets[hb]
    return result.to(device)


def _forward_default_net(model, graph):
    """Run one forward pass of the default Net given a batched HeteroData."""
    batched_cluster = reindex_clusters(graph)
    return model(
        graph['hits'].x,
        graph['hits', 'hits'].edge_index,
        graph['clusters', 'clusters'].edge_index,
        batched_cluster,
        graph.batch_dict['hits'],
    )


def forward_hetero(model, graph):
    """Run one forward pass of a HeteroGNN — takes x_dict/edge_index_dict."""
    return model(graph.x_dict, graph.edge_index_dict)


# ── Split / loader helpers ───────────────────────────────────────────────

def split_dataset(dataset, cfg: TrainConfig):
    """Train/test random split. Uses cfg.seed for reproducibility."""
    torch.manual_seed(cfg.seed)
    n_total = len(dataset)
    n_train = int(round(n_total * cfg.train_frac))
    n_test  = n_total - n_train
    return random_split(dataset, [n_train, n_test])


def build_loaders(dataset, cfg: TrainConfig):
    train_ds, test_ds = split_dataset(dataset, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers_loader)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers_loader)
    return train_loader, test_loader


# ── Epoch loops ──────────────────────────────────────────────────────────

def train_epoch(
    model,
    loader,
    optimizer,
    plan: device_mod.DeviceMap,
    loss_weights: LossWeights,
    forward_fn: Callable = _forward_default_net,
    loss_fn: Callable = default_net_loss,
    verbose: bool = True,
    grad_clip_norm: Optional[float] = None,
) -> float:
    model.train()
    losses = []
    iterator = tqdm(loader, desc='train') if verbose else loader
    for graph in iterator:
        graph = graph.to(plan.main)
        optimizer.zero_grad()
        preds = forward_fn(model, graph)
        loss_dict = loss_fn(preds, graph, loss_weights)
        loss = loss_dict['total']
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float('nan')


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    plan: device_mod.DeviceMap,
    loss_weights: LossWeights,
    forward_fn: Callable = _forward_default_net,
    loss_fn: Callable = default_net_loss,
    verbose: bool = True,
) -> float:
    model.eval()
    losses = []
    iterator = tqdm(loader, desc='eval') if verbose else loader
    for graph in iterator:
        graph = graph.to(plan.main)
        preds = forward_fn(model, graph)
        loss_dict = loss_fn(preds, graph, loss_weights)
        losses.append(loss_dict['total'].item())
    return float(np.mean(losses)) if losses else float('nan')


# ── Orchestrator ─────────────────────────────────────────────────────────

def fit(
    model,
    train_loader,
    test_loader,
    cfg: TrainConfig,
    plan: Optional[device_mod.DeviceMap] = None,
    forward_fn: Callable = _forward_default_net,
    loss_fn: Callable = default_net_loss,
) -> dict:
    """Full training loop. Returns a dict with loss history and checkpoint path.

    Model must already be on-device (call `device_mod.to_device(model, plan)`
    before invoking). We do not construct the model here so notebooks can
    initialise lazy PyG modules (SAGEConv with in_channels=-1) with a dummy
    forward pass before fit() starts.
    """
    if plan is None:
        plan = device_mod.plan_for(model, cfg.device, cfg.cpu_pinned)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.scheduler_milestones),
        gamma=cfg.scheduler_gamma,
    )

    ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
    train_history: list[float] = []
    val_history: list[float] = []
    min_loss = float('inf')
    best_epoch = -1

    if cfg.verbose:
        print(f'fit(): {device_mod.summarize(plan)}  '
              f'epochs={cfg.epochs}  batch={cfg.batch_size}  lr={cfg.lr}')

    t0 = time.time()
    for epoch in range(cfg.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, plan, cfg.loss_weights,
            forward_fn=forward_fn, loss_fn=loss_fn,
            verbose=cfg.verbose, grad_clip_norm=cfg.grad_clip_norm,
        )
        scheduler.step()
        val_loss = eval_epoch(
            model, test_loader, plan, cfg.loss_weights,
            forward_fn=forward_fn, loss_fn=loss_fn, verbose=cfg.verbose,
        )
        # Free accumulated MPS/CUDA cache before the next epoch — otherwise
        # long runs on MPS grow their allocator until unified memory is
        # exhausted and the kernel dies with no traceback.
        device_mod.empty_cache(plan.main)
        train_history.append(train_loss)
        val_history.append(val_loss)

        watch = train_loss if cfg.save_best_on == 'train' else val_loss
        if watch < min_loss:
            min_loss = watch
            best_epoch = epoch
            save_checkpoint(
                ckpt_path, model,
                arch_name=cfg.arch_name, arch_kwargs=cfg.arch_kwargs,
                epoch=epoch,
                metrics={'train_loss': train_loss, 'val_loss': val_loss},
            )

        if cfg.verbose:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'epoch {epoch:03d}  train {train_loss:.4f}  '
                  f'val {val_loss:.4f}  lr {lr_now:.2e}'
                  + ('  [best]' if epoch == best_epoch else ''))

    return {
        'train_loss': train_history,
        'val_loss': val_history,
        'best_epoch': best_epoch,
        'best_loss': min_loss,
        'checkpoint_path': ckpt_path,
        'elapsed_sec': time.time() - t0,
    }
