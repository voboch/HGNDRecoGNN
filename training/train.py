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
    # None -> use `seed`. Set to pin the split while varying `seed`.
    split_seed: Optional[int] = None

    # ── Optimizer / schedule ─────────────────────────────────────────────
    epochs: int = 20
    start_epoch: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    # 'multistep' is the reference notebook's schedule and stays the default so
    # published numbers reproduce. 'cosine' decays smoothly to `lr_min` over the
    # whole budget instead of dropping 10x at fixed epochs; with the default
    # milestones and epochs=20 the LR is already 1e-5 by epoch 15, so a quarter
    # of the budget trains at 1 % of the initial rate. 'plateau' decays only
    # when validation stops improving.
    scheduler: str = 'multistep'          # 'multistep' | 'cosine' | 'plateau'
    scheduler_milestones: tuple[int, ...] = (10, 15, 25)
    scheduler_gamma: float = 0.1
    warmup_epochs: int = 0                # linear ramp from lr/warmup_factor
    warmup_factor: float = 10.0
    lr_min: float = 0.0                   # cosine floor
    plateau_patience: int = 3
    grad_clip_norm: Optional[float] = None

    # ── Device ───────────────────────────────────────────────────────────
    # Resume. A mismatched legacy optim.pt/last.pt pair is refused by default;
    # see _legacy_resume_is_consistent.
    resume_allow_mismatch: bool = False

    device: Optional[str] = None
    cpu_pinned: Optional[tuple[str, ...]] = None

    # ── Loss ─────────────────────────────────────────────────────────────
    loss_weights: LossWeights = field(default_factory=LossWeights)

    # ── Provenance / I/O ─────────────────────────────────────────────────
    arch_name: str = 'net_default'
    arch_kwargs: dict = field(default_factory=dict)
    checkpoint_dir: str = 'checkpoints'
    checkpoint_name: str = 'model.pt'
    # 'val' selects the checkpoint that generalises; 'train' selects the one
    # that fits the training split best, which on this model is the most
    # overfit epoch (the converged SMASH run reports train 1.268 against val
    # 1.354). The default was 'train' through every result in reports/; it is
    # now 'val'. Pass save_best_on='train' to reproduce an older run exactly.
    save_best_on: str = 'val'     # 'val' | 'train'
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

def seed_everything(seed: int) -> None:
    # Seeds torch (CPU and all CUDA devices) and numpy together.
    #
    # Call this BEFORE the model is constructed. fit() and split_dataset()
    # seed on entry, but by then scripts/train.py has already built the
    # model, so the initial weights -- including the lazy SAGEConv shapes
    # materialised by the dummy forward pass -- came from whatever RNG state
    # the process happened to be in. Two runs with the same --seed therefore
    # started from different weights.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, cfg: TrainConfig):
    """Train/test random split. Uses cfg.split_seed, else cfg.seed."""
    # cfg.split_seed, when set, pins the split independently of cfg.seed,
    # so repeating a training run with a different training seed does not
    # also reshuffle which events are held out. It defaults to cfg.seed,
    # the behaviour every result in reports/ was produced with.
    seed = cfg.seed if getattr(cfg, "split_seed", None) is None else cfg.split_seed
    torch.manual_seed(seed)
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

RESUME_NAME = 'resume.pt'


def _write_resume_state(path, model, optimizer, scheduler, *, next_epoch,
                        min_loss, best_epoch, arch_name, arch_kwargs):
    """Write model, optimizer, scheduler, epoch and RNG as ONE atomic file.

    The previous contract wrote `optim.pt` (carrying next_epoch) and `last.pt`
    (carrying the weights) as two separate files, with `optim.pt` committed
    first. A job stopped between the two left next_epoch one ahead of the
    weights on disk, and the resume path paired them without checking, so
    training continued with an optimizer and a weight set from different
    epochs. Keeping everything in one file replaced atomically removes the
    window: a reader sees either the whole previous generation or the whole
    new one.

    RNG state is included so a resumed run draws the same batch order and the
    same augmentation as an uninterrupted one would.
    """
    blob = {
        'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'next_epoch': int(next_epoch),
        'min_loss': float(min_loss),
        'best_epoch': int(best_epoch),
        'arch_name': arch_name,
        'arch_kwargs': dict(arch_kwargs or {}),
        'rng': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
        },
        'format_version': 1,
    }
    if torch.cuda.is_available():
        blob['rng']['cuda'] = torch.cuda.get_rng_state_all()
    tmp = f'{path}.tmp'
    try:
        torch.save(blob, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _legacy_resume_is_consistent(opt_blob, last_path):
    """Does the old optim.pt / last.pt pair describe one generation?

    Returns (ok, detail). `optim.pt` says which epoch to start at; `last.pt`
    holds the weights from the epoch before that. Anything else means the run
    was interrupted between the two writes.
    """
    # A MISSING last.pt is not an inconsistency. scripts/train.py
    # --init-from loads the weights before fit() is called, and in that
    # flow fit() needs only the optimizer state, so there is nothing on
    # disk to disagree with. Only a last.pt that EXISTS and names a
    # different generation is a fault.
    if not os.path.exists(last_path):
        return True, (f'{os.path.basename(last_path)} absent; weights '
                      f'are the caller responsibility via --init-from')
    try:
        last = torch.load(last_path, map_location='cpu', weights_only=False)
    except Exception as exc:                       # truncated by a kill
        return False, f'{last_path} will not load ({exc})'
    next_epoch = int(opt_blob.get('next_epoch', -1))
    last_epoch = int(last.get('epoch', -1))
    if last_epoch + 1 != next_epoch:
        return False, (f'optim.pt says next_epoch={next_epoch} but last.pt '
                       f'holds epoch={last_epoch}; expected '
                       f'{next_epoch - 1}. The run was interrupted between '
                       f'the two writes.')
    return True, f'epoch {last_epoch} -> {next_epoch}'


def _build_scheduler(cfg, optimizer):
    """Build the LR schedule. Returns (scheduler, needs_validation_metric).

    All three options step once per epoch, which is what the training loop
    does; a per-batch schedule such as OneCycle would need the inner loop
    restructured and is deliberately not offered here rather than offered
    half-working.
    """
    sched_name = str(cfg.scheduler).lower()
    lrs = torch.optim.lr_scheduler

    if sched_name == 'multistep':
        dead = sorted(m for m in cfg.scheduler_milestones if m >= cfg.epochs)
        fired = sorted(m for m in cfg.scheduler_milestones if m < cfg.epochs)
        if dead and cfg.verbose:
            # (10, 15, 25) against the default epochs=20 never reaches 25. The
            # schedule was written for the reference's 25-epoch budget, so a
            # shorter run silently truncates it.
            #
            # Everything quoted here is COMPUTED from the milestones that
            # actually fire. An earlier version indexed [1] on the sorted
            # list, which raised IndexError on a single-milestone schedule --
            # a configuration the CLI accepts (--scheduler-milestones 25).
            final_lr = cfg.lr * cfg.scheduler_gamma ** len(fired)
            where = (f'reaches {final_lr:.1e} at epoch {fired[-1]}'
                     if fired else
                     f'never drops below {cfg.lr:.1e}')
            print(f'fit(): WARNING milestones {dead} are >= epochs='
                  f'{cfg.epochs} and will never fire. The schedule is being '
                  f'truncated: of {tuple(cfg.scheduler_milestones)} only '
                  f'{tuple(fired)} fire, so the LR {where} and stays there '
                  f'for the rest of the run.')
        core = lrs.MultiStepLR(optimizer,
                               milestones=list(cfg.scheduler_milestones),
                               gamma=cfg.scheduler_gamma)
        needs_metric = False

    elif sched_name == 'cosine':
        span = max(1, cfg.epochs - cfg.warmup_epochs)
        core = lrs.CosineAnnealingLR(optimizer, T_max=span, eta_min=cfg.lr_min)
        needs_metric = False

    elif sched_name == 'plateau':
        core = lrs.ReduceLROnPlateau(optimizer, mode='min',
                                     factor=cfg.scheduler_gamma,
                                     patience=cfg.plateau_patience)
        needs_metric = True

    else:
        raise ValueError(
            f"unknown scheduler {cfg.scheduler!r}; "
            f"expected 'multistep', 'cosine' or 'plateau'")

    if cfg.warmup_epochs > 0:
        if needs_metric:
            # SequentialLR cannot drive ReduceLROnPlateau, which needs a metric
            # passed to step(). Refuse rather than silently drop the warmup.
            raise ValueError("warmup_epochs is not supported with the "
                             "'plateau' scheduler; use 'cosine' or 'multistep'")
        warm = lrs.LinearLR(optimizer,
                            start_factor=1.0 / cfg.warmup_factor,
                            end_factor=1.0,
                            total_iters=cfg.warmup_epochs)
        core = lrs.SequentialLR(optimizer, schedulers=[warm, core],
                                milestones=[cfg.warmup_epochs])
    return core, needs_metric



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

    seed_everything(cfg.seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler, scheduler_needs_metric = _build_scheduler(cfg, optimizer)

    ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)

    # Resuming. Long runs on a shared cluster do not always reach their
    # walltime -- two 20-epoch jobs here were cancelled by the scheduler at
    # about three and a half hours -- so a run has to be continuable across
    # several submissions. The optimizer and scheduler state live beside the
    # checkpoint rather than inside it, which keeps the checkpoint format
    # unchanged and loadable by every existing consumer.
    start_epoch = int(getattr(cfg, 'start_epoch', 0) or 0)
    opt_path = os.path.join(cfg.checkpoint_dir, 'optim.pt')
    train_history: list[float] = []
    val_history: list[float] = []
    min_loss = float('inf')
    best_epoch = -1
    resume_path = os.path.join(cfg.checkpoint_dir, RESUME_NAME)
    last_path = os.path.join(cfg.checkpoint_dir, 'last.pt')
    if start_epoch and os.path.exists(resume_path):
        blob = torch.load(resume_path, map_location='cpu', weights_only=False)
        model.load_state_dict(blob['model'])
        optimizer.load_state_dict(blob['optimizer'])
        scheduler.load_state_dict(blob['scheduler'])
        min_loss = float(blob.get('min_loss', min_loss))
        best_epoch = int(blob.get('best_epoch', best_epoch))
        rng = blob.get('rng') or {}
        if 'torch' in rng:
            torch.set_rng_state(rng['torch'].cpu()
                                if hasattr(rng['torch'], 'cpu')
                                else rng['torch'])
        if 'numpy' in rng:
            np.random.set_state(rng['numpy'])
        if cfg.verbose:
            print(f'fit(): resumed model, optimizer, schedule and RNG from '
                  f'{resume_path} at epoch {start_epoch} '
                  f'(lr {scheduler.get_last_lr()[0]:.2e})')
    elif start_epoch and os.path.exists(opt_path):
        blob = torch.load(opt_path, map_location='cpu', weights_only=False)
        # Legacy two-file layout. It can describe two different epochs, so it
        # is checked rather than trusted; see _legacy_resume_is_consistent.
        ok, detail = _legacy_resume_is_consistent(blob, last_path)
        if not ok and not getattr(cfg, 'resume_allow_mismatch', False):
            raise RuntimeError(
                f'inconsistent resume state in {cfg.checkpoint_dir}: {detail}\n'
                f'Refusing to continue: the optimizer and the weights would '
                f'come from different epochs. Delete {RESUME_NAME}/optim.pt to '
                f'restart the run, or set resume_allow_mismatch=True to '
                f'continue anyway and accept the mismatch.')
        if not ok and cfg.verbose:
            print(f'fit(): WARNING resuming despite {detail}')
        optimizer.load_state_dict(blob['optimizer'])
        scheduler.load_state_dict(blob['scheduler'])
        min_loss = float(blob.get('min_loss', min_loss))
        best_epoch = int(blob.get('best_epoch', best_epoch))
        if cfg.verbose:
            print(f'fit(): resumed optimizer and schedule from {opt_path} '
                  f'at epoch {start_epoch} (lr {scheduler.get_last_lr()[0]:.2e})')
    elif start_epoch and cfg.verbose:
        print(f'fit(): starting at epoch {start_epoch} with a fresh optimizer; '
              f'no {opt_path} found, so Adam moments restart')

    if cfg.verbose:
        print(f'fit(): {device_mod.summarize(plan)}  '
              f'epochs={cfg.epochs}  batch={cfg.batch_size}  lr={cfg.lr}')

    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, plan, cfg.loss_weights,
            forward_fn=forward_fn, loss_fn=loss_fn,
            verbose=cfg.verbose, grad_clip_norm=cfg.grad_clip_norm,
        )
        val_loss = eval_epoch(
            model, test_loader, plan, cfg.loss_weights,
            forward_fn=forward_fn, loss_fn=loss_fn, verbose=cfg.verbose,
        )
        # Stepped AFTER validation so 'plateau' can see the metric it decays
        # on. For the epoch-indexed schedules the order is immaterial, and the
        # printed LR below now refers to the epoch just finished rather than
        # the next one.
        if scheduler_needs_metric:
            scheduler.step(val_loss)
        else:
            scheduler.step()
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

        # Written every epoch, not only on improvement: a job cancelled
        # mid-run must be resumable from where it stopped, which needs the
        # latest optimizer state and the next epoch index, not the best one.
        #
        # ORDER MATTERS. last.pt and optim.pt are kept for the existing job
        # scripts and for anything that loads a plain checkpoint, but they are
        # two files and cannot be committed together. resume.pt holds the same
        # generation in ONE atomic file and is written LAST, so it is the
        # commit point: if the job dies earlier, resume.pt still describes the
        # previous complete epoch and nothing pairs mismatched state.
        save_checkpoint(
            os.path.join(cfg.checkpoint_dir, 'last.pt'), model,
            arch_name=cfg.arch_name, arch_kwargs=cfg.arch_kwargs,
            epoch=epoch,
            metrics={'train_loss': train_loss, 'val_loss': val_loss},
        )
        torch.save({'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'next_epoch': epoch + 1,
                    'min_loss': min_loss,
                    'best_epoch': best_epoch},
                   os.path.join(cfg.checkpoint_dir, 'optim.pt.tmp'))
        os.replace(os.path.join(cfg.checkpoint_dir, 'optim.pt.tmp'),
                   os.path.join(cfg.checkpoint_dir, 'optim.pt'))
        _write_resume_state(
            os.path.join(cfg.checkpoint_dir, RESUME_NAME),
            model, optimizer, scheduler,
            next_epoch=epoch + 1, min_loss=min_loss, best_epoch=best_epoch,
            arch_name=cfg.arch_name, arch_kwargs=cfg.arch_kwargs,
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
