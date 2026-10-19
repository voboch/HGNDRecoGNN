#!/usr/bin/env python
"""Train an HGND model on a preprocessed sharded dataset.

Phase 1 only supports `--model net_default`. Phase 2 will read from the
model registry.

Usage:
  python -m HGNDRecoGNN.scripts.train \\
      --root notebooks/cache/ndet_dataset_smash_defaultSpot \\
      --model net_default --epochs 20 --batch-size 512 \\
      --checkpoint-dir checkpoints
"""

from __future__ import annotations

import argparse
import os
import random
import sys


def _add_package_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.abspath(os.path.join(here, '..', '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


def _with_energy_flattening(weights, dataset, bins: int,
                            max_graphs: int = 200_000):
    """Attach spectrum-flattening weights built from the dataset's clusters.

    Built once over the training partition rather than per batch: a batch of a few hundred
    graphs holds too few high-energy clusters for its own occupancy to be
    anything but noise, and weights that move between batches become gradient
    noise on the head they are meant to correct.

    Only `max_graphs` graphs are read, strided across the cache so the sample
    spans every shard. The v3 caches hold up to 13.9 M graphs and walking all
    of them would cost more than the epoch it precedes; 200 k graphs give tens
    of thousands of signal clusters, which fixes ten bin occupancies well
    inside the precision the weights are used to.
    """
    import dataclasses

    import torch

    from HGNDRecoGNN.training.losses import build_energy_flatten_weights

    if not hasattr(weights, 'energy_flatten'):
        raise SystemExit(
            f'--energy-flatten is only defined for the default Net loss; '
            f'{type(weights).__name__} has no energy term to reweight.')

    n = len(dataset)
    stride = max(1, n // max_graphs)
    energies = []
    for i in range(0, n, stride):
        e = getattr(dataset[i], 'clenergy', None)
        if e is not None:
            e = e.flatten()
            energies.append(e[e > 0])
    if not energies:
        raise SystemExit('--energy-flatten: no positive cluster energies found')
    pooled = torch.cat(energies)

    edges, bw = build_energy_flatten_weights(pooled, bins=bins)
    out = dataclasses.replace(weights, energy_flatten=True,
                              energy_bin_edges=edges, energy_bin_weights=bw)
    print(f'Energy flattening: {pooled.numel():,} signal clusters from '
          f'{len(range(0, n, stride)):,} of {n:,} graphs (stride {stride}), '
          f'{bins} bins over [{float(edges[0]):.2f}, {float(edges[-1]):.2f}] GeV')
    print('  bin weights: '
          + ', '.join(f'{float(x):.2f}' for x in bw))
    return out


def _net_default_arch_flags(args) -> dict:
    """The opt-in net_default architecture kwargs implied by `args`.

    Separated from main() so the mapping can be tested without a dataset, a
    GPU or pyg-lib. A wrong flag here is only discovered after a GPU
    allocation has been spent, which is the class of error
    tests/test_train_cli.py exists to catch.
    """
    out: dict = {}
    if getattr(args, 'cluster_pooling', None) is not None:
        out['cluster_pooling'] = args.cluster_pooling
    if getattr(args, 'cluster_size_feature', False):
        out['cluster_size_feature'] = True
    if getattr(args, 'hit_residual', False):
        out['hit_residual'] = True
    if getattr(args, 'hit_normalization', None) is not None:
        out['hit_normalization'] = args.hit_normalization
    return out


def _schedule_overrides(args) -> dict:
    """TrainConfig overrides for the schedule / optimisation flags.

    Only keys the user set explicitly are returned, so TrainConfig's own
    defaults stay in force for everything else.
    """
    out: dict = {}
    for attr, key in (('scheduler', 'scheduler'),
                      ('warmup_epochs', 'warmup_epochs'),
                      ('lr_min', 'lr_min'),
                      ('grad_clip_norm', 'grad_clip_norm'),
                      ('split_seed', 'split_seed')):
        val = getattr(args, attr, None)
        if val is not None:
            out[key] = val
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description='Train an HGND model.')
    parser.add_argument('--root', required=True,
                        help='Preprocessed dataset root (contains processed/).')
    parser.add_argument('--csv-dir', default=None,
                        help='CSV source dir. Only needed if dataset must be (re)built.')
    parser.add_argument('--model', default='net_default',
                        help='Model to train — any name from '
                             'HGNDRecoGNN.models.available().')
    parser.add_argument('--hidden', type=int, default=None,
                        help='Hidden dim override (per-model default when omitted).')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Num layers override (per-model default when omitted).')
    parser.add_argument('--heads', type=int, default=4,
                        help='Attention heads (GAT / HGT only).')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout override (per-model default when omitted).')
    parser.add_argument('--k-pe', type=int, default=4,
                        help='LapPE dim (spectral_* models only).')
    parser.add_argument('--num-shards', type=int, default=None,
                        help='Cap loaded shards. None = all.')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--init-from', default=None,
                        help='Checkpoint to load weights from before training. '
                             'Use with --start-epoch to continue a run that the '
                             'scheduler cancelled.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='First epoch index. Continues the LR schedule and, '
                             'if checkpoint-dir/optim.pt exists, the optimizer '
                             'state.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-frac', type=float, default=0.5)
    parser.add_argument('--device', default=None,
                        help='cuda | mps | cpu | auto (default auto).')
    parser.add_argument('--num-workers-loader', type=int, default=0)
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--checkpoint-name', default='model.pt')
    parser.add_argument('--preload', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='Preload all shards into RAM (default: on). '
                             'Use --no-preload to disable.')
    parser.add_argument('--preload-max-gb', type=float, default=None,
                        help='Cap the preload footprint (GB). Trims shard '
                             'count if the estimate would exceed this. '
                             'Recommended on MPS / small-RAM machines.')
    parser.add_argument('--save-best-on', default='val',
                        choices=['train', 'val'],
                        help='Which loss drives checkpoint selection. '
                             'val (default) keeps the epoch that generalises '
                             'best. train selects the lowest training loss, '
                             'which on this model is the most overfit epoch '
                             '(the converged SMASH run reports train 1.268 '
                             'against val 1.354); it was the default through '
                             'every result in reports/, so pass it to '
                             'reproduce one of those runs exactly.')
    parser.add_argument('--energy-flatten', action='store_true',
                        help='reweight the cluster-energy loss by the inverse '
                             'occupancy of each true-energy bin, as the '
                             'reference notebook does in its second training '
                             'stage. Plain MSE on a falling spectrum pulls '
                             'high-energy clusters down; see '
                             'reports/dombay_reference_audit.md.')
    parser.add_argument('--energy-flatten-bins', type=int, default=10,
                        help='bins for --energy-flatten (reference: 10)')
    parser.add_argument('--energy-flatten-sample', type=int, default=200_000,
                        help='graphs read to build the --energy-flatten '
                             'weights, strided across the cache')
    parser.add_argument('--scheduler-milestones', type=int, nargs='*',
                        default=None,
                        help='Override MultiStepLR milestones (e.g. 10 15 25).')
    parser.add_argument('--split-seed', type=int, default=None,
                        help='Seed for the train/test split only. Defaults to '
                             '--seed. Set it to repeat training with a '
                             'different --seed while holding out the same '
                             'events.')
    parser.add_argument('--scheduler', choices=('multistep', 'cosine', 'plateau'),
                        default=None,
                        help='LR schedule. multistep (default) is the '
                             'reference notebook schedule; cosine decays '
                             'smoothly over the whole budget; plateau decays '
                             'only when validation stalls.')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                        help='Linear LR warmup over this many epochs '
                             '(cosine/multistep only).')
    parser.add_argument('--lr-min', type=float, default=None,
                        help='Cosine floor (default 0).')
    parser.add_argument('--grad-clip-norm', type=float, default=None,
                        help='Clip gradients to this global norm. Unset by '
                             'default, matching every published run.')
    parser.add_argument('--cluster-pooling',
                        choices=('mean', 'mean_max_sum', 'mean_max_sum_proj'),
                        default=None,
                        help="Cluster-branch pooling. 'mean' (default) is "
                             "avg_pool_x alone and is invariant to cluster "
                             "size; 'mean_max_sum' concatenates max and sum, "
                             "which triples the DynamicEdgeConv input and adds "
                             "4.19M parameters at H=512, confounding any gain "
                             "with capacity; 'mean_max_sum_proj' projects the "
                             "concatenation back to hidden_channels first, "
                             "keeping that layer's shape identical to 'mean'.")
    parser.add_argument('--cluster-size-feature', action='store_true',
                        help='Append log1p(cluster hit count) to the cluster '
                             'vector. Cluster size alone separates signal '
                             'from background at AUC 0.729.')
    parser.add_argument('--hit-residual', action='store_true',
                        help='Residual connections across the SAGEConv stack '
                             '(parameter-free; helps at num_layers=8).')
    parser.add_argument('--hit-normalization', choices=('batch', 'layer'), default=None,
                        help='SAGE hit-layer normalization; batch preserves legacy weights.')
    parser.add_argument('--validation-aggregation', choices=('population', 'batch_mean'),
                        default='population', help='Native per-head population means; '
                        'batch_mean reproduces historical checkpoint selection.')
    args = parser.parse_args()

    _add_package_to_path()

    import numpy as np
    import torch
    from HGNDRecoGNN import device as device_mod
    from HGNDRecoGNN import models as model_registry
    from HGNDRecoGNN.data.graph_dataset import HGNDGraphDataset
    from HGNDRecoGNN.training import TrainConfig, fit
    from HGNDRecoGNN.training.train import build_loaders

    # Seed before dataset construction and the builder's lazy initialisation.
    # Seeding only in build_loaders()/fit() leaves the initial weights random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'PyTorch {torch.__version__}')

    ds_kwargs = dict(
        root=args.root,
        hits_csv_dir=args.csv_dir or args.root,   # only used if reprocessing
        num_workers=8,
        num_shards=args.num_shards,
    )
    dataset = HGNDGraphDataset(**ds_kwargs)
    if args.preload:
        dataset.preload(max_gb=args.preload_max_gb)
    print(f'Dataset: {len(dataset)} graphs')

    # Build arch_kwargs, forwarding only explicit user overrides so each
    # model's own defaults take effect otherwise.
    arch_kwargs: dict = {}
    if args.hidden is not None:
        # net_default expects `hidden_channels`; others expect `hidden`.
        if args.model == 'net_default':
            arch_kwargs['hidden_channels'] = args.hidden
        else:
            arch_kwargs['hidden'] = args.hidden
    if args.num_layers is not None:
        arch_kwargs['num_layers'] = args.num_layers
    if args.dropout is not None:
        arch_kwargs['dropout'] = args.dropout
    if args.model.startswith('hetero_gat') or args.model in ('hetero_hgt',
                                                             'thesis_gat',
                                                             'spectral_gat'):
        arch_kwargs['heads'] = args.heads
    if args.model.startswith('spectral_'):
        arch_kwargs['k_pe'] = args.k_pe
    # net_default's opt-in architecture flags. Folded in HERE, before the model
    # is constructed, and not later beside the TrainConfig: arch_kwargs is both
    # the constructor's argument list and what gets recorded in the checkpoint,
    # so adding them afterwards would log a configuration the model never had.
    if args.model == 'net_default':
        arch_kwargs.update(_net_default_arch_flags(args))
    elif args.cluster_pooling or args.cluster_size_feature or args.hit_residual or args.hit_normalization:
        parser.error('--cluster-pooling, --cluster-size-feature and '
                     '--hit-residual apply to --model net_default only')

    # Seed BEFORE constructing the model. fit() seeds on entry, but the
    # weights -- and the lazy SAGEConv shapes materialised by
    # build_default_net's dummy forward -- are created here, so without
    # this two runs with the same --seed began from different weights.
    from HGNDRecoGNN.training.train import seed_everything
    seed_everything(args.seed)
    model, spec = model_registry.get(args.model, dataset, **arch_kwargs)
    print(f'Model: {args.model} — {spec.description}')

    if args.init_from:
        from HGNDRecoGNN.training import load_checkpoint
        prev = load_checkpoint(args.init_from)
        missing, unexpected = model.load_state_dict(prev.state_dict, strict=False)
        if missing or unexpected:
            raise SystemExit(
                f'--init-from {args.init_from}: state_dict does not match this '
                f'architecture ({len(missing)} missing, {len(unexpected)} '
                f'unexpected keys). Check --hidden/--num-layers.')
        print(f'Initialised from {args.init_from} '
              f'(epoch {prev.epoch}, metrics {prev.metrics})')

    plan = device_mod.plan_for_spec(model, args.device,
                                    spec.default_cpu_pinned)
    device_mod.to_device(model, plan)
    print(device_mod.summarize(plan))
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    cfg_kwargs: dict = dict(
        batch_size=args.batch_size,
        num_workers_loader=args.num_workers_loader,
        seed=args.seed,
        train_frac=args.train_frac,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        arch_name=args.model,
        arch_kwargs=arch_kwargs,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        loss_weights=spec.default_loss_weights,
        save_best_on=args.save_best_on,
        validation_aggregation=args.validation_aggregation,
    )
    if args.scheduler_milestones is not None:
        cfg_kwargs['scheduler_milestones'] = tuple(args.scheduler_milestones)
    cfg_kwargs.update(_schedule_overrides(args))
    cfg = TrainConfig(**cfg_kwargs)
    print(f'Checkpoint policy: save_best_on={cfg.save_best_on}')

    train_loader, test_loader = build_loaders(dataset, cfg)
    if args.energy_flatten:
        cfg.loss_weights = _with_energy_flattening(
            spec.default_loss_weights, train_loader.dataset, args.energy_flatten_bins,
            args.energy_flatten_sample)
    result = fit(model, train_loader, test_loader, cfg, plan=plan,
                 forward_fn=spec.forward_fn, loss_fn=spec.loss_fn)

    print(f'\ndone: best epoch {result["best_epoch"]} '
          f'(loss {result["best_loss"]:.4f})  '
          f'checkpoint: {result["checkpoint_path"]}  '
          f'elapsed: {result["elapsed_sec"]:.1f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
