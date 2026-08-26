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
import sys


def _add_package_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.abspath(os.path.join(here, '..', '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)


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
    args = parser.parse_args()

    _add_package_to_path()

    import torch
    from HGNDRecoGNN import device as device_mod
    from HGNDRecoGNN import models as model_registry
    from HGNDRecoGNN.data.graph_dataset import HGNDGraphDataset
    from HGNDRecoGNN.training import TrainConfig, fit
    from HGNDRecoGNN.training.train import build_loaders

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

    model, spec = model_registry.get(args.model, dataset, **arch_kwargs)
    print(f'Model: {args.model} — {spec.description}')

    plan = device_mod.plan_for(model, args.device,
                               cpu_pinned=spec.default_cpu_pinned or None)
    device_mod.to_device(model, plan)
    print(device_mod.summarize(plan))
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    cfg = TrainConfig(
        batch_size=args.batch_size,
        num_workers_loader=args.num_workers_loader,
        seed=args.seed,
        train_frac=args.train_frac,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        arch_name=args.model,
        arch_kwargs=arch_kwargs,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        loss_weights=spec.default_loss_weights,
    )

    train_loader, test_loader = build_loaders(dataset, cfg)
    result = fit(model, train_loader, test_loader, cfg, plan=plan,
                 forward_fn=spec.forward_fn, loss_fn=spec.loss_fn)

    print(f'\ndone: best epoch {result["best_epoch"]} '
          f'(loss {result["best_loss"]:.4f})  '
          f'checkpoint: {result["checkpoint_path"]}  '
          f'elapsed: {result["elapsed_sec"]:.1f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
