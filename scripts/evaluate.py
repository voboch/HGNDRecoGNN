#!/usr/bin/env python
"""Run a trained model on a dataset and export prediction DataFrames.

Usage:
  python -m HGNDRecoGNN.scripts.evaluate \\
      --root      notebooks/cache/ndet_dataset_smash_defaultSpot \\
      --checkpoint checkpoints/defaultSpot_net_default/model.pt \\
      --out-dir   results/defaultSpot_net_default \\
      --suffix    smash
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
    p = argparse.ArgumentParser(description='Predict + export DataFrames.')
    p.add_argument('--root', required=True,
                   help='Preprocessed dataset root.')
    p.add_argument('--checkpoint', required=True,
                   help='v1 checkpoint written by scripts/train.py.')
    p.add_argument('--out-dir', required=True,
                   help='Where pred_{hits,clusters,edges}_<suffix>.pkl land.')
    p.add_argument('--suffix', default='smash',
                   help='Filename suffix — matches results_smash.ipynb by default.')
    p.add_argument('--num-shards', type=int, default=None,
                   help='Cap loaded shards.')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', default=None,
                   help='cuda | mps | cpu | auto.')
    p.add_argument('--split', default='test', choices=['test', 'all'],
                   help='Run on the held-out test split (default) or the '
                        'whole dataset. Use "all" when evaluating a checkpoint '
                        'on a dataset it was never trained on (e.g. cross-'
                        'dataset sensitivity).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-frac', type=float, default=0.5,
                   help='Only used with --split test to reproduce the split.')
    p.add_argument('--preload-max-gb', type=float, default=None,
                   help='Cap the preload footprint (GB). Trims shard count '
                        'if the estimate would exceed this.')
    args = p.parse_args()

    _add_package_to_path()

    from torch_geometric.loader import DataLoader
    from HGNDRecoGNN import device as device_mod
    from HGNDRecoGNN import models as model_registry
    from HGNDRecoGNN.data.graph_dataset import HGNDGraphDataset
    from HGNDRecoGNN.training import predict, load_checkpoint, TrainConfig
    from HGNDRecoGNN.training.train import build_loaders
    from HGNDRecoGNN.training.eval import save_predictions

    ckpt = load_checkpoint(args.checkpoint)
    print(f'ckpt: arch={ckpt.arch_name}  kwargs={ckpt.arch_kwargs}  '
          f'epoch={ckpt.epoch}  metrics={ckpt.metrics}')

    dataset = HGNDGraphDataset(root=args.root, hits_csv_dir=args.root,
                               num_shards=args.num_shards,
                               allow_stale_schema=True)
    dataset.preload(max_gb=args.preload_max_gb)

    model, spec = model_registry.get(ckpt.arch_name, dataset, **ckpt.arch_kwargs)
    model.load_state_dict(ckpt.state_dict)
    plan = device_mod.plan_for(model, args.device,
                               cpu_pinned=spec.default_cpu_pinned or None)
    device_mod.to_device(model, plan)

    if args.split == 'test':
        cfg = TrainConfig(batch_size=args.batch_size, seed=args.seed,
                          train_frac=args.train_frac)
        _, loader = build_loaders(dataset, cfg)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    preds = predict(model, loader, plan=plan, forward_fn=spec.forward_fn)
    paths = save_predictions(preds, args.out_dir, suffix=args.suffix)
    for k, p_ in paths.items():
        print(f'  {k:9s}  {len(preds[k]):>9,} rows  →  {p_}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
