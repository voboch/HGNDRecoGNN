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


def _restandardise(dataset, cache_root: str, target_dir: str) -> None:
    """Re-express preloaded hit features in another scaler's standardisation.

    The cache stores `(x - mean_src) / scale_src` for the nine scaled features,
    with the raw `SurfaceHit` flag appended as a tenth column. Recovering the
    physical value and re-standardising is one affine map per feature:

        x_dst = (x_src * scale_src + mean_src - mean_dst) / scale_dst

    `SurfaceHit` is left alone. Top and bottom halves carry separate scalers,
    selected per graph by `istop`, matching how the cache was written.
    """
    import numpy as np
    import pandas as pd
    import torch

    def _load(d):
        return (pd.read_pickle(os.path.join(d, 'scaler_top.pkl')),
                pd.read_pickle(os.path.join(d, 'scaler_bot.pkl')))

    src_dir = os.path.join(cache_root, 'processed')
    src_top, src_bot = _load(src_dir)
    dst_top, dst_bot = _load(target_dir)

    if os.path.realpath(src_dir) == os.path.realpath(target_dir):
        print(f'restandardise: source and target are the same scaler '
              f'({target_dir}) — nothing to do')
        return

    def _coef(src, dst):
        # x_dst = a * x_src + b
        a = src.scale_ / dst.scale_
        b = (src.mean_ - dst.mean_) / dst.scale_
        return (torch.as_tensor(a, dtype=torch.float32),
                torch.as_tensor(b, dtype=torch.float32))

    a_top, b_top = _coef(src_top, dst_top)
    a_bot, b_bot = _coef(src_bot, dst_bot)
    n = a_top.numel()

    print(f'restandardise: {cache_root}/processed -> {target_dir}')
    print(f'  max |a-1| = {float((a_top - 1).abs().max()):.5f} (top), '
          f'{float((a_bot - 1).abs().max()):.5f} (bot)')
    print(f'  max |b|   = {float(b_top.abs().max()):.5f} (top), '
          f'{float(b_bot.abs().max()):.5f} (bot)')

    n_done = 0
    for g in dataset._preloaded:
        istop = bool(torch.as_tensor(g.istop).flatten()[0].item())
        a, b = (a_top, b_top) if istop else (a_bot, b_bot)
        x = g['hits'].x
        x[:, :n] = x[:, :n] * a + b
        n_done += 1
    print(f'  rewrote {n_done:,} graphs')


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
    p.add_argument('--restandardise-to', default=None, metavar='SCALER_DIR',
                   help='Re-express the cached hit features in the '
                        'standardisation of the scalers in SCALER_DIR '
                        '(a processed/ directory holding scaler_top.pkl and '
                        'scaler_bot.pkl). Each cache was standardised with a '
                        'scaler fitted to its own sample, so the same physical '
                        'neutron reaches the network differently depending on '
                        'which sample it came from: across the three U_sym '
                        'caches the eToF scale differs by 0.41 %% and its mean '
                        'by 0.019 GeV, which biases the reconstructed energy '
                        'by ~0.8 %% in the direction that cancels the U_sym '
                        'signal. Standardisation is affine and invertible, so '
                        'this undoes it exactly. Point it at the scaler the '
                        'checkpoint was trained with. See '
                        'reports/dombay_reference_audit.md.')
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

    if args.restandardise_to:
        _restandardise(dataset, args.root, args.restandardise_to)

    model, spec = model_registry.get(ckpt.arch_name, dataset, **ckpt.arch_kwargs)
    model.load_state_dict(ckpt.state_dict)
    plan = device_mod.plan_for_spec(model, args.device,
                                    spec.default_cpu_pinned)
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
