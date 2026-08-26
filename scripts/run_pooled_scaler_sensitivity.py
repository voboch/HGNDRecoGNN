"""Re-evaluate the HPC checkpoint on all three SMASH datasets after
remapping per-dataset StandardScaler outputs into a single pooled
StandardScaler basis.

Motivation
----------
The 2026-08-26 sensitivity smoke run (`results/sensitivity_hpc_smoke/`)
showed a 37% closure deficit on defaultSpot vs 80% closure on the two
OOD datasets — the opposite of the expected pattern. The paper-critic
review flagged per-dataset scaler contamination as the leading
candidate (Sec. 6 item 1 of the paper draft).

This script tests that hypothesis without a full re-preprocess by:
  1. Loading the parquet raw-hits caches for defaultSpot / zeroSpot /
     bigSpot from `notebooks/cache/ndet_dataset_smash_<name>[_smoke]/
     processed/_hits_cache_*.parquet`.
  2. Fitting a POOLED `StandardScaler` on the concatenation of top and
     bot hits across all three datasets (top and bot pooled
     separately, matching the per-dataset convention).
  3. Building a PyG runtime `transform` that maps each hit feature
     row from per-dataset-scaled space back into pooled-scaled space
     via the analytic identity
         x_new = a * x_old + b
     with a_i = sigma_old_i / sigma_new_i and
          b_i = (mu_old_i - mu_new_i) / sigma_new_i.
  4. Running `HGNDRecoGNN.training.predict` on the transformed dataset
     for each of the three roots.
  5. Saving `pred_{hits,clusters,edges}_hpc_pooled.pkl` to
     `results/sensitivity_hpc_pooled_smoke/<name>_hpc/` and then
     invoking `HGNDRecoGNN.scripts.sensitivity` to produce the
     comparison plots and CSVs.

Usage
-----
    conda activate pyg
    export PYTHONPATH=/Users/vovvy/Project/BM@N/HGND:$PYTHONPATH
    python -m HGNDRecoGNN.scripts.run_pooled_scaler_sensitivity
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from HGNDRecoGNN import device as device_mod
from HGNDRecoGNN import models as model_registry
from HGNDRecoGNN.data.graph_dataset import HGNDGraphDataset, FEATURES
from HGNDRecoGNN.training import predict, load_checkpoint, TrainConfig
from HGNDRecoGNN.training.train import build_loaders
from HGNDRecoGNN.training.eval import save_predictions


REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / 'checkpoints_hpc' / 'defaultSpot_v2_seed42_20ep_after4278192' / 'model.pt'
OUT_ROOT = REPO / 'results' / 'sensitivity_hpc_pooled_smoke'
CACHE_ROOT = REPO / 'notebooks' / 'cache'
DATASETS = [
    ('defaultSpot', 'ndet_dataset_smash_defaultSpot',       18.0),
    ('zeroSpot',    'ndet_dataset_smash_zeroSpot_smoke',     0.0),
    ('bigSpot',     'ndet_dataset_smash_bigSpot_smoke',     90.0),
]
NUM_SHARDS = 4
BATCH_SIZE = 64
PRELOAD_MAX_GB = 0.6

# Features excluding the last one (SurfaceHit is a raw 0/1, never scaled).
NUM_SCALED = len(FEATURES) - 1


def _hits_cache_path(root: Path) -> Path:
    hits = list((root / 'processed').glob('_hits_cache_*.parquet'))
    if not hits:
        raise FileNotFoundError(f'no parquet cache under {root}/processed')
    return hits[0]


def _load_hits_split(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (top_hits, bot_hits) feature matrices for scaler fitting."""
    df = pd.read_parquet(_hits_cache_path(root))
    top = df.loc[df.fY > 0, FEATURES[:-1]].to_numpy(dtype=np.float64)
    bot = df.loc[df.fY < 0, FEATURES[:-1]].to_numpy(dtype=np.float64)
    return top, bot


def fit_pooled_scaler() -> tuple[StandardScaler, StandardScaler]:
    tops, bots = [], []
    for _, root_dir, _ in DATASETS:
        root = CACHE_ROOT / root_dir
        t, b = _load_hits_split(root)
        print(f'  {root_dir:>52s}  top={t.shape[0]:>9,}  bot={b.shape[0]:>9,}')
        tops.append(t)
        bots.append(b)
    top_all = np.concatenate(tops, axis=0)
    bot_all = np.concatenate(bots, axis=0)
    print(f'  pooled  top={top_all.shape[0]:,}  bot={bot_all.shape[0]:,}')
    pooled_top = StandardScaler().fit(top_all)
    pooled_bot = StandardScaler().fit(bot_all)
    return pooled_top, pooled_bot


def _load_per_dataset_scalers(root: Path) -> tuple[StandardScaler, StandardScaler]:
    proc = root / 'processed'
    return pd.read_pickle(proc / 'scaler_top.pkl'), pd.read_pickle(proc / 'scaler_bot.pkl')


def _remap_coeffs(old: StandardScaler, new: StandardScaler) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (a, b) so that new_scaled = a * old_scaled + b elementwise."""
    a = torch.as_tensor(old.scale_ / new.scale_,               dtype=torch.float32)
    b = torch.as_tensor((old.mean_ - new.mean_) / new.scale_, dtype=torch.float32)
    return a, b  # both [NUM_SCALED]


def make_transform(a_top, b_top, a_bot, b_bot):
    """Return a PyG runtime transform mapping per-dataset-scaled
    features into pooled-scaled features. istop is a per-graph scalar
    tensor stored at graph.istop."""
    def _fn(graph):
        x = graph['hits'].x        # [N_hits, NUM_SCALED+1]  (last col = SurfaceHit)
        is_top = bool(int(graph.istop.item()))
        a, b = (a_top, b_top) if is_top else (a_bot, b_bot)
        scaled = x[:, :NUM_SCALED] * a.to(x.device) + b.to(x.device)
        graph['hits'].x = torch.cat([scaled, x[:, NUM_SCALED:]], dim=1)
        return graph
    return _fn


def verify_remap(root: Path, transform, pooled_top, pooled_bot,
                 old_top, old_bot, n_probe: int = 3) -> None:
    """Sanity-check: after transform, features on a few graphs should
    match what we'd get by re-scaling raw hits from parquet with the
    pooled scaler."""
    dataset = HGNDGraphDataset(root=str(root), hits_csv_dir=str(root),
                               num_shards=NUM_SHARDS, allow_stale_schema=True)
    dataset.preload(max_gb=PRELOAD_MAX_GB, verbose=False)
    for i in range(n_probe):
        g = dataset[i]
        x_before = g['hits'].x.clone()
        is_top = bool(int(g.istop.item()))
        old, new = (old_top, pooled_top) if is_top else (old_bot, pooled_bot)
        # Round-trip: undo old scaler → apply pooled scaler
        raw = old.inverse_transform(x_before[:, :NUM_SCALED].numpy())
        expected = new.transform(raw)
        g2 = transform(g)
        got = g2['hits'].x[:, :NUM_SCALED].numpy()
        diff = np.abs(expected - got).max()
        print(f'  graph {i}: istop={is_top}  max|expected-got|={diff:.2e}')
        assert diff < 1e-4, 'remap coefficients disagree with round-trip'


def evaluate_one(name: str, root_dir: str, transform) -> tuple[dict, Path]:
    root = CACHE_ROOT / root_dir
    print(f'\n=== evaluate {name}  ({root_dir}) ===')
    dataset = HGNDGraphDataset(root=str(root), hits_csv_dir=str(root),
                               num_shards=NUM_SHARDS, allow_stale_schema=True,
                               transform=transform)
    dataset.preload(max_gb=PRELOAD_MAX_GB)

    ckpt = load_checkpoint(str(CKPT))
    model, spec = model_registry.get(ckpt.arch_name, dataset, **ckpt.arch_kwargs)
    model.eval()
    with torch.no_grad():
        probe = next(iter(DataLoader(dataset, batch_size=8, shuffle=False)))
        spec.forward_fn(model, probe)
    model.load_state_dict(ckpt.state_dict, strict=True)

    plan = device_mod.plan_for(model, None, cpu_pinned=spec.default_cpu_pinned or None)
    device_mod.to_device(model, plan)

    cfg = TrainConfig(batch_size=BATCH_SIZE, seed=42, train_frac=0.5)
    _, loader = build_loaders(dataset, cfg)
    t0 = time.time()
    preds = predict(model, loader, plan=plan, forward_fn=spec.forward_fn, verbose=False)
    print(f'  inference: {time.time()-t0:.1f}s   '
          f'hits={len(preds["hits"]):,}  clusters={len(preds["clusters"]):,}  '
          f'edges={len(preds["edges"]):,}')

    out_dir = OUT_ROOT / f'{name}_hpc'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(preds, str(out_dir), suffix='hpc_pooled')
    return preds, out_dir


def run_sensitivity() -> None:
    args = [
        sys.executable, '-m', 'HGNDRecoGNN.scripts.sensitivity',
        '--out-dir', str(OUT_ROOT / 'sensitivity'),
        '--dataset-cache-root', str(CACHE_ROOT),
        '--threshold', '0.5',
    ]
    for name, _root_dir, _u in DATASETS:
        pkl = OUT_ROOT / f'{name}_hpc' / 'pred_clusters_hpc_pooled.pkl'
        args += ['--dataset', f'{name}={pkl}']
    print('\n=== sensitivity CLI ===')
    print(' '.join(args))
    subprocess.run(args, check=True)


def main() -> int:
    print('=== fitting pooled scaler ===')
    pooled_top, pooled_bot = fit_pooled_scaler()

    print('\n=== per-dataset remap coefficients (top) ===')
    print(f'  {"feature":10s}  {"a=σ_old/σ_new":>14s}  {"b":>10s}')
    coeffs = {}
    for name, root_dir, _ in DATASETS:
        root = CACHE_ROOT / root_dir
        old_top, old_bot = _load_per_dataset_scalers(root)
        a_top, b_top = _remap_coeffs(old_top, pooled_top)
        a_bot, b_bot = _remap_coeffs(old_bot, pooled_bot)
        coeffs[name] = (a_top, b_top, a_bot, b_bot, old_top, old_bot)
        print(f'\n  --- {name} (top) ---')
        for i, feat in enumerate(FEATURES[:-1]):
            print(f'    {feat:10s}  {float(a_top[i]):>14.4f}  {float(b_top[i]):>10.4f}')

    print('\n=== verifying remap on defaultSpot ===')
    a_top, b_top, a_bot, b_bot, old_top, old_bot = coeffs['defaultSpot']
    tf = make_transform(a_top, b_top, a_bot, b_bot)
    verify_remap(CACHE_ROOT / 'ndet_dataset_smash_defaultSpot', tf,
                 pooled_top, pooled_bot, old_top, old_bot)

    print('\n=== evaluating with pooled-scaled inputs ===')
    for name, root_dir, _ in DATASETS:
        a_top, b_top, a_bot, b_bot, *_ = coeffs[name]
        tf = make_transform(a_top, b_top, a_bot, b_bot)
        evaluate_one(name, root_dir, tf)

    run_sensitivity()
    print('\ndone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
