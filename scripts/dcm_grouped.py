"""Preregistered DCM schedule comparison with a reserved source-run test.

The delivered DCM export has three run directories. Keep each whole directory
in one partition, including both detector arms. The test therefore contains
one unseen source run; it cannot estimate between-generation-job variance.
Reuse frozen graphs/noise and undo their old affine scaling before fitting
new scalers on training hits only. No graph or target is rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np


def group_partition(groups, seed=42):
    """Three delivered source runs -> one train, one validation, one test.

    The partition is deliberately independent of training seed and outcomes.
    With only three runs, fractional event-level splitting would hide the
    source-run dependence the experiment is meant to test.
    """
    groups = sorted(set(groups))
    if len(groups) != 3:
        raise ValueError(f'Expected the three delivered source runs, got {groups}')
    random.Random(seed).shuffle(groups)
    return dict(zip(('train', 'validation', 'test'), ([g] for g in groups)))


def split_indices(graph_groups, partition):
    lookup = {g: k for k, gs in partition.items() for g in gs}
    if len(lookup) != sum(map(len, partition.values())):
        raise ValueError('Source run occurs in multiple partitions')
    if set(graph_groups) != set(lookup):
        raise ValueError('Graph/source-run coverage differs from manifest')
    return {k: np.array([i for i, g in enumerate(graph_groups) if lookup[g] == k],
                        dtype=np.int64) for k in partition}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_data(root, preload=True):
    from HGNDRecoGNN.data.graph_dataset import HGNDGraphDataset
    meta = json.loads((Path(root) / 'processed/meta.json').read_text())
    ds = HGNDGraphDataset(root=str(root), hits_csv_dir=str(root),
                          num_shards=meta['num_shards'], allow_stale_schema=True)
    if preload:
        ds.preload()
    if len(ds) != meta['num_graphs']:
        raise ValueError('Partial preload is forbidden for this experiment')
    return ds, meta


def prepare(ds, root, raw, campaign):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from HGNDRecoGNN.scripts.reco_truth_join import row_offsets, to_local
    from HGNDRecoGNN.data.graph_dataset import FEATURES

    out = campaign / 'prepared'
    if (out / 'manifest.json').exists():
        raise FileExistsError('Prepared manifest already exists; do not overwrite')
    out.mkdir(parents=True, exist_ok=True)
    offsets = row_offsets(str(raw))
    rows = np.array([int(ds[i].Row.reshape(-1)[0]) for i in range(len(ds))])
    loc = to_local(rows, offsets)
    if loc.unmapped.any():
        raise ValueError('Cached Row cannot be mapped to original source files')
    groups = loc.run.astype(str).tolist()
    partition = group_partition(groups)
    indices = split_indices(groups, partition)
    # Both sides of each physical event must remain in a single partition.
    ownership = {}
    for k, ix in indices.items():
        for row in rows[ix]:
            if ownership.setdefault(int(row), k) != k:
                raise AssertionError('Physical event leaks between partitions')

    old = [pd.read_pickle(Path(root) / f'processed/scaler_{s}.pkl')
           for s in ('bot', 'top')]
    new = [StandardScaler(), StandardScaler()]
    nfeat = len(FEATURES) - 1
    train = indices['train']
    # Bound memory by 1024 graphs. This is an I/O chunk, not a physics cut.
    for start in range(0, len(train), 1024):
        chunks = [[], []]
        for i in train[start:start + 1024]:
            g = ds[int(i)]
            side = int(g.istop.item())
            scaled = g['hits'].x[:, :nfeat].numpy().astype(np.float64)
            chunks[side].append(old[side].inverse_transform(scaled))
        for side, arrays in enumerate(chunks):
            if arrays:
                new[side].partial_fit(np.concatenate(arrays))
    a = np.stack([old[s].scale_ / new[s].scale_ for s in (0, 1)])
    b = np.stack([(old[s].mean_ - new[s].mean_) / new[s].scale_ for s in (0, 1)])
    np.savez_compressed(out / 'indices_and_scaling.npz', **indices, a=a, b=b)
    for side, name in enumerate(('bot', 'top')):
        pd.to_pickle(new[side], out / f'scaler_{name}.pkl')
    offsets.to_csv(out / 'source_row_offsets.csv', index=False)
    manifest = {
        'protocol': 'dcm_source_run_holdout_v1', 'data_root': str(root),
        'raw_root': str(raw), 'split_seed': 42, 'partition_source_runs': partition,
        'graphs': {k: len(v) for k, v in indices.items()},
        'physical_events': {k: len(set(rows[v])) for k, v in indices.items()},
        'scaler_fit': 'training hits only; recovered from frozen float32 features',
        'frozen_graphs_and_time_noise': True, 'training_seeds': [42, 123, 456],
        'arms': ['multistep', 'cosine'], 'epochs_both_arms': 30,
        'architecture': {'hidden_channels': 512, 'num_layers': 8},
        'threshold_policy': 'first validation threshold reaching count purity 0.70',
        'threshold_grid': {'start': 0.001, 'stop': 0.999, 'count': 401},
        'threshold_justification': '0.70 is the published comparison convention; '
            'the full validation purity/efficiency scan is retained and plotted',
        'limits': 'One source run per partition. No between-run test variance. '
            'New initialization; no legacy checkpoint is reused. '
            'No independent claim of reproducing the published event estimator.',
        'sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    }
    write_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


def make_subset(ds, indices, a, b):
    import torch
    from torch.utils.data import Dataset

    class Remapped(Dataset):
        def __len__(self):
            return len(indices)

        def __getitem__(self, i):
            # PyG preload returns shared objects: clone before an affine remap.
            g = ds[int(indices[i])].clone()
            side = int(g.istop.item())
            x = g['hits'].x
            n = a.shape[1]
            x[:, :n] = x[:, :n] * torch.as_tensor(a[side], dtype=x.dtype) + \
                torch.as_tensor(b[side], dtype=x.dtype)
            return g

    return Remapped()


def evaluate(model, spec, loader, plan, out, threshold=None):
    from HGNDRecoGNN.training.eval import predict
    from HGNDRecoGNN.scripts.reco_performance import (
        roc_auc, threshold_scan, energy_performance)
    pred = predict(model, loader, plan=plan, forward_fn=spec.forward_fn)
    cl = pred['clusters']
    scan = threshold_scan(cl, np.linspace(0.001, 0.999, 401))
    if threshold is None:
        eligible = scan[scan.purity_count >= 0.70]
        if eligible.empty:
            raise RuntimeError('Validation sample never reaches count purity 0.70')
        threshold = float(eligible.iloc[0].threshold)
    out.mkdir(parents=True, exist_ok=True)
    cl.to_pickle(out / 'pred_clusters_dcm.pkl')
    scan.to_csv(out / 'threshold_scan.csv', index=False)
    one = threshold_scan(cl, [threshold]).iloc[0].to_dict()
    result = {'n_clusters': len(cl), 'roc_auc': roc_auc(cl.cl_score, cl.cl_label),
              'working_threshold': threshold, 'working_point': one,
              'energy_performance': energy_performance(cl, threshold).to_dict('records')}
    write_json(out / 'performance.json', result)
    if out.name == 'validation':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from HGNDRecoGNN.scripts.plot_b_full import apply_style
        apply_style()
        fig, ax = plt.subplots(figsize=(6.84, 3.5))
        ax.plot(scan.threshold, scan.purity_count, label='cluster purity')
        ax.plot(scan.threshold, scan.efficiency_count, ls='--', label='cluster efficiency')
        ax.axhline(0.70, color='0.3', ls=':', label='comparison convention: purity 0.70')
        ax.axvline(threshold, color='0.3', ls='--')
        ax.set(xlabel='cluster score threshold', ylabel='fraction', ylim=(0, 1.03))
        ax.text(.02, .03, 'DCM Xe+CsI simulation; validation source run\n'
                'reconstructed selection against cluster MC truth', transform=ax.transAxes)
        ax.legend()
        fig.savefig(out / 'threshold_scan.pdf', bbox_inches='tight')
        plt.close(fig)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['prepare', 'smoke', 'train', 'test'], required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--arm', choices=['multistep', 'cosine'], default='multistep')
    p.add_argument('--seed', type=int, choices=[42, 123, 456], default=42)
    args = p.parse_args(argv)
    import torch
    from torch_geometric.loader import DataLoader
    from HGNDRecoGNN import models, device
    from HGNDRecoGNN.training.train import TrainConfig, fit, seed_everything
    from HGNDRecoGNN.training.checkpoint import load_checkpoint

    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    # A smoke needs only its small train/validation subsets. Loading the full
    # cache for preparation AND each smoke arm turns an execution check into
    # three long Lustre reads and can exhaust the test partition walltime.
    ds, _ = load_data(args.root, preload=args.phase != 'smoke')
    prep = args.campaign / 'prepared'
    if args.phase == 'prepare':
        prepare(ds, args.root, args.raw_root, args.campaign)
        return
    manifest = json.loads((prep / 'manifest.json').read_text())
    for name, value in manifest['sha256'].items():
        if digest(prep / name) != value:
            raise ValueError(f'Prepared artifact changed: {name}')
    if manifest['data_root'] != args.root or manifest['raw_root'] != args.raw_root:
        raise ValueError('Inputs differ from the frozen manifest')
    bundle = np.load(prep / 'indices_and_scaling.npz')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not train on a login node')
    smoke = args.phase == 'smoke'
    out = args.campaign / (f'smoke_{args.arm}_{args.seed}' if smoke
                           else f'{args.arm}_{args.seed}')
    out.mkdir(exist_ok=True)
    epoch_budget = 1 if smoke else manifest['epochs_both_arms']
    train_idx = bundle['train'][:512] if smoke else bundle['train']
    val_idx = bundle['validation'][:256] if smoke else bundle['validation']
    subsets = {k: make_subset(ds, ix, bundle['a'], bundle['b'])
               for k, ix in [('train', train_idx), ('validation', val_idx)]}
    seed_everything(args.seed)
    arch = manifest['architecture']
    model, spec = models.get('net_default', subsets['train'], **arch)
    plan = device.plan_for_spec(model, 'cuda', spec.default_cpu_pinned)
    device.to_device(model, plan)
    cfg = TrainConfig(epochs=epoch_budget, batch_size=256, seed=args.seed,
                      scheduler=args.arm, warmup_epochs=0, save_best_on='val',
                      arch_kwargs=arch, checkpoint_dir=str(out / 'checkpoints'))
    if smoke:
        cfg.scheduler_milestones = ()
    if args.phase in ('train', 'smoke'):
        resume = out / 'checkpoints/resume.pt'
        if (out / 'training_complete.json').exists():
            raise FileExistsError('Training already completed; refusing a duplicate')
        if resume.exists():
            state = torch.load(resume, weights_only=False, map_location='cpu')
            cfg.start_epoch = int(state['next_epoch'])
        loaders = {k: DataLoader(v, batch_size=256, shuffle=(k == 'train'))
                   for k, v in subsets.items()}
        history = fit(model, loaders['train'], loaders['validation'], cfg, plan,
                      forward_fn=spec.forward_fn, loss_fn=spec.loss_fn)
        if not all(np.isfinite(v) for k in ('train_loss', 'val_loss') for v in history[k]):
            raise RuntimeError('Nonfinite training/validation loss')
        checkpoint = load_checkpoint(str(out / 'checkpoints/model.pt'))
        model.load_state_dict(checkpoint.state_dict)
        # The smoke verifies execution, not classifier quality after two steps.
        # Select everything there so an untrained model cannot fail a purity
        # target meant only for the complete experiment.
        result = evaluate(model, spec, loaders['validation'], plan, out / 'validation',
                          threshold=0.0 if smoke else None)
        write_json(out / 'training_complete.json', {
            'arm': args.arm, 'seed': args.seed, 'epochs': epoch_budget,
            'selected_epoch': checkpoint.epoch, 'checkpoint_sha256': digest(out / 'checkpoints/model.pt'),
            'split_manifest_sha256': digest(prep / 'manifest.json'),
            'validation_threshold': result['working_threshold'], 'test_opened': False})
        return
    # Test is evaluated only after every preregistered training arm completed.
    for arm in manifest['arms']:
        for seed in manifest['training_seeds']:
            if not (args.campaign / f'{arm}_{seed}/training_complete.json').exists():
                raise RuntimeError('Do not open test until all six arms finish')
    if (out / 'test/performance.json').exists():
        raise FileExistsError('Test already evaluated; refusing repeated opening')
    frozen = json.loads((out / 'training_complete.json').read_text())
    path = out / 'checkpoints/model.pt'
    if digest(path) != frozen['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed before test')
    model.load_state_dict(load_checkpoint(str(path)).state_dict)
    test = make_subset(ds, bundle['test'], bundle['a'], bundle['b'])
    result = evaluate(model, spec, DataLoader(test, batch_size=256), plan,
                      out / 'test', threshold=frozen['validation_threshold'])
    write_json(out / 'test_complete.json', {'arm': args.arm, 'seed': args.seed,
        'threshold_source': 'validation only', 'n_test_graphs': len(test),
        'test_source_runs': manifest['partition_source_runs']['test'],
        'checkpoint_sha256': frozen['checkpoint_sha256'], 'roc_auc': result['roc_auc']})


if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    if '--phase' in argv and argv[argv.index('--phase') + 1] == 'smoke':
        # Same interpreter: expensive torch/PyG imports happen once. Only
        # train/validation shards are lazily read; the reserved test is unused.
        for arm in ('multistep', 'cosine'):
            main(argv + ['--arm', arm])
    else:
        main(argv)
