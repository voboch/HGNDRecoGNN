"""Bounded validation-only DCM architecture/training controls after the re-audit.

The previous test was opened; this campaign never evaluates it. Reuse its
frozen train/validation partition and train-fitted scalers. Four recipes and
three paired seeds are fixed before launch; no follow-up sweep is automatic.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from .dcm_grouped import digest, write_json, make_subset

RECIPES = {
    'reference': {},
    'layernorm': {'hit_normalization': 'layer'},
    'pool': {'cluster_pooling': 'mean_max_sum'},
    'low_lr': {'lr': 1e-4},
}
SEEDS = (42, 123, 456)


def recipe(name, seed, smoke=False):
    overrides = dict(RECIPES[name])
    lr = overrides.pop('lr', 1e-3)
    # H512/L8, 30 epochs, milestones and batch size are reference choices,
    # not optima. One decade in rate is a sensitivity control, not a fitted
    # optimum. Pooling alone also increases capacity; report that confounding.
    return {'recipe': name, 'seed': seed, 'epochs': 1 if smoke else 30,
            'batch_size': 256, 'lr': lr, 'weight_decay': 1e-5,
            'scheduler': 'multistep', 'scheduler_milestones': [] if smoke else [10, 15, 25],
            'scheduler_gamma': .1, 'warmup_epochs': 0,
            'architecture': {'hidden_channels': 512, 'num_layers': 8,
                             'hit_normalization': 'batch', 'cluster_pooling': 'mean',
                             'cluster_size_feature': False, 'hit_residual': False, **overrides},
            'loss_weights': {'hit': 1., 'link': 1., 'cluster_cls': 1.,
                             'cluster_energy': 1., 'cluster_link': 1.},
            'grad_clip_norm': None,
            'save_best_on': 'val', 'validation_aggregation': 'population',
            'test_evaluation': False}


def validate_prepared(prepared):
    manifest = json.loads((prepared/'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(prepared/name) != expected:
            raise ValueError(f'Prepared artifact changed: {name}')
    bundle = np.load(prepared/'indices_and_scaling.npz')
    development = set(np.concatenate([bundle['train'], bundle['validation']]))
    if set(bundle['train']).intersection(bundle['validation']) or development.intersection(bundle['test']):
        raise ValueError('Partition overlap')
    return manifest, bundle


def read_retry(path):
    for attempt in range(5):
        try:
            return Path(path).read_bytes()
        except OSError:
            if attempt == 4:raise
            time.sleep(2**attempt)


def selected_shards(indices, shard_lengths):
    # Prepared indices refer to preload's concatenation, not sid*shard_size.
    # The builder flushes short shards at phase boundaries as well as the end.
    stops = np.cumsum(shard_lengths)
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) and (indices.min() < 0 or indices.max() >= stops[-1]):
        raise ValueError('Selected index outside the frozen cache population')
    return sorted(set(map(int, np.searchsorted(stops, indices, side='right'))))


def _collect(shard, start, wanted, graphs):
    for offset, graph in enumerate(shard):
        index = start+offset
        if index in wanted:graphs[index] = graph


def _check_coverage(graphs, wanted):
    if set(graphs) != wanted:
        missing = sorted(wanted-set(graphs))
        raise ValueError(f'Incomplete selected graph population: {len(missing)} missing; first={missing[:10]}')


def inventory_selected(root, indices, meta):
    """Inspect every shard once on the compute node; retain allowed indices only."""
    import torch
    wanted = set(map(int, indices)); graphs = {}; lengths = []; hashes = {}
    start = 0
    for sid in range(meta['num_shards']):
        data = read_retry(root/f'processed/shard_{sid}.pt')
        shard = torch.load(io.BytesIO(data), weights_only=False, map_location='cpu')
        lengths.append(len(shard)); hashes[str(sid)] = hashlib.sha256(data).hexdigest()
        _collect(shard, start, wanted, graphs)
        start += len(shard)
        if sid % 25 == 0 or sid+1 == meta['num_shards']:
            print(f'Cache inventory: {sid+1}/{meta["num_shards"]} shards, {start} flat graphs', flush=True)
        del shard, data
    if start != meta['num_graphs']:raise ValueError('Shard population disagrees with cache metadata')
    _check_coverage(graphs, wanted)
    return graphs, {'meta_sha256': digest(root/'processed/meta.json'), 'shards': hashes,
                    'shard_lengths': lengths, 'num_graphs': start,
                    'index_policy': 'numeric shard order, concatenated actual lengths; HGNDGraphDataset.preload'}


def load_selected(root, indices, shard_lengths, expected):
    import torch
    wanted = set(map(int, indices)); graphs = {}
    starts = np.concatenate([[0], np.cumsum(shard_lengths)])
    for sid in selected_shards(indices, shard_lengths):
        data = read_retry(root/f'processed/shard_{sid}.pt')
        if hashlib.sha256(data).hexdigest() != expected[str(sid)]:
            raise ValueError(f'Cache content changed: shard {sid}')
        shard = torch.load(io.BytesIO(data), weights_only=False, map_location='cpu')
        if len(shard) != shard_lengths[sid]:raise ValueError(f'Shard length changed: {sid}')
        _collect(shard, int(starts[sid]), wanted, graphs)
        del shard, data
    _check_coverage(graphs, wanted)
    return graphs  # make_subset uses original indices and clones before remap.


def validate_population(graphs, bundle, manifest, prepared):
    """Check full allowed population and source identity before any gradient step."""
    import pandas as pd
    from .reco_truth_join import to_local
    offsets = pd.read_csv(prepared/'source_row_offsets.csv', dtype={'run': str, 'stem': str})
    result = {}
    for part in ('train', 'validation'):
        indices = list(map(int, bundle[part]))
        rows = np.array([int(graphs[i].Row.reshape(-1)[0]) for i in indices], dtype=np.int64)
        sides = np.array([int(graphs[i].istop.item()) for i in indices], dtype=np.int64)
        local = to_local(rows, offsets)
        if local.unmapped.any() or set(local.run) != set(manifest['partition_source_runs'][part]):
            raise ValueError(f'Graph/source-run identity changed: {part}')
        identity = np.column_stack([indices, rows, sides]).astype('<i8')
        clusters = sum(int(graphs[i].cllabel.numel()) for i in indices)
        counts = {'graphs': len(indices), 'physical_events': len(set(rows)),
                  'clusters': clusters,
                  'signal_clusters': sum(int(graphs[i].cllabel.sum()) for i in indices),
                  'source_runs': sorted(set(local.run)),
                  'ordered_index_row_side_sha256': hashlib.sha256(identity.tobytes()).hexdigest()}
        if counts['graphs'] != manifest['graphs'][part] or counts['physical_events'] != manifest['physical_events'][part]:
            raise ValueError(f'Prepared population changed: {part}')
        if len(set(zip(rows, sides))) != len(indices):raise ValueError(f'Duplicate event/side: {part}')
        if part == 'validation' and clusters != 101107:raise ValueError('Validation cluster population changed')
        result[part] = counts
    return result


def score_validation(model, spec, loader, plan, out):
    from HGNDRecoGNN.training.eval import predict
    from HGNDRecoGNN.scripts.reco_performance import roc_auc, threshold_scan, energy_performance
    cl = predict(model, loader, plan=plan, forward_fn=spec.forward_fn)['clusters']
    if len(cl) != 101107:raise ValueError('Validation cluster population changed')
    out.mkdir(exist_ok=True)
    cl.to_pickle(out/'pred_clusters_dcm.pkl')
    scan = threshold_scan(cl, np.linspace(.001, .999, 401))
    scan.to_csv(out/'threshold_scan.csv', index=False)
    eligible = scan[scan.purity_count >= .70]
    # An unreachable conventional purity is a model outcome, not permission
    # to invent a new cut or silently discard that recipe/seed.
    threshold = float(eligible.iloc[0].threshold) if len(eligible) else None
    point = threshold_scan(cl, [threshold]).iloc[0].to_dict() if threshold is not None else None
    result = {'scored_on': 'reused validation source run 988974; development only',
              'n_clusters': len(cl), 'n_signal_clusters': int(cl.cl_label.sum()),
              'roc_auc': roc_auc(cl.cl_score, cl.cl_label),
              'working_threshold': threshold, 'working_point': point,
              'threshold_policy': 'first validation grid point reaching count purity 0.70; conventional target',
              'energy_performance': energy_performance(cl, threshold).to_dict('records')
                    if threshold is not None else [], 'test_opened_by_this_campaign': False}
    def finite(x):
        if isinstance(x, dict):return {k: finite(v) for k, v in x.items()}
        if isinstance(x, list):return [finite(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x):return None
        return x
    write_json(out/'performance.json', finite(result))
    return result


def run(args, name, seed, manifest, bundle, frozen, graphs=None):
    import fcntl
    prefix = 'smoke_' if args.phase == 'smoke' else ''
    out = args.campaign/f'{prefix}{name}_{seed}'
    out.mkdir(exist_ok=True)
    # Kernel locks release on process death; a crash does not require unsafe
    # deletion of a stale lock, and two submissions cannot write one run.
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(args, name, seed, manifest, bundle, frozen, graphs)


def _run(args, name, seed, manifest, bundle, frozen, graphs):
    import torch
    from torch_geometric.loader import DataLoader
    from HGNDRecoGNN import device, models
    from HGNDRecoGNN.training.train import TrainConfig, fit, seed_everything, _restore_rng_state
    from HGNDRecoGNN.training.checkpoint import load_checkpoint
    smoke = args.phase == 'smoke'
    config = recipe(name, seed, smoke)
    config['split_manifest_sha256'] = digest(args.prepared/'manifest.json')
    config['data_manifest_sha256'] = digest(args.campaign/'data_manifest.json')
    config['source_manifest_sha256'] = digest(args.campaign/'source_manifest.json')
    out = args.campaign/(f'smoke_{name}_{seed}' if smoke else f'{name}_{seed}')
    out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():raise FileExistsError('Completed task; refusing duplicate')
    if (out/'effective_config.json').exists():
        if json.loads((out/'effective_config.json').read_text()) != config:
            raise ValueError('Recipe or input identity changed; use a new campaign')
    else:write_json(out/'effective_config.json', config)
    write_json(out/'status.json', {'state': 'loading', 'test_opened_by_this_campaign': False})
    train_idx = bundle['train'][:512] if smoke else bundle['train']
    val_idx = bundle['validation'][:256] if smoke else bundle['validation']
    indices = np.concatenate([train_idx, val_idx])
    assert not set(map(int, indices)).intersection(map(int, bundle['test']))
    ds = graphs if graphs is not None else load_selected(
        Path(manifest['data_root']), indices, frozen['shard_lengths'], frozen['shards'])
    if not smoke:
        coverage = validate_population(ds, bundle, manifest, args.prepared)
        if coverage != frozen['population']:raise ValueError('Full population differs from passed smoke')
        write_json(out/'population_verified.json', coverage)
    subsets = {k: make_subset(ds, ix, bundle['a'], bundle['b'])
               for k, ix in [('train', train_idx), ('validation', val_idx)]}
    seed_everything(seed)
    model, spec = models.get('net_default', subsets['train'], **config['architecture'])
    plan = device.plan_for_spec(model, 'cuda', spec.default_cpu_pinned)
    device.to_device(model, plan)
    count = sum(p.numel() for p in model.parameters())
    write_json(out/'model_manifest.json', {'parameters': count,
               'actual_hit_normalization': model.hit_normalization,
               'actual_cluster_pooling': model.cluster_pooling,
               'cluster_input_channels': model.cluster_in_channels,
               'architecture': config['architecture']})
    cfg = TrainConfig(epochs=config['epochs'], batch_size=config['batch_size'], lr=config['lr'],
                      seed=seed, arch_kwargs=config['architecture'], checkpoint_dir=str(out/'checkpoints'),
                      scheduler_milestones=tuple(config['scheduler_milestones']),
                      validation_aggregation='population', save_best_on='val')
    resume = out/'checkpoints/resume.pt'
    if resume.exists():cfg.start_epoch = int(torch.load(resume, weights_only=False, map_location='cpu')['next_epoch'])
    from dataclasses import asdict
    write_json(out/'train_config.json', asdict(cfg))
    write_json(out/'status.json', {'state': 'training', 'test_opened_by_this_campaign': False})
    loaders = {k: DataLoader(v, batch_size=256, shuffle=(k=='train')) for k,v in subsets.items()}
    fit(model, loaders['train'], loaders['validation'], cfg, plan,
        forward_fn=spec.forward_fn, loss_fn=spec.loss_fn)
    state = torch.load(resume, weights_only=False, map_location='cpu')
    checkpoint = load_checkpoint(str(out/'checkpoints/model.pt'))
    if state['next_epoch'] != config['epochs'] or checkpoint.epoch != state['best_epoch']:
        raise ValueError('Incomplete budget or selected checkpoint/resume disagreement')
    write_json(out/'checkpoints/history.json', state['history'])
    if len(state['history']) != config['epochs']:raise ValueError('Full component history is required')
    model.load_state_dict(checkpoint.state_dict)
    if smoke:
        batch = next(iter(loaders['validation'])).to(plan.main);model.eval()
        with torch.no_grad():outputs = spec.forward_fn(model, batch)
        assert all(torch.isfinite(x).all() for x in outputs)
        rng = {'cuda': torch.cuda.get_rng_state_all()}
        expected = torch.rand(16, device='cuda');_restore_rng_state(rng)
        assert torch.equal(torch.rand(16, device='cuda'), expected)
        # Check event isolation with the real GPU kNN, including cluster and
        # edge ordering, rather than relying only on a local stand-in layer.
        pair = [subsets['validation'][0], subsets['validation'][1]]
        separate = []
        with torch.no_grad():
            for graph in DataLoader(pair, batch_size=1):
                separate.append(spec.forward_fn(model, graph.to(plan.main)))
            together = spec.forward_fn(model, next(iter(DataLoader(pair, batch_size=2))).to(plan.main))
        for i in range(5):
            assert torch.allclose(torch.cat([x[i] for x in separate]), together[i], rtol=1e-4, atol=1e-5)
    else:
        write_json(out/'status.json', {'state': 'validation', 'test_opened_by_this_campaign': False})
        score_validation(model, spec, loaders['validation'], plan, out/'validation')
    write_json(out/'complete.json', {'recipe': name, 'seed': seed, 'epochs': config['epochs'],
               'selected_epoch': checkpoint.epoch, 'checkpoint_sha256': digest(out/'checkpoints/model.pt'),
               'parameters': count, 'split_manifest_sha256': config['split_manifest_sha256'],
               'test_opened_by_this_campaign': False, 'smoke': smoke})
    write_json(out/'status.json', {'state': 'completed', 'test_opened_by_this_campaign': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['smoke', 'train'], required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--task', type=int, choices=range(12), default=0)
    args = p.parse_args()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU allocation required')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    manifest, bundle = validate_prepared(args.prepared)
    root = Path(manifest['data_root']);meta = json.loads((root/'processed/meta.json').read_text())
    data_manifest = args.campaign/'data_manifest.json'
    if args.phase == 'smoke':
        if data_manifest.exists():raise FileExistsError('Smoke manifest exists; inspect before any retry')
        indices = np.concatenate([bundle['train'], bundle['validation']])
        graphs, frozen = inventory_selected(root, indices, meta)
        frozen['population'] = validate_population(graphs, bundle, manifest, args.prepared)
        write_json(data_manifest, frozen)
        # Exercise the exact full training loader, not only the inventory path.
        # Release its first population before the second pass to bound RAM.
        del graphs
        graphs = load_selected(root, indices, frozen['shard_lengths'], frozen['shards'])
        if validate_population(graphs, bundle, manifest, args.prepared) != frozen['population']:
            raise ValueError('Training-loader population differs from flat inventory')
        write_json(args.campaign/'full_population_verified.json', frozen['population'])
        for name in RECIPES:run(args, name, 42, manifest, bundle, frozen, graphs)
        write_json(args.campaign/'smoke_complete.json', {'recipes': list(RECIPES), 'cuda_rng_replay': True,
                   'real_knn_batch_invariance': True, 'full_train_validation_coverage': True,
                   'full_training_loader_replay': True,
                   'data_manifest_sha256': digest(data_manifest)})
    else:
        if not (args.campaign/'smoke_complete.json').exists():raise RuntimeError('Successful GPU smoke required')
        passed = json.loads((args.campaign/'smoke_complete.json').read_text())
        if not passed.get('full_training_loader_replay') or passed['data_manifest_sha256'] != digest(data_manifest):
            raise ValueError('Full-population smoke receipt missing or data manifest changed')
        frozen = json.loads(data_manifest.read_text())
        if digest(root/'processed/meta.json') != frozen['meta_sha256']:raise ValueError('Cache metadata changed')
        name = list(RECIPES)[args.task % 4];seed = SEEDS[args.task//4]
        if (args.campaign/f'{name}_{seed}/complete.json').exists():raise FileExistsError('Completed task')
        try:run(args, name, seed, manifest, bundle, frozen)
        except BaseException as exc:
            out = args.campaign/f'{name}_{seed}'
            if out.is_dir():write_json(out/'status.json', {'state': 'failed', 'error': repr(exc)})
            raise


if __name__ == '__main__':main()
