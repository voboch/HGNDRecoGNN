"""Check the DCM ladder on a test GPU before submitting production jobs."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from HGNDRecoGNN import device, models
from HGNDRecoGNN.data.graph_dataset import HGNDGraphDataset
from HGNDRecoGNN.training.checkpoint import load_checkpoint, save_checkpoint
from HGNDRecoGNN.training.train import TrainConfig, _build_scheduler


VARIANTS = {
    'ref': (25, 'multistep', 0, {}),
    'epochs': (30, 'multistep', 0, {}),
    'sched': (30, 'cosine', 0, {}),
    'warmup': (30, 'cosine', 3, {}),
    'size': (30, 'cosine', 3, {'cluster_size_feature': True}),
    'pool': (30, 'cosine', 3, {'cluster_size_feature': True,
                              'cluster_pooling': 'mean_max_sum'}),
    'residual': (30, 'cosine', 3, {'cluster_size_feature': True,
                                  'cluster_pooling': 'mean_max_sum',
                                  'hit_residual': True}),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--batch-size', type=int, default=256)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('A test GPU is required; CPU fallback is forbidden here.')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # net_default consumes only hits/clusters, so the existing v2 DCM cache is
    # sufficient. One explicitly selected shard avoids rebuilding that cache.
    ds = HGNDGraphDataset(root=args.root, hits_csv_dir=args.root, num_shards=1,
                          allow_stale_schema=True)
    ds.preload()
    cpu_batch = next(iter(DataLoader(ds, batch_size=args.batch_size)))
    records = []
    for name, (epochs, schedule, warmup, extra) in VARIANTS.items():
        torch.manual_seed(42)
        arch = {'hidden_channels': 512, 'num_layers': 8, **extra}
        model, spec = models.get('net_default', ds, **arch)
        plan = device.plan_for_spec(model, 'cuda', spec.default_cpu_pinned)
        device.to_device(model, plan)
        graph = cpu_batch.clone().to(plan.main)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        cfg = TrainConfig(epochs=epochs, scheduler=schedule,
                          warmup_epochs=warmup, verbose=False)
        scheduler, _ = _build_scheduler(cfg, opt)
        model.train()
        before = next(model.parameters()).detach().clone()
        predictions = spec.forward_fn(model, graph)
        loss = spec.loss_fn(predictions, graph, spec.default_loss_weights)['total']
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'{name}: nonfinite loss')
        loss.backward()
        for param in model.parameters():
            if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                raise RuntimeError(f'{name}: nonfinite gradient')
        opt.step()
        scheduler.step()
        if torch.equal(before, next(model.parameters()).detach()):
            raise RuntimeError(f'{name}: first parameter did not update')
        ckpt_path = out / name / 'model.pt'
        save_checkpoint(str(ckpt_path), model, 'net_default', arch, epoch=0)
        checkpoint = load_checkpoint(str(ckpt_path))
        restored, restored_spec = models.get(checkpoint.arch_name, ds,
                                               **checkpoint.arch_kwargs)
        restored.load_state_dict(checkpoint.state_dict)
        device.to_device(restored, plan)
        model.eval()
        restored.eval()
        with torch.no_grad():
            expected = spec.forward_fn(model, graph)
            actual = restored_spec.forward_fn(restored, graph)
        if not all(torch.allclose(a, b) for a, b in zip(expected, actual)):
            raise RuntimeError(f'{name}: checkpoint round trip changed predictions')
        records.append({'variant': name, 'loss': float(loss.detach()),
                        'checkpoint': str(ckpt_path), 'round_trip': True})
        print(json.dumps(records[-1]), flush=True)
        del model, restored, opt, scheduler, predictions, loss, before
        del expected, actual, graph, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    result = {'status': 'passed', 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'batch_size': args.batch_size,
              'variants': records}
    (out / 'smoke.json').write_text(json.dumps(result, indent=2) + '\n')
    print('All seven variants passed real CUDA forward/backward and checkpoint reload.',
          flush=True)


if __name__ == '__main__':
    main()
