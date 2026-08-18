"""Inference / prediction export.

Extracted from `notebooks/preprocessing_dataloader.ipynb` cell 10. Returns
the same three DataFrames (per-hit, per-cluster, per-edge) with the same
column names so `notebooks/results_smash.ipynb` keeps working unchanged.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import pandas as pd
import torch
from tqdm import tqdm

from .. import device as device_mod
from .train import _forward_default_net


@torch.no_grad()
def predict(
    model,
    loader,
    plan: Optional[device_mod.DeviceMap] = None,
    forward_fn: Callable = _forward_default_net,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run `model` on `loader` and return three DataFrames.

    Returns dict with keys `hits`, `clusters`, `edges`. Column layout is
    identical to the v1 notebook:
      hits:     Row, Instance, ClusterID, score, label, istop
      clusters: Row, ClusterID, cl_score, e_pred, cl_istop, cl_label, e_true
      edges:    link_score, link_label
    """
    if plan is None:
        plan = device_mod.plan_for(model, None, None)

    all_hit_rows:  list[int]   = []
    all_hit_inst:  list[int]   = []
    all_hit_cl:    list[int]   = []
    all_hit_score: list[float] = []
    all_hit_label: list[int]   = []
    all_hit_istop: list[int]   = []

    all_link_score: list[float] = []
    all_link_label: list[int]   = []

    all_cl_rows:  list[int]   = []
    all_cl_id:    list[int]   = []
    all_cl_score: list[float] = []
    all_cl_label: list[int]   = []
    all_cl_epred: list[float] = []
    all_cl_etrue: list[float] = []
    all_cl_istop: list[int]   = []

    model.eval()
    t0 = time.time()
    iterator = tqdm(loader, desc='predict') if verbose else loader

    for graph in iterator:
        graph = graph.to(plan.main)
        preds = forward_fn(model, graph)
        link_scores, hit_scores, cl_scores, cl_energy, _cl_link, cl_batch = preds

        hit_batch = graph.batch_dict['hits']
        n_graphs  = int(hit_batch.max().item()) + 1
        _, hit_counts = hit_batch.unique(return_counts=True)

        for gi in range(n_graphs):
            n_h = int(hit_counts[gi].item())
            all_hit_rows.extend([int(graph.Row[gi].item())] * n_h)
            all_hit_istop.extend([int(graph.istop[gi].item())] * n_h)
            all_hit_inst.extend(range(n_h))

        all_hit_cl.extend(graph.cluster.squeeze(-1).cpu().tolist())
        all_hit_label.extend(graph['hits'].y.cpu().long().tolist())
        all_hit_score.extend(hit_scores.cpu().tolist())

        all_link_label.extend(graph.true_links.cpu().long().tolist())
        all_link_score.extend(link_scores.cpu().tolist())

        _, cl_counts = cl_batch.unique(return_counts=True)
        for gi in range(n_graphs):
            n_c = int(cl_counts[gi].item())
            all_cl_rows.extend([int(graph.Row[gi].item())] * n_c)
            all_cl_istop.extend([int(graph.istop[gi].item())] * n_c)
            all_cl_id.extend(range(n_c))

        all_cl_label.extend(graph.cllabel.cpu().long().tolist())
        all_cl_score.extend(cl_scores.cpu().tolist())
        all_cl_etrue.extend(graph.clenergy.cpu().float().tolist())
        all_cl_epred.extend(cl_energy.cpu().tolist())

    elapsed = time.time() - t0
    if verbose:
        print(f'predict(): {len(all_hit_score)} hits  '
              f'{len(all_cl_score)} clusters  {len(all_link_score)} edges  '
              f'({elapsed:.1f}s)')

    hits_df = pd.DataFrame({
        'Row':       all_hit_rows,
        'Instance':  all_hit_inst,
        'ClusterID': all_hit_cl,
        'score':     all_hit_score,
        'label':     all_hit_label,
        'istop':     all_hit_istop,
    }).astype({'Row': int, 'Instance': int, 'ClusterID': int, 'istop': int})

    clusters_df = pd.DataFrame({
        'Row':       all_cl_rows,
        'ClusterID': all_cl_id,
        'cl_score':  all_cl_score,
        'e_pred':    all_cl_epred,
        'cl_istop':  all_cl_istop,
        'cl_label':  all_cl_label,
        'e_true':    all_cl_etrue,
    }).astype({'Row': int, 'ClusterID': int, 'cl_istop': int, 'cl_label': int})

    edges_df = pd.DataFrame({
        'link_score': all_link_score,
        'link_label': all_link_label,
    })

    return {'hits': hits_df, 'clusters': clusters_df, 'edges': edges_df}


def save_predictions(preds: dict[str, pd.DataFrame], out_dir: str,
                     suffix: str = 'smash') -> dict[str, str]:
    """Pickle the three DataFrames using v1 filenames (`pred_*_<suffix>.pkl`)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for key, df in preds.items():
        p = os.path.join(out_dir, f'pred_{key}_{suffix}.pkl')
        df.to_pickle(p)
        paths[key] = p
    return paths
