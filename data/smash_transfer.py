"""Native DCM graph/feature view of hash-verified SMASH observed hits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
from torch_geometric.nn import radius_graph

from .graph_dataset import FEATURES, _dfToF

# The transfer view preserves DCM's trained graph recipe. SMASH's 1.5 ns
# audit candidates are a separate diagnostic and cannot replace this window.
RADIUS_CELLS = 3.6
TIME_NS = 1.0


def transfer_pack(frame, scalers, affine_a, affine_b, geometry_bounds):
    """Keep per-event/arm namespaces, local cluster/edge IDs, and hit order."""
    if not len(frame):
        return None
    rows = frame.Row.to_numpy(np.int64)
    sides = (frame.fY.to_numpy() > 0).astype(np.int64)
    order = np.lexsort((np.arange(len(rows)), rows, sides))
    frame = frame.iloc[order].copy()
    rows, sides = rows[order], sides[order]
    keys = pd.MultiIndex.from_arrays([sides, rows])
    batch, uniques = pd.factorize(keys, sort=False)
    node_lengths = np.bincount(batch)
    node_ptr = np.r_[0, np.cumsum(node_lengths)]
    pos = torch.tensor(frame[['LayerId', 'RowId', 'ColumnId']].to_numpy(np.float32))
    times = torch.tensor(frame[['fTime']].to_numpy(np.float32))
    batch_t = torch.tensor(batch)
    spatial = radius_graph(pos, r=RADIUS_CELLS, batch=batch_t, loop=True, max_num_neighbors=32)
    temporal = radius_graph(times, r=TIME_NS, batch=batch_t, loop=True, max_num_neighbors=32)
    stride = len(frame)+1
    edges = spatial[:, torch.isin(spatial[0]*stride+spatial[1], temporal[0]*stride+temporal[1])].numpy()
    if edges.size and not np.array_equal(batch[edges[0]], batch[edges[1]]):
        raise ValueError('Inter-event edges in transfer view')
    edge_batch = batch[edges[0]]
    edge_order = np.argsort(edge_batch, kind='stable')
    edges, edge_batch = edges[:, edge_order], edge_batch[edge_order]
    edge_lengths = np.bincount(edge_batch, minlength=len(node_lengths))
    edge_ptr = np.r_[0, np.cumsum(edge_lengths)]
    adj = csr_matrix((np.ones(edges.shape[1]), edges), shape=(len(frame), len(frame)))
    nc, labels = connected_components(adj, directed=False)
    cluster_batch = np.full(nc, -1, dtype=np.int64)
    cluster_batch[labels] = batch
    cluster_lengths = np.bincount(cluster_batch, minlength=len(node_lengths))
    cluster_ptr = np.r_[0, np.cumsum(cluster_lengths)]
    local_cluster = labels - cluster_ptr[batch]
    local_edges = edges - node_ptr[edge_batch]
    distance = np.linalg.norm(frame[['fX', 'fY', 'fZ']].to_numpy()/100., axis=1)
    frame['eToF'] = _dfToF(distance, frame.fTime.to_numpy())
    frame['SurfaceHit'] = False
    for col, (lo, hi) in geometry_bounds.items():
        if not frame[col].between(lo, hi).all():
            raise ValueError('SMASH cell outside verified DCM geometry')
        frame['SurfaceHit'] |= (frame[col] == lo) | (frame[col] == hi)
    x = np.empty((len(frame), len(FEATURES)), dtype=np.float32)
    for side in (0, 1):
        mask = sides == side
        # Match DCM cache float32 storage followed by its frozen train-only
        # affine remap, rather than silently fitting a new SMASH scaler.
        if mask.any():
            standardized = scalers[side].transform(frame.loc[mask, FEATURES[:-1]]).astype(np.float32)
            x[mask, :-1] = standardized * np.asarray(affine_a[side], np.float32) + np.asarray(affine_b[side], np.float32)
    x[:, -1] = frame.SurfaceHit.to_numpy(np.float32)
    ek = frame.Ekin.to_numpy()
    prompt = ((frame.PDG.abs().to_numpy() == 2112) & frame.Side.isin([5, 6]).to_numpy()
              & (ek > .3) & (ek >= _dfToF(distance, frame.fTime.to_numpy()+.3))
              & (ek <= _dfToF(distance, frame.fTime.to_numpy()-.3)))
    cl_label = np.zeros(nc, dtype=np.float32)
    cl_energy = np.zeros(nc, dtype=np.float32)
    np.maximum.at(cl_label, labels, prompt)
    np.maximum.at(cl_energy, labels, np.where(prompt, ek, 0))
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transfer features')
    return {'x': torch.tensor(x), 'pe': torch.zeros((len(x), 4)),
            'edges': torch.tensor(local_edges), 'cluster': torch.tensor(local_cluster),
            'label': torch.tensor(cl_label), 'energy': torch.tensor(cl_energy),
            'row': torch.tensor([int(key[1]) for key in uniques]),
            'side': torch.tensor([int(key[0]) for key in uniques]),
            'index': torch.arange(len(uniques)), 'node_ptr': torch.tensor(node_ptr),
            'edge_ptr': torch.tensor(edge_ptr), 'cluster_ptr': torch.tensor(cluster_ptr)}
