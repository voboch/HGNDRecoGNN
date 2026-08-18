"""Adapters that shim HGNDRecoGNN's HeteroData into the shapes older
models expect.

Currently provides:

- `compute_lap_pe_per_cluster(x, edge_index, cluster_ids, k_pe=4)` —
  per-cluster Laplacian positional encoding used by the thesis
  spectral-gated models. Since HGNDRecoGNN clusters are connected
  components of the hit-hit graph, the full Laplacian is block-diagonal
  and each block's eigenvectors are independent. We loop over clusters
  in Python for clarity; a batched scipy-sparse variant is a Phase 2b
  optimization if this becomes a bottleneck.
"""

from __future__ import annotations

import numpy as np
import torch


def _lap_pe_dense(A: np.ndarray, k_pe: int) -> np.ndarray:
    """Ported from GNN_for_neutron_reconstruction — sign-fixed LapPE."""
    n = A.shape[0]
    if n <= 1 or k_pe <= 0:
        return np.zeros((n, k_pe), dtype=np.float32)

    deg = A.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    m = deg > 0
    inv_sqrt[m] = 1.0 / np.sqrt(deg[m])
    D = np.diag(inv_sqrt)
    L = np.eye(n, dtype=np.float64) - D @ A @ D

    evals, evecs = np.linalg.eigh(L)
    # Skip the trivial (constant) eigenvector at index 0.
    pe = evecs[:, 1:1 + k_pe]

    # Sign-fix each eigenvector by its largest-magnitude entry.
    for j in range(pe.shape[1]):
        idx = int(np.argmax(np.abs(pe[:, j])))
        if pe[idx, j] < 0:
            pe[:, j] *= -1.0

    # Right-pad if fewer eigenvalues than k_pe (small clusters).
    if pe.shape[1] < k_pe:
        pad = np.zeros((n, k_pe - pe.shape[1]), dtype=np.float64)
        pe = np.concatenate([pe, pad], axis=1)

    return pe.astype(np.float32)


def compute_lap_pe_per_cluster(
    edge_index: torch.Tensor,
    cluster_ids: torch.Tensor,
    n_hits: int,
    k_pe: int = 4,
) -> torch.Tensor:
    """Per-cluster Laplacian PE for all hits in a batched graph.

    Parameters
    ----------
    edge_index : (2, E) LongTensor
        Hit-hit edges of the batched graph. Cross-cluster edges are ignored
        (they shouldn't exist because clusters are connected components,
        but we guard against inconsistency).
    cluster_ids : (n_hits,) LongTensor
        Per-hit cluster id (must be contiguous 0..K-1 across the batch).
    n_hits : int
    k_pe : int

    Returns
    -------
    (n_hits, k_pe) float32 tensor on CPU. Callers should `.to(device)` it.
    """
    if n_hits == 0:
        return torch.zeros((0, k_pe), dtype=torch.float32)

    ei_np = edge_index.detach().cpu().numpy() if edge_index.numel() > 0 \
            else np.zeros((2, 0), dtype=np.int64)
    cl_np = cluster_ids.detach().cpu().numpy()

    pe_full = np.zeros((n_hits, k_pe), dtype=np.float32)

    # Group hit indices by cluster id.
    order = np.argsort(cl_np, kind='stable')
    sorted_cl = cl_np[order]
    boundaries = np.concatenate([
        [0],
        np.where(np.diff(sorted_cl) != 0)[0] + 1,
        [len(sorted_cl)],
    ])

    # Reverse index: hit id → position within its cluster.
    hit_to_local = np.empty(n_hits, dtype=np.int64)

    for b in range(len(boundaries) - 1):
        cluster_hit_ids = order[boundaries[b]:boundaries[b + 1]]
        n_c = len(cluster_hit_ids)
        if n_c <= 1:
            continue
        # local ordering
        for local_i, gid in enumerate(cluster_hit_ids):
            hit_to_local[gid] = local_i
        # subgraph edges: keep those where both endpoints are in this cluster
        src, dst = ei_np
        mask = np.isin(src, cluster_hit_ids) & np.isin(dst, cluster_hit_ids)
        if not mask.any():
            continue
        sub_src = hit_to_local[src[mask]]
        sub_dst = hit_to_local[dst[mask]]
        A = np.zeros((n_c, n_c), dtype=np.float64)
        A[sub_src, sub_dst] = 1.0
        A[sub_dst, sub_src] = 1.0
        np.fill_diagonal(A, 0.0)

        pe = _lap_pe_dense(A, k_pe)
        pe_full[cluster_hit_ids] = pe

    return torch.from_numpy(pe_full)


def batched_cluster_index(graph) -> tuple[torch.Tensor, int]:
    """Return per-hit global cluster id (contiguous 0..K-1) and cluster count.

    Uses `graph.cluster` (per-hit local id) + `graph.batch_dict['hits']`
    to build a batch-global cluster id space so that scatter operations
    treat clusters from different events as distinct.
    """
    from ..training.train import reindex_clusters
    ids_1based = reindex_clusters(graph)   # returns ids starting at 1
    ids_0based = (ids_1based - 1).long()
    n_clusters = int(ids_0based.max().item()) + 1 if ids_0based.numel() else 0
    return ids_0based, n_clusters
