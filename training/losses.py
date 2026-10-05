"""Loss functions for HGND models.

Each model type has its own composite loss. Currently:

- `default_net_loss` — the v1 4-way BCE + 1 MSE for the default Net.
  Inputs are already sigmoid'd inside the model, so BCE (not BCEWithLogits).
- `hetero_cluster_loss` — cluster-level BCEWithLogits + masked SmoothL1
  for the vkr26 HeteroGNN family (sage / gat / hgt).
- `spectral_hit_loss` — hit-level BCEWithLogits + hinge for thesis
  spectral-gated models (added in models/spectral.py port).
- `build_energy_flatten_weights` — per-bin weights that flatten the cluster
  energy spectrum, for the reference notebook's second-stage energy loss.

The training loop picks the right loss via each model's `ModelSpec.loss_fn`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ── Weight dataclasses ──────────────────────────────────────────────────

@dataclass
class LossWeights:
    """Default Net (4-way BCE + 1 MSE)."""
    hit: float = 1.0
    link: float = 1.0
    cluster_cls: float = 1.0
    cluster_energy: float = 1.0
    cluster_link: float = 1.0

    # Spectrum-flattened energy loss, the reference notebook's second training
    # stage (EgdeClass_Cluster_Jan25-xecs.ipynb cells 63 and 74-79). Plain MSE
    # on a falling spectrum is dominated by the populated low-energy bins, and
    # the energy head then pulls high-energy clusters down: the converged
    # checkpoint is within +/-1 % to 2.6 GeV and drifts to -9.4 % at 4-5 GeV.
    # Reweighting each cluster by the inverse occupancy of its true-energy bin
    # removes that pull. The high band of the hardness ratio has no upper edge,
    # so it is exactly the region this affects.
    #
    # Off by default: every result in reports/ predates it, and turning it on
    # silently would make them incomparable. Set `energy_flatten=True` and
    # attach weights from `build_energy_flatten_weights`.
    energy_flatten: bool = False
    energy_bin_edges: torch.Tensor | None = None
    energy_bin_weights: torch.Tensor | None = None


@dataclass
class HeteroClusterWeights:
    """vkr26 HeteroGNN (cluster-level cls + energy)."""
    cluster_cls: float = 1.0
    cluster_energy: float = 0.5   # λ_E from vkr26


# ── Spectrum flattening ─────────────────────────────────────────────────

def build_energy_flatten_weights(
    energies,
    bins: int = 10,
    eps: float = 1e-6,
    min_count: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-bin weights that flatten a cluster-energy spectrum.

    Follows the reference notebook: histogram the *positive* true energies into
    `bins`, then weight each bin by the count a flat spectrum would have put
    there divided by the count actually there. A bin holding a tenth of the
    clusters it would hold under a flat spectrum gets ten times the weight.

    Build this once from the training set, not per batch — a batch of 256
    graphs holds too few high-energy clusters for its own occupancy to be
    anything but noise, and weights that move between batches turn into
    gradient noise on the head they are meant to correct.

    Returns
    -------
    (edges, weights)
        `edges` has `bins + 1` entries, `weights` has `bins`. Pass both to
        `LossWeights(energy_flatten=True, energy_bin_edges=edges,
        energy_bin_weights=weights)`.
    """
    e = torch.as_tensor(energies, dtype=torch.float32).flatten()
    e = e[e > 0]
    if e.numel() == 0:
        raise ValueError('no positive cluster energies to build weights from')

    counts = torch.histc(e, bins=bins, min=float(e.min()), max=float(e.max()))
    edges = torch.linspace(float(e.min()), float(e.max()), bins + 1)
    target = e.numel() / bins
    w = target / (counts + eps)

    # Cap the weight a sparsely populated bin can earn. Exact flattening gives
    # every bin an equal share of the loss regardless of how many clusters it
    # holds, which is the intent in the bulk and a trap in the tail: on the DCM
    # sample the top two bins hold 29 of 40,225 signal clusters, 0.07 %, and
    # would carry 20 % of the energy loss at a weight ratio of 1461. The
    # gradient of the energy head would then be set by a couple of dozen
    # examples. Clipping at the weight a bin of `min_count` clusters would earn
    # keeps the reweighting in the regime where it is estimating something.
    if min_count > 0:
        w = torch.clamp(w, max=float(target / min_count))
        # Clipping removes weight without giving it back, which would quietly
        # shrink the energy term inside the composite loss and change the
        # balance against the classification terms. Renormalise so the mean
        # weight over the training spectrum stays 1, as it is by construction
        # without the cap: the reweighting then redistributes emphasis within
        # the energy term instead of rescaling it.
        idx = torch.bucketize(e, edges[1:-1])
        mean_w = w[idx].mean()
        if float(mean_w) > 0:
            w = w / mean_w
    return edges, w


def _energy_flatten_weights(e_true: torch.Tensor,
                            weights: 'LossWeights') -> torch.Tensor | None:
    """Look up each cluster's bin weight, or None when flattening is off."""
    if not weights.energy_flatten:
        return None
    edges = weights.energy_bin_edges
    bw = weights.energy_bin_weights
    if edges is None or bw is None:
        raise ValueError(
            'energy_flatten=True needs energy_bin_edges and '
            'energy_bin_weights; build them with build_energy_flatten_weights')

    edges = edges.to(e_true.device)
    bw = bw.to(e_true.device)
    # torch.bucketize on the interior edges reproduces np.digitize(edges[1:-1]),
    # so energies below the first edge and above the last fall in the end bins
    # instead of indexing out of range.
    idx = torch.bucketize(e_true.detach(), edges[1:-1])
    return bw[idx]


# ── Losses ──────────────────────────────────────────────────────────────

def default_net_loss(
    preds: tuple,
    graph,
    weights: LossWeights = LossWeights(),
) -> dict[str, torch.Tensor]:
    """Composite loss for the default Net. See models/default_net.py."""
    link_scores, hit_scores, cl_scores, cl_energy, cl_link_scores, _cl_batch = preds

    hit_loss = F.binary_cross_entropy(hit_scores, graph['hits'].y.float())
    link_loss = F.binary_cross_entropy(link_scores, graph.true_links.float())

    cl_mask = graph.cllabel.bool()
    cluster_cls_loss = F.binary_cross_entropy(cl_scores, graph.cllabel.float())
    if cl_mask.any():
        e_pred = cl_energy[cl_mask]
        e_true = graph.clenergy[cl_mask].float()
        w = _energy_flatten_weights(e_true, weights)
        if w is None:
            cluster_energy_loss = F.mse_loss(e_pred, e_true)
        else:
            cluster_energy_loss = (w * (e_pred - e_true) ** 2).mean()
    else:
        cluster_energy_loss = torch.zeros((), device=cl_energy.device)

    conn_mask = graph['clusters'].connection_mask.bool()
    if conn_mask.any():
        cluster_link_loss = F.binary_cross_entropy(
            cl_link_scores[conn_mask],
            graph['clusters'].y[conn_mask].float(),
        )
    else:
        cluster_link_loss = torch.zeros((), device=cl_link_scores.device)

    total = (weights.hit             * hit_loss
             + weights.link          * link_loss
             + weights.cluster_cls   * cluster_cls_loss
             + weights.cluster_energy * cluster_energy_loss
             + weights.cluster_link  * cluster_link_loss)

    return {
        'hit': hit_loss,
        'link': link_loss,
        'cluster_cls': cluster_cls_loss,
        'cluster_energy': cluster_energy_loss,
        'cluster_link': cluster_link_loss,
        'total': total,
    }


def hetero_cluster_loss(
    preds: tuple,
    graph,
    weights: HeteroClusterWeights = HeteroClusterWeights(),
) -> dict[str, torch.Tensor]:
    """Loss for HeteroGNN family — (cls_logits, e_pred) on cluster nodes.

    Matches vkr26 masked multitask: BCEWithLogits over all clusters, plus
    SmoothL1 on the signal-only subset (y_cls > 0.5). Energy target is
    `graph.clenergy` (in GeV, from _build_single_graph).
    """
    cls_logits, e_pred = preds
    y_cls = graph.cllabel.float()
    y_e   = graph.clenergy.float()

    cluster_cls_loss = F.binary_cross_entropy_with_logits(cls_logits, y_cls)

    mask = y_cls.bool()
    if mask.any():
        cluster_energy_loss = F.smooth_l1_loss(e_pred[mask], y_e[mask])
    else:
        cluster_energy_loss = torch.zeros((), device=e_pred.device)

    total = (weights.cluster_cls    * cluster_cls_loss
             + weights.cluster_energy * cluster_energy_loss)

    return {
        'cluster_cls': cluster_cls_loss,
        'cluster_energy': cluster_energy_loss,
        'total': total,
    }
