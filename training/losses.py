"""Loss functions for HGND models.

Each model type has its own composite loss. Currently:

- `default_net_loss` — the v1 4-way BCE + 1 MSE for the default Net.
  Inputs are already sigmoid'd inside the model, so BCE (not BCEWithLogits).
- `hetero_cluster_loss` — cluster-level BCEWithLogits + masked SmoothL1
  for the vkr26 HeteroGNN family (sage / gat / hgt).
- `spectral_hit_loss` — hit-level BCEWithLogits + hinge for thesis
  spectral-gated models (added in models/spectral.py port).

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


@dataclass
class HeteroClusterWeights:
    """vkr26 HeteroGNN (cluster-level cls + energy)."""
    cluster_cls: float = 1.0
    cluster_energy: float = 0.5   # λ_E from vkr26


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
        cluster_energy_loss = F.mse_loss(cl_energy[cl_mask],
                                         graph.clenergy[cl_mask].float())
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
