"""Student backbones on frozen HGND hits and connected-component candidates.

The ten HGND features, detector timing, candidates and MC targets belong to
the main pipeline. Only the backbone/pooling/heads come from the student.
Previously trained six-feature notebook checkpoints cannot be loaded here.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

from . import ModelSpec, register
from .student_ufc import MODEL_CLASSES
from ..training.losses import HeteroClusterWeights, hetero_cluster_loss

PIPELINE_MODELS = {
    "student_sage": "baseline_graphsage",
    "student_dynedge": "baseline_dynamicedgeconv",
    "student_spectral_sage": "spectral_gated_graphsage",
    "student_spectral_dynedge": "spectral_gated_dynamicedgeconv",
}


def candidate_view(graph, spectral=False):
    """Make an observable-only view with one batch ID per frozen candidate.

    Sorting keeps torch_cluster kNN batches contiguous even when hits from
    several connected components were interleaved in the original event.
    Candidate IDs follow the native cluster-node order, including singletons.
    """
    x = graph["hits"].x
    local = graph.cluster.reshape(-1).long()
    hit_batch = graph["hits"].batch
    cluster_ptr = graph["clusters"].ptr
    ids = local + cluster_ptr[hit_batch]
    n_clusters = int(graph["clusters"].num_nodes)
    counts = torch.bincount(ids, minlength=n_clusters)
    if len(counts) != n_clusters or bool((counts == 0).any()):
        raise ValueError("Frozen cluster nodes and hit membership disagree")
    order = torch.argsort(ids, stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order), device=order.device)
    edges = graph["hits", "hits"].edge_index
    # No labels, truth IDs, true energies or cluster features enter this view.
    view = Data(x=x[order], edge_index=inverse[edges], batch=ids[order],
                n_hits=counts.to(x.dtype))
    if spectral:
        if "student_pe" not in graph["hits"]:
            raise ValueError("Spectral models require topology-only student_pe")
        view.pe = graph["hits"].student_pe[order]
    return view


class StudentPipeline(nn.Module):
    def __init__(self, model_key, in_dim, **kwargs):
        super().__init__()
        self.spectral = model_key.startswith("spectral_")
        self.core = MODEL_CLASSES[model_key](in_dim=in_dim, **kwargs)

    def forward(self, graph):
        return self.core(candidate_view(graph, self.spectral))


def _forward(model, graph):
    return model(graph)


def _register(name, model_key):
    @register(name)
    def factory(dataset, **kwargs):
        model = StudentPipeline(model_key, int(dataset[0]["hits"].x.shape[1]), **kwargs)
        return model, ModelSpec(
            forward_fn=_forward, loss_fn=hetero_cluster_loss,
            default_loss_weights=HeteroClusterWeights(),
            description="Student backbone/heads on native frozen HGND candidates",
            tags=("student", "cluster", "native-hgnd-inputs"))
    return factory


for _name, _key in PIPELINE_MODELS.items():
    _register(_name, _key)
