"""Default HGND reconstruction Net — extracted verbatim from
`notebooks/preprocessing_dataloader.ipynb` cell 11.

Behavior is identical to the v1 notebook: hit-level branch runs on the main
device (MPS/CUDA/CPU), cluster-level branch is pinned to CPU because
DynamicEdgeConv has no MPS kernel. The `_cpu` suffix on cluster modules is
preserved to keep state_dict keys stable across the refactor — do not
rename them. Phase 2 will introduce a device-aware variant that uses
`HGNDRecoGNN.device.DeviceMap` for CUDA fast path.
"""

from __future__ import annotations

import torch
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq
from torch_geometric.nn import (
    BatchNorm, DynamicEdgeConv, EdgeConv, GraphConv, SAGEConv, Sequential,
)
from torch_geometric.nn.pool import avg_pool_x
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import scatter


class Net(torch.nn.Module):
    """Hit-level (EdgeConv → SAGEConv → GraphConv) + cluster-level (DynamicEdgeConv).

    Output tuple: `(link_scores, hit_scores, cluster_scores, cluster_energy,
                    cluster_link_scores, cluster_batch)` — sigmoid'd where
    applicable, matching the v1 loss expectations.
    """

    def __init__(self, hidden_channels: int, num_layers: int, num_features: int,
                 cluster_pooling: str = 'mean', cluster_size_feature: bool = False,
                 hit_residual: bool = False, hit_normalization: str = 'batch'):
        """`cluster_pooling`, `cluster_size_feature` and `hit_residual` all
        default to the v1 behaviour, so an existing checkpoint still loads and
        every published number is reproduced unchanged. See the three blocks
        below for why each one exists.

        cluster_pooling: 'mean' (v1) or 'mean_max_sum'.
            The cluster branch is fed `avg_pool_x`, i.e. the MEAN of its hits'
            embeddings, which is invariant to how many hits the cluster has.
            Measured on the SMASH defaultSpot cache, cluster size alone
            separates signal from background with ROC AUC 0.729 (signal mean
            6.53 hits, background 2.98; 48.5 % of background clusters are
            single-hit against 15.5 % of signal). A mean cannot see that
            directly -- it can only be inferred through the hit graph -- so
            max and sum are concatenated alongside it.

        cluster_size_feature: append log1p(cluster size) to the cluster vector.
            The same quantity, stated explicitly rather than left to be
            inferred. Note `graph['clusters'].x` exists in the cache but is
            filled with zeros (data/graph_dataset.py) and is never read by this
            model, so the size is computed here from the hit->cluster
            assignment instead and no cache has to be rebuilt.

        hit_residual: residual connections across the SAGEConv stack.
            At the reference setting num_layers=8 the hit branch is eight
            message-passing layers deep with no skip connection anywhere,
            which is the regime where GNN node features oversmooth. Applied
            only where the shapes already match, so it adds no parameters.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_features = num_features
        if cluster_pooling not in ('mean', 'mean_max_sum'):
            raise ValueError(
                f"cluster_pooling must be 'mean' or 'mean_max_sum', "
                f"got {cluster_pooling!r}")
        self.cluster_pooling = cluster_pooling
        self.cluster_size_feature = bool(cluster_size_feature)
        self.hit_residual = bool(hit_residual)
        if hit_normalization not in ('batch', 'layer'):
            raise ValueError(f'Unknown hit normalization: {hit_normalization}')
        self.hit_normalization = hit_normalization

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        nn = Seq(Lin(num_features * 2, 64), ReLU(),
                 Lin(64, 64), ReLU(),
                 Lin(64, 64), ReLU())
        self.convs.append(EdgeConv(nn, aggr='sum'))

        nn = Seq(Lin(128, 128), ReLU(),
                 Lin(128, 128), ReLU(),
                 Lin(128, 256), ReLU())
        self.convs.append(EdgeConv(nn, aggr='sum'))

        for _ in range(num_layers - 4):
            self.convs.append(SAGEConv(-1, hidden_channels))
            # LayerNorm uses no running statistics from training batches.
            # This is an opt-in source-shift control, not an established fix.
            norm = (BatchNorm(hidden_channels) if hit_normalization == 'batch'
                    else torch.nn.LayerNorm(hidden_channels))
            self.batch_norms.append(norm)
        self.convs.append(GraphConv(hidden_channels, hidden_channels))
        self.convs.append(GraphConv(hidden_channels, hidden_channels))

        self.edge_out = Sequential('x', [
            (Linear(hidden_channels * 2, hidden_channels), 'x -> x'),
            ReLU(inplace=True),
            Linear(hidden_channels, hidden_channels),
            ReLU(inplace=True),
            Linear(hidden_channels, 1),
        ])

        self.hitcl_out = Sequential('x', [
            (Linear(hidden_channels, hidden_channels // 2), 'x -> x'),
            ReLU(inplace=True),
            Linear(hidden_channels // 2, hidden_channels // 2),
            ReLU(inplace=True),
            Linear(hidden_channels // 2, 1),
        ])

        # Cluster-level modules — pinned to CPU on MPS (DynamicEdgeConv/knn
        # is CPU-only there). The `_cpu` suffix is kept for state_dict stability.
        # Width of one pooled cluster vector before DynamicEdgeConv, which
        # concatenates a pair of them and so sees twice this.
        n_pool = 3 if self.cluster_pooling == 'mean_max_sum' else 1
        cl_in = hidden_channels * n_pool + (1 if self.cluster_size_feature else 0)
        self.cluster_in_channels = cl_in

        nn_cl = Seq(Lin(cl_in * 2, hidden_channels * 4), ReLU(),
                    Lin(hidden_channels * 4, hidden_channels * 4), ReLU(),
                    Lin(hidden_channels * 4, hidden_channels), ReLU())
        self.cluster_conv_cpu = DynamicEdgeConv(nn_cl, k=100, aggr='sum')

        self.clclass_out_cpu = Sequential('x', [
            (Linear(hidden_channels, hidden_channels // 2), 'x -> x'),
            ReLU(inplace=True),
            Linear(hidden_channels // 2, hidden_channels // 2),
            ReLU(inplace=True),
            Linear(hidden_channels // 2, 1),
        ])

        self.clenergy_out_cpu = Sequential('x', [
            (Linear(hidden_channels, hidden_channels // 2), 'x -> x'),
            ReLU(inplace=True),
            Linear(hidden_channels // 2, hidden_channels // 2),
            ReLU(inplace=True),
            Linear(hidden_channels // 2, 1),
        ])

        self.cl_edge_out_cpu = Sequential('x', [
            (Linear(hidden_channels * 2, hidden_channels), 'x -> x'),
            ReLU(inplace=True),
            Linear(hidden_channels, hidden_channels),
            ReLU(inplace=True),
            Linear(hidden_channels, 1),
        ])

    def to(self, device):
        """Move the model to `device`, pinning the cluster branch only on MPS.

        DynamicEdgeConv has no MPS kernel, so on Apple silicon the cluster
        branch has to run on CPU.  That routing was previously applied on every
        device, with two costs on CUDA: the GPU sat idle while the most
        expensive layer ran on CPU, which on a shared cluster gets the job
        reaped by the idle-GPU watchdog, and the hit features had to be
        detached to cross the device boundary, so the score and energy heads
        could not shape the representation they consume.

        On CUDA and CPU the branch now stays with the rest of the model. The
        `_cpu` suffixes are kept because they are baked into existing
        checkpoints' state_dict keys.
        """
        super().to(device)
        if torch.device(device).type == 'mps':
            self.cluster_conv_cpu.cpu()
            self.clclass_out_cpu.cpu()
            self.clenergy_out_cpu.cpu()
            self.cl_edge_out_cpu.cpu()
        return self

    def _cluster_split_from(self, x: torch.Tensor) -> bool:
        """Whether the cluster branch sits on a different device from the hits.

        Compared against the hit features rather than tested for `cpu`, so a
        CPU-only run keeps the gradient: there is no device boundary to cross
        there, and detaching would preserve the MPS workaround where it costs
        something and buys nothing.
        """
        return next(self.cluster_conv_cpu.parameters()).device != x.device

    def _augment_cluster_features(self, cl_mean, cl_idx, cl_feat):
        """Concatenate max/sum pooling and the cluster size onto the mean.

        Row alignment is the whole difficulty here. `avg_pool_x` does not index
        its output by the raw cluster id: it first RELABELS the ids to be
        consecutive (`consecutive_cluster`) and returns one row per unique
        cluster in that new order. Scattering with the raw ids therefore
        misaligns every row as soon as the ids are non-consecutive or do not
        start at zero -- silently, whenever max(id) happens to fit inside the
        row count. The same relabelling is applied here so that row i of each
        extra pool belongs to the same cluster as row i of the mean.

        With both flags off this returns `cl_mean` untouched.
        """
        if self.cluster_pooling == 'mean' and not self.cluster_size_feature:
            return cl_mean

        idx, _ = consecutive_cluster(cl_idx)
        n = cl_mean.size(0)
        parts = [cl_mean]
        if self.cluster_pooling == 'mean_max_sum':
            parts.append(scatter(cl_feat, idx, dim=0, dim_size=n, reduce='max'))
            parts.append(scatter(cl_feat, idx, dim=0, dim_size=n, reduce='sum'))
        if self.cluster_size_feature:
            ones = torch.ones(cl_feat.size(0), 1,
                              dtype=cl_feat.dtype, device=cl_feat.device)
            size = scatter(ones, idx, dim=0, dim_size=n, reduce='sum')
            # log1p: sizes are heavy-tailed, and a raw count would enter the
            # first Linear on a different scale from the standardised features.
            parts.append(torch.log1p(size))
        return torch.cat(parts, dim=-1)

    def forward(self, x, edge_index, edge_index_cl, clusters, batch):
        x = self.convs[0](x, edge_index).relu()
        x = self.convs[1](x, edge_index).relu()

        for conv, bn in zip(self.convs[2:-2], self.batch_norms):
            h = bn(conv(x, edge_index)).relu()
            # Only where the width already matches: the first SAGEConv widens
            # 256 -> hidden_channels and has nothing to add to.
            x = x + h if (self.hit_residual and h.shape == x.shape) else h

        row, col = edge_index
        new_edge_attr = self.edge_out(torch.cat([x[row], x[col]], dim=-1))
        new_edge_attr = torch.sigmoid(new_edge_attr).squeeze(1)

        x = self.convs[-2](x, edge_index, new_edge_attr)
        x = x.relu()
        x = self.convs[-1](x, edge_index)
        x = x.relu()

        # Cluster branch. When it is pinned to CPU (MPS only, since
        # DynamicEdgeConv has no MPS kernel) the hit features must be detached
        # to cross the device boundary, and the branch then trains only on its
        # own losses. When it shares the device with the hit branch the
        # gradient flows through, so the score and energy heads shape the hit
        # representation they consume — which is what "end-to-end" should mean.
        _split = self._cluster_split_from(x)
        if _split:
            cl_idx, cl_feat, cl_b = clusters.cpu(), x.detach().cpu(), batch.cpu()
        else:
            cl_idx, cl_feat, cl_b = clusters, x, batch
        cl_x, cl_batch = avg_pool_x(cl_idx, cl_feat, cl_b)
        cl_x = self._augment_cluster_features(cl_x, cl_idx, cl_feat)
        cl_x = self.cluster_conv_cpu(cl_x, cl_batch)

        cl_cl = self.clclass_out_cpu(cl_x).sigmoid()
        cl_e  = self.clenergy_out_cpu(cl_x)

        row_cl, col_cl = edge_index_cl.cpu() if _split else edge_index_cl
        cl_connection = self.cl_edge_out_cpu(
            torch.cat([cl_x[row_cl], cl_x[col_cl]], dim=-1)
        )
        cl_connection = torch.sigmoid(cl_connection).squeeze(1)

        x = self.hitcl_out(x)
        x = torch.sigmoid(x)

        _dev = x.device
        return (new_edge_attr,
                x.squeeze(1),
                cl_cl.squeeze(1).to(_dev),
                cl_e.squeeze(1).to(_dev),
                cl_connection.to(_dev),
                cl_batch.to(_dev))


def build_default_net(dataset, hidden_channels: int = 512,
                      num_layers: int = 8, **arch_kwargs) -> Net:
    """Construct and lazily-initialise the default Net.

    PyG's `SAGEConv(-1, ...)` defers parameter shape inference to the first
    forward pass. We do one dummy pass on CPU with `dataset[0]` so the
    optimizer sees fully-materialised parameters.
    """
    num_features = dataset[0]['hits'].num_features
    model = Net(hidden_channels, num_layers, num_features, **arch_kwargs)

    with torch.no_grad():
        g0 = dataset[0].clone()
        cl0 = g0.cluster.squeeze(-1) + 1
        b0 = torch.zeros(g0['hits'].x.size(0), dtype=torch.long)
        model.eval()
        model(
            g0['hits'].x,
            g0['hits', 'hits'].edge_index,
            g0['clusters', 'clusters'].edge_index,
            cl0, b0,
        )
    model.train()
    return model
