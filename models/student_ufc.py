"""Faithful local ports of Billa Lewis L's six UFC notebook architectures.

Source: GNN-Based Neutron Reconstruction in High-Granularity Detector Data
(2026). See student_ufc_sources.json for notebook hashes and syntax repairs.
State-dict module names are preserved. These models consume homogeneous UFC
candidate graphs; they do not replace the heterogeneous default Net.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    AttentionalAggregation, DynamicEdgeConv, GATv2Conv, SAGEConv,
    global_add_pool, global_max_pool, global_mean_pool,
)


class BaselineResidualSAGEBlock(nn.Module):

    def __init__(self, hidden_dim, dropout=0.1, aggr='mean'):
        super().__init__()
        self.conv = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr=aggr)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        return self.norm(x + h)


class BaselineClusterGNN_GraphSAGE_UFC(nn.Module):

    def __init__(self, in_dim, hidden=128, num_layers=4, dropout=0.1, aggr='mean'):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.aggr = aggr
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([BaselineResidualSAGEBlock(hidden_dim=hidden, dropout=dropout, aggr=aggr) for _ in range(num_layers)])
        self.attn_pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)))
        pooled_dim = hidden * 4
        reg_in_dim = pooled_dim + 1
        self.shared_cls = nn.Sequential(nn.Linear(pooled_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.shared_reg = nn.Sequential(nn.Linear(reg_in_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.cls_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.reg_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else x.new_zeros(x.size(0), dtype=torch.long)
        n_hits = data.n_hits.view(-1, 1).float()
        log_n_hits = torch.log1p(n_hits)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, edge_index)
        g_attn = self.attn_pool(h, batch)
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g_add = global_add_pool(h, batch)
        g = torch.cat([g_attn, g_mean, g_max, g_add], dim=1)
        h_cls = self.shared_cls(g)
        g_reg = torch.cat([g, log_n_hits], dim=1)
        h_reg = self.shared_reg(g_reg)
        logit = self.cls_head(h_cls).squeeze(-1)
        e_pred = F.softplus(self.reg_head(h_reg).squeeze(-1))
        return (logit, e_pred)


class BaselineResidualGATv2Block(nn.Module):

    def __init__(self, hidden_dim, heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % heads == 0, 'hidden_dim must be divisible by heads'
        self.conv = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads, concat=True, dropout=dropout, add_self_loops=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        return self.norm(x + h)


class BaselineClusterGNN_GATv2_UFC(nn.Module):

    def __init__(self, in_dim, hidden=128, num_layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.heads = int(heads)
        self.dropout_p = float(dropout)
        assert self.hidden % self.heads == 0
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([BaselineResidualGATv2Block(hidden_dim=hidden, heads=heads, dropout=dropout) for _ in range(num_layers)])
        self.attn_pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)))
        pooled_dim = hidden * 4
        reg_in_dim = pooled_dim + 1
        self.shared_cls = nn.Sequential(nn.Linear(pooled_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.shared_reg = nn.Sequential(nn.Linear(reg_in_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.cls_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.reg_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else x.new_zeros(x.size(0), dtype=torch.long)
        n_hits = data.n_hits.view(-1, 1).float()
        log_n_hits = torch.log1p(n_hits)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, edge_index)
        g_attn = self.attn_pool(h, batch)
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g_add = global_add_pool(h, batch)
        g = torch.cat([g_attn, g_mean, g_max, g_add], dim=1)
        h_cls = self.shared_cls(g)
        g_reg = torch.cat([g, log_n_hits], dim=1)
        h_reg = self.shared_reg(g_reg)
        logit = self.cls_head(h_cls).squeeze(-1)
        e_pred = F.softplus(self.reg_head(h_reg).squeeze(-1))
        return (logit, e_pred)


class BaselineResidualDynamicEdgeConvBlock(nn.Module):

    def __init__(self, hidden_dim, k=8, dropout=0.1):
        super().__init__()
        self.k = int(k)
        nn_edge = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.conv = DynamicEdgeConv(nn=nn_edge, k=self.k, aggr='max')
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        h = self.conv(x, batch)
        h = self.dropout(h)
        return self.norm(x + h)


class BaselineClusterGNN_DynamicEdgeConv_UFC(nn.Module):
    """
    Final baseline Dynamic EdgeConv:
      - UFC segmentation defines cluster candidates
      - DynamicEdgeConv recomputes local neighborhoods inside each cluster
      - no spectral gate
      - no LapPE use in forward
      - EventRank handled in training loss
      - global_add_pool included
      - log1p(n_hits) injected into regression head
    """

    def __init__(self, in_dim, hidden=128, num_layers=4, k=8, dropout=0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.k = int(k)
        self.dropout_p = float(dropout)
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([BaselineResidualDynamicEdgeConvBlock(hidden_dim=hidden, k=k, dropout=dropout) for _ in range(num_layers)])
        self.attn_pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)))
        pooled_dim = hidden * 4
        reg_in_dim = pooled_dim + 1
        self.shared_cls = nn.Sequential(nn.Linear(pooled_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.shared_reg = nn.Sequential(nn.Linear(reg_in_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.cls_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.reg_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, data):
        x = data.x
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else x.new_zeros(x.size(0), dtype=torch.long)
        n_hits = data.n_hits.view(-1, 1).float()
        log_n_hits = torch.log1p(n_hits)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, batch)
        g_attn = self.attn_pool(h, batch)
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g_add = global_add_pool(h, batch)
        g = torch.cat([g_attn, g_mean, g_max, g_add], dim=1)
        h_cls = self.shared_cls(g)
        g_reg = torch.cat([g, log_n_hits], dim=1)
        h_reg = self.shared_reg(g_reg)
        logit = self.cls_head(h_cls).squeeze(-1)
        e_pred = F.softplus(self.reg_head(h_reg).squeeze(-1))
        return (logit, e_pred)


class ResidualSAGEBlock(nn.Module):

    def __init__(self, hidden_dim, dropout=0.1, aggr='mean'):
        super().__init__()
        self.conv = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr=aggr)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        return self.norm(x + h)


class SpectralGate_spectral_gated_graphsage(nn.Module):
    """
    Pure spectral gate:

        z_i = |LapPE_i|
        g_i = tanh(MLP_gate(z_i))
        x_tilde_i = x_base_i * (1 + g_i)
    """

    def __init__(self, pe_dim, hidden_dim, pe_hidden=64, dropout=0.1, gate_init_std=0.001):
        super().__init__()
        self.pe_dim = int(pe_dim)
        self.hidden_dim = int(hidden_dim)
        self.gate_init_std = float(gate_init_std)
        if self.pe_dim <= 0:
            raise ValueError('SpectralMultiplicativeGate requires pe_dim > 0.')
        self.gate = nn.Sequential(nn.Linear(self.pe_dim, pe_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(pe_hidden, hidden_dim))
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=self.gate_init_std)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, h_base, pe):
        z = pe.abs()
        g = torch.tanh(self.gate(z))
        h = h_base * (1.0 + g)
        return (h, g)


class ClusterGNN_GraphSAGE_UFC_SpectralGated(nn.Module):

    def __init__(self, in_dim, pe_dim, hidden=128, pe_hidden=64, num_layers=4, dropout=0.1, aggr='mean', gate_init_std=0.001):
        super().__init__()
        self.in_dim = int(in_dim)
        self.pe_dim = int(pe_dim)
        self.hidden = int(hidden)
        self.pe_hidden = int(pe_hidden)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout)
        self.aggr = aggr
        self.gate_init_std = float(gate_init_std)
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.spectral_gate = SpectralGate_spectral_gated_graphsage(pe_dim=pe_dim, hidden_dim=hidden, pe_hidden=pe_hidden, dropout=dropout, gate_init_std=gate_init_std)
        self.blocks = nn.ModuleList([ResidualSAGEBlock(hidden_dim=hidden, dropout=dropout, aggr=aggr) for _ in range(num_layers)])
        self.attn_pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)))
        pooled_dim = hidden * 4
        reg_in_dim = pooled_dim + 1
        self.shared_cls = nn.Sequential(nn.Linear(pooled_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.shared_reg = nn.Sequential(nn.Linear(reg_in_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.cls_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.reg_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.last_gate_mean = float('nan')
        self.last_gate_std = float('nan')
        self.last_gate_abs_mean = float('nan')

    def forward(self, data):
        x = data.x
        pe = data.pe
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else x.new_zeros(x.size(0), dtype=torch.long)
        n_hits = data.n_hits.view(-1, 1).float()
        log_n_hits = torch.log1p(n_hits)
        h_base = self.input_proj(x)
        h, spectral_gate = self.spectral_gate(h_base=h_base, pe=pe)
        for block in self.blocks:
            h = block(h, edge_index)
        g_attn = self.attn_pool(h, batch)
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g_add = global_add_pool(h, batch)
        g = torch.cat([g_attn, g_mean, g_max, g_add], dim=1)
        h_cls = self.shared_cls(g)
        g_reg = torch.cat([g, log_n_hits], dim=1)
        h_reg = self.shared_reg(g_reg)
        logit = self.cls_head(h_cls).squeeze(-1)
        e_pred = F.softplus(self.reg_head(h_reg).squeeze(-1))
        with torch.no_grad():
            self.last_gate_mean = float(spectral_gate.detach().mean().cpu())
            self.last_gate_std = float(spectral_gate.detach().std().cpu())
            self.last_gate_abs_mean = float(spectral_gate.detach().abs().mean().cpu())
        return (logit, e_pred)


class SpectralGate_spectral_gated_gatv2(nn.Module):
    """
    Pure spectral gate:

        z_i = |LapPE_i|
        g_i = tanh(MLP_gate(z_i))
        x_tilde_i = x_base_i * (1 + g_i)
    """

    def __init__(self, pe_dim, hidden_dim, pe_hidden=64, dropout=0.1, gate_init_std=0.001):
        super().__init__()
        self.pe_dim = int(pe_dim)
        self.hidden_dim = int(hidden_dim)
        self.gate_init_std = float(gate_init_std)
        if self.pe_dim <= 0:
            raise ValueError('SpectralMultiplicativeGate requires pe_dim > 0.')
        self.gate = nn.Sequential(nn.Linear(self.pe_dim, pe_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(pe_hidden, hidden_dim))
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=self.gate_init_std)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, h_base, pe):
        z = pe.abs()
        g = torch.tanh(self.gate(z))
        h = h_base * (1.0 + g)
        return (h, g)


class ResidualGATv2Block(nn.Module):

    def __init__(self, hidden_dim, heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % heads == 0
        self.conv = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads, concat=True, dropout=dropout, add_self_loops=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        return self.norm(x + h)


class ClusterGNN_GATv2_UFC_SpectralGated(nn.Module):

    def __init__(self, in_dim, pe_dim, hidden=128, pe_hidden=64, num_layers=4, heads=4, dropout=0.1, gate_init_std=0.001):
        super().__init__()
        self.in_dim = int(in_dim)
        self.pe_dim = int(pe_dim)
        self.hidden = int(hidden)
        self.pe_hidden = int(pe_hidden)
        self.num_layers = int(num_layers)
        self.heads = int(heads)
        self.dropout_p = float(dropout)
        self.gate_init_std = float(gate_init_std)
        assert self.hidden % self.heads == 0
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.spectral_gate = SpectralGate_spectral_gated_gatv2(pe_dim=pe_dim, hidden_dim=hidden, pe_hidden=pe_hidden, dropout=dropout, gate_init_std=gate_init_std)
        self.blocks = nn.ModuleList([ResidualGATv2Block(hidden_dim=hidden, heads=heads, dropout=dropout) for _ in range(num_layers)])
        self.attn_pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)))
        pooled_dim = hidden * 4
        reg_in_dim = pooled_dim + 1
        self.shared_cls = nn.Sequential(nn.Linear(pooled_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.shared_reg = nn.Sequential(nn.Linear(reg_in_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.cls_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.reg_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.last_gate_mean = float('nan')
        self.last_gate_std = float('nan')
        self.last_gate_abs_mean = float('nan')

    def forward(self, data):
        x = data.x
        pe = data.pe
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else x.new_zeros(x.size(0), dtype=torch.long)
        n_hits = data.n_hits.view(-1, 1).float()
        log_n_hits = torch.log1p(n_hits)
        h_base = self.input_proj(x)
        h, spectral_gate = self.spectral_gate(h_base=h_base, pe=pe)
        for block in self.blocks:
            h = block(h, edge_index)
        g_attn = self.attn_pool(h, batch)
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g_add = global_add_pool(h, batch)
        g = torch.cat([g_attn, g_mean, g_max, g_add], dim=1)
        h_cls = self.shared_cls(g)
        g_reg = torch.cat([g, log_n_hits], dim=1)
        h_reg = self.shared_reg(g_reg)
        logit = self.cls_head(h_cls).squeeze(-1)
        e_pred = F.softplus(self.reg_head(h_reg).squeeze(-1))
        with torch.no_grad():
            self.last_gate_mean = float(spectral_gate.detach().mean().cpu())
            self.last_gate_std = float(spectral_gate.detach().std().cpu())
            self.last_gate_abs_mean = float(spectral_gate.detach().abs().mean().cpu())
        return (logit, e_pred)


class SpectralGate_spectral_gated_dynamicedgeconv(nn.Module):
    """
    Pure spectral gate:

        z_i = |LapPE_i|
        g_i = tanh(MLP_gate(z_i))
        x_tilde_i = x_base_i * (1 + g_i)
    """

    def __init__(self, pe_dim, hidden_dim, pe_hidden=64, dropout=0.1, gate_init_std=0.001):
        super().__init__()
        self.pe_dim = int(pe_dim)
        self.hidden_dim = int(hidden_dim)
        self.gate_init_std = float(gate_init_std)
        if self.pe_dim <= 0:
            raise ValueError('SpectralMultiplicativeGate requires pe_dim > 0.')
        self.gate = nn.Sequential(nn.Linear(self.pe_dim, pe_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(pe_hidden, hidden_dim))
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=self.gate_init_std)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, h_base, pe):
        z = pe.abs()
        g = torch.tanh(self.gate(z))
        h = h_base * (1.0 + g)
        return (h, g)


class ResidualDynamicEdgeConvBlock(nn.Module):

    def __init__(self, hidden_dim, k=8, dropout=0.1):
        super().__init__()
        self.k = int(k)
        nn_edge = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.conv = DynamicEdgeConv(nn=nn_edge, k=self.k, aggr='max')
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        h = self.conv(x, batch)
        h = self.dropout(h)
        return self.norm(x + h)


class ClusterGNN_DynamicEdgeConv_UFC_SpectralGated(nn.Module):

    def __init__(self, in_dim, pe_dim, hidden=128, pe_hidden=64, num_layers=4, k=8, dropout=0.1, gate_init_std=0.001):
        super().__init__()
        self.in_dim = int(in_dim)
        self.pe_dim = int(pe_dim)
        self.hidden = int(hidden)
        self.pe_hidden = int(pe_hidden)
        self.num_layers = int(num_layers)
        self.k = int(k)
        self.dropout_p = float(dropout)
        self.gate_init_std = float(gate_init_std)
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.spectral_gate = SpectralGate_spectral_gated_dynamicedgeconv(pe_dim=pe_dim, hidden_dim=hidden, pe_hidden=pe_hidden, dropout=dropout, gate_init_std=gate_init_std)
        self.blocks = nn.ModuleList([ResidualDynamicEdgeConvBlock(hidden_dim=hidden, k=k, dropout=dropout) for _ in range(num_layers)])
        self.attn_pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)))
        pooled_dim = hidden * 4
        reg_in_dim = pooled_dim + 1
        self.shared_cls = nn.Sequential(nn.Linear(pooled_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.shared_reg = nn.Sequential(nn.Linear(reg_in_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.cls_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.reg_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.last_gate_mean = float('nan')
        self.last_gate_std = float('nan')
        self.last_gate_abs_mean = float('nan')

    def forward(self, data):
        x = data.x
        pe = data.pe
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else x.new_zeros(x.size(0), dtype=torch.long)
        n_hits = data.n_hits.view(-1, 1).float()
        log_n_hits = torch.log1p(n_hits)
        h_base = self.input_proj(x)
        h, spectral_gate = self.spectral_gate(h_base=h_base, pe=pe)
        for block in self.blocks:
            h = block(h, batch)
        g_attn = self.attn_pool(h, batch)
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g_add = global_add_pool(h, batch)
        g = torch.cat([g_attn, g_mean, g_max, g_add], dim=1)
        h_cls = self.shared_cls(g)
        g_reg = torch.cat([g, log_n_hits], dim=1)
        h_reg = self.shared_reg(g_reg)
        logit = self.cls_head(h_cls).squeeze(-1)
        e_pred = F.softplus(self.reg_head(h_reg).squeeze(-1))
        with torch.no_grad():
            self.last_gate_mean = float(spectral_gate.detach().mean().cpu())
            self.last_gate_std = float(spectral_gate.detach().std().cpu())
            self.last_gate_abs_mean = float(spectral_gate.detach().abs().mean().cpu())
        return (logit, e_pred)


MODEL_CLASSES = {
    'baseline_graphsage': BaselineClusterGNN_GraphSAGE_UFC,
    'baseline_gatv2': BaselineClusterGNN_GATv2_UFC,
    'baseline_dynamicedgeconv': BaselineClusterGNN_DynamicEdgeConv_UFC,
    'spectral_gated_graphsage': ClusterGNN_GraphSAGE_UFC_SpectralGated,
    'spectral_gated_gatv2': ClusterGNN_GATv2_UFC_SpectralGated,
    'spectral_gated_dynamicedgeconv': ClusterGNN_DynamicEdgeConv_UFC_SpectralGated,
}


def build_student_model(model_key, config=None):
    """Construct an exact architecture, using saved checkpoint config when given.

    Missing optional settings retain the source notebooks' constructor defaults.
    A provided checkpoint must include its recorded core dimensions.
    """
    cls = MODEL_CLASSES[model_key]
    if config is None:
        kwargs = {"in_dim": 6}
        if "pe_dim" in inspect.signature(cls).parameters:
            kwargs["pe_dim"] = 4
    else:
        kwargs = {"in_dim": int(config["in_dim"]),
                  "hidden": int(config["hidden_dim"]),
                  "num_layers": int(config["num_layers"]),
                  "dropout": float(config["dropout"])}
        for parameter in ("heads", "pe_dim", "pe_hidden", "aggr", "gate_init_std"):
            if parameter in inspect.signature(cls).parameters and parameter in config:
                kwargs[parameter] = config[parameter]
        if "pe_dim" in inspect.signature(cls).parameters and "pe_dim" not in kwargs:
            raise ValueError("Spectral checkpoint must record pe_dim")
        if "k" in inspect.signature(cls).parameters and "dynamic_knn" in config:
            kwargs["k"] = int(config["dynamic_knn"])
        if "pe_hidden" in inspect.signature(cls).parameters and "pe_hidden" not in kwargs:
            kwargs["pe_hidden"] = max(kwargs["hidden"] // 2, 32)
    if kwargs["in_dim"] != 6:
        raise ValueError("Student UFC observable recipe has exactly six input features")
    return cls(**kwargs)


def load_student_checkpoint(path, model_key, device="cpu"):
    """Load saved tensor weights and primitive config with strict key checking."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary containing config and state_dict")
    model = build_student_model(model_key, checkpoint)
    key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[key], strict=True)
    return model.to(device).eval(), checkpoint
