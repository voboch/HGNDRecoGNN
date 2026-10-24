"""Observable UFC candidates with optional, separately attached MC targets.

The numerical graph/feature recipe is inherited from the student's audited
notebooks. Changing time_column creates a different recipe fingerprint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

# Source notebook's ToF conventions, retained for numerical reproduction.
MN_MEV = 939.55
C = 299_792_458.0


@dataclass(frozen=True)
class UFCRecipe:
    """Inherited student settings; none was optimised by this compatibility port.

    Geometry divisors are the recorded detector maxima, not per-sample fits.
    The time column is mandatory so perfect and detector timing stay explicit.
    """

    time_column: str
    edep_threshold_GeV: float = 0.003
    radius_cells: float = 5.0
    edge_dt_ns: float = 2.5
    max_radius_neighbors: int = 32
    min_hits: int = 2
    layer_max: float = 8.0
    row_max: float = 11.0
    col_max: float = 11.0
    k_pe: int = 4

    def __post_init__(self):
        if self.time_column not in ("fTime_nonoise", "fTime"):
            raise ValueError("Choose fTime_nonoise or fTime explicitly")
        if self.radius_cells <= 0 or self.edge_dt_ns <= 0 or self.min_hits < 1:
            raise ValueError("Invalid UFC geometry/time settings")
        if min(self.layer_max, self.row_max, self.col_max) <= 0:
            raise ValueError("Geometry divisors must be positive")

    def fingerprint(self):
        payload = {"version": "student-ufc-observables-v1", **asdict(self)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_observables(frame, recipe):
    """Return sorted measured hits and every input event, without reading truth.

    Invalid measured hits are removed. Their event remains in the inventory.
    _hit_id is the original positional row, independent of a pandas index.
    """
    required = ["Row", "fELoss", "fX", "fY", "fZ", "LayerId", "RowId",
                "ColumnId", recipe.time_column]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing observable columns: {sorted(missing)}")
    work = frame[required].apply(pd.to_numeric, errors="coerce").copy()
    rows = work.Row.to_numpy(dtype=np.float64)
    if not np.isfinite(rows).all() or not np.equal(rows, np.floor(rows)).all():
        raise ValueError("Event Row must be a finite integer")
    inventory = np.sort(work.Row.astype(np.int64).unique())
    work["_hit_id"] = np.arange(len(work), dtype=np.int64)
    work = work[np.isfinite(work[required]).all(axis=1)]
    work = work[work.fELoss > recipe.edep_threshold_GeV].copy()
    for name in ("Row", "LayerId", "RowId", "ColumnId"):
        if not np.equal(work[name], np.floor(work[name])).all():
            raise ValueError(f"{name} must be integral")
        work[name] = work[name].astype(np.int64)
    for name in ("fELoss", "fX", "fY", "fZ", recipe.time_column):
        work[name] = work[name].astype(np.float32)
    for name, maximum in (("LayerId", recipe.layer_max), ("RowId", recipe.row_max),
                          ("ColumnId", recipe.col_max)):
        if not work[name].between(0, maximum).all():
            raise ValueError(f"{name} exceeds recorded geometry divisor {maximum}")
    return work.sort_values("Row", kind="mergesort"), inventory


def build_ufc_event(hits, recipe, source_id, spectral=False):
    """Build candidates for one event; MC columns never enter this function."""
    if not len(hits):
        return []
    if hits.Row.nunique() != 1:
        raise ValueError("build_ufc_event expects exactly one event")
    layer = hits.LayerId.to_numpy(dtype=np.float32)
    rowid = hits.RowId.to_numpy(dtype=np.float32)
    colid = hits.ColumnId.to_numpy(dtype=np.float32)
    time = hits[recipe.time_column].to_numpy(dtype=np.float32)
    coords = hits[["fX", "fY", "fZ"]].to_numpy(dtype=np.float32)
    distance = np.sqrt((coords * coords).sum(axis=1)).astype(np.float32) / np.float32(100.)
    etof = tof_energy_neutron_GeV(distance, time)
    features = normalize_node_features(
        layer, rowid, colid, hits.fELoss.to_numpy(dtype=np.float32), time, etof,
        recipe.layer_max, recipe.row_max, recipe.col_max, recipe.edge_dt_ns)
    pos = torch.tensor(np.stack([layer, rowid, colid], axis=1), dtype=torch.float32)
    t = torch.tensor(time, dtype=torch.float32)
    edges = radius_graph(pos, r=recipe.radius_cells, loop=False,
                         max_num_neighbors=recipe.max_radius_neighbors)
    edges = edges[:, (t[edges[0]] - t[edges[1]]).abs() <= recipe.edge_dt_ns]
    edges = torch.unique(torch.cat([edges, edges.flip(0)], dim=1), dim=1)
    components = uf_components(edges, len(hits))
    graphs = []
    hit_ids = hits._hit_id.to_numpy(dtype=np.int64)
    for component in range(int(components.max()) + 1):
        indices = np.where(components == component)[0]
        if len(indices) < recipe.min_hits:
            continue
        inside = torch.zeros(len(hits), dtype=torch.bool)
        inside[indices] = True
        local_edges = edges[:, inside[edges[0]] & inside[edges[1]]]
        mapping = -torch.ones(len(hits), dtype=torch.long)
        mapping[indices] = torch.arange(len(indices), dtype=torch.long)
        local_edges = mapping[local_edges]
        graph = Data(x=torch.tensor(features[indices], dtype=torch.float32),
                     edge_index=local_edges,
                     row=torch.tensor([int(hits.Row.iloc[0])], dtype=torch.long),
                     cluster=torch.tensor([component], dtype=torch.long),
                     n_hits=torch.tensor([float(len(indices))]),
                     hit_ids=torch.tensor(hit_ids[indices], dtype=torch.long),
                     source_id=str(source_id))
        if spectral:
            graph.pe = torch.tensor(lap_pe_from_dense_adj(
                edge_index_to_dense_adj(local_edges, len(indices)), recipe.k_pe))
        graphs.append(graph)
    return graphs


def signal_hit_targets(frame):
    """Student MC target definition; no additional Ekin>0.3 cut is introduced."""
    required = ["Row", "PDG", "Side", "Id", "Ekin", "eToF_up_2sig", "eToF_dn_2sig"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing MC target columns: {sorted(missing)}")
    target = frame[required].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(target).all(axis=None):
        raise ValueError("Invalid MC targets; observable candidates must not be filtered by truth")
    mask = ((target.PDG == 2112) & target.Side.isin([5, 6])
            & (target.Ekin <= target.eToF_up_2sig) & (target.Ekin >= target.eToF_dn_2sig))
    return mask.to_numpy(), target


def attach_targets(graphs, raw_frame):
    """Attach labels after candidate construction, without changing x or edges."""
    if not graphs:
        return
    hit_ids = torch.cat([g.hit_ids for g in graphs]).numpy()
    selected = raw_frame.iloc[hit_ids]
    mask, values = signal_hit_targets(selected)
    offset = 0
    for graph in graphs:
        size = graph.num_nodes
        local_mask = mask[offset:offset + size]
        local = values.iloc[offset:offset + size]
        graph.y_cls = torch.tensor([float(local_mask.any())])
        energy = local.Ekin.to_numpy()[local_mask]
        graph.y_E = torch.tensor([float(energy.max()) if len(energy) else 0.], dtype=torch.float32)
        graph.signal_ids = torch.tensor(sorted(set(local.Id.to_numpy()[local_mask].astype(np.int64))),
                                        dtype=torch.long)
        offset += size


def truth_inventory(observables, raw_frame, source_id, energy_min):
    """Eligible truth objects after the hit cut, including unrepresented objects.

    This inventory does not include neutrons with no retained detector hit.
    energy_min is explicit so it cannot be confused with generator acceptance.
    """
    mask, values = signal_hit_targets(raw_frame.iloc[observables._hit_id.to_numpy()])
    signal = values.loc[mask & (values.Ekin.to_numpy() > energy_min)]
    return {(str(source_id), int(row)): sorted(set(group.Id.astype(np.int64)))
            for row, group in signal.groupby("Row", sort=False)}


def tof_energy_neutron_GeV(l_m, t_ns):
    l_m = np.asarray(l_m, dtype=np.float64)
    t_ns = np.asarray(t_ns, dtype=np.float64)
    t_s = t_ns * 1e-09
    t_s = np.clip(t_s, 1e-15, None)
    beta = l_m / (C * t_s)
    beta = np.clip(beta, 1e-14, 1.0 - 1e-14)
    E_MeV = MN_MEV * (1.0 / np.sqrt(1.0 - beta ** 2) - 1.0)
    E_MeV = np.clip(E_MeV, 0.0, 10000.0)
    return (E_MeV * 0.001).astype(np.float32)


def uf_components(edge_index: torch.Tensor, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if edge_index.numel() == 0:
        return np.arange(n, dtype=np.int64)
    ei = edge_index.detach().cpu().numpy()
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in ei.T:
        ru, rv = (find(u), find(v))
        if ru != rv:
            parent[rv] = ru
    comp = np.array([find(i) for i in range(n)], dtype=np.int64)
    uniq = {u: i for i, u in enumerate(np.unique(comp))}
    return np.array([uniq[c] for c in comp], dtype=np.int64)


def normalize_node_features(layer, rowid, colid, edep, t_meas, eToF, layer_max, row_max, col_max, dt_ns=2.5):
    x0 = (layer / max(layer_max, 1.0)).astype(np.float32)
    x1 = (rowid / max(row_max, 1.0)).astype(np.float32)
    x2 = (colid / max(col_max, 1.0)).astype(np.float32)
    x3 = np.log1p(edep * 1000.0).astype(np.float32)
    t0 = (t_meas - t_meas.min()).astype(np.float32)
    x4 = (t0 / max(float(dt_ns), 1e-06)).astype(np.float32)
    x5 = (eToF / 10.0).astype(np.float32)
    return np.stack([x0, x1, x2, x3, x4, x5], axis=1).astype(np.float32)


def edge_index_to_dense_adj(edge_index: torch.Tensor, n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float64)
    if edge_index.numel() == 0 or n <= 0:
        return A
    ei = edge_index.detach().cpu().numpy()
    A[ei[0], ei[1]] = 1.0
    A[ei[1], ei[0]] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def normalized_laplacian_dense(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    deg = A.sum(axis=1)
    inv_sqrt_deg = np.zeros_like(deg)
    m = deg > 0
    inv_sqrt_deg[m] = 1.0 / np.sqrt(deg[m])
    D_inv_sqrt = np.diag(inv_sqrt_deg)
    L = np.eye(n, dtype=np.float64) - D_inv_sqrt @ A @ D_inv_sqrt
    return L


def sign_fix_evecs(V: np.ndarray) -> np.ndarray:
    V = V.copy()
    if V.size == 0:
        return V
    for j in range(V.shape[1]):
        idx = np.argmax(np.abs(V[:, j]))
        if V[idx, j] < 0:
            V[:, j] *= -1.0
    return V


def lap_pe_from_dense_adj(A: np.ndarray, k_pe: int=4) -> np.ndarray:
    n = A.shape[0]
    if n <= 1 or k_pe <= 0:
        return np.zeros((n, k_pe), dtype=np.float32)
    L = normalized_laplacian_dense(A)
    evals, evecs = np.linalg.eigh(L)
    start = 1
    end = min(n, start + k_pe)
    pe = evecs[:, start:end]
    pe = sign_fix_evecs(pe)
    if pe.shape[1] < k_pe:
        pad = np.zeros((n, k_pe - pe.shape[1]), dtype=np.float64)
        pe = np.concatenate([pe, pad], axis=1)
    return pe.astype(np.float32)
