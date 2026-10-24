"""Tensor-only storage of native HGND candidates with topology-only LapPE.

Storage changes; frozen feature values, hit order, edges, singleton candidates,
cluster labels/energy targets and original event/side keys are preserved.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ..models.adapters import compute_lap_pe_per_cluster


def pack_native(graphs, indices, a, b):
    fields = {k: [] for k in ("x", "pe", "edges", "cluster", "label", "energy",
                              "row", "side", "index")}
    pointers = {k: [0] for k in ("node_ptr", "edge_ptr", "cluster_ptr")}
    for i in indices:
        g = graphs[int(i)]
        side = int(g.istop.item())
        x = g["hits"].x.clone()
        n = a.shape[1]
        # Exactly the main pipeline's train-fitted affine remap; no refitting.
        x[:, :n] = x[:, :n] * torch.as_tensor(a[side], dtype=x.dtype) + torch.as_tensor(b[side], dtype=x.dtype)
        cluster = g.cluster.reshape(-1).long()
        edge = g["hits", "hits"].edge_index
        nc = int(g["clusters"].num_nodes)
        if not torch.equal(torch.unique(cluster), torch.arange(nc)):
            raise ValueError("Cluster membership is not the native node order")
        if edge.numel() and not torch.equal(cluster[edge[0]], cluster[edge[1]]):
            raise ValueError("Frozen connected components contain cross-cluster edges")
        pe = compute_lap_pe_per_cluster(edge, cluster, len(x), k_pe=4)
        for key, value in (("x", x), ("pe", pe), ("edges", edge),
                           ("cluster", cluster), ("label", g.cllabel),
                           ("energy", g.clenergy)):
            fields[key].append(value)
        for key, value in (("row", int(g.Row.item())), ("side", side), ("index", int(i))):
            fields[key].append(value)
        for key, size in (("node_ptr", len(x)), ("edge_ptr", edge.shape[1]), ("cluster_ptr", nc)):
            pointers[key].append(pointers[key][-1] + size)
    result = {key: torch.cat(values, dim=1 if key == "edges" else 0)
              for key, values in fields.items() if key not in ("row", "side", "index")}
    result.update({key: torch.tensor(fields[key], dtype=torch.long) for key in ("row", "side", "index")})
    result.update({key: torch.tensor(values, dtype=torch.long) for key, values in pointers.items()})
    if not all(torch.isfinite(result[key]).all() for key in ("x", "pe", "energy")):
        raise ValueError("Nonfinite native input/target")
    return result


class NativePackDataset(torch.utils.data.Dataset):
    def __init__(self, pack):
        self.pack = pack

    def __len__(self):
        return len(self.pack["row"])

    def __getitem__(self, i):
        p = self.pack
        n0, n1 = map(int, p["node_ptr"][i:i+2])
        e0, e1 = map(int, p["edge_ptr"][i:i+2])
        c0, c1 = map(int, p["cluster_ptr"][i:i+2])
        g = HeteroData()
        g["hits"].x = p["x"][n0:n1]
        g["hits"].student_pe = p["pe"][n0:n1]
        g["hits", "hits"].edge_index = p["edges"][:, e0:e1]
        g["clusters"].x = torch.zeros((c1-c0, 1), dtype=torch.float32)
        g.cluster = p["cluster"][n0:n1, None]
        g.cllabel = p["label"][c0:c1]
        g.clenergy = p["energy"][c0:c1]
        g.Row = p["row"][i:i+1]
        g.istop = p["side"][i:i+1]
        return g


def population(pack):
    identity = torch.stack([pack[k] for k in ("index", "row", "side")], dim=1).numpy().astype("<i8")
    import hashlib
    return {"graphs": len(pack["row"]), "physical_events": len(np.unique(pack["row"].numpy())),
            "clusters": len(pack["label"]), "signal_clusters": int(pack["label"].sum()),
            "ordered_index_row_side_sha256": hashlib.sha256(identity.tobytes()).hexdigest()}
