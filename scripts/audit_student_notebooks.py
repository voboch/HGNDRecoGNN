"""Audit deposited UFC notebooks without running their Colab side effects.

The only source repairs allowed by the probe are the recorded missing list
bracket and dangling assertion commas. Original notebook bytes stay unchanged.
Numerical defaults in extracted definitions belong to the student's recipe.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hgnd-student-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import nn
from torch.nn import functional as F
from torch_geometric import nn as pyg_nn
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repaired_source(source):
    repairs = []
    if '"eToF_up_2sig", "eToF_dn_2sig",\n    int_cols' in source:
        source = source.replace(
            '"eToF_up_2sig", "eToF_dn_2sig",\n    int_cols',
            '"eToF_up_2sig", "eToF_dn_2sig",\n    ]\n    int_cols',
        )
        repairs.append("Close float_cols list before int_cols")
    source, count = re.subn(r"(?m)^(\s*assert [^\n]+),\s*$", r"\1", source)
    if count:
        repairs.append(f"Remove {count} dangling assertion comma(s)")
    return source, repairs


def namespace(notebook, repair=False):
    ns = dict(
        np=np, pd=pd, torch=torch, nn=nn, F=F, Data=Data,
        DataLoader=DataLoader, copy=copy, gc=gc, os=os, json=json,
        plt=plt, LogNorm=LogNorm, random=random,
    )
    ns.update({n: getattr(pyg_nn, n) for n in (
        "radius_graph", "SAGEConv", "GATv2Conv", "DynamicEdgeConv",
        "AttentionalAggregation", "global_mean_pool", "global_max_pool",
        "global_add_pool",
    )})
    ns.update({n: getattr(metrics, n) for n in (
        "roc_auc_score", "average_precision_score", "roc_curve",
        "precision_recall_curve", "precision_recall_fscore_support",
        "confusion_matrix",
    )})
    evidence = {"sha256": digest(notebook), "syntax_errors": [], "repairs": [],
                "saved_auc_outputs": [], "definitions": {}}
    for index, cell in enumerate(json.loads(notebook.read_text())["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        # Installation/magic cells are Colab code, not Python syntax failures.
        if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            continue
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            for value in re.findall(r"(?m)^ROC AUC:\s*(\d+\.\d+)", text):
                evidence["saved_auc_outputs"].append({"cell_zero_based": index, "auc": float(value)})
        try:
            ast.parse(source)
        except SyntaxError as exc:
            evidence["syntax_errors"].append({"cell_zero_based": index,
                                              "line_in_cell": exc.lineno,
                                              "message": exc.msg})
            if not repair:
                continue
            source, changes = repaired_source(source)
            evidence["repairs"].extend({"cell_zero_based": index, "change": c} for c in changes)
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    ns[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        for node in definitions:
            evidence["definitions"][node.name] = {"cell_zero_based": index,
                "ast_sha256": hashlib.sha256(ast.dump(node).encode()).hexdigest()}
        # Execute definitions only: no notebook imports, installation, I/O or training.
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook), "exec"), ns)
    ns.update(MN_MEV=939.55, C=299_792_458.0)
    return ns, evidence


def architecture_probe(ns):
    candidates = [(name, cls) for name, cls in ns.items()
                  if isinstance(cls, type) and "ClusterGNN" in name]
    name, cls = candidates[0]
    kwargs = {"in_dim": 6}
    if "pe_dim" in __import__("inspect").signature(cls).parameters:
        kwargs["pe_dim"] = 4
    torch.manual_seed(42)  # Repeated diagnostic input, not a trained seed result.
    model = cls(**kwargs).eval()
    graphs = []
    # Retained boundary: two hits; k boundary: k=8; above-boundary: nine hits.
    for n in (2, 8, 9):
        pairs = torch.combinations(torch.arange(n), r=2).T
        edges = torch.cat([pairs, pairs.flip(0)], dim=1)
        graphs.append(Data(x=torch.randn(n, 6), edge_index=edges,
                           pe=torch.randn(n, 4), n_hits=torch.tensor([float(n)]),
                           y_cls=torch.tensor([1.]), y_E=torch.tensor([1.]),
                           row=torch.tensor([1]), cluster=torch.tensor([0])))
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        output = model(batch)
        separate = [model(Batch.from_data_list([g])) for g in graphs]
        changed = batch.clone()
        changed.y_cls.zero_()
        changed.y_E.fill_(999.)
        changed.row.fill_(999)
        changed.cluster.fill_(999)
        truth_changed = model(changed)
    assert all(t.shape == (3,) and torch.isfinite(t).all() for t in output)
    batch_delta = max((output[k] - torch.cat([o[k] for o in separate])).abs().max().item()
                      for k in range(2))
    assert batch_delta < 1e-5  # Float32 reduction tolerance, not a performance cut.
    assert all(torch.equal(a, b) for a, b in zip(output, truth_changed))
    clone = cls(**kwargs).eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(output, clone(batch)))
    model.train()
    logit, energy = model(batch)
    loss = F.binary_cross_entropy_with_logits(logit, batch.y_cls) + F.l1_loss(energy, batch.y_E)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return {"class": name, "parameters": sum(p.numel() for p in model.parameters()),
            "cpu_forward_backward": True, "strict_state_roundtrip": True,
            "truth_fields_not_read_by_forward": True, "batch_vs_single_max_abs": batch_delta,
            "input": "Data(x[*,6], edge_index, n_hits; pe[*,4] for spectral models)",
            "output": "two tensors: classification logit and positive energy"}


def event_metric_probe(ns):
    class FixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            return batch.x[batch.ptr[:-1], 0], torch.ones(batch.num_graphs)

    fn = ns["run_multitask_baseline_graphsage_inference_and_make_plots"]
    graphs = [Data(x=torch.tensor([[v], [v]]), edge_index=torch.tensor([[0, 1], [1, 0]]),
                   y_cls=torch.tensor([1.]), y_E=torch.tensor([1.]),
                   tof_ref=torch.tensor([1.]), row=torch.tensor([100]),
                   cluster=torch.tensor([i]), n_hits=torch.tensor([2.]))
              for i, v in enumerate((2., -2.))]
    results = {}
    with tempfile.TemporaryDirectory(prefix="hgnd-student-event-") as temp:
        root = Path(temp)
        torch.save(graphs, root / "graphs_shard_0000.pt")
        for size in (1, 2):
            # Allow only the PyG types in our own constructed fixture on torch>=2.6.
            with contextlib.redirect_stdout(io.StringIO()), torch.serialization.safe_globals(
                [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]
            ):
                out = fn(FixedModel(), str(root), str(root / f"out{size}"),
                         thresholds=[0.5], batch_size=size, return_cluster_df=True)
            results[str(size)] = out["threshold_df"].to_dict("records")
            plt.close("all")
    return results


def population_probe(ns, pickle_path):
    frame = pd.read_pickle(pickle_path)
    timing = (frame.fTime - frame.fTime_nonoise).describe().to_dict()
    duplicates = int(frame.duplicated(["Row", "Instance"]).sum())
    metadata = {"path": str(pickle_path), "sha256": digest(pickle_path),
                "rows": len(frame), "columns": list(frame.columns),
                "time_noise_ns": timing, "duplicate_Row_Instance": duplicates,
                "attrs": frame.attrs}
    del frame
    arrays, starts, ends, rows, lm, rm, cm = ns["prepare_arrays_and_slices"](pickle_path, verbose=False)
    splits = ns["make_event_splits"](rows)
    assert not any(np.intersect1d(splits[a], splits[b]).size
                   for a, b in ((0, 1), (0, 2), (1, 2)))
    metadata.update(events=len(rows), neutron_hits=int((arrays[2] == 2112).sum()),
                    signal_hits=int(arrays[-1].sum()),
                    signal_hits_below_or_equal_0p3=int(((arrays[-1] > 0) & (arrays[4] <= .3)).sum()),
                    split_sizes=[len(x) for x in splits], row_split_intersections=0,
                    split_row_sha256=[hashlib.sha256(rows[x].tobytes()).hexdigest() for x in splits])
    split_at = np.empty(len(rows), dtype=np.int8)
    for i, ids in enumerate(splits):
        split_at[ids] = i
    totals = [dict(events=0, events_with_graph=0, candidate_graphs=0, positive_graphs=0,
                   unique_signal_objects=0, represented_signal_objects=0,
                   positive_components_with_multiple_objects=0,
                   duplicated_signal_objects_across_components=0) for _ in splits]
    start_time = time.monotonic()
    checked_truth_invariance = 0
    for i, (a, b, row) in enumerate(zip(starts, ends, rows)):
        graphs, meta = ns["build_cluster_graphs_for_event"](
            arrays, int(a), int(b), int(row), layer_max=lm, row_max=rm, col_max=cm)
        total = totals[int(split_at[i])]
        total["events"] += 1
        total["events_with_graph"] += int(bool(graphs))
        total["candidate_graphs"] += len(graphs)
        total["positive_graphs"] += len(meta)
        all_objects = set(arrays[1][a:b][arrays[-1][a:b] > 0].tolist())
        represented = set().union(*(m["signal_ids"] for m in meta)) if meta else set()
        total["unique_signal_objects"] += len(all_objects)
        total["represented_signal_objects"] += len(represented)
        total["positive_components_with_multiple_objects"] += sum(len(m["signal_ids"]) > 1 for m in meta)
        listed = [v for m in meta for v in m["signal_ids"]]
        total["duplicated_signal_objects_across_components"] += len(listed) - len(set(listed))
        if graphs and checked_truth_invariance < 32:
            # Local event copies: change only MC fields, keep measured arrays.
            altered = [x[a:b].copy() for x in arrays]
            for col in (1, 2, 3, 4, 13):
                altered[col].fill(0)
            other, _ = ns["build_cluster_graphs_for_event"](
                tuple(altered), 0, int(b-a), int(row), layer_max=lm, row_max=rm, col_max=cm)
            assert len(graphs) == len(other)
            assert all(torch.equal(g.x, h.x) and torch.equal(g.edge_index, h.edge_index)
                       for g, h in zip(graphs, other))
            checked_truth_invariance += 1
        if (i + 1) % 20000 == 0:
            print(f"Preprocessed {i+1}/{len(rows)} events in {time.monotonic()-start_time:.1f}s", flush=True)
    metadata.update(populations=dict(zip(("train", "val", "test"), totals)),
                    truth_feature_topology_invariance_events=checked_truth_invariance,
                    elapsed_graph_seconds=time.monotonic()-start_time)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[2]
    parser.add_argument("--notebooks", type=Path, default=project / "GNN_for_neutron_reconstruction")
    parser.add_argument("--pickle", type=Path, default=project / "NeutronRecoGNN/data/Jan26/hits_Jan_26.pkl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", action="store_true", help="Rebuild all candidate counts; no training")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"script_sha256": digest(__file__), "torch": torch.__version__,
              "notebooks": {}, "no_training": True, "trained_auc_reproduced": False}
    spaces = {}
    for path in sorted(args.notebooks.glob("*.ipynb")):
        ns, evidence = namespace(path, repair=True)
        spaces[path.name] = ns
        if path.name != "Extra_plots.ipynb":
            evidence["architecture_probe"] = architecture_probe(ns)
        result["notebooks"][path.name] = evidence
        print(path.name, "checked", flush=True)
    baseline = spaces["baseline_GraphSAGE.ipynb"]
    result["event_metric_probe"] = event_metric_probe(baseline)
    if args.population:
        result["input_population"] = population_probe(baseline, args.pickle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
