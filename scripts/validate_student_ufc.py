"""Compare the standalone UFC port with notebook definitions and local inputs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.data.student_ufc import (  # noqa: E402
    UFCRecipe, attach_targets, build_ufc_event, prepare_observables, signal_hit_targets,
)
from HGNDRecoGNN.models.student_ufc import MODEL_CLASSES, build_student_model  # noqa: E402
from HGNDRecoGNN.scripts.audit_student_notebooks import namespace  # noqa: E402
from HGNDRecoGNN.scripts.student_ufc import sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[2]
    parser.add_argument("--pickle", type=Path, default=project / "NeutronRecoGNN/data/Jan26/hits_Jan_26.pkl")
    parser.add_argument("--notebooks", type=Path, default=project / "GNN_for_neutron_reconstruction")
    parser.add_argument("--full-population", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    recipe = UFCRecipe(time_column="fTime_nonoise")
    raw = pd.read_pickle(args.pickle)
    obs, event_rows = prepare_observables(raw, recipe)
    reference, _ = namespace(args.notebooks / "Challenger_ufc_GraphSAGE.ipynb", repair=True)
    arrays, starts, ends, rows, lm, rm, cm = reference["prepare_arrays_and_slices"](args.pickle, verbose=False)
    splits = reference["make_event_splits"](rows)
    assert np.array_equal(event_rows, rows)
    selected = splits[0][:32]  # Engineering sample from training only, no AUC.
    groups = {int(row): event for row, event in obs[obs.Row.isin(rows[selected])].groupby("Row")}
    real_graphs = []
    feature_delta = 0.
    pe_delta = 0.
    for index in selected:
        row = int(rows[index])
        port = build_ufc_event(groups[row], recipe, "local-fixture", spectral=True)
        attach_targets(port, raw)
        original, _ = reference["build_cluster_graphs_for_event"](
            arrays, int(starts[index]), int(ends[index]), row,
            layer_max=lm, row_max=rm, col_max=cm)
        assert len(port) == len(original)
        for a, b in zip(port, original):
            assert torch.equal(a.edge_index, b.edge_index)
            assert torch.equal(a.y_cls, b.y_cls) and torch.equal(a.y_E, b.y_E)
            assert torch.equal(a.cluster, b.cluster)
            feature_delta = max(feature_delta, (a.x - b.x).abs().max().item())
            pe_delta = max(pe_delta, (a.pe - b.pe).abs().max().item())
        real_graphs.extend(port)
    assert feature_delta < 1e-6 and pe_delta < 1e-6
    batch = Batch.from_data_list(real_graphs)
    source_manifest = json.loads((project / "HGNDRecoGNN/models/student_ufc_sources.json").read_text())
    model_parity = {}
    fixture_state = None
    for key in MODEL_CLASSES:
        notebook = args.notebooks / source_manifest[key]["notebook"]
        assert sha256(notebook) == source_manifest[key]["sha256"]
        ns, _ = namespace(notebook, repair=True)
        cls = next(value for name, value in ns.items()
                   if isinstance(value, type) and "ClusterGNN" in name)
        torch.manual_seed(42)
        original = cls(in_dim=6, pe_dim=4).eval() if key.startswith("spectral_") else cls(in_dim=6).eval()
        model = build_student_model(key).eval()
        model.load_state_dict(original.state_dict(), strict=True)
        with torch.no_grad():
            expected = original(batch)
            actual = model(batch)
        delta = max((a - b).abs().max().item() for a, b in zip(expected, actual))
        assert delta < 1e-5
        model_parity[key] = {"max_abs_error": delta, "state_dict_keys_identical": True,
                             "parameters": sum(p.numel() for p in model.parameters())}
        if key == "baseline_graphsage":
            fixture_state = copy.deepcopy(original.state_dict())
        print(key, "parity", delta, flush=True)
    del arrays, batch
    # Exercise the real command using explicitly untrained, temporary weights
    # and observable-only hits. The partial run suppresses all quality AUCs.
    with tempfile.TemporaryDirectory(prefix="hgnd-ufc-cli-") as temp:
        directory = Path(temp)
        path = directory / "untrained_fixture.pt"
        torch.save({"state_dict": fixture_state, "in_dim": 6, "hidden_dim": 128,
                    "num_layers": 4, "dropout": .1, "fixture_status": "untrained_random_weights"}, path)
        columns = [c for c in raw if c not in ("PDG", "Side", "Id", "Ekin", "fMotherId",
                                              "eToF_up_2sig", "eToF_dn_2sig")]
        raw.loc[raw.Row.isin(rows[selected]), columns].to_pickle(directory / "observables.pkl")
        subprocess.run([sys.executable, str(project / "HGNDRecoGNN/scripts/student_ufc.py"),
                        "infer", "--model", "baseline_graphsage", "--checkpoint", str(path),
                        "--hits-pickle", str(directory / "observables.pkl"),
                        "--time-column", "fTime_nonoise", "--split", "all", "--seed", "42",
                        "--limit-events", "32", "--output", str(directory / "result")], check=True)
        predictions = pd.read_csv(directory / "result/predictions.csv")
        metrics = json.loads((directory / "result/metrics.json").read_text())
        assert "true_y" not in predictions and metrics["roc_auc"] is None
        assert len(predictions) == len(real_graphs)
    result = {"input_sha256": sha256(args.pickle), "training_sample_events": len(selected),
              "sample_candidates": len(real_graphs), "feature_max_abs_error": feature_delta,
              "lap_pe_max_abs_error": pe_delta, "six_model_parity": model_parity,
              "observable_only_cli_smoke": True, "trained_auc_reproduced": False,
              "weights_in_parity_test": "seeded random engineering fixture, not student trained weights",
              "jobs_submitted": 0, "source_988976_evaluated": False}
    if args.full_population:
        mask, _ = signal_hit_targets(raw)
        split_of_row = np.empty(len(rows), dtype=np.int8)
        for i, indices in enumerate(splits):
            split_of_row[indices] = i
        totals = [{"events": 0, "events_with_candidates": 0, "candidates": 0, "positives": 0}
                  for _ in splits]
        started = time.monotonic()
        for count, (row, hits) in enumerate(obs.groupby("Row", sort=True), 1):
            graphs = build_ufc_event(hits, recipe, "local-fixture")
            total = totals[int(split_of_row[np.searchsorted(rows, row)])]
            total["events"] += 1
            total["events_with_candidates"] += int(bool(graphs))
            total["candidates"] += len(graphs)
            total["positives"] += sum(bool(mask[g.hit_ids.numpy()].any()) for g in graphs)
            if count % 20000 == 0:
                print("Port population", count, "events", round(time.monotonic()-started, 1), "seconds", flush=True)
        expected = ((168108, 45740), (36269, 9924), (36343, 9857))
        assert all((v["candidates"], v["positives"]) == target for v, target in zip(totals, expected))
        result["full_population"] = dict(zip(("train", "val", "test"), totals))
    source_files = [Path(__file__), project / "HGNDRecoGNN/scripts/student_ufc.py",
                    project / "HGNDRecoGNN/data/student_ufc.py",
                    project / "HGNDRecoGNN/models/student_ufc.py",
                    project / "HGNDRecoGNN/models/student_ufc_sources.json",
                    project / "HGNDRecoGNN/training/student_ufc.py"]
    result["execution_source_sha256"] = {str(p.relative_to(project)): sha256(p) for p in source_files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
