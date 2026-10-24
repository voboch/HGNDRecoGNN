"""Recipe fidelity, coherent recovery and test-barrier checks."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from HGNDRecoGNN.models.student_ufc import MODEL_CLASSES
from HGNDRecoGNN.scripts.audit_student_notebooks import namespace
from HGNDRecoGNN.scripts.student_reproduce import require_complete
from HGNDRecoGNN.training.student_reproduce import (
    StudentTrainConfig, better_checkpoint, pack_graphs, set_seeds, train, unpack_graphs,
)


@pytest.fixture
def parts():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(123)
    graphs = []
    for i in range(18):
        n = (2, 8, 9)[i % 3]
        pairs = torch.combinations(torch.arange(n), r=2).T
        edges = torch.cat([pairs, pairs.flip(0)], dim=1)
        graphs.append(Data(x=torch.randn(n, 6, generator=generator), edge_index=edges,
                           pe=torch.randn(n, 4, generator=generator), row=torch.tensor([i // 2]),
                           cluster=torch.tensor([i % 2]), n_hits=torch.tensor([float(n)]),
                           y_cls=torch.tensor([float(i % 2)]),
                           y_E=torch.tensor([float(i % 2) * (1 + i / 10)])))
    return [graphs[:6], graphs[6:12]], [graphs[12:]]


def test_packed_parts_preserve_order_edges_and_targets(parts):
    for part in parts[0] + parts[1]:
        packed = pack_graphs(part)
        for spectral in (False, True):
            restored = unpack_graphs(packed, spectral)
            assert len(restored) == len(part)
            for a, b in zip(part, restored):
                for key in ("x", "edge_index", "row", "cluster", "n_hits", "y_cls", "y_E"):
                    assert torch.equal(a[key], b[key])
                assert ("pe" in b) == spectral
                if spectral:
                    assert torch.equal(a.pe, b.pe)


@pytest.mark.parametrize("model_key", list(MODEL_CLASSES))
def test_two_epoch_training_matches_original_notebook(model_key, parts, tmp_path):
    import json
    project = Path(__file__).resolve().parents[2]
    metadata = json.loads((project / "HGNDRecoGNN/models/student_ufc_sources.json").read_text())
    path = project / "GNN_for_neutron_reconstruction" / metadata[model_key]["notebook"]
    if not path.is_file():
        pytest.skip("Original notebooks are not present in this checkout")
    ns, _ = namespace(path, repair=True)
    function = next(v for k, v in ns.items() if k.startswith("train_multitask_") and k.endswith("trainval"))
    for split, shards in zip(("train", "val"), parts):
        (tmp_path / split).mkdir()
        for i, graphs in enumerate(shards):
            torch.save(graphs, tmp_path / split / f"graphs_shard_{i:04d}.pt")
    config = replace(StudentTrainConfig(model_key, 42), max_epochs=2, batch_size=4)
    set_seeds(config.seed)
    with contextlib.redirect_stdout(io.StringIO()), torch.serialization.safe_globals(
            [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
        original, original_history = function(str(tmp_path / "train"), str(tmp_path / "val"),
                                              str(tmp_path / "original.pt"), epochs=2, batch_size=4,
                                              pos_weight=config.pos_weight, print_every_shards=0)
    restored_parts = [[unpack_graphs(pack_graphs(p), model_key.startswith("spectral_")) for p in split]
                      for split in parts]
    model, history, finished = train(config, *restored_parts, tmp_path / "port",
                                     data_hash="fixture-data", source_hash="fixture-source", device="cpu")
    assert finished
    for expected, actual in zip(original_history.to_dict("records"), history):
        for key, value in expected.items():
            assert actual[key] == value, (model_key, key, actual[key], value)
    assert original.state_dict().keys() == model.state_dict().keys()
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[name]), (model_key, name)


def test_resume_preserves_optimizer_rng_history_and_best(parts, tmp_path):
    config = replace(StudentTrainConfig("baseline_graphsage", 7), max_epochs=3, batch_size=4)
    full, full_history, finished = train(config, *parts, tmp_path / "full", data_hash="data",
                                         source_hash="source", device="cpu")
    assert finished
    _, partial, finished = train(config, *parts, tmp_path / "resume", data_hash="data",
                                  source_hash="source", device="cpu", stop_after_epochs=1)
    assert not finished and len(partial) == 1
    resumed, resumed_history, finished = train(config, *parts, tmp_path / "resume", data_hash="data",
                                              source_hash="source", device="cpu", resume=True)
    assert finished
    for expected, actual in zip(full_history, resumed_history):
        assert {k: v for k, v in expected.items() if k != "seconds"} == {
            k: v for k, v in actual.items() if k != "seconds"}
    for name, tensor in full.state_dict().items():
        assert torch.equal(tensor, resumed.state_dict()[name])
    for data, source in (("changed", "source"), ("data", "changed")):
        with pytest.raises(ValueError, match="changed across"):
            train(config, *parts, tmp_path / "resume", data_hash=data, source_hash=source,
                  device="cpu", resume=True)
    with pytest.raises(ValueError, match="changed across"):
        train(replace(config, learning_rate=.002), *parts, tmp_path / "resume", data_hash="data",
              source_hash="source", device="cpu", resume=True)


def test_checkpoint_comparator_retains_primary_metric_and_delta():
    best = dict(val_score_sep_z=2., val_roc_auc=.98, val_pr_auc=.95, val_total=.2)
    assert not better_checkpoint({**best, "val_score_sep_z": 1.99, "val_roc_auc": .999}, best, .0001)
    assert better_checkpoint({**best, "val_score_sep_z": 2.00001, "val_roc_auc": .981}, best, .0001)
    assert better_checkpoint({**best, "val_total": .19}, best, .0001)
    assert not better_checkpoint(best.copy(), best, .0001)


def test_test_barrier_rejects_missing_runs(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_complete(tmp_path, "data", "source")


def test_score_moments_use_notebook_float64():
    from HGNDRecoGNN.training.student_reproduce import score_statistics
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .2, .8, .9], dtype=np.float32)
    result = score_statistics(y, p)
    q = p.astype(np.float64)
    assert result["score_sep_z"] == (q[2:].mean()-q[:2].mean()) / np.sqrt(q[2:].var()+q[:2].var())
