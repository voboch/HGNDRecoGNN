"""Original student optimisation/selection semantics with coherent recovery.

All scientific defaults below reproduce the six audited notebooks. Validation
losses retain their graph-weighted minibatch means; checkpoint selection first
uses score separation z, then ROC-AUC, PR-AUC and loss with min_delta=1e-4.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..models.student_ufc import build_student_model


MODEL_ORDER = ("baseline_graphsage", "baseline_gatv2", "baseline_dynamicedgeconv",
               "spectral_gated_graphsage", "spectral_gated_gatv2", "spectral_gated_dynamicedgeconv")
SEEDS = (42, 7, 21)


@dataclass(frozen=True)
class StudentTrainConfig:
    model_key: str
    seed: int
    max_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = .001
    weight_decay: float = .01  # Notebook AdamW default; not the HGND default Net setting.
    lambda_reg: float = .05
    lambda_rank: float = .20
    rank_margin: float = .20
    grad_clip: float = 1.
    patience: int = 12
    min_delta: float = .0001
    pos_weight: float = 122368 / 45740  # Frozen full training candidates only.


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ranking_loss(logits, labels, rows, margin):
    losses = []
    for row in torch.unique(rows):
        mask = rows == row
        if torch.sum(mask) < 2:
            continue
        scores = logits[mask]
        target = labels[mask]
        positive = scores[target > .5]
        negative = scores[target <= .5]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        losses.append(F.relu(margin - positive[:, None] + negative[None, :]).mean())
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def score_statistics(labels, scores):
    # The notebook promotes sigmoid float32 outputs before both moments/AUC.
    scores = np.asarray(scores, dtype=np.float64)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Each full training/validation population needs both classes")
    signal, background = scores[labels == 1], scores[labels == 0]
    difference = float(signal.mean()) - float(background.mean())
    separation = difference / np.sqrt(max(float(signal.var() + background.var()), 1e-12))
    return {"roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores)),
            "score_sep_mean_diff": difference, "score_sep_z": float(separation)}


def better_checkpoint(row, best, delta):
    if best is None:
        return True
    for name, sign in (("val_score_sep_z", 1), ("val_roc_auc", 1),
                       ("val_pr_auc", 1), ("val_total", -1)):
        if sign * row[name] > sign * best[name] + delta:
            return True
        if sign * row[name] < sign * best[name] - delta:
            return False
    return False


def run_epoch(model, parts, config, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = dict(total=0., cls=0., rank=0., reg=0.)
    count = 0
    predictions, labels = [], []
    gates = {name: [] for name in ("gate_mean", "gate_std", "gate_abs_mean")}
    positive_weight = torch.tensor([config.pos_weight], dtype=torch.float32, device=device)
    for graphs in parts:
        loader = DataLoader(graphs, batch_size=config.batch_size, shuffle=training,
                            num_workers=0, pin_memory=(device.type == "cuda"))
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                logits, energies = model(batch)
                target = batch.y_cls.view(-1)
                energy_target = batch.y_E.view(-1)
                cls = F.binary_cross_entropy_with_logits(logits.view(-1), target,
                                                         pos_weight=positive_weight)
                rank = ranking_loss(logits.view(-1), target, batch.row.view(-1), config.rank_margin)
                signal = (target > .5) & (energy_target > 0)
                reg = (F.l1_loss(energies.view(-1)[signal], energy_target[signal])
                       if torch.any(signal) else torch.zeros((), dtype=torch.float32, device=device))
                loss = cls + config.lambda_rank * rank + config.lambda_reg * reg
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite student training loss")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
            batch_count = target.numel()
            for key, value in (("total", loss), ("cls", cls), ("rank", rank), ("reg", reg)):
                totals[key] += float(value.detach().item()) * batch_count
            count += batch_count
            predictions.append(logits.sigmoid().detach().cpu().numpy())
            labels.append(target.detach().cpu().numpy().astype(int))
            if config.model_key.startswith("spectral_"):
                for name in gates:
                    gates[name].append(getattr(model, "last_" + name))
        gc.collect()
    if not count:
        raise ValueError("Empty student epoch")
    return {**{name: value / count for name, value in totals.items()},
            **score_statistics(np.concatenate(labels), np.concatenate(predictions)), "graphs": count,
            **{name: float(np.nanmean(values)) for name, values in gates.items() if values}}


def pack_graphs(graphs):
    """Tensor-only storage; graph ordering and original shard boundaries stay fixed."""
    nodes = np.cumsum([0] + [g.num_nodes for g in graphs])
    edges = np.cumsum([0] + [g.num_edges for g in graphs])
    result = {"node_ptr": torch.tensor(nodes), "edge_ptr": torch.tensor(edges),
              "x": torch.cat([g.x for g in graphs]),
              "edge_index": torch.cat([g.edge_index for g in graphs], dim=1),
              "pe": torch.cat([g.pe for g in graphs])}
    for name in ("row", "cluster", "n_hits", "y_cls", "y_E"):
        result[name] = torch.cat([getattr(g, name) for g in graphs])
    return result


def unpack_graphs(pack, spectral):
    result = []
    for i in range(len(pack["row"])):
        a, b = int(pack["node_ptr"][i]), int(pack["node_ptr"][i + 1])
        c, d = int(pack["edge_ptr"][i]), int(pack["edge_ptr"][i + 1])
        graph = Data(x=pack["x"][a:b], edge_index=pack["edge_index"][:, c:d])
        for name in ("row", "cluster", "n_hits", "y_cls", "y_E"):
            setattr(graph, name, pack[name][i:i + 1])
        if spectral:
            graph.pe = pack["pe"][a:b]
        result.append(graph)
    return result


def checkpoint_metadata(config):
    return {"in_dim": 6, "hidden_dim": 128, "num_layers": 4, "dropout": .1,
            "aggr": "mean", "heads": 4, "dynamic_knn": 8, "pe_dim": 4, "pe_hidden": 64,
            "gate_init_std": .001, "seed": config.seed, "model_key": config.model_key,
            "checkpoint_selection_metric": "val_score_sep_z_then_roc_auc_then_pr_auc_then_val_loss",
            "training_config": asdict(config)}


def train(config, train_parts, val_parts, output, *, data_hash, source_hash,
          device="cuda", resume=False, stop_after_epochs=None):
    """Run a fixed recipe; an interrupted seed continues from one atomic state."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"config": asdict(config), "data_sha256": data_hash, "source_sha256": source_hash}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    resume_path = output / "resume.pt"
    if resume_path.exists() and not resume:
        raise FileExistsError("Existing atomic state: explicit resume is required")
    set_seeds(config.seed)
    device = torch.device(device)
    model = build_student_model(config.model_key).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    history, best_row, best_state, patience, start_epoch = [], None, None, 0, 1
    if resume:
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        if blob["identity_hash"] != identity_hash:
            raise ValueError("Source/data/recipe changed across student resume")
        model.load_state_dict(blob["model"], strict=True)
        optimizer.load_state_dict(blob["optimizer"])
        history, best_row, best_state = blob["history"], blob["best_row"], blob["best_state"]
        patience, start_epoch = blob["patience"], blob["next_epoch"]
        random.setstate(blob["rng_python"])
        np.random.set_state(blob["rng_numpy"])
        torch.set_rng_state(blob["rng_torch"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in blob["rng_cuda"]])
    atomic_json(output / "identity.json", {**identity, "identity_hash": identity_hash})
    finished = bool(history and (patience >= config.patience or start_epoch > config.max_epochs))
    for epoch in range(start_epoch, config.max_epochs + 1):
        if finished:
            break
        started = time.monotonic()
        training = run_epoch(model, train_parts, config, device, optimizer)
        validation = run_epoch(model, val_parts, config, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in training.items()},
               **{f"val_{k}": v for k, v in validation.items()},
               "seconds": time.monotonic() - started}
        history.append(row)
        if better_checkpoint(row, best_row, config.min_delta):
            best_row, best_state, patience = copy.deepcopy(row), copy.deepcopy(model.state_dict()), 0
            atomic_torch(output / "best.pt", {**checkpoint_metadata(config), "epoch": epoch,
                         "state_dict": best_state, "validation": best_row})
        else:
            patience += 1
        blob = {"identity_hash": identity_hash, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history,
                "best_row": best_row, "best_state": best_state, "patience": patience,
                "next_epoch": epoch + 1, "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        atomic_torch(resume_path, blob)
        atomic_json(output / "history.json", history)
        finished = patience >= config.patience or epoch == config.max_epochs
        atomic_json(output / "status.json", {"state": "complete" if finished else "training",
                    "epoch": epoch, "best_epoch": best_row["epoch"], "patience": patience,
                    "val_roc_auc": row["val_roc_auc"], "epoch_seconds": row["seconds"]})
        print(json.dumps({"model": config.model_key, "seed": config.seed,
                          "epoch": epoch, "best_epoch": best_row["epoch"],
                          "val_auc": row["val_roc_auc"], "seconds": row["seconds"]}), flush=True)
        if finished or (stop_after_epochs is not None and len(history) - start_epoch + 1 >= stop_after_epochs):
            break
    model.load_state_dict(best_state, strict=True)
    return model.eval(), history, finished
