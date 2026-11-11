"""Fit paired regression branches on a frozen accepted SMASH representation."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from HGNDRecoGNN.scripts.smash_train import (  # noqa: E402
    TASKS, NativePackDataset, PackStore, build_model, epoch_checkpoint_name,
    extract_members, load_epoch, load_split, seed_everything, sha, task_partitions,
    write_json,
)
from HGNDRecoGNN.scripts.smash_transfer_infer import check_prediction  # noqa: E402

# Dombay notebook cells 52, 63, 74 and 76–79. Retain event-sized batches.
BINS, EPOCHS, BATCH = 10, 15, 256
LR, DECAY, STEP, GAMMA = 1e-4, 1e-5, 5, 0.1
ARMS = ("unit_mse", "energy_weighted_mse")


def checked_json(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Input digest changed: {path}")
    return json.loads(path.read_text())


def frozen_features(model, pack, main, positives=False):
    """Capture the regression input on the production forward path."""
    inputs, scores, old_energy = [], [], []
    captured = []

    def capture(_, args):
        captured.append(args[0])

    hook = model.core.shared_reg.register_forward_pre_hook(capture)
    try:
        for graph in DataLoader(NativePackDataset(pack), batch_size=BATCH, shuffle=False):
            with torch.no_grad():
                graph = graph.to(main)
                logits, energy = model(graph)
                feature, = captured
                captured.clear()
                if positives:
                    feature = feature[graph.cllabel.bool()]
                else:
                    scores.append(logits.sigmoid().cpu())
                    old_energy.append(energy.cpu())
                inputs.append(feature.cpu())
    finally:
        hook.remove()
    features = torch.cat(inputs)
    if positives:
        return features
    return features, torch.cat(scores).numpy(), torch.cat(old_energy).numpy()


def cache_positives(model, store, ids, fold, main):
    inputs, energies, lengths = [], [], []
    for cid, _ in store.entries(ids, "B", fold):
        pack = store.load(cid, "B", fold)
        positive = pack["label"].bool()
        prefix = np.r_[0, np.cumsum(positive.numpy(), dtype=np.int64)]
        lengths.append(np.diff(prefix[pack["cluster_ptr"].numpy()]))
        inputs.append(frozen_features(model, pack, main, positives=True))
        energies.append(pack["energy"][positive])
    features, target = torch.cat(inputs), torch.cat(energies)
    offsets = np.r_[0, np.cumsum(np.concatenate(lengths), dtype=np.int64)]
    if len(features) != len(target) or len(target) != offsets[-1]:
        raise ValueError("Frozen features and native target pointers differ")
    if not torch.isfinite(features).all() or not (target > 0).all():
        raise ValueError("Nonfinite representation or nonpositive native target")
    return features, target, offsets


def energy_weights(target):
    values = target.numpy()
    counts, edges = np.histogram(values, bins=BINS)
    if counts.sum() != len(values) or (counts > 0).sum() < 2:
        raise ValueError("Training targets do not support the Dombay weights")
    bin_weights = np.divide(
        len(values) / BINS, counts, out=np.zeros(BINS, dtype=np.float64),
        where=counts > 0,
    )
    weights = bin_weights[np.digitize(values, edges[1:-1])]
    if not (weights > 0).all():
        raise ValueError("Assigned an empty energy bin")
    # With an empty bin the Dombay mean is below one. Keep exact relative
    # occupancy weighting and normalise to isolate it from the loss scale.
    bin_weights /= weights.mean()
    weights = bin_weights[np.digitize(values, edges[1:-1])]
    return weights, {"edges_GeV": edges.tolist(), "counts": counts.tolist(),
                     "weights": bin_weights.tolist(), "max_weight": float(weights.max()),
                     "effective_positives": float(weights.sum() ** 2 / np.square(weights).sum())}


def positive_indices(offsets, events):
    starts = offsets[events]
    counts = offsets[events + 1] - starts
    group_starts = np.cumsum(counts) - counts
    return np.arange(counts.sum()) + np.repeat(starts - group_starts, counts)


def branch_energy(branch, features):
    return nn.functional.softplus(branch(features).squeeze(-1))


@torch.no_grad()
def mse(branch, inputs, targets, main):
    branch.eval()
    total = 0.0
    for features, truth in zip(inputs.split(8192), targets.split(8192), strict=True):
        error = branch_energy(branch, features.to(main)) - truth.to(main)
        total += error.square().sum(dtype=torch.float64).item()
    return total / len(targets)


def fit_branch(original, train, validation, weights, seed, main, name):
    seed_everything(seed)
    order_rng = np.random.default_rng(seed)
    branch = deepcopy(original).to(main)
    branch.requires_grad_(True)
    optimizer = torch.optim.Adam(branch.parameters(), lr=LR, weight_decay=DECAY)
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, [STEP], gamma=GAMMA)
    features, targets, offsets = train
    features, targets = features.to(main), targets.to(main)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=main)
    history = [{"epoch": 0, "validation_mse": mse(branch, *validation[:2], main)}]
    for epoch in range(1, EPOCHS + 1):
        branch.train()
        order = order_rng.permutation(len(offsets) - 1)
        total = 0.0
        for start in range(0, len(order), BATCH):
            indices = positive_indices(offsets, order[start:start + BATCH])
            if not len(indices):
                continue
            idx = torch.as_tensor(indices, device=main)
            optimizer.zero_grad(set_to_none=True)
            residual = branch_energy(branch, features[idx]) - targets[idx]
            loss = (weights[idx] * residual.square()).mean()
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(indices)
        row = {"epoch": epoch, "learning_rate": schedule.get_last_lr()[0],
               "training_objective": total / len(targets),
               "validation_mse": mse(branch, *validation[:2], main)}
        if not np.isfinite(list(row.values())).all():
            raise ValueError(f"Nonfinite fine-tuning history: {name} epoch {epoch}")
        print(name, row, flush=True)
        history.append(row)
        schedule.step()
    branch.eval()
    if not any(not torch.equal(a, b) for a, b in zip(
        original.parameters(), branch.parameters(), strict=True
    )):
        raise ValueError(f"Regression branch did not update: {name}")
    return branch, history


@torch.no_grad()
def score(branch, features, main):
    return torch.cat([branch_energy(branch, x.to(main)).cpu() for x in features.split(BATCH)]).numpy()


def run(args):
    plan = json.loads(args.plan.read_text())
    fold, seed = TASKS[args.task]
    if sha(args.split) != plan["split_sha256"]:
        raise ValueError("Split digest changed")
    split = load_split(args.split)
    train_ids, val_ids = task_partitions(split, fold)
    test_ids = sorted(split["folds"][fold]["test"])
    inputs = plan["folds"][fold]
    entry = next(x for x in inputs["runs"] if x["task"] == args.task)
    receipt = checked_json(Path(entry["receipt"]), entry["receipt_sha256"])
    summary_name = "train/complete.json"
    complete_path = extract_members(receipt["bundle_dir"], receipt["manifest_sha256"],
                                    [summary_name], args.work / "train")[summary_name]
    complete = checked_json(complete_path, entry["complete_sha256"])
    epoch = complete["selection"]["e_en"]
    name = "checkpoints/" + epoch_checkpoint_name(epoch)
    member = "train/" + name
    checkpoint = extract_members(receipt["bundle_dir"], receipt["manifest_sha256"], [member],
                                 args.work / "train", check_archive=False)[member]
    prep = plan["prepare"]
    if not (
        complete["state"] == "complete" and complete["evaluable"]
        and not complete["test_opened"] and (complete["fold"], complete["seed"]) == (fold, seed)
        and complete["identity"]["split_sha256"] == plan["split_sha256"]
        and complete["identity"]["prepare_manifest_sha256"] == prep["manifest_sha256"]
        and complete["identity"]["train_files"] == train_ids
        and complete["identity"]["validation_files"] == val_ids
    ):
        raise ValueError("Selected checkpoint belongs to a different baseline")
    store = PackStore.from_bundle(prep["bundle_dir"], prep["manifest_sha256"], args.work / "prepare")
    if store.receipt["split_sha256"] != plan["split_sha256"]:
        raise ValueError("Pack and split identities differ")
    store.fetch(train_ids + val_ids, "B", fold)
    sample = next(cid for cid, _ in store.entries(train_ids, "B", fold))
    _, model, _, placement = build_model(
        seed, NativePackDataset(store.load(sample, "B", fold)), "cuda"
    )
    load_epoch(model, checkpoint.parent, epoch, complete["files"][name])
    model.eval().requires_grad_(False)
    original = nn.Sequential(deepcopy(model.core.shared_reg), deepcopy(model.core.reg_head))
    main = placement.main
    print("Caching train and validation representations", flush=True)
    train = cache_positives(model, store, train_ids, fold, main)
    validation = cache_positives(model, store, val_ids, fold, main)
    weights, occupancy = energy_weights(train[1])
    args.output.mkdir(parents=True, exist_ok=False)
    branches = {}
    for arm, weight in zip(ARMS, (np.ones_like(weights), weights), strict=True):
        branch, history = fit_branch(original, train, validation, weight, seed, main, arm)
        branches[arm] = branch
        torch.save(branch.state_dict(), args.output / f"{arm}.pt")
        write_json(args.output / f"{arm}_history.json", history)
    del train, validation
    torch.cuda.empty_cache()

    eval_input = inputs["evaluation"]
    summary_name = "evaluation/summary.json"
    eval_paths = extract_members(eval_input["bundle_dir"], eval_input["manifest_sha256"],
                                 [summary_name], args.work / "baseline")
    eval_info = checked_json(eval_paths[summary_name], eval_input["summary_files"][summary_name])
    saved = eval_info["predictions"]["B0"][str(seed)]
    info = eval_info["training"][str(seed)]
    if not (
        eval_info["fold"] == fold and not eval_info["test_threshold_tuning"]
        and eval_info["test_files"] == test_ids
        and set(saved) == set(test_ids) and info["complete_sha256"] == sha(complete_path)
        and info["selection"]["e_en"] == epoch
    ):
        raise ValueError("Saved baseline prediction identity differs")
    csv_paths = extract_members(eval_input["bundle_dir"], eval_input["manifest_sha256"],
                                ["evaluation/" + rec["path"] for rec in saved.values()],
                                args.work / "baseline", check_archive=False)
    store.fetch(test_ids, "B", fold)
    print("Scoring only the fixed held-out files", flush=True)
    maximum_error = 0.0
    for cid in test_ids:
        pack = store.load(cid, "B", fold)
        reference = saved[cid]
        path = csv_paths["evaluation/" + reference["path"]]
        if sha(path) != reference["sha256"]:
            raise ValueError(f"Saved raw prediction digest changed: {cid}")
        frame = pd.read_csv(path)
        keys = frame[["Row", "ClusterID", "cl_istop"]].to_numpy(dtype=np.int64)
        if frame.duplicated(["Row", "ClusterID", "cl_istop"]).any():
            raise ValueError(f"Duplicate candidate: {cid}")
        # Accepted CSV rounds floats to nine significant digits. Recover the
        # original float32 target for the inherited exact native target guard.
        frame["e_true"] = frame.e_true.astype(np.float32)
        check_prediction(pack, frame)
        features, _, raw_replay = frozen_features(model, pack, main)
        old = frame.e_pred.to_numpy(dtype=np.float32)
        # Allow only float32 GPU reduction drift between the original and
        # current nodes. The reported baseline uses the original stored energy.
        np.testing.assert_allclose(raw_replay, old, atol=2e-5, rtol=2e-5, err_msg=cid)
        maximum_error = max(maximum_error, float(np.abs(raw_replay - old).max()))
        energies = {arm: score(branch, features, main) for arm, branch in branches.items()}
        if not all(np.isfinite(x).all() and (x >= 0).all() for x in energies.values()):
            raise ValueError(f"Invalid branch energy: {cid}")
        target = args.output / "predictions" / (cid + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, keys=keys, **energies)
    write_json(args.output / "summary.json", {
        "state": "complete", "task": args.task, "fold": fold, "seed": seed,
        "plan_sha256": sha(args.plan), "split_sha256": plan["split_sha256"],
        "prepare_manifest_sha256": prep["manifest_sha256"], "baseline_complete_sha256": sha(complete_path),
        "baseline_checkpoint_sha256": sha(checkpoint), "initial_epoch": epoch,
        "epochs": EPOCHS, "batch_graphs": BATCH, "arms": list(ARMS), "train_files": train_ids,
        "validation_files": val_ids, "test_files": test_ids, "training_weights": occupancy,
        "raw_replay_max_abs_GeV": maximum_error, "source_sums_sha256": sha(args.source_sums),
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plan", "split", "work", "output", "source-sums"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--task", type=int, choices=range(len(TASKS)), required=True)
    run(parser.parse_args())
