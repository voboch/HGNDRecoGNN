"""Train and score the registered SMASH architecture challengers on native packs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.data.student_pipeline import HeteroCandidateDataset, NativePackDataset
from HGNDRecoGNN.scripts import smash_train as base
from HGNDRecoGNN.scripts import smash_train_prepare as prep
from HGNDRecoGNN.scripts.reco_performance import threshold_scan

MODELS = ("student_dynedge", "smash_hetero_sage")
TASKS = [(model, fold, seed) for model in MODELS for fold, seed in base.TASKS]
PURITY_TARGETS = range(60, 96, 5)  # Inherited grid used to select the hardness cut.
# FP32 reductions may sum in a different order after batching or CUDA restart.
RTOL, ATOL = 1e-5, 2e-6


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_pinned(path, expected):
    require(prep.sha(path) == expected, f"Digest changed: {path}")
    return json.loads(Path(path).read_text())


def dataset_type(model):
    return HeteroCandidateDataset if model == "smash_hetero_sage" else NativePackDataset


def native_store(args, registered):
    prepare = registered["prepare"]
    store = base.PackStore.from_bundle(prepare["bundle_dir"], prepare["manifest_sha256"],
                                       args.work / "native")
    require(store.receipt["state"] == "complete"
            and store.receipt["split_sha256"] == registered["split_sha256"]
            and prep.sha(store.receipt_path) == prepare["summary_sha256"], "Native preparation changed")
    return store


def identity_for(registered, split, source, store, task):
    model, fold, seed = TASKS[task]
    train, validation = base.task_partitions(split, fold)
    for role, ids in (("train", train), ("validation", validation)):
        require(all(store.receipt["files"][cid]["roles"][str(fold)] == role for cid in ids),
                f"Pack role changed in fold {fold}: {role}")
    return {"model": model, "fold": fold, "seed": seed, "epochs": base.EPOCHS,
            "split_sha256": registered["split_sha256"],
            "prepare_manifest_sha256": registered["prepare"]["manifest_sha256"],
            "prepare_summary_sha256": prep.sha(store.receipt_path),
            "source_sums_sha256": source["sums_sha256"],
            "plan_sha256": registered["plan_sha256"],
            "train_files": train, "validation_files": validation}


def fit(args, identity, store):
    model_key, fold, seed = (identity[k] for k in ("model", "fold", "seed"))
    require(0 <= args.start_epoch < args.end_epoch <= base.EPOCHS, "Invalid epoch interval")
    mirror = args.mirror
    require(mirror is not None, "A registered mirror is required for training")
    state_path = mirror / "state.json"
    if args.start_epoch:
        state = json.loads(state_path.read_text())
        require(state["next_epoch"] == args.start_epoch, "Mirror is at a different boundary")
        require(json.loads((mirror / "effective_config.json").read_text())["identity"] == identity,
                "Mirror belongs to another scientific task")
    else:
        require(not state_path.exists(), "Task already has a mirror")
    factory = dataset_type(model_key)
    train_ids, val_ids = identity["train_files"], identity["validation_files"]
    store.fetch(train_ids + val_ids, "B", fold)
    train_ds = factory(base.concat_packs(list(base.load_packs(store, train_ids, "B", fold).values())))
    val_packs = base.load_packs(store, val_ids, "B", fold)
    val_ds = factory(base.concat_packs(list(val_packs.values())))
    finished = args.end_epoch == base.EPOCHS
    run = base.train_run(args.work / "fit", seed, train_ds, val_ds, "cuda", base.EPOCHS, identity,
                         mirror=mirror, resume_mirror=mirror if args.start_epoch else None,
                         stop_after=None if finished else args.end_epoch - 1, model_key=model_key)
    state = json.loads(state_path.read_text())
    require(state["next_epoch"] == args.end_epoch and state["job"] == os.environ["SLURM_JOB_ID"],
            "Job did not publish its registered checkpoint boundary")
    for name, digest in state["sha256"].items():
        require(prep.sha(mirror / name) == digest, f"Checkpoint mirror digest changed: {name}")
    summary = {"state": "complete", "schema": "smash-architecture-fit-v1", **identity,
               "scientific_seed_complete": finished, "start_epoch": args.start_epoch,
               "epochs_committed": args.end_epoch, "job": os.environ["SLURM_JOB_ID"],
               "mirror_state_sha256": prep.sha(state_path)}
    if finished:
        model, spec, placement, selection, hashes, training = run
        checkpoint_dir = args.work / "fit/checkpoints"
        epochs = {head: selection[head] for head in ("e_cls", "e_en")}
        selected = {}
        for epoch in sorted(set(epochs.values())):
            name = base.epoch_checkpoint_name(epoch)
            target = args.output / "checkpoints" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(checkpoint_dir / name, target)
            digest = hashes[f"checkpoints/{name}"]
            require(prep.sha(target) == digest, f"Selected checkpoint changed: {name}")
            selected[f"checkpoints/{name}"] = digest
        cls = epochs["e_cls"]
        base.load_epoch(model, checkpoint_dir, cls, selected[f"checkpoints/{base.epoch_checkpoint_name(cls)}"],
                        model_key=model_key)
        frames = {cid: base.score_pack(model, spec, placement, pack, dataset_type=factory)
                  for cid, pack in val_packs.items()}
        scan = threshold_scan(base.pooled(frames), base.THRESHOLDS)
        cuts = {}
        for target in PURITY_TARGETS:
            feasible = scan.loc[scan.purity_count >= target / 100]
            cuts[str(target)] = float(feasible.iloc[0].threshold) if len(feasible) else None
        scan.to_csv(args.output / "validation_threshold_scan.csv", index=False)
        for name in ("history.json", "effective_config.json", "environment.json"):
            shutil.copyfile(args.work / "fit" / name, args.output / name)
        summary.update(selection=epochs, checkpoints=selected, cuts=cuts, validation_score_epoch=cls,
                       validation_scan_sha256=prep.sha(args.output / "validation_threshold_scan.csv"),
                       history_sha256=prep.sha(args.output / "history.json"), training=training)
    prep.write_json(args.output / "summary.json", summary)


def score(args, identity, registered, split, store):
    model_key, fold, seed = (identity[k] for k in ("model", "fold", "seed"))
    require(args.fit_receipt and args.fit_receipt_sha256, "A pinned complete fit receipt is required")
    publication = read_pinned(args.fit_receipt, args.fit_receipt_sha256)
    bundle = publication["bundle_dir"]
    digest = publication["manifest_sha256"]
    root = args.work / "model"
    manifest = base.extract_members(bundle, digest, ["summary.json"], root)["summary.json"]
    trained = read_pinned(manifest, publication["summary_files"]["summary.json"])
    require(trained["schema"] == "smash-architecture-fit-v1" and trained["scientific_seed_complete"]
            and trained["epochs_committed"] == base.EPOCHS
            and all(trained[k] == v for k, v in identity.items()), "Fit does not match this scoring task")
    if args.test_scope == "odd":
        require(args.odd_release and args.odd_release_sha256, "Odd scoring requires a frozen policy receipt")
        release = read_pinned(args.odd_release, args.odd_release_sha256)
        require(release["plan_sha256"] == registered["plan_sha256"]
                and release["fit_receipt_sha256"][str(args.task)] == args.fit_receipt_sha256,
                "Frozen policy receipt does not cover this fit")
    names = list(trained["checkpoints"])
    paths = base.extract_members(bundle, digest, names, root, check_archive=False)
    for member, path in paths.items():
        require(prep.sha(path) == trained["checkpoints"][member], f"Checkpoint pin failed: {member}")
    ids = sorted(set(split["folds"][fold]["test"]) & set(registered["evaluation"][args.test_scope]))
    require(ids and all(store.receipt["files"][cid]["roles"][str(fold)] == "test" for cid in ids),
            "Scoring files differ from their held-out fold")
    packs = base.load_packs(store, ids, "B", fold)
    require(packs, "No candidates in this scoring scope")
    factory = dataset_type(model_key)
    _, model, spec, placement = base.build_model(seed, factory(next(iter(packs.values()))), "cuda",
                                                 model_key=model_key)
    by_head = {}
    for head, epoch in trained["selection"].items():
        name = f"checkpoints/{base.epoch_checkpoint_name(epoch)}"
        base.load_epoch(model, root / "checkpoints", epoch, trained["checkpoints"][name], model_key=model_key)
        by_head[head] = {cid: base.score_pack(model, spec, placement, pack, dataset_type=factory)
                         for cid, pack in packs.items()}
    frames = base.combine(by_head["e_cls"], by_head["e_en"])
    predictions = {}
    for cid, frame in frames.items():
        frame["e_pred_raw"] = frame.e_pred.to_numpy()
        relative = f"predictions/{cid}.csv"
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[base.PREDICTION_COLUMNS].to_csv(path, index=False, float_format="%.9g")
        predictions[cid] = {"path": relative, "sha256": prep.sha(path)}
    prep.write_json(args.output / "summary.json", {"state": "complete",
                    "schema": "smash-architecture-predictions-v1", **identity,
                    "test_scope": args.test_scope, "selection": trained["selection"],
                    "cuts": trained["cuts"], "test_files": ids,
                    "empty_after_cuts": sorted(set(ids) - packs.keys()), "predictions": predictions,
                    "fit_receipt_sha256": args.fit_receipt_sha256,
                    "odd_release_sha256": args.odd_release_sha256})


def witness_pack(sizes, seed):
    """Interleave hits from candidates of known size in one event-side graph."""
    cluster = torch.tensor([c for i in range(max(sizes)) for c, n in enumerate(sizes) if i < n])
    n, nc = len(cluster), len(sizes)
    hits = torch.randn(n, 10, generator=torch.Generator().manual_seed(seed))
    edges = (cluster[:, None] == cluster).nonzero().T.contiguous()
    return {"x": hits, "pe": torch.zeros(n, 4), "edges": edges, "cluster": cluster,
            "label": torch.arange(nc) % 2, "energy": torch.arange(nc, dtype=torch.float32),
            "row": torch.tensor([seed]), "side": torch.tensor([0]), "index": torch.tensor([0]),
            "node_ptr": torch.tensor([0, n]), "edge_ptr": torch.tensor([0, edges.shape[1]]),
            "cluster_ptr": torch.tensor([0, nc])}


def compare_nested(a, b, *, exact, path="state"):
    if isinstance(a, dict):
        require(a.keys() == b.keys(), f"Different keys at {path}")
        for key in a:
            compare_nested(a[key], b[key], exact=exact, path=f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare_nested(x, y, exact=exact, path=f"{path}[{i}]")
    elif torch.is_tensor(a):
        rtol, atol = (0, 0) if exact else (RTOL, ATOL)
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=rtol, atol=atol,
                                   msg=lambda msg: f"{path}: {msg}")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, float) and not exact:
        np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=path)
    else:
        require(a == b, f"Different values at {path}")


def floating_pairs(a, b):
    if isinstance(a, dict):
        for key in a:
            yield from floating_pairs(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        for x, y in zip(a, b, strict=True):
            yield from floating_pairs(x, y)
    elif torch.is_tensor(a) and a.is_floating_point():
        yield a.double(), b.double()
    elif isinstance(a, float):
        yield torch.tensor([a], dtype=torch.float64), torch.tensor([b], dtype=torch.float64)


def numerical_drift(a, b):
    """Describe CUDA arithmetic drift independently of the exact restore gate."""
    result = {}
    for key in ("model", "optimizer", "history"):
        squared = maximum = 0.
        count = 0
        for x, y in floating_pairs(a[key], b[key]):
            delta = x - y
            require(torch.isfinite(delta).all(), f"Nonfinite CUDA drift in {key}")
            if delta.numel():
                squared += float(delta.square().sum())
                maximum = max(maximum, float(delta.abs().max()))
                count += delta.numel()
        result[key] = {"values": count, "max_abs": maximum, "rms": (squared / count) ** .5}
    return result


def witness(args, model_key, seed):
    factory = dataset_type(model_key)
    # Exercise singleton, below-k, exact-k and above-k sizes with interleaved hits.
    sizes = (1, 2, 7, 8, 9)
    first = witness_pack(sizes, seed)
    second = witness_pack((3, 1), seed + 1)
    dataset = factory(base.concat_packs([first, second]))
    _, model, spec, placement = base.build_model(seed, dataset, "cuda", model_key=model_key)
    model.eval()
    first_graph, second_graph = dataset[0], dataset[1]

    def outputs(graphs):
        batch = Batch.from_data_list(graphs).to(placement.main)
        return spec.forward_fn(model, batch)

    with torch.no_grad():
        original = outputs([first_graph])
        combined = outputs([first_graph, second_graph])
        changed = first_graph.clone()
        changed["hits"].x[changed.cluster.reshape(-1) == len(sizes) - 1] += 2
        perturbed = outputs([changed])
        truth_free = first_graph.clone()
        for field in ("cllabel", "clenergy", "Row", "istop"):
            del truth_free[field]
        no_truth = outputs([truth_free])
        for a, b, c, d in zip(original, combined, perturbed, no_truth, strict=True):
            require(a.shape == (len(sizes),) and b.shape == (len(sizes) + 2,)
                    and torch.isfinite(b).all(), "A witness lost a candidate or yielded nonfinite outputs")
            torch.testing.assert_close(a, b[:len(a)], rtol=RTOL, atol=ATOL)
            torch.testing.assert_close(a[:-1], c[:-1], rtol=RTOL, atol=ATOL)
            torch.testing.assert_close(a, d, rtol=RTOL, atol=ATOL)
        require(any(not torch.allclose(a[-1], c[-1]) for a, c in zip(original, perturbed, strict=True)),
                "Perturbing a candidate's own hits had no effect")
    if model_key == "smash_hetero_sage":
        batch = Batch.from_data_list([first_graph, second_graph])
        membership = batch["hits", "belongs_to", "clusters"].edge_index
        require(torch.equal(membership[0], torch.arange(len(batch["hits"].x)))
                and torch.equal(batch["hits"].batch[membership[0]],
                                batch["clusters"].batch[membership[1]])
                and torch.equal(membership.flip(0), batch["clusters", "contains", "hits"].edge_index),
                "Heterogeneous batching did not preserve hit membership")
    batch = Batch.from_data_list([first_graph, second_graph]).to(placement.main)
    model.train()
    loss = spec.loss_fn(spec.forward_fn(model, batch), batch, spec.default_loss_weights)["total"]
    loss.backward()
    require(torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters()
                                        if p.grad is not None), "Witness gradients are nonfinite")

    root = args.work / "resume_witness"
    trial = {"probe": True, "model": model_key, "seed": seed}
    def run(name, **kwargs):
        return base.train_run(root / name, seed, dataset, dataset, "cuda", 2, trial,
                              model_key=model_key, **kwargs)
    mirror = root / "epoch_0_mirror"
    boundary = {}

    def read_state(checkpoints):
        return torch.load(checkpoints / base.RESUME_NAME, map_location="cpu", weights_only=False)

    def capture(epoch, checkpoints):
        if epoch != 0:
            return
        base.Mirror(checkpoints, mirror, static=(checkpoints.parent / "effective_config.json",))(epoch, None)
        state = json.loads((mirror / "state.json").read_text())
        require(state["next_epoch"] == 1, "Wrong mirror generation")
        digest = prep.sha(checkpoints / base.RESUME_NAME)
        require(digest == state["sha256"][state["resume"]] == prep.sha(mirror / state["resume"]),
                "Atomic resume blob changed while mirroring")
        name = base.epoch_checkpoint_name(epoch)
        require(prep.sha(checkpoints / name) == state["sha256"][name] == prep.sha(mirror / name),
                "Epoch checkpoint changed while mirroring")
        boundary.update(blob=read_state(checkpoints), sha256=digest, state_sha256=prep.sha(mirror / "state.json"))

    def check_live(state):
        expected = boundary["blob"]
        require(prep.sha(root / "resumed/checkpoints" / base.RESUME_NAME) == boundary["sha256"],
                "Restored atomic blob differs from the uninterrupted checkpoint")
        compare_nested({key: expected[key] for key in state}, state, exact=True, path="live_restore")
        boundary["exact_live_restore"] = True

    continuous = run("continuous", epoch_observer=capture)
    control = run("control")
    resumed = run("resumed", resume_mirror=mirror, resume_observer=check_live)
    require(boundary.get("exact_live_restore"), "The live resume state was never observed")
    a, b, c = (read_state(root / name / "checkpoints") for name in ("continuous", "control", "resumed"))
    compare_nested(boundary["blob"]["history"], c["history"][:1], exact=True, path="history_prefix")
    drift = {"uninterrupted_pair": numerical_drift(a, b),
             "same_boundary_resume": numerical_drift(a, c)}
    print(f"CUDA witness drift: {json.dumps(drift)}", flush=True)
    for peer in (b, c):
        for key in ("scheduler", "rng", "next_epoch", "best_epoch"):
            compare_nested(a[key], peer[key], exact=True, path=key)
        compare_nested(a["optimizer"]["param_groups"], peer["optimizer"]["param_groups"],
                       exact=True, path="optimizer_groups")
        steps = [{key: value["step"] for key, value in blob["optimizer"]["state"].items()}
                 for blob in (a, peer)]
        compare_nested(*steps, exact=True, path="optimizer_steps")
        compare_nested(a["history"], peer["history"], exact=False, path="cuda_loss_history")
    for head in ("e_cls", "e_en", "e_comp"):
        require(continuous[3][head] == control[3][head] == resumed[3][head], "CUDA witness changed selected epoch")
    with torch.no_grad():
        output_batch = Batch.from_data_list([first_graph, second_graph]).to(placement.main)
        output_ref = spec.forward_fn(continuous[0].eval(), output_batch)
        for trial in (control, resumed):
            output = spec.forward_fn(trial[0].eval(), output_batch)
            compare_nested(output_ref, output, exact=False, path="cuda_predictions")
    return {"candidate_sizes": sizes, "truth_free_forward": True, "candidate_isolation": True,
            "resume_training": True, "exact_live_restore": True, "cuda_drift": drift,
            "boundary_resume_sha256": boundary["sha256"], "boundary_state_sha256": boundary["state_sha256"],
            "selected_epochs": {h: resumed[3][h] for h in ("e_cls", "e_en", "e_comp")},
            "rtol": RTOL, "atol": ATOL}


def probe(args, identity, store, checks, archive_check_seconds):
    model_key, fold, seed = (identity[k] for k in ("model", "fold", "seed"))
    factory = dataset_type(model_key)
    started = time.monotonic()
    cohorts = {}
    for role in ("train", "validation"):
        ordered = sorted((cid for cid in identity[role + "_files"]
                          if not store.receipt["files"][cid]["empty_after_cuts"]),
                         key=lambda cid: (store.receipt["files"][cid]["population"]["edges"]
                                          / store.receipt["files"][cid]["population"]["graphs"], cid))
        # Sample ranks from the whole density range, including both endpoints.
        sampled = [ordered[i] for i in np.linspace(0, len(ordered) - 1, 8, dtype=int)]
        cohorts[role] = {"sampled": sampled,
                         "graphs_full_fold": sum(store.receipt["files"][cid]["population"]["graphs"]
                                                 for cid in ordered)}
    store.fetch(cohorts["train"]["sampled"] + cohorts["validation"]["sampled"], "B", fold)
    datasets = {}
    for role, cohort in cohorts.items():
        packs = base.load_packs(store, cohort["sampled"], "B", fold)
        datasets[role] = factory(base.concat_packs(list(packs.values())))
    stage_seconds = time.monotonic() - started
    _, model, spec, placement = base.build_model(seed, datasets["train"], "cuda", model_key=model_key)
    torch.cuda.reset_peak_memory_stats()
    n_batches, batch_size = 24, 256
    validation = datasets["validation"]
    # Both the warm-up and the timed slice traverse all eight validation packs.
    positions = np.concatenate([np.linspace(0, len(validation) - 1, count * batch_size, dtype=int)
                                for count in (5, n_batches)])
    validation = torch.utils.data.Subset(validation, positions.tolist())
    rates = base.throughput(model, spec, placement, datasets["train"], validation,
                            n_batches=n_batches, batch_size=batch_size)
    estimate = sum(cohort["graphs_full_fold"] * rates[role]["seconds_per_graph"]
                   for role, cohort in cohorts.items())
    prep.write_json(args.output / "summary.json", {"schema": "smash-architecture-probe-v2",
                    "state": "complete", **identity, "checks": checks, "cohorts": cohorts,
                    "input_archive_check_seconds": archive_check_seconds,
                    "sample_stage_seconds": stage_seconds, "epoch_seconds_projection": estimate,
                    "parameters": sum(p.numel() for p in model.parameters()), "rates": rates,
                    "gpu": torch.cuda.get_device_name(), "environment": base.environment(),
                    "job": os.environ["SLURM_JOB_ID"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("probe", "fit", "score"), required=True)
    parser.add_argument("--task", choices=range(len(TASKS)), type=int, required=True)
    for name in ("plan", "split", "source-sums", "work", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("mirror", "fit-receipt", "odd-release"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--fit-receipt-sha256")
    parser.add_argument("--odd-release-sha256")
    parser.add_argument("--test-scope", choices=("even", "odd"), default="even")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--end-epoch", type=int, default=base.EPOCHS)
    args = parser.parse_args()
    require(os.getenv("SLURM_JOB_ID") and torch.cuda.is_available(),
            "Use a scheduled GPU allocation for this campaign")
    source = base.source_identity(args.source_sums)
    registered = json.loads(args.plan.read_text())
    registered["plan_sha256"] = prep.sha(args.plan)
    require(prep.sha(args.split) == registered["split_sha256"], "Registered split changed")
    split = prep.load_split(args.split)
    holdout = set().union(*(set(f["test"]) for f in split["folds"]))
    even, odd = (set(registered["evaluation"][scope]) for scope in ("even", "odd"))
    require(len(even) == 286 and len(odd) == 282 and not even & odd and even | odd == holdout,
            "Registered evaluation populations differ from the original test folds")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.phase == "probe":
        model_key, _, seed = TASKS[args.task]
        checks = witness(args, model_key, seed)
    started = time.monotonic()
    store = native_store(args, registered)
    archive_check_seconds = time.monotonic() - started
    identity = identity_for(registered, split, source, store, args.task)
    if args.phase == "probe":
        probe(args, identity, store, checks, archive_check_seconds)
    elif args.phase == "fit":
        fit(args, identity, store)
    else:
        score(args, identity, registered, split, store)


if __name__ == "__main__":
    main()
