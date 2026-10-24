"""Finite four-model native HGND check on the approved source-run split."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN import device, models  # noqa: E402
from HGNDRecoGNN.data.student_pipeline import NativePackDataset, pack_native, population  # noqa: E402
from HGNDRecoGNN.models.student_pipeline import PIPELINE_MODELS  # noqa: E402
from HGNDRecoGNN.scripts.dcm_development import load_selected, validate_population, validate_prepared  # noqa: E402
from HGNDRecoGNN.scripts.dcm_grouped import digest, write_json  # noqa: E402
from HGNDRecoGNN.scripts.student_reproduce import environment, verify_source  # noqa: E402
from HGNDRecoGNN.training.checkpoint import load_checkpoint  # noqa: E402
from HGNDRecoGNN.training.eval import predict  # noqa: E402
from HGNDRecoGNN.training.train import TrainConfig, fit, seed_everything  # noqa: E402

MODELS = tuple(PIPELINE_MODELS)
SEEDS = (42, 123, 456)  # Approved main-pipeline training seeds, not student split seeds.
TASKS = [(name, seed) for seed in SEEDS for name in MODELS]
SPLIT_HASH = "1ab9285e3315dc0f043b7d5116255f1459978477dc9a9b4d95e046649163278b"
FROZEN_HASH = "1722523f8708a10d3b6eac43c303fd4780ea31e4df671daf25d37ca69c7a7c29"


def atomic_torch(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def recipe(name, seed, smoke=False):
    # H128/L4/dropout .1 and PE4/k8 preserve the audited student architecture.
    # Optimizer, 30 epochs, batch and MultiStepLR inherit the main reference
    # budget; this is a transfer check, not an optimisation of this budget.
    arch = {"hidden": 128, "num_layers": 4, "dropout": .1}
    if "spectral" in name:
        arch.update(pe_dim=4, pe_hidden=64, gate_init_std=.001)
    if "dynedge" in name:
        arch["k"] = 8
    return {"model": name, "seed": seed, "architecture": arch,
            "epochs": 1 if smoke else 30, "batch_size": 256,
            "optimizer": "Adam", "lr": .001, "weight_decay": 1e-5,
            "scheduler": "multistep", "milestones": [] if smoke else [10, 15, 25],
            "gamma": .1, "validation_aggregation": "population",
            "loss": "native hetero_cluster_loss: BCEWithLogits + 0.5 signal SmoothL1",
            "selector": "minimum population validation composite loss",
            "threshold": "first of 401 validation cuts [0.001,0.999] reaching count purity 0.70",
            "previously_opened_test": True}


def frozen_inputs(args):
    if digest(args.prepared / "manifest.json") != SPLIT_HASH:
        raise ValueError("Approved split manifest changed")
    if digest(args.frozen_data) != FROZEN_HASH:
        raise ValueError("Approved cache hash inventory changed")
    manifest, bundle = validate_prepared(args.prepared)
    expected = {"train": ["988975"], "validation": ["988974"], "test": ["988976"]}
    if manifest["partition_source_runs"] != expected:
        raise ValueError("Not the approved source-run split")
    if sorted(np.concatenate([bundle[k] for k in expected]).tolist()) != list(range(283398)):
        raise ValueError("Split does not cover the approved graph population exactly")
    frozen = json.loads(args.frozen_data.read_text())
    if digest(Path(manifest["data_root"]) / "processed/meta.json") != frozen["meta_sha256"]:
        raise ValueError("Frozen cache metadata changed")
    return manifest, bundle, frozen


def prepare(args, source_hash):
    args.campaign.mkdir(parents=True, exist_ok=False)
    (args.campaign / "packs").mkdir()
    manifest, bundle, frozen = frozen_inputs(args)
    # Select only train/validation. A shard may contain other indices, but they
    # are not collected, transformed, fitted or scored in this phase.
    indices = np.concatenate([bundle[k] for k in ("train", "validation")])
    graphs = load_selected(Path(manifest["data_root"]), indices, frozen["shard_lengths"], frozen["shards"])
    verified = validate_population(graphs, bundle, manifest, args.prepared)
    if verified != frozen["population"]:
        raise ValueError("Native train/validation population differs from approved inventory")
    result = {"source_sha256": source_hash, "split_sha256": SPLIT_HASH,
              "cache_inventory_sha256": FROZEN_HASH, "population": verified,
              "partition_source_runs": manifest["partition_source_runs"],
              "test_read_or_scored": False, "files": {}}
    for part in ("train", "validation"):
        pack = pack_native(graphs, bundle[part], bundle["a"], bundle["b"])
        expected = {k: v for k, v in verified[part].items() if k != "source_runs"}
        if population(pack) != expected:
            raise ValueError("Packed population differs from frozen HGND population")
        target = args.campaign / "packs" / f"{part}.pt"
        atomic_torch(target, pack)
        result["files"][target.name] = digest(target)
        print(f"Prepared {part}: {population(pack)}", flush=True)
        del pack
    write_json(args.campaign / "prepared.json", result)


def load_part(args, part, source_hash, limit=None):
    receipt = json.loads((args.campaign / "prepared.json").read_text())
    if receipt["source_sha256"] != source_hash or receipt["split_sha256"] != SPLIT_HASH:
        raise ValueError("Prepared/source identity differs")
    path = args.campaign / "packs" / f"{part}.pt"
    if digest(path) != receipt["files"][path.name]:
        raise ValueError("Native tensor pack changed")
    data = NativePackDataset(torch.load(path, map_location="cpu", weights_only=True))
    return data if limit is None else torch.utils.data.Subset(data, range(min(limit, len(data))))


def score(model, spec, loader, plan, out, part, threshold=None):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from HGNDRecoGNN.scripts.reco_performance import threshold_scan, energy_performance
    out.mkdir(exist_ok=False)
    cl = predict(model, loader, plan=plan, forward_fn=spec.forward_fn, verbose=False)["clusters"]
    keys = ["Row", "cl_istop", "ClusterID"]
    if cl.duplicated(keys).any() or not np.isfinite(cl[["cl_score", "e_pred", "e_true"]]).all(axis=None):
        raise ValueError("Invalid/duplicate native predictions")
    scan = None
    if part == "validation":
        # Inherited comparison convention; a stricter cut trades efficiency
        # for purity. Save the full objective instead of only the first cut.
        scan = threshold_scan(cl, np.linspace(.001, .999, 401))
        eligible = scan.loc[scan.purity_count >= .70]
        threshold = float(eligible.iloc[0].threshold) if len(eligible) else None
        scan.to_csv(out / "threshold_scan.csv", index=False)
        plot_scan(scan, threshold, out)
    signal = cl.loc[(cl.cl_label == 1) & (cl.e_true > 0)]
    residual = signal.e_pred.to_numpy() - signal.e_true.to_numpy()
    result = {"scored_on": f"native HGND {part} source run " + ("988974" if part == "validation" else "988976"),
              "n_clusters": len(cl), "n_signal": int(cl.cl_label.sum()),
              "roc_auc": float(roc_auc_score(cl.cl_label, cl.cl_score)),
              "average_precision": float(average_precision_score(cl.cl_label, cl.cl_score)),
              "working_threshold": threshold,
              "working_point": threshold_scan(cl, [threshold]).iloc[0].to_dict() if threshold is not None else None,
              "energy_all_signal": {"n": len(signal), "mae_GeV": float(np.mean(np.abs(residual))),
                                    "rmse_GeV": float(np.sqrt(np.mean(residual**2))),
                                    "bias_GeV": float(np.mean(residual))},
              "energy_selected_signal": energy_performance(cl, threshold).to_dict("records") if threshold is not None else [],
              "energy_population": "native MC-positive clusters with e_true>0; not neutron-track resolution",
              "previously_opened_test": part == "test"}
    cl.to_csv(out / "predictions.csv", index=False, float_format="%.9g")
    result["predictions_sha256"] = digest(out / "predictions.csv")
    write_json(out / "metrics.json", clean(result))
    return clean(result)


def plot_scan(scan, threshold, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from HGNDRecoGNN.scripts.plot_b_full import apply_style
    apply_style()
    fig, ax = plt.subplots(figsize=(6.84, 3.5))
    ax.plot(scan.threshold, scan.purity_count, label="count purity")
    ax.plot(scan.threshold, scan.efficiency_count, ls="--", label="count efficiency")
    ax.axhline(.70, color=".3", ls=":", label="inherited purity target 0.70")
    if threshold is not None:
        ax.axvline(threshold, color=".3", ls="--")
    ax.set(xlabel="reconstructed cluster score cut (dimensionless)", ylabel="fraction (dimensionless)", ylim=(0, 1.03))
    ax.text(.02, .03, "Reconstructed selection / MC truth labels\nDCM reused validation 988974; uncertainty not estimated", transform=ax.transAxes)
    ax.legend()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"threshold_scan.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build(name, seed, dataset, smoke=False):
    seed_everything(seed)
    config = recipe(name, seed, smoke)
    model, spec = models.get(name, dataset, **config["architecture"])
    plan = device.plan_for_spec(model, "cuda", spec.default_cpu_pinned)
    device.to_device(model, plan)
    return config, model, spec, plan


def run_train(args, source_hash, name, seed, smoke=False):
    prefix = "smoke_" if smoke else ""
    out = args.campaign / f"{prefix}{name}_{seed}"
    out.mkdir(exist_ok=False)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train = load_part(args, "train", source_hash, 512 if smoke else None)
        validation = load_part(args, "validation", source_hash, 256 if smoke else None)
        config, model, spec, plan = build(name, seed, train, smoke)
        config.update(source_sha256=source_hash, split_sha256=SPLIT_HASH,
                      prepared_sha256=digest(args.campaign / "prepared.json"),
                      parameters=sum(p.numel() for p in model.parameters()))
        write_json(out / "effective_config.json", config)
        write_json(out / "environment.json", environment())
        cfg = TrainConfig(epochs=config["epochs"], batch_size=256, seed=seed,
                          arch_name=name, arch_kwargs=config["architecture"],
                          lr=config["lr"], weight_decay=config["weight_decay"],
                          loss_weights=spec.default_loss_weights, checkpoint_dir=str(out / "checkpoints"),
                          scheduler_milestones=tuple(config["milestones"]),
                          save_best_on="val", validation_aggregation="population", verbose=False)
        write_json(out / "train_config.json", asdict(cfg))
        loaders = {"train": DataLoader(train, batch_size=256, shuffle=True),
                   "validation": DataLoader(validation, batch_size=256, shuffle=False)}
        print(f"Training {name} seed {seed}, {config['parameters']} parameters", flush=True)
        fit(model, loaders["train"], loaders["validation"], cfg, plan,
            forward_fn=spec.forward_fn, loss_fn=spec.loss_fn)
        resume = torch.load(out / "checkpoints/resume.pt", weights_only=False, map_location="cpu")
        if resume["next_epoch"] != cfg.epochs or len(resume["history"]) != cfg.epochs:
            raise ValueError("Incomplete training budget")
        if any(x["validation"]["aggregation"] != "population" for x in resume["history"]):
            raise ValueError("Population selector was not applied")
        selected = int(np.argmin([x["validation"]["total"] for x in resume["history"]]))
        checkpoint = load_checkpoint(str(out / "checkpoints/model.pt"))
        if checkpoint.epoch != selected or resume["best_epoch"] != selected:
            raise ValueError("Selected checkpoint is not the first validation minimum")
        write_json(out / "history.json", resume["history"])
        model.load_state_dict(checkpoint.state_dict)
        if smoke:
            check_isolation(model, spec, validation, plan)
            # Atomic resume must describe the actual last model/optimizer;
            # never combine newer separate files from different generations.
            last = load_checkpoint(str(out / "checkpoints/last.pt"))
            if last.epoch + 1 != resume["next_epoch"]:
                raise ValueError("Atomic resume and last checkpoint disagree")
            for key, value in last.state_dict.items():
                torch.testing.assert_close(value, resume["model"][key], rtol=0, atol=0)
            replay = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            replay.load_state_dict(resume["optimizer"])
            from HGNDRecoGNN.training.train import _restore_rng_state
            rng = {"cuda": torch.cuda.get_rng_state_all()}
            expected = torch.rand(16, device=plan.main)
            _restore_rng_state(rng)
            torch.testing.assert_close(torch.rand(16, device=plan.main), expected, rtol=0, atol=0)
            metric = {"working_threshold": None, "engineering_only": True}
        else:
            metric = score(model, spec, loaders["validation"], plan, out / "validation", "validation")
        receipt = {"model": name, "seed": seed, "epochs": cfg.epochs, "selected_epoch": selected,
                   "config": config, "validation_threshold": metric["working_threshold"],
                   "validation_sha256": digest(out / "validation/metrics.json") if not smoke else None,
                   "files": {str(p.relative_to(out)): digest(p) for p in
                             (out / "checkpoints/model.pt", out / "checkpoints/resume.pt", out / "history.json")}}
        write_json(out / "complete.json", receipt)
        print(f"Completed {name} seed {seed}, selected epoch {selected}", flush=True)


def check_isolation(model, spec, data, plan):
    from torch_geometric.data import Batch
    # Native singleton candidates are retained. Include a deliberately tiny
    # observable graph as well as full events in the actual GPU kNN check.
    tiny = data[0].clone()
    tiny["hits"].x = tiny["hits"].x[:1]
    tiny["hits"].student_pe = torch.zeros((1, 4))
    tiny["hits", "hits"].edge_index = torch.tensor([[0], [0]])
    tiny["clusters"].x = torch.zeros((1, 1))
    tiny.cluster = torch.zeros((1, 1), dtype=torch.long)
    tiny.cllabel = tiny.cllabel[:1]
    tiny.clenergy = tiny.clenergy[:1]
    graphs = [data[0], data[1], tiny]
    model.eval()
    with torch.no_grad():
        separate = [spec.forward_fn(model, Batch.from_data_list([g]).to(plan.main)) for g in graphs]
        together = spec.forward_fn(model, Batch.from_data_list(graphs).to(plan.main))
        for head in range(2):
            torch.testing.assert_close(torch.cat([v[head] for v in separate]), together[head], rtol=1e-4, atol=1e-5)
        # Forward uses no truth fields. Removing them must leave scores equal.
        observable = Batch.from_data_list(graphs).to(plan.main)
        del observable.cllabel
        del observable.clenergy
        again = spec.forward_fn(model, observable)
        for a, b in zip(together, again):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def require_complete(args, source_hash):
    receipts = []
    for name, seed in TASKS:
        out = args.campaign / f"{name}_{seed}"
        receipt = json.loads((out / "complete.json").read_text())
        config = receipt["config"]
        if (receipt["model"] != name or receipt["seed"] != seed or receipt["epochs"] != 30
                or config["source_sha256"] != source_hash or config["split_sha256"] != SPLIT_HASH
                or config["prepared_sha256"] != digest(args.campaign / "prepared.json")):
            raise ValueError("Incomplete or inconsistent training cohort")
        for file, expected in receipt["files"].items():
            if digest(out / file) != expected:
                raise ValueError("Completed training artifact changed")
        if digest(out / "validation/metrics.json") != receipt["validation_sha256"]:
            raise ValueError("Frozen validation threshold artifact changed")
        receipts.append(receipt)
    return receipts


def test(args, source_hash):
    receipts = require_complete(args, source_hash)  # Barrier BEFORE reading test graphs.
    out = args.campaign / "test_evaluation"
    out.mkdir(exist_ok=False)  # No automatic repeated opening, even after failure.
    manifest, bundle, frozen = frozen_inputs(args)
    graphs = load_selected(Path(manifest["data_root"]), bundle["test"], frozen["shard_lengths"], frozen["shards"])
    pack = pack_native(graphs, bundle["test"], bundle["a"], bundle["b"])
    pop = population(pack)
    if pop["graphs"] != 131173 or pop["physical_events"] != 91081 or pop["clusters"] != 222251:
        raise ValueError("Test population differs from approved HGND cohort")
    from HGNDRecoGNN.scripts.reco_truth_join import to_local
    import pandas as pd
    offsets = pd.read_csv(args.prepared / "source_row_offsets.csv", dtype={"run": str, "stem": str})
    local = to_local(pack["row"].numpy(), offsets)
    if local.unmapped.any() or set(local.run) != {"988976"}:
        raise ValueError("Test graphs do not belong only to approved source 988976")
    atomic_torch(out / "test_pack.pt", pack)
    write_json(out / "population.json", {**pop, "test_pack_sha256": digest(out / "test_pack.pt")})
    del graphs
    dataset = NativePackDataset(pack)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    results = []
    for receipt in receipts:
        name, seed = receipt["model"], receipt["seed"]
        _, model, spec, plan = build(name, seed, dataset)
        path = args.campaign / f"{name}_{seed}" / "checkpoints/model.pt"
        model.load_state_dict(load_checkpoint(str(path)).state_dict)
        metric = score(model, spec, loader, plan, out / f"{name}_{seed}", "test", receipt["validation_threshold"])
        metric.update(model=name, seed=seed, checkpoint_sha256=digest(path), selected_epoch=receipt["selected_epoch"])
        results.append(metric)
        write_json(out / "per_seed.json", results)
        del model
        torch.cuda.empty_cache()
    summary = {}
    for name in MODELS:
        rows = [r for r in results if r["model"] == name]
        if sorted(r["seed"] for r in rows) != sorted(SEEDS):
            raise ValueError("Missing test seed")
        summary[name] = {}
        for key in ("roc_auc", "average_precision"):
            values = np.array([r[key] for r in rows])
            summary[name][key] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)), "n_seeds": 3}
        for key in ("mae_GeV", "rmse_GeV", "bias_GeV"):
            values = np.array([r["energy_all_signal"][key] for r in rows])
            summary[name][key] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)), "n_seeds": 3}
    write_json(out / "summary.json", {"models": summary, "population": pop, "source_sha256": source_hash,
               "split_sha256": SPLIT_HASH, "uncertainty": "sample SD over training seeds; no between-source variance",
               "test_scope": "previously opened source run 988976; not an independent confirmatory test",
               "test_threshold_tuning": False})
    write_json(out / "complete.json", {"runs": len(results), "summary_sha256": digest(out / "summary.json"),
                                     "per_seed_sha256": digest(out / "per_seed.json")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["prepare", "smoke", "train", "test"])
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--frozen-data", type=Path, required=True)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--task", type=int, choices=range(len(TASKS)), default=0)
    args = p.parse_args()
    torch.set_num_threads(int(os.getenv("SLURM_CPUS_PER_TASK", "4")))
    source_hash = verify_source(args.source_manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("This campaign must execute on an allocated CUDA GPU")
    try:
        if args.phase == "prepare":
            prepare(args, source_hash)
        elif args.phase == "smoke":
            for name in MODELS:
                run_train(args, source_hash, name, SEEDS[0], smoke=True)
            write_json(args.campaign / "smoke_complete.json", {"models": list(MODELS), "source_sha256": source_hash,
                       "prepared_sha256": digest(args.campaign / "prepared.json"), "test_scored": False,
                       "gpu": environment(), "singleton_and_batch_isolation": "passed"})
        elif args.phase == "train":
            smoke = json.loads((args.campaign / "smoke_complete.json").read_text())
            if smoke["source_sha256"] != source_hash or smoke["prepared_sha256"] != digest(args.campaign / "prepared.json"):
                raise ValueError("Passed GPU smoke differs from requested campaign")
            run_train(args, source_hash, *TASKS[args.task])
        else:
            test(args, source_hash)
    except BaseException as exc:
        if args.campaign.exists():
            write_json(args.campaign / f"failure_{args.phase}_{args.task}_{os.getenv('SLURM_JOB_ID', 'local')}.json",
                       {"error": repr(exc), "traceback": traceback.format_exc(), "environment": environment()})
        raise


if __name__ == "__main__":
    main()
