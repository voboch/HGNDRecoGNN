"""One no-floor SMASH arm using the accepted trainer, recipe, graphs and scalers.

Run a planned 15-epoch segment or its continuation to 30 total epochs. No
baseline retraining, test scoring, cut refit or soft calibration is performed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.data.label_ablation import PatchStore, apply_patch, candidate_keys  # noqa: E402
from HGNDRecoGNN.data.student_pipeline import NativePackDataset  # noqa: E402
from HGNDRecoGNN.scripts import smash_train as base  # noqa: E402
from HGNDRecoGNN.scripts import smash_train_prepare as prep  # noqa: E402


def cross_label_metrics(frame, cut):
    """Fixed-score common-reference diagnostics, including zero-energy positives."""
    out = {}
    score, energy = frame.cl_score.to_numpy(float), frame.e_pred.to_numpy(float)
    selected = score > cut
    for ref in ("native", "disabled"):
        label = frame["label_"+ref].to_numpy() == 1
        out[ref] = {"n": len(frame), "positive": int(label.sum()),
                    "auc": float(roc_auc_score(label, score)) if np.unique(label).size == 2 else None,
                    "ap": float(average_precision_score(label, score)) if label.any() else None,
                    "selected": int(selected.sum()), "selected_positive": int((selected & label).sum()),
                    "purity": float((selected & label).sum()/selected.sum()) if selected.any() else None,
                    "efficiency": float((selected & label).sum()/label.sum()) if label.any() else None}
    native, disabled = frame.label_native.to_numpy() == 1, frame.label_disabled.to_numpy() == 1
    for name, mask in (("common_positive", native & disabled), ("added_soft", disabled & ~native)):
        target = frame.energy_disabled.to_numpy(float)[mask]
        residual = energy[mask] - target
        out[name] = {"n": int(mask.sum()), "zero_energy_targets": int((target == 0).sum()),
                     "raw_bias_GeV": float(residual.mean()) if len(residual) else None,
                     "raw_mae_GeV": float(np.abs(residual).mean()) if len(residual) else None}
    return out


def train(args):
    if not os.getenv("SLURM_JOB_ID") or args.device != "cuda":
        raise ValueError("This campaign trains only in a scheduled GPU job")
    source = base.source_identity(args.source_sums)
    split = prep.load_split(args.split)
    if prep.sha(args.split) != "848ebf492b254b75e09297b95f89d7c2773f7877afe5e629520d520904a59886":
        raise ValueError("Registered split changed")
    # This is the human exception, recorded explicitly, not a GPU-smoke PASS.
    authorization = json.loads(args.authorization.read_text())
    if (authorization.get("human_instruction") != "продолжи запуски пока что без тестов, подготовь статус на завтра"
            or "waives" not in authorization.get("exception", "")):
        raise ValueError("The explicit human test-queue exception is required")
    fold, seed = base.TASKS[args.task]
    store = base.PackStore.from_bundle(args.prepare_bundle, args.prepare_manifest_sha256, args.work / "native")
    patches = PatchStore(args.patches, args.patch_manifest_sha256, prep.sha)
    manifest = patches.manifest
    if (store.receipt["state"] != "complete" or store.receipt["split_sha256"] != prep.sha(args.split)
            or manifest["split_sha256"] != prep.sha(args.split)
            or manifest["prepare_sha256"] != prep.sha(store.receipt_path)
            or prep.sha(store.receipt_path) != "2b58f9ccce0efccb4c66afa96e93c4bbd124c1c61a22593baa05c26544de7133"
            or manifest["source_sums_sha256"] != source["sums_sha256"]
            or manifest.get("label_floor_GeV", "missing") is not None):
        raise ValueError("Native packs, patches and split are not the same generation")
    baseline = json.loads(args.baseline_complete.read_text())
    if (prep.sha(args.baseline_complete) != args.baseline_complete_sha256
            or baseline["fold"] != fold or baseline["seed"] != seed or baseline["state"] != "complete"
            or baseline["epochs"] != 30 or not baseline["evaluable"]):
        raise ValueError("Frozen baseline receipt differs from this fold/seed")
    cut = baseline["cuts"]["B0"]
    if cut is None:
        raise ValueError("No accepted baseline operating point")
    role = split["folds"][fold]
    train_ids, val_ids = role["train"], role["validation"]

    def load(ids):
        native = base.load_packs(store, ids, "B", fold)
        changed, references = {}, {}
        for cid, pack in native.items():
            digest = store.receipt["files"][cid]["packs"][f"B/fold_{fold}"]["sha256"]
            patch = patches.load(cid, digest)
            changed[cid], references[cid] = apply_patch(pack, patch), patch
        return changed, references

    train_packs, _ = load(train_ids)
    val_packs, val_references = load(val_ids)
    train_ds = NativePackDataset(base.concat_packs(list(train_packs.values())))
    del train_packs
    val_ds = NativePackDataset(base.concat_packs(list(val_packs.values())))
    identity = {"arm": "SMASH_no_label_energy_clause", "source": source,
                "split_sha256": prep.sha(args.split), "prepare_sha256": prep.sha(store.receipt_path),
                "patch_manifest_sha256": args.patch_manifest_sha256,
                "authorization_sha256": prep.sha(args.authorization),
                "baseline_complete_sha256": args.baseline_complete_sha256,
                "fold": fold, "seed": seed, "epochs_total": 30}
    # Job and segment IDs are deliberately absent from scientific identity.
    execution = {"job": os.getenv("SLURM_JOB_ID"), "task": args.task, "segment": args.segment,
                 "test_queue_waived_by_human": True, "GPU_smoke_pass_claimed": False}
    if args.segment == 1:
        if args.resume_mirror is not None or (args.mirror / "state.json").exists():
            raise ValueError("First segment cannot duplicate an existing mirrored run")
    else:
        if args.resume_mirror is None:
            raise ValueError("Second segment requires its coherent mirror")
        state = json.loads((args.resume_mirror / "state.json").read_text())
        if state["next_epoch"] != 15:
            raise ValueError("Continuation must start at the registered boundary of 15 epochs")
    result = base.train_run(args.output / "train", seed, train_ds, val_ds, args.device, 30, identity,
                            mirror=args.mirror, resume_mirror=args.resume_mirror,
                            stop_after=14 if args.segment == 1 else None)
    if args.segment == 1:
        state = json.loads((args.mirror / "state.json").read_text())
        if state["next_epoch"] != 15:
            raise ValueError("Planned segment did not export a coherent 15-epoch boundary")
        for name, digest in state["sha256"].items():
            if prep.sha(args.mirror / name) != digest:
                raise ValueError("Segment mirror digest mismatch")
        prep.write_json(args.output / "segment.json", {"state": "planned_boundary", "complete_seed": False,
                        "epochs_committed": 15, "identity": identity, "execution": execution,
                        "mirror_state_sha256": prep.sha(args.mirror / "state.json")})
        prep.write_json(args.output / "summary.json", {"state": "complete", "phase": "planned_segment_1",
                        "scientific_seed_complete": False, "epochs_committed": 15,
                        "requires_continuation": True, "identity": identity, "execution": execution})
        return
    model, spec, plan, selection, hashes, run = result
    checkpoint_dir = args.output / "train/checkpoints"
    base.load_epoch(model, checkpoint_dir, selection["e_cls"], hashes[f"checkpoints/{base.epoch_checkpoint_name(selection['e_cls'])}"])
    scores = base.score_files(model, spec, plan, val_packs)
    base.load_epoch(model, checkpoint_dir, selection["e_en"], hashes[f"checkpoints/{base.epoch_checkpoint_name(selection['e_en'])}"])
    energies = base.score_files(model, spec, plan, val_packs)
    frames = base.combine(scores, energies)
    export, metrics = {}, {}
    for cid, frame in frames.items():
        keys = frame[["Row", "cl_istop", "ClusterID"]].to_numpy(np.int64)
        ref = val_references[cid]
        if not np.array_equal(keys, ref["candidate_keys"].numpy()):
            raise ValueError("Validation prediction keys differ from target patch")
        for ref_name in ("native", "disabled"):
            frame["label_"+ref_name] = ref[ref_name+"_label"].numpy()
            frame["energy_"+ref_name] = ref[ref_name+"_energy"].numpy()
        path = args.output / "validation" / (cid+".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.10g")
        export[cid] = {"path": str(path.relative_to(args.output)), "sha256": prep.sha(path)}
    merged = base.pooled(frames)
    metrics["pooled"] = cross_label_metrics(merged, cut)
    for tag in base.TAGS:
        metrics[tag] = cross_label_metrics(merged.loc[merged.tag == tag], cut)
    prep.write_json(args.output / "complete.json", {"schema": "smash-label-ablation-train-v1", "state": "complete",
                    "complete_seed": True, "epochs": 30, "fold": fold, "seed": seed, "identity": identity,
                    "execution": execution, "selection": selection, "training_hashes": hashes, "run": run,
                    "operating_point": {"cut": cut, "source": "accepted baseline B0, same fold/seed",
                                        "calibration": None, "cut_refitted": False},
                    "validation_only": True, "validation_metrics": metrics, "predictions": export,
                    "test_outcomes_opened": False, "soft_calibration_fitted": False})
    prep.write_json(args.output / "summary.json", {"state": "complete", "phase": "training_30_epochs",
                    "scientific_seed_complete": True, "epochs_committed": 30,
                    "complete_sha256": prep.sha(args.output / "complete.json"), "identity": identity,
                    "execution": execution})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-sums", "split", "authorization", "prepare-bundle", "patches", "baseline-complete", "work", "output", "mirror"):
        p.add_argument("--"+key, type=Path, required=True)
    for key in ("prepare-manifest-sha256", "patch-manifest-sha256", "baseline-complete-sha256"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--resume-mirror", type=Path)
    p.add_argument("--task", type=int, choices=range(9), required=True)
    p.add_argument("--segment", type=int, choices=(1, 2), required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    train(a)
