"""CPU target preparation for all registered SMASH train/validation roles.

Re-observe with the accepted deterministic recipe and replay native targets;
write only compact label/energy patches. No model or F-outcome metrics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.data.label_ablation import apply_patch, make_patch  # noqa: E402
from HGNDRecoGNN.scripts import smash_train_prepare as prep  # noqa: E402
from HGNDRecoGNN.scripts.smash_train import PackStore, source_identity  # noqa: E402
from HGNDRecoGNN.scripts.label_energy_audit import bound_path  # noqa: E402


def run(args):
    started = time.monotonic()
    source = source_identity(args.source_sums)
    plan = json.loads(args.plan.read_text())
    base = args.plan.resolve().parent
    paths = {k: bound_path(plan[k], base) for k in ("split", "prepare")}
    if (prep.sha(paths["split"]) != "848ebf492b254b75e09297b95f89d7c2773f7877afe5e629520d520904a59886"
            or prep.sha(paths["prepare"]) != "2b58f9ccce0efccb4c66afa96e93c4bbd124c1c61a22593baa05c26544de7133"):
        raise ValueError("Registered split/prepare input pin changed")
    split = prep.load_split(paths["split"])
    accepted = json.loads(paths["prepare"].read_text())
    if accepted["state"] != "complete" or accepted["split_sha256"] != prep.sha(paths["split"]):
        raise ValueError("Incomplete or mismatched accepted preparation")
    root = Path(__file__).resolve().parents[1]
    # Accepted Stage A source pins: the no-floor math/ToF reader is inherited,
    # not silently replaced alongside the target-only training adapter.
    inherited = {
        "analysis/label_energy_scan.py": "5f3e9d6112be4af37b5a785078ffec755bc0c1cb210f280d993f65b3e42184da",
        "data/graph_dataset.py": "cb75aa6b38052a5b8ac15149df5f6b33abeb70b3945c725ae2b5b9123f36c648",
    }
    if any(prep.sha(root / name) != digest for name, digest in inherited.items()):
        raise ValueError("Accepted Stage A target/ToF reader changed")
    for name in ("scripts/smash_train_prepare.py", "scripts/audit_smash_sources.py", "data/smash_transfer.py"):
        if prep.sha(root / name) != accepted["scripts_sha256"][name]:
            raise ValueError("Accepted observation/graph recipe changed: " + name)
    ids = sorted({cid for fold in split["folds"] for role in ("train", "validation") for cid in fold[role]})
    if set(ids) != set(split["contents"]) or len(ids) != 578:
        raise ValueError("Registered train/validation population changed")
    args.output.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=False)
    store = PackStore.from_bundle(Path(plan["prepare_bundle"]), plan["prepare_bundle_manifest_sha256"], args.work / "native")
    if prep.sha(store.receipt_path) != prep.sha(paths["prepare"]):
        raise ValueError("Bundle receipt differs from the accepted preparation")

    def process(cid):
        c = split["contents"][cid]
        _, obs = prep.observe_file((cid, c, split["tags"][c["tag"]]["raw_root"], str(args.work / "observed"), None))
        item = accepted["files"][cid]
        if obs["sha256"] != item["observation"]["sha256"] or item["raw_sha256"] != c["sha256"]:
            raise ValueError("Re-observation differs from accepted bytes: " + cid)
        if item["empty_after_cuts"]:
            if obs["observed_hits"]:
                raise ValueError("Previously empty observation became nonempty")
            return cid, {"empty_after_cuts": True, "observation_sha256": obs["sha256"]}
        fold = next(f["fold"] for f in split["folds"] if cid in f["train"] or cid in f["validation"])
        # Fetches are performed serially below: bundle extraction is shared state.
        native = store.load(cid, "B", fold)
        patch, audit = make_patch(prep.load_observation(obs["path"]), native)
        apply_patch(native, patch)
        rel = Path("patches") / (cid + ".pt")
        prep.atomic_torch(args.output / rel, patch)
        return cid, {**audit, "path": rel.as_posix(), "sha256": prep.sha(args.output / rel),
                     "observation_sha256": obs["sha256"], "roles": item["roles"],
                     "native_pack_sha256": {k: v["sha256"] for k, v in item["packs"].items()},
                     "empty_after_cuts": False}

    for fold in range(3):
        wanted = [cid for cid in ids if not accepted["files"][cid]["empty_after_cuts"]
                  and next(f["fold"] for f in split["folds"] if cid in f["train"] or cid in f["validation"]) == fold]
        store.fetch(wanted, "B", fold)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = dict(pool.map(process, ids))
    summary = {"schema": "smash-label-patches-v1", "state": "complete", "files": records,
               "split_sha256": prep.sha(paths["split"]), "prepare_sha256": prep.sha(paths["prepare"]),
               "plan_sha256": prep.sha(args.plan), "source_sums_sha256": source["sums_sha256"], "source": source,
               "seconds": time.monotonic()-started, "label_floor_GeV": None,
               "scope": "all registered train/validation roles, including unbound; no F metrics or model calls",
               "primary_binding_required": False, "additional_unit_tests_run": False}
    prep.write_json(args.output / "summary.json", summary)
    return {"state": "complete", "files": len(records), "summary_sha256": prep.sha(args.output / "summary.json")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "source-sums", "work", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    print(json.dumps(run(p.parse_args())))
