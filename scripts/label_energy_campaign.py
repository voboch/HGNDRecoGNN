"""Stage the fixed S-only label census on a compute node, then run its audit.

Reuses accepted graphs and the exact original observation recipe. No model
is constructed, fitted or scored. The smoke uses the first content ID in
each potential x even-part stratum; full uses all 286 selection groups.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts import label_energy_audit as audit  # noqa: E402
from HGNDRecoGNN.scripts import smash_train_prepare as prep  # noqa: E402


def select_groups(split, scope):
    groups = {f"{g['tag']}/{g['kept']}": g for g in split["groups"]
              if g["truth_bound"] and g["part"] in (0, 2, 4)}
    if scope == "full":
        if len(groups) != 286:
            raise ValueError("Full census must retain the registered S286 membership")
        return groups
    if scope != "smoke":
        raise ValueError("Unknown phase")
    chosen = {}
    for tag in ("zeroSpot", "defaultSpot", "bigSpot"):
        for part in (0, 2, 4):
            cid = min(c for c, g in groups.items() if g["tag"] == tag and g["part"] == part)
            chosen[cid] = groups[cid]
    return chosen


def run(args):
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    root = args.plan.resolve().parent
    paths = {key: audit.bound_path(plan[key], root) for key in ("config", "split", "prepare", "census")}
    split = json.loads(paths["split"].read_text())
    prepared = json.loads(paths["prepare"].read_text())
    # Re-observation uses precisely the accepted source recipe, not a current
    # reimplementation. The new audit/campaign files are separately versioned.
    repo = Path(__file__).resolve().parents[1]
    for name in ("scripts/smash_train_prepare.py", "scripts/audit_smash_sources.py", "data/smash_transfer.py"):
        if audit.sha256(repo / name) != prepared["scripts_sha256"][name]:
            raise ValueError("Original observation/graph code changed: " + name)
    groups = select_groups(split, args.phase)
    if args.phase == "full":
        if args.smoke is None:
            raise ValueError("A verified same-source CPU smoke is required")
        smoke = json.loads(args.smoke.read_text())
        if (smoke["state"] != "complete" or smoke["phase"] != "smoke"
                or smoke["plan_sha256"] != audit.sha256(args.plan)
                or smoke["source_sums_sha256"] != audit.sha256(args.source_sums)
                or smoke["files"] != 9 or not smoke["all_native_replays_exact"]):
            raise ValueError("Smoke did not pass on this exact plan/source")
    args.output.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=False)
    wanted = []
    for cid, group in groups.items():
        fold = (group["part"] // 2 - 1) % 3
        wanted.append("prepare/" + prepared["files"][cid]["packs"][f"B/fold_{fold}"]["path"])
    extracted = prep.extract_members(Path(plan["prepare_bundle"]), plan["prepare_bundle_manifest_sha256"],
                                     wanted, args.work / "native")
    normalized = prep.load_split(paths["split"])

    def observe(cid):
        content = normalized["contents"][cid]
        task = (cid, content, normalized["tags"][content["tag"]]["raw_root"], str(args.work / "reobserve"), None)
        _, record = prep.observe_file(task)
        if record["sha256"] != prepared["files"][cid]["observation"]["sha256"]:
            raise ValueError("Re-observation differs from the accepted bytes: " + cid)
        return cid, record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        observations = dict(pool.map(observe, sorted(groups)))
    records = {}
    for cid, group in groups.items():
        fold = (group["part"] // 2 - 1) % 3
        pack = prepared["files"][cid]["packs"][f"B/fold_{fold}"]
        records[cid] = {"observation": {"path": observations[cid]["path"], "sha256": observations[cid]["sha256"]},
                        "pack": {"path": str(extracted["prepare/" + pack["path"]]), "sha256": pack["sha256"]}}
    manifest = {"schema": "label-energy-audit-inputs-v1", "scope": args.phase,
                "config_sha256": audit.sha256(paths["config"]), "files": records,
                **{key: {"path": str(paths[key]), "sha256": audit.sha256(paths[key])}
                   for key in ("split", "prepare", "census")}}
    manifest_path = args.output / "inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.phase}: bound {len(groups)} observations and native validation packs", flush=True)
    result = audit.run(manifest_path, paths["config"], args.output / "audit")
    summary = {"schema": "label-energy-campaign-v1", "state": "complete", "phase": args.phase,
               "files": len(result["files"]), "all_native_replays_exact": all(x["native_replay_exact"] for x in result["files"].values()),
               "direct_primary_proof_bound": all(x["primary_binding"] for x in result["files"].values()),
               "plan_sha256": audit.sha256(args.plan), "source_sums_sha256": audit.sha256(args.source_sums),
               "seconds": time.monotonic() - started,
               "scope": "S-only label census, no training/inference or report-set scan"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "source-sums", "work", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args)))
