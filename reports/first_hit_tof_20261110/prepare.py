"""Freeze the previously accepted isolated keys before reading hit times."""

import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "reports/primary_prediction_response_20261109"))
from analyze import checked, read_geometry, digest  # noqa: E402

P = ROOT / "reports/primary_association_20261106"
plan = json.loads((P / "plan.json").read_text())
summary = json.loads((P / "result/summary.json").read_text())
accept = json.loads((P / "acceptance.json").read_text())
checked(P / "result/summary.json", accept["summary_sha256"])
checked(P / "plan.json", accept["plan_sha256"])
records = {r["content"]: r for r in summary["files"]}
native = {
    r["content"]: r
    for r in json.loads(
        (ROOT / "reports/native_features_20261106/plan.json").read_text()
    )["files"]
}
parts = {}
for r in plan["files"]:
    cid = r["content"]
    record = records[cid]
    raw = checked(P / "result" / cid / "neutrons.npz", record["files"]["neutrons.npz"])
    keys, _, energy, _, _, _, prim, cluster, _ = read_geometry(raw, record, native[cid])
    parts[cid] = cluster
    r["keys_sha256"] = native[cid]["keys_sha256"]
    r["isolated_keys_sha256"] = digest(np.asarray(keys.tolist())[cluster], "<i8")
    r["isolated_truth_sha256"] = digest(energy[prim], "<f8")
    r["isolated_count"] = len(cluster)
assert len(parts) == 286 and sum(map(len, parts.values())) == 142_912
np.savez_compressed(HERE / "selection.npz", **parts)
plan["expected_isolated"] = 142_912
(HERE / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
