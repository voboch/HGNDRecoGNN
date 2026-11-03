"""Accept the actual 4482805 export; no models, fits, tests or F metrics."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.data.label_ablation import tensor_sha
from HGNDRecoGNN.scripts.smash_input_bundle import sha, unpack

folder = ROOT / "results/label_ablation_20261103/prepare_4482805"
receipt = json.loads((HERE / "prepare_4482805_completion.json").read_text())
pub = receipt["publication"]
assert "COMPLETED|0:0|00:07:02" in receipt["sacct"]
assert sha(folder / "manifest.json") == pub["manifest_sha256"]
assert sha(folder / "inputs.tar") == pub["archive_sha256"]
out = folder / "extracted"
if not out.exists():
    unpack(folder / "inputs.tar", folder / "manifest.json", out)
manifest = json.loads((folder / "manifest.json").read_text())
for f in manifest["files"]:
    assert sha(out / f["member"]) == f["sha256"]
summary = json.loads((out / "summary.json").read_text())
assert sha(out / "summary.json") == pub["summary_files"]["summary.json"]
assert summary["state"] == "complete" and summary["schema"] == "smash-label-patches-v1"
assert summary["label_floor_GeV"] is None
assert summary["source_sums_sha256"] == "921e04a4e6d1faec8e8b1a757e2a08df2b02f34d2a2f35e9d4e34c0ef750eb3c"
assert summary["plan_sha256"] == sha(HERE / "launch_plan.json")
prepare_path = ROOT / "results/smash_train_20261028/prepare_4465983/summary.json"
split_path = ROOT / "results/smash_train_20261028/split/split_manifest.json"
assert summary["prepare_sha256"] == sha(prepare_path)
assert summary["split_sha256"] == sha(split_path)
prepare = json.loads(prepare_path.read_text())
split = json.loads(split_path.read_text())
old = json.loads((ROOT / "reports/energy_label_results_20261103/audit_summary.json").read_text())
assert len(summary["files"]) == 578 and set(summary["files"]) == set(prepare["files"])
selection = {g["tag"]+"/"+g["kept"] for g in split["groups"] if g["truth_bound"] and g["part"] in (0, 2, 4)}
totals = {"clusters": 0, "native_positive": 0, "disabled_positive": 0,
          "added": 0, "disabled_zero_energy_positive": 0, "added_float32_at_0p3": 0}
stotals = dict(totals)
for cid, record in summary["files"].items():
    native = prepare["files"][cid]
    assert not record["empty_after_cuts"] and record["native_replay_exact"]
    assert record["observation_sha256"] == native["observation"]["sha256"]
    assert record["roles"] == native["roles"]
    assert record["native_pack_sha256"] == {k: v["sha256"] for k, v in native["packs"].items()}
    path = out / record["path"]
    assert sha(path) == record["sha256"]
    p = torch.load(path, weights_only=True, map_location="cpu")
    keys = p["candidate_keys"]
    assert tensor_sha(keys) == record["candidate_keys_sha256"] and keys.dtype == torch.int64
    assert keys.ndim == 2 and keys.shape[1] == 3
    assert np.unique(keys.numpy(), axis=0).shape[0] == len(keys)
    y, d = p["native_label"], p["disabled_label"]
    e, f = p["native_energy"], p["disabled_energy"]
    assert all(v.shape == (len(keys),) and v.dtype == torch.float32 and torch.isfinite(v).all() for v in (y, d, e, f))
    assert torch.all((y == 0) | (y == 1)) and torch.all((d == 0) | (d == 1))
    assert torch.all(d >= y) and torch.equal(e[y.bool()], f[y.bool()])
    assert torch.all(e >= 0) and torch.all(f >= 0)
    assert torch.all(e[~y.bool()] == 0) and torch.all(f[~d.bool()] == 0)
    added = d.bool() & ~y.bool()
    assert torch.all(f[added] <= np.float32(.3))
    for row in record["ledger"]:
        a, b = (y, e) if row["label_floor_GeV"] == .3 else (d, f)
        assert row["signal_clusters"] == int(a.sum())
        assert row["positive_zero_energy_targets"] == int((a.bool() & (b == 0)).sum())
        if cid in selection:
            original = next(z for z in old["files"][cid]["ledger"] if z["label_floor_GeV"] == row["label_floor_GeV"])
            assert row["signal_clusters"] == original["signal_clusters"]
    stats = {"clusters": len(keys), "native_positive": int(y.sum()), "disabled_positive": int(d.sum()),
             "added": int(added.sum()), "disabled_zero_energy_positive": int((d.bool() & (f == 0)).sum()),
             "added_float32_at_0p3": int((added & (f == np.float32(.3))).sum())}
    for key, value in stats.items():
        totals[key] += value
        if cid in selection:
            stotals[key] += value
assert len(selection) == 286 and stotals["added"] == 5480 and stotals["native_positive"] == 322058
proof = {"state": "accepted", "job": "4482805", "slurm": "COMPLETED/0:0,7:02,cn-035",
         "files": 578, "archive_members_verified": len(manifest["files"]),
         "source_sums_sha256": summary["source_sums_sha256"], "plan_sha256": summary["plan_sha256"],
         "manifest_sha256": pub["manifest_sha256"], "archive_sha256": pub["archive_sha256"],
         "patch_summary_sha256": sha(out / "summary.json"),
         "source_observation_and_native_replay": "all578 guarded on compute; local metadata/pins exact",
         "local_patch_checks": "all patch SHAs, keys, native/disabled support, common targets, dtype and ledger counts exact",
         "inventory_totals": totals, "S286_totals": stotals,
         "full_inventory_counts_are_not_F_model_metrics": True, "new_model_outcomes": False,
         "additional_unit_tests_run": False, "source": str(folder.relative_to(ROOT)),
         "accept_script_sha256": sha(Path(__file__))}
(HERE / "prepare_4482805_acceptance.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(proof, ensure_ascii=False, indent=2))
