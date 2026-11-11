"""Pin the published legacy population, actual test-fold packs and checkpoints."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results/smash_train_20261028"


def main():
    pins = {}

    def load(path):
        payload = (ROOT / path).read_bytes()
        pins[str(path)] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    first = load(Path("reports/first_hit_tof_20261110/plan.json"))
    frozen = load(Path("reports/energy_head_dombay_20261111/plan.json"))
    cuts = load(Path("reports/rn_score_cut_20261110/choice.json"))["per_fold_seed_score_cuts"]
    prepare = load((RESULTS / "prepare_4465983/summary.json").relative_to(ROOT))
    receipt = load((RESULTS / "prepare_4465983/receipt.json").relative_to(ROOT))
    assert receipt == frozen["prepare"]
    assert prepare["split_sha256"] == frozen["split_sha256"]
    index, folds = {}, {}
    for job in (4466339, 4467086, 4467091):
        evaluation = load((RESULTS / f"evaluate_{job}/evaluation_summary.json").relative_to(ROOT))
        fold = evaluation["fold"]
        for cid in evaluation["test_files"]:
            assert cid not in index
            index[cid] = fold
        run_pins = {r["task"]: r for r in frozen["folds"][fold]["runs"]}
        runs = {}
        for seed, record in evaluation["training"].items():
            pin = run_pins[record["task"]]
            assert pin["complete_sha256"] == record["complete_sha256"]
            runs[seed] = {**pin, "selection": record["selection"], "cut": cuts[f"{fold}/{seed}"]}
        scaler = prepare["scalers"][f"fold_{fold}"]["pickle"]
        folds[fold] = {"runs": runs, "scaler": {**scaler, "member": "prepare/" + scaler["path"]}}
    files = []
    for row in first["files"]:
        cid = row["content"]
        fold = index[cid]
        native = prepare["files"][cid]
        assert row["part"] in (0, 2, 4) and native["roles"][str(fold)] == "test"
        assert native["raw_sha256"] == row["sha256"] and native["observation"] == row["observation"]
        assert native["population"]["hits"] == row["observation"]["observed_hits"]
        assert row["observation"]["rejected_pdg0_hits"] == 0
        pack = native["packs"][f"B/fold_{fold}"]
        fields = ("content", "tag", "member", "part", "raw_root", "sha256",
                  "observation", "processed_span", "keys_sha256")
        files.append({**{key: row[key] for key in fields}, "fold": fold,
                      "pack": {**pack, "member": "prepare/" + pack["path"]},
                      "population": native["population"]})
    assert len(files) == len({r["content"] for r in files}) == 286
    for fold, details in folds.items():
        details["witness"] = min(r["content"] for r in files if r["fold"] == fold)
    assert sum(r["population"]["hits"] for r in files) == first["expected_hits"]
    assert sum(r["population"]["clusters"] for r in files) == first["expected_clusters"]
    plan = {"schema": 1, "prepare": receipt, "split_sha256": frozen["split_sha256"],
            "geometry": prepare["dcm_inputs"]["geometry_bounds"], "folds": folds,
            "files": files, "plan_input_sha256": pins,
            "caps": {"partition": "rocky", "qos": "normal", "account": "proj_1855",
                     "constraint": "type_d", "cpus": 4, "gpus": 0, "mem": 0,
                     "time": "00:45:00", "workers": 2, "jobs": 1, "requeue": False}}
    (HERE / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
