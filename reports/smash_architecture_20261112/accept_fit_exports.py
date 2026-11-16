"""Verify the published 72 fit segments and selected weights of 18 models.

This is acceptance of existing experiment artifacts, not a new training/test run.
Only the accepted selected checkpoints are deserialized, locally, on CPU.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tarfile

import torch


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, obj):
    raw = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    if path.exists():
        need(path.read_text() == raw, f"Existing acceptance differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--extract", type=Path, required=True)
    args = parser.parse_args()
    report = Path(__file__).resolve().parent
    repo = report.parents[1]
    launch = json.loads((report / "fit_launch_plan.json").read_text())
    registered_bytes = (report / "plan.json").read_bytes()
    source = launch["source"]
    need(sha(registered_bytes) == source["plan_sha256"], "Scientific plan changed")
    registered = json.loads(registered_bytes)
    prepared_bytes = (repo / "results/smash_train_20261028/prepare_4465983/summary.json").read_bytes()
    need(sha(prepared_bytes) == registered["prepare"]["summary_sha256"], "Preparation changed")
    prepared = json.loads(prepared_bytes)
    tasks = [(m, f, s) for m in ("student_dynedge", "smash_hetero_sage")
             for f in range(3) for s in (42, 123, 456)]
    summaries, publications, all_members = {}, {}, 0
    terminal = {}
    raw_sacct = (repo / "reports/project_audit_20261116/monitor/sacct_raw.txt").read_text()
    for row in raw_sacct.splitlines():
        if row.startswith("45117"):
            sid, state, code, elapsed, start, end = row.split("|")
            need(state == "COMPLETED" and code == "0:0", f"Unfinished segment {sid}")
            terminal[sid] = {"state": state, "exit_code": code, "seconds": int(elapsed),
                             "start_msk": start, "end_msk": end}
    boundaries = [(0, 8), (8, 16), (16, 23), (23, 30)]
    with tarfile.open(args.archive) as outer:
        names = outer.getnames()
        need(len(names) == len(set(names)) == 216, "Duplicate/missing publication members")
        for name in names:
            match = re.fullmatch(r"fit_(\d+)_(\d+)_(\d+)_(\d+)\.json", name)
            if not match:
                continue
            task, start, end, job = map(int, match.groups())
            need(task in range(18) and (start, end) in boundaries, f"Unexpected case {name}")
            key = (task, start, end)
            need(key not in summaries, f"Duplicate segment {key}")
            raw_receipt = outer.extractfile(name).read()
            receipt = json.loads(raw_receipt)
            stem = name[:-5]
            mb = outer.extractfile(stem + "/manifest.json").read()
            ab = outer.extractfile(stem + "/inputs.tar").read()
            manifest = json.loads(mb)
            need(receipt["state"] == manifest["state"] == "complete", name)
            need(receipt["bundle_dir"] == source["campaign"] + "/" + stem, "Wrong campaign")
            need(sha(mb) == receipt["manifest_sha256"], "Publication manifest changed")
            need(sha(ab) == receipt["archive_sha256"] == manifest["archive_sha256"], "Archive SHA")
            need(len(ab) == receipt["archive_bytes"] == manifest["archive_bytes"], "Archive length")
            files = {r["member"]: r for r in manifest["files"]}
            need(len(files) == receipt["files"] == len(manifest["files"]), "Member counts")
            with tarfile.open(fileobj=io.BytesIO(ab)) as inner:
                need(set(inner.getnames()) == set(files) and len(inner.getnames()) == len(files), "Archive inventory")
                data = {}
                for member, record in files.items():
                    path = Path(member)
                    need(not path.is_absolute() and ".." not in path.parts, "Unsafe member")
                    b = inner.extractfile(member).read()
                    need(len(b) == record["bytes"] and sha(b) == record["sha256"], "Member SHA: " + member)
                    data[member] = b
            all_members += len(data)
            need(sha(data["summary.json"]) == receipt["summary_files"]["summary.json"], "Summary binding")
            summary = json.loads(data["summary.json"])
            model, fold, seed = tasks[task]
            expected = {"model": model, "fold": fold, "seed": seed, "epochs": 30,
                        "split_sha256": registered["split_sha256"],
                        "prepare_manifest_sha256": registered["prepare"]["manifest_sha256"],
                        "prepare_summary_sha256": registered["prepare"]["summary_sha256"],
                        "source_sums_sha256": source["source_sums_sha256"],
                        "plan_sha256": source["plan_sha256"],
                        "train_files": sorted(prepared["folds"][fold]["train"]),
                        "validation_files": sorted(prepared["folds"][fold]["validation"])}
            need(all(summary[k] == v for k, v in expected.items()), f"Identity changed: {stem}")
            need(summary["schema"] == "smash-architecture-fit-v1" and summary["state"] == "complete"
                 and summary["job"] == str(job) and summary["start_epoch"] == start
                 and summary["epochs_committed"] == end
                 and summary["scientific_seed_complete"] == (end == 30), "Fit completion metadata")
            stage = boundaries.index((start, end))
            sid = f"{4511720 + stage}_{task}"
            need(sid in terminal and terminal[sid]["seconds"] <= launch["segments"][stage]["time_limit_seconds"], "Slurm/cap")
            summaries[key] = summary
            publications[key] = {"receipt": source["receipts"] + "/" + name,
                                 "sha256": sha(raw_receipt), "publication": receipt,
                                 "summary_sha256": sha(data["summary.json"]), "slurm": terminal[sid]}
            dest = args.extract / stem
            dest.mkdir(parents=True, exist_ok=True)
            for member, b in data.items():
                target = dest / member
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    need(target.read_bytes() == b, "Previously extracted bytes differ")
                else:
                    target.write_bytes(b)
    need(len(summaries) == 72, "Expected 72 segments")
    accepted, index = {}, {}
    for task, (model, fold, seed) in enumerate(tasks):
        summary = summaries[(task, 23, 30)]
        pub = publications[(task, 23, 30)]
        dest = args.extract / Path(pub["receipt"]).stem
        config = json.loads((dest / "effective_config.json").read_text())
        history_raw = (dest / "history.json").read_bytes()
        hist = json.loads(history_raw)
        need(sha(history_raw) == summary["history_sha256"], "History digest")
        need([h["epoch"] for h in hist] == list(range(30)), "Missing history epochs")
        need(config["identity"] == {k: summary[k] for k in config["identity"]}, "Effective identity")
        need(config["resumed"]["next_epoch"] == 23
             and config["resumed"]["state_sha256"] == summaries[(task, 16, 23)]["mirror_state_sha256"]
             and config["resumed"]["interrupted_job"] == summaries[(task, 16, 23)]["job"], "Final resume boundary")
        need(config["epochs"] == 30 and config["batch_size"] == 256 and config["optimizer"] == "Adam"
             and config["lr"] == .001 and config["weight_decay"] == 1e-5
             and config["milestones"] == [10, 15, 25] and config["gamma"] == .1, "Recipe changed")
        need(not summary["training"]["mirror"]["failures"], "Unresolved final mirror failures")
        for h in hist:
            v = h["validation"]
            need(v["aggregation"] == "population", "Validation aggregation")
            need(math.isclose(h["lr_used"], .001 * .1 ** sum(h["epoch"] >= m for m in [10, 15, 25]), rel_tol=1e-12), "Schedule reset")
            for role in ("train", "validation"):
                pop = [prepared["files"][c]["population"] for c in config["identity"][role + "_files"]]
                expected_counts = {"cluster_cls": sum(p["clusters"] for p in pop),
                                   "cluster_energy": sum(p["signal_clusters"] for p in pop)}
                need(h[role]["counts"] == expected_counts, "Training/validation candidate denominator")
                need(all(math.isfinite(float(x)) for x in h[role]["components"].values()), "Nonfinite history")
        selection = {head: min(range(30), key=lambda i: hist[i]["validation"]["components"][component])
                     for head, component in (("e_cls", "cluster_cls"), ("e_en", "cluster_energy"))}
        need(selection == summary["selection"] and summary["validation_score_epoch"] == selection["e_cls"], "Head selection")
        expected_checkpoints = {f"checkpoints/epoch_{e:03d}.pt" for e in selection.values()}
        need(set(summary["checkpoints"]) == expected_checkpoints, "Selected checkpoint inventory")
        for name, expected_sha in summary["checkpoints"].items():
            b = (dest / name).read_bytes()
            need(sha(b) == expected_sha, "Selected checkpoint binding")
            blob = torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
            need(blob["format_version"] == 1 and blob["arch_name"] == model
                 and blob["arch_kwargs"] == config["architecture"]
                 and name == f"checkpoints/epoch_{blob['epoch']:03d}.pt", "Checkpoint metadata")
            need(all(torch.isfinite(t).all().item() for t in blob["state_dict"].values()), "Nonfinite weights")
        scan_bytes = (dest / "validation_threshold_scan.csv").read_bytes()
        need(sha(scan_bytes) == summary["validation_scan_sha256"], "Validation scan digest")
        scan = list(csv.DictReader(io.StringIO(scan_bytes.decode())))
        need(len(scan) == 401 and all(math.isclose(float(r["threshold"]), .001 + i * .998/400, abs_tol=1e-12)
                                    for i, r in enumerate(scan)), "Threshold grid")
        cuts = {str(t): next((float(r["threshold"]) for r in scan if float(r["purity_count"]) >= t/100), None)
                for t in range(60, 96, 5)}
        need(cuts == summary["cuts"], "Cut rule differs")
        accepted[str(task)] = {"model": model, "fold": fold, "seed": seed, "epochs": 30,
                              "selection": selection, "budget_limited": {h: e == 29 for h, e in selection.items()},
                              "cuts": cuts, "parameters": config["parameters"],
                              "validation_loss_at_selection": {h: hist[e]["validation"]["components"] for h, e in selection.items()},
                              "history_sha256": summary["history_sha256"], "checkpoints": summary["checkpoints"],
                              "publication": pub, "resume_boundary_verified": 23}
        index[str(task)] = {"receipt": pub["receipt"], "sha256": pub["sha256"]}
    record = {"schema": "smash-architecture-fit-acceptance-v1", "state": "accepted",
              "source": source, "archive_sha256": sha(args.archive.read_bytes()),
              "segments": 72, "models": 18, "archive_members_verified": all_members,
              "checks": ["72 Slurm completions/caps", "all publication/archive/member SHA and sizes",
                         "model/fold/seed/roles/native source identity", "30 contiguous population-loss epochs",
                         "native train/validation candidate counts each epoch", "unreset LR schedule",
                         "final resume matches published 23-epoch boundary", "per-head earliest validation minima",
                         "selected weights SHA, metadata and finite tensors", "all 401 threshold points and eight cuts"],
              "limits": "No new inference; exact RNG restore uses inherited live probe and trainer guards. Full intermediate resume blobs were not downloaded.",
              "tasks": accepted}
    write(report / "fit_acceptance_20261116.json", record)
    write(report / "fit_index_20261116.json", {"schema": "smash-architecture-fit-index-v1", "tasks": index})
    print(json.dumps({k: record[k] for k in ("state", "segments", "models", "archive_members_verified")}, indent=2))
    print("Selected head epochs", {k: v["selection"] for k, v in accepted.items()})


if __name__ == "__main__":
    main()
