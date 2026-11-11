"""Accept verified Slurm bundles and compact per-task provenance."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
ARMS = ("unit_mse", "energy_weighted_mse")
SEEDS = (42, 123, 456)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    result = sha256()
    for part in iter(lambda: stream.read(1 << 20), b""):
        result.update(part)
    return result.hexdigest()


def verify_file(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Digest changed: {path}")


def accept(bundle, target, evidence, task, job):
    source = read(HERE / "source_package.json")
    receipt = read(bundle / "receipt.json")
    assert receipt["state"] == "complete"
    assert receipt["bundle_dir"] == f"{source['campaign']}/task_{task}_{job}"
    manifest = bundle / "manifest.json"
    archive = bundle / "inputs.tar"
    verify_file(manifest, receipt["manifest_sha256"])
    verify_file(archive, receipt["archive_sha256"])
    manifest_data = read(manifest)
    assert manifest_data["archive_sha256"] == receipt["archive_sha256"]
    records = {x["member"]: x for x in manifest_data["files"]}
    assert len(records) == len(manifest_data["files"]) == receipt["files"]
    temp = target.with_name(target.name + ".partial")
    assert not target.exists()
    temp.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(archive, "r") as tar:
        for item in tar:
            name = PurePosixPath(item.name)
            assert not name.is_absolute() and ".." not in name.parts and item.isfile()
            assert str(name) == item.name and item.name not in seen
            ref = records[item.name]
            assert item.size == ref["bytes"]
            path = temp / item.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(item) as stream, path.open("xb") as output:
                shutil.copyfileobj(stream, output)
            verify_file(path, ref["sha256"])
            seen.add(item.name)
    assert seen == set(records)
    verify_file(temp / "summary.json", receipt["summary_files"]["summary.json"])
    summary = read(temp / "summary.json")
    fold, seed_index = divmod(task, 3)
    plan = read(HERE / "plan.json")
    run = next(x for x in plan["folds"][fold]["runs"] if x["task"] == task)
    assert summary["state"] == "complete" and summary["task"] == task
    assert (summary["fold"], summary["seed"]) == (fold, SEEDS[seed_index])
    assert summary["baseline_complete_sha256"] == run["complete_sha256"]
    assert summary["plan_sha256"] == source["plan_sha256"] == digest(HERE / "plan.json")
    assert summary["source_sums_sha256"] == source["source_sums_sha256"]
    assert summary["prepare_manifest_sha256"] == plan["prepare"]["manifest_sha256"]
    assert summary["split_sha256"] == plan["split_sha256"]
    assert summary["epochs"] == 15 and tuple(summary["arms"]) == ARMS
    files = set(summary["test_files"])
    assert len(files) == (192, 189, 187)[fold]
    assert files.isdisjoint(summary["train_files"]) and files.isdisjoint(summary["validation_files"])
    paths = {f"predictions/{cid}.npz" for cid in files}
    assert {x for x in seen if x.startswith("predictions/")} == paths
    history = {}
    for arm in ARMS:
        name = arm + "_history.json"
        rows = read(temp / name)
        assert [x["epoch"] for x in rows] == list(range(16))
        assert [x["learning_rate"] for x in rows[1:6]] == [1e-4] * 5
        assert [x["learning_rate"] for x in rows[6:]] == [1e-5] * 10
        history[arm] = rows
    evidence.mkdir(parents=True, exist_ok=True)
    item = evidence / f"task_{task}.json"
    with item.open("x") as file:
        json.dump({"job_id": job, "summary": summary, "histories": history,
                   "receipt": receipt, "members_verified": len(seen),
                   "branch_sha256": {arm: records[arm + ".pt"]["sha256"] for arm in ARMS}}, file, indent=2)
        file.write("\n")
    temp.rename(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "target", "evidence"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("task", "job"):
        parser.add_argument("--" + name, required=True, type=int)
    args = parser.parse_args()
    accept(args.bundle, args.target, args.evidence, args.task, args.job)
