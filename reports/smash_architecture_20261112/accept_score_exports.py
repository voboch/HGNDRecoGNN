"""Accept one complete scope of locally downloaded SMASH score publications."""

from argparse import ArgumentParser
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory


REPORT = Path(__file__).resolve().parent
REPO = REPORT.parents[1]
RESULTS = REPO / "results/smash_architecture_20261112"
BASE = REPO / "results/smash_train_20261028"
sys.path.insert(0, str(REPO.parent))
from HGNDRecoGNN.scripts.smash_input_bundle import unpack  # noqa: E402

# The frozen scorer numbers tasks by model, then fold, then seed.
TASKS = [(model, fold, seed) for model in ("student_dynedge", "smash_hetero_sage")
         for fold in range(3) for seed in (42, 123, 456)]
TASK_IDS = {str(task) for task in range(len(TASKS))}
COLUMNS = ["Row", "ClusterID", "cl_score", "e_pred", "cl_istop", "cl_label", "e_true",
           "cluster_size", "e_pred_raw"]


def need(ok, why):
    if not ok:
        raise ValueError(why)


def read_json(path, expected=None):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    need(expected is None or digest == expected, f"SHA-256 differs: {path}")
    return json.loads(raw), digest


def receipt_inventory(directory, scope, sums):
    pattern = re.compile(rf"score_{scope}_(\d+)_(\d+)\.json")
    paths = {}
    for path in sorted(directory.glob(f"score_{scope}_*.json")):
        match = pattern.fullmatch(path.name)
        need(match is not None, f"Unexpected score receipt name: {path}")
        task = match[1]
        need(task in TASK_IDS and task not in paths, f"Unexpected or duplicate score task: {path}")
        paths[task] = path
    need(set(paths) == TASK_IDS, f"Missing score tasks: {sorted(TASK_IDS - paths.keys(), key=int)}")
    if sums is None:
        return paths, {}, None

    raw = sums.read_bytes()
    pins = {}
    for line in raw.decode().splitlines():
        record = re.fullmatch(r"([0-9a-f]{64}) [ *](\S+)", line)
        need(record is not None and record[2] not in pins, f"Malformed receipt checksum list: {sums}")
        pins[record[2]] = record[1]
    need(set(pins) == {path.name for path in paths.values()}, "Receipt checksum inventory differs")
    return paths, pins, sha256(raw).hexdigest()


def accepted_inputs(args):
    # Both scopes reuse the fit pins registered before even scoring.
    launch, _ = read_json(REPORT / "score_even_launch_plan_20261116.json")
    acceptance, _ = read_json(REPORT / "fit_acceptance_20261116.json", launch["fit_acceptance_sha256"])
    index, _ = read_json(REPORT / "fit_index_20261116.json", launch["fit_index_sha256"])
    source = launch["source"]
    need(acceptance["schema"] == "smash-architecture-fit-acceptance-v1"
         and acceptance["state"] == "accepted" and acceptance["source"] == source
         and acceptance["models"] == len(TASKS) and set(acceptance["tasks"]) == TASK_IDS
         and index["schema"] == "smash-architecture-fit-index-v1" and set(index["tasks"]) == TASK_IDS,
         "Accepted fits or fit index differ from the registered campaign")
    plan, _ = read_json(REPORT / "plan.json", source["plan_sha256"])
    split, _ = read_json(BASE / "split/split_manifest.json", plan["split_sha256"])
    prepared, _ = read_json(BASE / "prepare_4465983/summary.json", plan["prepare"]["summary_sha256"])
    need(prepared["state"] == "complete" and prepared["split_sha256"] == plan["split_sha256"],
         "Prepared inputs differ from the registered split")

    release_digest = None
    if args.scope == "odd":
        need(args.odd_release is not None and args.odd_release_sha256,
             "Odd score acceptance requires the published release and SHA-256")
        release, release_digest = read_json(args.odd_release, args.odd_release_sha256)
        need(release["schema"] == "smash-architecture-odd-release-v1"
             and release["plan_sha256"] == source["plan_sha256"]
             and release["fit_receipt_sha256"] == {t: v["sha256"] for t, v in index["tasks"].items()},
             "Odd release differs from the registered plan or accepted fits")
    else:
        need(args.odd_release is None and args.odd_release_sha256 is None, "Odd release applies to odd scoring")

    contents = {g["id"]: f"{g['tag']}/{g['kept']}" for g in split["groups"]}
    roles = [{role: sorted(contents[g] for g in fold["partitions"][short])
              for role, short in (("train", "train"), ("validation", "val"), ("test", "test"))}
             for fold in (split["folds"][str(fold)] for fold in range(3))]
    scope_files = set(plan["evaluation"][args.scope])
    fits = {}
    for task, (model, fold, seed) in enumerate(TASKS):
        task = str(task)
        accepted = acceptance["tasks"][task]
        publication = accepted["publication"]
        fit_pin = index["tasks"][task]
        need(fit_pin == {"receipt": publication["receipt"], "sha256": publication["sha256"]},
             f"Fit index differs from acceptance: task {task}")
        fit_root = RESULTS / "fits" / Path(fit_pin["receipt"]).stem
        fit, _ = read_json(fit_root / "summary.json", publication["summary_sha256"])
        identity = {"model": model, "fold": fold, "seed": seed, "epochs": accepted["epochs"],
                    "split_sha256": plan["split_sha256"], "plan_sha256": source["plan_sha256"],
                    "source_sums_sha256": source["source_sums_sha256"],
                    "prepare_manifest_sha256": plan["prepare"]["manifest_sha256"],
                    "prepare_summary_sha256": plan["prepare"]["summary_sha256"],
                    "train_files": roles[fold]["train"], "validation_files": roles[fold]["validation"]}
        comparison = {field: accepted[field] for field in ("selection", "cuts")}
        need(fit["schema"] == "smash-architecture-fit-v1" and fit["state"] == "complete"
             and fit["scientific_seed_complete"] and fit["epochs_committed"] == identity["epochs"]
             and all(fit[k] == v for k, v in (identity | comparison).items()),
             f"Fit summary differs from accepted identity or cuts: task {task}")
        test = sorted(scope_files.intersection(roles[fold]["test"]))
        empty = [cid for cid in test if prepared["files"][cid]["empty_after_cuts"]]
        need(test and all(prepared["files"][cid]["roles"][str(fold)] == "test" for cid in test),
             f"Test inventory differs from prepared roles: fold {fold}")
        fits[task] = identity | comparison | {"test_scope": args.scope, "test_files": test,
                     "empty_after_cuts": empty, "fit_receipt_sha256": fit_pin["sha256"],
                     "odd_release_sha256": release_digest}
    return launch, plan, prepared["files"], fits, release_digest


def verify_csv(path, population):
    keys, signals = set(), 0
    with path.open(newline="") as stream:
        reader = csv.reader(stream, strict=True)
        need(next(reader, None) == COLUMNS, f"Prediction columns differ: {path}")
        for fields in reader:
            try:
                row, cluster, score, energy, side, label, truth, size, raw = map(float, fields)
            except ValueError as error:
                raise ValueError(f"Non-numeric or malformed prediction: {path}:{reader.line_num}") from error
            key = row, cluster, side  # ClusterID starts over on each detector side.
            if (not all(map(isfinite, (row, cluster, score, energy, side, label, truth, size, raw)))
                    or not (row >= 0 and cluster >= 0 and row.is_integer() and cluster.is_integer()
                            and 0 <= score <= 1 and energy == raw >= 0 and truth >= 0
                            and side in (0, 1) and label in (0, 1) and size > 0 and size.is_integer())
                    or key in keys):
                raise ValueError(f"Invalid raw prediction or duplicate candidate: {path}:{reader.line_num}")
            keys.add(key)
            signals += int(label)
    need(len(keys) > 0 and len(keys) == population["clusters"] and signals == population["signal_clusters"],
         f"Prediction candidate counts differ from preparation: {path}")
    return len(keys)


def accept_publication(receipt_path, receipt_pin, destination, source, expected, prepared):
    receipt, receipt_digest = read_json(receipt_path, receipt_pin)
    bundle = receipt_path.with_suffix("")
    manifest_path, archive = bundle / "manifest.json", bundle / "inputs.tar"
    manifest, manifest_digest = read_json(manifest_path, receipt["manifest_sha256"])
    members = {row["member"]: row for row in manifest["files"]}
    need(receipt["state"] == manifest["state"] == "complete"
         and receipt["bundle_dir"] == f"{source['campaign']}/{bundle.name}"
         and receipt["archive_sha256"] == manifest["archive_sha256"]
         and archive.stat().st_size == receipt["archive_bytes"] == manifest["archive_bytes"]
         and len(members) == len(manifest["files"]) == receipt["files"],
         f"Invalid score publication: {receipt_path}")
    unpack(archive, manifest_path, destination)
    verified, _ = read_json(destination / "bundle_verified.json")
    need(verified["manifest_sha256"] == manifest_digest
         and verified["archive_sha256"] == receipt["archive_sha256"], f"Bundle changed during extraction: {bundle}")

    summary, summary_digest = read_json(destination / "summary.json", receipt["summary_files"]["summary.json"])
    predictions = summary["predictions"]
    expected_files = set(expected["test_files"]) - set(expected["empty_after_cuts"])
    need(summary["schema"] == "smash-architecture-predictions-v1" and summary["state"] == "complete"
         and all(summary[key] == value for key, value in expected.items())
         and set(predictions) == expected_files
         and set(members) == {"summary.json"} | {f"predictions/{cid}.csv" for cid in expected_files},
         f"Score fit identity, cuts, or test inventory differ: {bundle}")
    rows = 0
    for cid, entry in predictions.items():
        member = f"predictions/{cid}.csv"
        need(entry["path"] == member and entry["sha256"] == members[member]["sha256"],
             f"Prediction member does not match its manifest: {bundle}/{member}")
        rows += verify_csv(destination / member, prepared[cid]["population"])

    return {"receipt": receipt_path.name, "receipt_sha256": receipt_digest, "summary_sha256": summary_digest,
            "manifest_sha256": manifest_digest, "archive_sha256": receipt["archive_sha256"],
            "files": len(predictions), "rows": rows}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("publications", type=Path, help="Directory of score receipt, manifest and archive triplets")
    parser.add_argument("--scope", required=True, choices=("even", "odd"))
    parser.add_argument("--extract", type=Path, required=True, help="New local directory for accepted bundles")
    parser.add_argument("--receipt-sums", type=Path, help="Published sha256sum list of the 18 score receipts")
    parser.add_argument("--odd-release", type=Path)
    parser.add_argument("--odd-release-sha256")
    args = parser.parse_args()

    output = args.extract.absolute()
    need(not output.exists(), f"Extraction directory already exists: {output}")
    paths, pins, sums_digest = receipt_inventory(args.publications, args.scope, args.receipt_sums)
    launch, plan, prepared, expected, release_digest = accepted_inputs(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".score-accept-") as staging:
        stage = Path(staging) / "accepted"
        tasks = {}
        for task, path in sorted(paths.items(), key=lambda item: int(item[0])):
            name = path.stem
            record = accept_publication(path, pins.get(path.name), stage / name, launch["source"],
                                        expected[task], prepared)
            tasks[task] = record | {"bundle": str(output / name)}

        paths = [record["bundle"] for record in tasks.values()]
        result = {"schema": "smash-architecture-score-acceptance-v1", "state": "accepted", "scope": args.scope,
                  "source_sums_sha256": launch["source"]["source_sums_sha256"],
                  "plan_sha256": launch["source"]["plan_sha256"], "split_sha256": plan["split_sha256"],
                  "fit_acceptance_sha256": launch["fit_acceptance_sha256"], "fit_index_sha256": launch["fit_index_sha256"],
                  "odd_release_sha256": release_digest, "receipt_sums_sha256": sums_digest,
                  "tasks": tasks, "prediction_bundles": paths}
        with (stage / "acceptance.json").open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
        need(not output.exists(), f"Extraction directory already exists: {output}")
        stage.rename(output)
    print(json.dumps({"acceptance": str(output / "acceptance.json"), "prediction_bundles": paths}, indent=2))


if __name__ == "__main__":
    main()
