"""Pin the odd cross-fit population and exact accepted truth counters."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "results/smash_train_20261028"
FIXED = ROOT / "reports/rn_score_cut_20261110/fixed_results"
TRUTH = ROOT / "results/truth_reach_20261026/aggregate_4453896_verified/result"
COUNTERS = ("nprim", "acc_n", "acc_p", "acc_n_hi", "acc_p_hi", "acc_n_lo",
            "acc_p_lo", "acc_n_lo_reach", "acc_p_lo_reach")
PINS = {
    BASE / "split/split_manifest.json": "848ebf492b254b75e09297b95f89d7c2773f7877afe5e629520d520904a59886",
    BASE / "prepare_4465983/summary.json": "2b58f9ccce0efccb4c66afa96e93c4bbd124c1c61a22593baa05c26544de7133",
    BASE / "heldout_rn/per_file_counts.json": "2f2f7ac1342651631d105701012776c539145166c4ae6826ad3522a239785218",
    FIXED.parent / "choice.json": "22c81979ca8064aed4248cb3d11bb4379048c8b2ea22a236688054e15c1a0880",
    TRUTH / "summary.json": "c6613d0b4b33d6a86b971c64c9cd0ca081491c0bcd3035ee8ff176edbdb166aa",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def check(condition, context):
    if not condition:
        raise ValueError(context)


def digest_counters(rows, values):
    return hashlib.sha256(np.column_stack((rows, values)).astype("<i8").tobytes()).hexdigest()


def main():
    for path, expected in PINS.items():
        check(sha(path) == expected, f"Accepted input changed: {path}")
    split = load(BASE / "split/split_manifest.json")
    prepared = load(BASE / "prepare_4465983/summary.json")
    check(prepared["state"] == "complete" and prepared["split_sha256"] == PINS[BASE / "split/split_manifest.json"],
          "Accepted prepare identity")
    history = load(BASE / "heldout_rn/per_file_counts.json")["primary"]["B1"]
    heldout = {row["content"]: row for records in history.values() for row in records}
    choice = load(FIXED.parent / "choice.json")
    check(choice["chosen_target_percent"] == 90 and len(choice["per_fold_seed_score_cuts"]) == 9,
          "Frozen score policy")
    reference = load(TRUTH / "summary.json")
    records = []
    proof_pins = {}
    geometry_sha = set()
    for tag in ("zeroSpot", "defaultSpot", "bigSpot"):
        path = TRUTH / tag / "proof.json"
        check(sha(path) == reference["samples"][tag]["proof_sha256"], f"{tag}: proof")
        proof_pins[str(path.relative_to(ROOT))] = sha(path)
        proof = load(path)
        geometry_sha.add(proof["source_sha256"]["hgnd_acceptance.py"])
        check(proof["status"] == "verified" and proof["coverage"] == "full"
              and proof["counter_mismatches"] == 0, f"{tag}: incomplete truth proof")
        events_path = path.with_name("events.pkl")
        progress_path = path.with_name("progress.json")
        for artifact in (events_path, progress_path):
            check(sha(artifact) == proof["artifacts_sha256"][artifact.name], str(artifact))
        events = pd.read_pickle(events_path)
        event_files = {idx: table.sort_values("Row") for idx, table in events.groupby("file_idx")}
        members = {member: idx for idx, member in enumerate(load(progress_path)["done"])}
        inputs = {row["member"]: row for row in proof["raw_files"]}
        for group in split["groups"]:
            if group["tag"] != tag or not group["truth_bound"] or group["part"] not in (1, 3, 5):
                continue
            member = group["kept"]
            cid = f"{tag}/{member}"
            native, original = prepared["files"][cid], heldout[cid]
            primary_member = group["primary"] + "_prim.csv"
            expected = inputs[primary_member]
            event = event_files[members[primary_member]]
            span = original["processed_span"]
            old = event.loc[event.Row < span]
            check(native["primary_member"] == group["primary"] and original["primary_member"] == primary_member
                  and len(event) == expected["events"] and np.array_equal(old.Row, np.arange(span)),
                  f"{cid}: primary member or processed span")
            truth = old[["acc_n_hi", "acc_n_lo_reach"]].sum().tolist()
            check(truth == original["truth_hi_lo"][1], f"{cid}: accepted MC bands")
            raw = {kind: group[f"{kind}_sha256"] for kind in ("hits", "vacs")}
            check(raw == native["raw_sha256"] and native["population"]["hits"] == native["observation"]["observed_hits"]
                  and native["observation"]["rejected_pdg0_hits"] == 0, f"{cid}: accepted observation")
            records.append({
                "content": cid, "tag": tag, "member": member, "part": group["part"],
                "fold": original["fold"], "processed_span": span, "raw_root": split["raw_roots"][tag],
                "sha256": raw, "observation": native["observation"],
                "primary_path": expected["path_read"], "primary_sha256": expected["sha256"],
                "certified_events": len(event),
                "counter_sha256": digest_counters(event.Row, event[list(COUNTERS)]),
                "open_truth_hi_lo": truth,
            })
    with np.load(FIXED / "files.npz") as fixed, np.load(FIXED / "draw_weights.npz") as draws:
        use = fixed["parts"] % 2 == 1
        lookup = {row["content"]: row for row in records}
        records = [lookup[cid] for cid in fixed["ids"][use]]
        check(len(lookup) == len(records) == 282 and np.array_equal(draws["ids"], fixed["ids"]),
              "Fixed-policy cohort")
        for record, span, part, fold, counts in zip(records, fixed["spans"][use], fixed["parts"][use],
                                                     fixed["folds"][use], fixed["truth"][use, 1], strict=True):
            check((record["processed_span"], record["part"], record["fold"]) == (span, part, fold)
                  and record["open_truth_hi_lo"] == counts.tolist(), record["content"])
    check(len(geometry_sha) == 1, "Geometry differs by potential")
    plan = {
        "files": records, "file_count": len(records), "events": sum(r["processed_span"] for r in records),
        "observed_hits": sum(r["observation"]["observed_hits"] for r in records),
        "old_source": "/home/vbocharnikov/hgnd_experiments/energy_label_20261102_cc7a4fb5e5",
        "old_source_sums_sha256": "cc7a4fb5e56c7744177e5e69d85ecb6b9896246eecf5ae04294d207b4008a8cb",
        "geometry_sha256": geometry_sha.pop(),
        "fixed_target_percent": choice["chosen_target_percent"],
        "frozen_cuts": choice["per_fold_seed_score_cuts"],
        "counter_columns": COUNTERS, "proof_sha256": proof_pins,
        "local_inputs_sha256": {str(p.relative_to(ROOT)): s for p, s in PINS.items()}
            | {str((FIXED / p).relative_to(ROOT)): sha(FIXED / p)
               for p in ("files.npz", "draw_weights.npz", "results.json")},
    }
    with (HERE / "plan.json").open("x") as output:
        json.dump(plan, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
