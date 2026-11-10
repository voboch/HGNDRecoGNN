"""Pair accepted odd-file counts with the exact rule and primary replay."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd

from prepare import BASE, FIXED, HERE, ROOT, check, load, sha

BENCHMARK = HERE.parent / "rule_tof_benchmark_20261110"
sys.path.insert(0, str(BENCHMARK))
from contrast import ARMS, TAGS, reduce_spectrum  # noqa: E402

EVALUATIONS = {
    4466339: "30e456e22b83d78b1d947247f338da771353b7918bbf3e5f3c3ebb7eecdcd6a7",
    4467086: "6306bc293bdce9011ac5800929e4fd373ee63a86de3089bfbcde57e7254f984a",
    4467091: "2b4fcbb19376874674a0b9fb7ef675a14780266f9755f1e734ce131e09b7154c",
}
SEEDS = ("42", "123", "456")
COLUMNS = ["Row", "ClusterID", "cl_istop", "cl_score", "e_pred"]


def registry():
    out = {}
    for job, expected in EVALUATIONS.items():
        root = BASE / f"evaluate_{job}"
        path = root / "evaluation_summary.json"
        check(sha(path) == expected, f"Evaluation identity changed: {job}")
        meta = load(path)
        check(meta["state"] == "complete" and not meta["test_threshold_tuning"], str(path))
        for cid in meta["test_files"]:
            check(cid not in out, f"Duplicate evaluation file {cid}")
            out[cid] = root / "extracted/evaluation", meta
    return out


def frozen(plan):
    for relative, expected in plan["local_inputs_sha256"].items():
        check(sha(ROOT / relative) == expected, f"Accepted input changed: {relative}")
    files = plan["files"]
    ids = [r["content"] for r in files]
    with np.load(FIXED / "files.npz") as fixed, np.load(FIXED / "draw_weights.npz") as saved:
        odd = np.isin(fixed["parts"], (1, 3, 5))
        check(ids == fixed["ids"][odd].tolist() and np.array_equal(saved["ids"], fixed["ids"]),
              "Odd file or draw order changed")
        draws = saved["weights"][:, odd]
        tags, parts, edges = fixed["tags"][odd], fixed["parts"][odd], fixed["edges"]
        hist = fixed["spectra"][odd, :, 1, 5, :]
        selected = fixed["objects"][odd, :, 1, 2]
    check(draws.shape == (400, 282) and (draws >= 0).all(), "Archived odd-file draws")
    for tag in TAGS.values():
        for part in (1, 3, 5):
            mask = (tags == tag) & (parts == part)
            check(mask.any() and (draws[:, mask].sum(axis=1) == mask.sum()).all(),
                  f"Bootstrap stratum size {tag}/{part}")
    expected = np.stack([
        hist[..., (edges[:-1] >= .3) & (edges[1:] <= 1)].sum(axis=-1),
        hist[..., (edges[:-1] >= 2) & (edges[1:] <= 4)].sum(axis=-1),
        hist[..., edges[:-1] >= 4].sum(axis=-1),
    ], axis=-1)
    evals = registry()
    for i, rec in enumerate(files):
        cid, fold = rec["content"], rec["fold"]
        base, meta = evals[cid]
        check(meta["fold"] == fold, f"{cid}: wrong evaluation fold")
        keys = None
        for j, seed in enumerate(SEEDS):
            member = meta["predictions"]["B1"][seed][cid]
            path = base / member["path"]
            check(sha(path) == member["sha256"], f"{cid}/{seed}: prediction digest")
            frame = pd.read_csv(path, usecols=COLUMNS)
            identity = frame[COLUMNS[:3]].to_numpy()
            check(not frame.duplicated(COLUMNS[:3]).any()
                  and frame.Row.min() >= 0 and frame.Row.max() + 1 == rec["processed_span"]
                  and (keys is None or np.array_equal(identity, keys)), f"{cid}: candidate keys")
            keys = identity
            e, score = frame.e_pred.to_numpy(), frame.cl_score.to_numpy()
            check(np.isfinite(e).all() and frame.cl_score.between(0, 1).all(), f"{cid}: score/energy")
            keep = score > plan["frozen_cuts"][f"{fold}/{seed}"]
            energy = e[keep]
            count = [np.count_nonzero((energy > .3) & (energy < 1)),
                     np.count_nonzero((energy >= 2) & (energy < 4)),
                     np.count_nonzero(energy >= 4)]
            check(len(energy) == selected[i, j] and np.array_equal(count, expected[i, j]),
                  f"{cid}/{seed}: frozen selected and strict-floor band counts differ")
    return ids, tags, draws, expected


def extracted(plan, plan_path, export):
    receipt_path, manifest_path = export / "receipt.json", export / "manifest.json"
    receipt, manifest = load(receipt_path), load(manifest_path)
    tar_path = export / "inputs.tar"
    check(receipt["state"] == "complete" and sha(manifest_path) == receipt["manifest_sha256"]
          and sha(tar_path) == manifest["archive_sha256"] == receipt["archive_sha256"]
          and tar_path.stat().st_size == manifest["archive_bytes"] == receipt["archive_bytes"],
          "Publication receipt and archive")
    pins = {r["member"]: r["sha256"] for r in manifest["files"]}
    with tarfile.open(tar_path) as archive:
        def member(name):
            with archive.extractfile(name) as stream:
                raw = stream.read()
            check(hashlib.sha256(raw).hexdigest() == pins[name], name)
            return raw

        names = archive.getnames()
        check(len(names) == len(pins) == 283 and set(names) == set(pins), "Archive inventory")
        summary = json.loads(member("summary.json"))
        check(summary["state"] == "complete" and summary["plan_sha256"] == sha(plan_path)
              and summary["observation_replay"] == summary["primary_counter_replay"] == "exact"
              and summary["rule_sha256"] == sha(BENCHMARK / "rule.py")
              and pins["summary.json"] == receipt["summary_files"]["summary.json"], "Replay summary")
        rows = {r["content"]: r for r in summary["files"]}
        check(len(rows) == len(summary["files"]) == len(plan["files"]), "Replay inventory")
        counts = []
        for rec in plan["files"]:
            cid = rec["content"]
            got = rows[cid]
            name = cid + ".npz"
            check(got["member"] == name and got["sha256"] == pins[name]
                  and got["processed_span"] == rec["processed_span"]
                  and got["observation_sha256"] == rec["observation"]["sha256"]
                  and got["hits"] == rec["observation"]["observed_hits"]
                  and got["truth_counter_sha256"] == rec["counter_sha256"]
                  and got["truth_source_sha256"] == rec["primary_sha256"], cid)
            truth = [got["truth_soft"], got["truth_hard"], got["truth_overflow"]]
            check([sum(truth[1:]), truth[0]] == rec["open_truth_hi_lo"], f"{cid}: MC count")
            with np.load(BytesIO(member(name))) as z:
                e, keep = z["energy"], z["selected"]
                ledger = dict(zip(summary["rule_rejection_keys"], z["rule_rejections"], strict=True))
                check((z["rows"] >= 0).all() and (z["rows"] < rec["processed_span"]).all()
                      and np.isfinite(e[keep]).all() and ((e[keep] > .3) & (e[keep] < 4)).all()
                      and keep.sum() == ledger["selected"], f"{cid}: rule count")
                rules = [np.count_nonzero(keep & (e < 1)), np.count_nonzero(keep & (e >= 2)),
                         ledger["energy_ceiling"]]
            counts.append([truth, rules])
    return np.asarray(counts), {p.name: sha(p) for p in (receipt_path, manifest_path, tar_path)}


def main(plan_path, export, output, counts_path):
    check(not output.exists() and not counts_path.exists(), "Odd result already exists")
    plan = load(plan_path)
    ids, tags, draws, gnn = frozen(plan)
    others, pins = extracted(plan, plan_path, export)
    # The two deterministic spectra repeat identically along the seed axis.
    reference_counts = np.broadcast_to(others[:, None], (len(ids), len(SEEDS), 2, 3))
    all_counts = np.concatenate([reference_counts, gnn[:, :, None]], axis=2)
    soft, hard, overflow = np.moveaxis(all_counts, -1, 0)
    weights = np.vstack([np.ones(len(ids), dtype=np.int64), draws])
    open_hard = hard + overflow
    within = reduce_spectrum(weights, tags, soft, hard)
    open_result = reduce_spectrum(weights, tags, soft, open_hard)
    historical = load(FIXED / "results.json")["scopes"]["odd_fixed_check"]
    # Validate open-arm full-sample and percentile values below against the accepted policy.
    for key, row in open_result["contrasts"].items():
        old = historical["contrasts"][key]
        original = old["calibrated_ladder"]["C4"]
        for method, expected in (("mc", old["truth_neutron"]), ("gnn", original["purity90"]),
                                 ("gnn_minus_mc", original["purity90_minus_truth"])):
            check(all(np.isclose(row[method][field], expected[field], rtol=0, atol=1e-12)
                      for field in ("value", "q16", "q84", "half_width_68")), f"{key}: fixed-policy replay")
    with counts_path.open("xb") as stream:
        np.savez_compressed(stream, content=ids, tags=tags, draws=draws, soft=soft, hard=hard,
                            overflow=overflow, arms=ARMS, seeds=SEEDS)
    record = {
        "file_count": len(ids), "event_count": plan["events"], "bootstrap_draws": len(draws),
        "seeds": SEEDS, "arms": ARMS, "fixed_target_percent": plan["fixed_target_percent"],
        "parts": [1, 3, 5], "plan_sha256": sha(plan_path), "publication_sha256": pins,
        "counts_sha256": sha(counts_path), "reducer_sha256": sha(Path(__file__)),
        "tail_counts_by_seed": overflow.sum(axis=0).tolist(),
        "tail_share_of_open_hard_percent_by_seed": (100 * overflow.sum(axis=0) / open_hard.sum(axis=0)).tolist(),
        "within_reach": within, "open_upper": open_result,
    }
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "export", "output", "counts"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    main(args.plan, args.export, args.output, args.counts)
