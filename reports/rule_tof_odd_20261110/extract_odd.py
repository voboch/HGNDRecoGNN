"""Replay frozen rule observations and extend certified MC bands on odd files."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

from extract import process as rule_process, sha
from geometry import in_acceptance
from rule import RULE_REJECTIONS

KINEMATICS = ("Px", "Py", "Pz", "vX", "vY", "vZ")


def primary(record, counters, local, stage):
    path = local / "primary.csv"
    stage(Path(record["primary_path"]), path, record["primary_sha256"])
    size = record["certified_events"]
    total = np.zeros((len(counters), size), dtype=np.int64)
    hard = 0
    for frame in pd.read_csv(path, usecols=["Row", "PDG", "Ekin", *KINEMATICS], chunksize=100_000):
        rows = frame.Row.to_numpy()
        if not np.issubdtype(rows.dtype, np.integer) or not ((0 <= rows) & (rows < size)).all():
            raise ValueError(f"Invalid truth row in {record['content']}")
        acc = in_acceptance(*(frame[k].to_numpy() for k in KINEMATICS))
        en, pdg = frame.Ekin.to_numpy(), frame.PDG.to_numpy()
        if not np.isfinite(en).all():
            raise ValueError(f"Invalid truth energy in {record['content']}")
        n, p = acc & (pdg == 2112), acc & (pdg == 2212)
        masks = {
            "nprim": np.ones(len(frame), dtype=bool), "acc_n": n, "acc_p": p,
            "acc_n_hi": n & (en >= 2), "acc_p_hi": p & (en >= 2),
            "acc_n_lo": n & (en < 1), "acc_p_lo": p & (en < 1),
            "acc_n_lo_reach": n & (en >= .3) & (en < 1),
            "acc_p_lo_reach": p & (en >= .3) & (en < 1),
        }
        for i, key in enumerate(counters):
            total[i] += np.bincount(rows[masks[key]], minlength=size)
        hard += np.count_nonzero(n & (en >= 2) & (en < 4) & (rows < record["processed_span"]))
    digest = hashlib.sha256(np.column_stack((np.arange(size), total.T)).astype("<i8").tobytes()).hexdigest()
    if digest != record["counter_sha256"]:
        raise ValueError(f"Certified per-event counter replay failed for {record['content']}")
    hi, lo = total[[counters.index("acc_n_hi"), counters.index("acc_n_lo_reach")],
                   :record["processed_span"]].sum(axis=1).tolist()
    if [hi, lo] != record["open_truth_hi_lo"] or not 0 <= hard <= hi:
        raise ValueError(f"Accepted truth bands changed for {record['content']}")
    return {"truth_soft": lo, "truth_hard": hard, "truth_overflow": hi - hard,
            "truth_counter_sha256": digest, "truth_source_sha256": record["primary_sha256"]}


def process(task):
    record, old_source, counters, work, output = task
    result = rule_process((record, old_source, work, output))
    from HGNDRecoGNN.scripts.smash_train_prepare import stage
    with tempfile.TemporaryDirectory(dir=work) as directory:
        result.update(primary(record, counters, Path(directory), stage))
    return result


def main(plan_path, work, output):
    plan = json.loads(plan_path.read_bytes())
    old = Path(plan["old_source"])
    if (sha(old / "SHA256SUMS") != plan["old_source_sums_sha256"]
            or sha(Path(__file__).with_name("geometry.py")) != plan["geometry_sha256"]):
        raise ValueError("Accepted observation or geometry source changed")
    work.mkdir()
    output.mkdir()
    tasks = [(r, str(old), plan["counter_columns"], work, output) for r in plan["files"]]
    if len(tasks) != 282 or any(r["part"] not in (1, 3, 5) for r in plan["files"]):
        raise ValueError("Odd file scope changed")
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = []
        for result in pool.map(process, tasks):
            results.append(result)
            if len(results) % 25 == 0:
                print(f"Replayed {len(results)}/{len(tasks)} odd files", flush=True)
    if (len(results) != plan["file_count"] or sum(r["hits"] for r in results) != plan["observed_hits"]
            or sum(r["processed_span"] for r in results) != plan["events"]):
        raise ValueError("Cohort changed")
    summary = {"state": "complete", "files": results, "observation_replay": "exact",
               "primary_counter_replay": "exact", "plan_sha256": sha(plan_path),
               "rule_sha256": sha(Path(__file__).with_name("rule.py")),
               "rule_rejection_keys": RULE_REJECTIONS}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("plan", "work", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    main(args.plan, args.work, args.output)
