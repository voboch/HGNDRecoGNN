"""Run the rule reference only after an exact replay of frozen observations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from rule import RULE_REJECTIONS, reconstruct


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process(task):
    record, source, work, output = task
    sys.path.insert(0, source)
    from HGNDRecoGNN.scripts import audit_smash_sources as audit
    from HGNDRecoGNN.scripts.smash_train_prepare import stage

    content = record["content"]
    with tempfile.TemporaryDirectory(dir=work) as directory:
        local = Path(directory)
        raw = {}
        for kind in ("hits", "vacs"):
            raw[kind] = local / (kind + ".csv")
            remote = Path(record["raw_root"]) / (record["member"] + "_" + kind + ".csv")
            stage(remote, raw[kind], record["sha256"][kind])
        hf = audit.read_csv(raw["hits"], audit.HITS)
        vf = audit.read_csv(raw["vacs"], audit.VACS)
        _, hits, rejected = audit.observe(hf, vf, record["member"] + "_hits.csv")
        columns = [name for name in hits.columns if name != "_merge"]
        observation = local / (Path(record["member"]).name + ".pt")
        torch.save({"columns": columns, "values": torch.tensor(hits[columns].to_numpy(np.float64))},
                   observation)
        digest = sha(observation)
        expected = record["observation"]
        if (digest != expected["sha256"] or rejected != 0
                or len(hits) != expected["observed_hits"]
                or hits.Row.max() + 1 != record["processed_span"]):
            raise ValueError(f"Frozen observation replay failed for {content}")
        cluster, rows, energies, selected, literal, rejected_rules = reconstruct(hits[audit.HITS])
        direct = (hits.PDG == 2112) & (hits.fMotherId == -1) & hits.Side.isin([5, 6])
        pairs = np.unique(np.column_stack([
            hits.loc[direct, ["Row", "Id"]].to_numpy(np.int64), cluster[direct],
        ]), axis=0)
        name = content + ".npz"
        artifact = output / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact, rows=rows, energy=energies, selected=selected,
                            literal=literal, edges=pairs, rule_rejections=rejected_rules)
        return {"content": content, "member": name, "sha256": sha(artifact),
                "observation_sha256": digest, "hits": len(hits),
                "processed_span": record["processed_span"]}


def main(plan_path, work, output):
    plan = json.loads(plan_path.read_bytes())
    old_source = Path(plan["old_source"])
    if sha(old_source / "SHA256SUMS") != plan["old_source_sums_sha256"]:
        raise ValueError("Frozen observation source changed")
    work.mkdir()
    output.mkdir()
    tasks = [(record, str(old_source), work, output) for record in plan["files"]]
    with ProcessPoolExecutor(max_workers=2) as executor:
        records = []
        for record in executor.map(process, tasks):
            records.append(record)
            if len(records) % 25 == 0:
                print(f"Replayed {len(records)}/{len(tasks)} files", flush=True)
    if len(records) != 286 or sum(record["hits"] for record in records) != 5833940:
        raise ValueError("Frozen cohort size changed")
    summary = {
        "state": "complete", "files": records, "native_observation_replay": "exact",
        "plan_sha256": sha(plan_path), "extractor_sha256": sha(Path(__file__)),
        "rule_sha256": sha(Path(__file__).with_name("rule.py")),
        "rule_rejection_keys": RULE_REJECTIONS,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("plan", "work", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    main(args.plan, args.work, args.output)
