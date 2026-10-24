"""Stratify saved student test scores by identical training candidate inputs.

No model fitting, inference, cut choice or new holdout is performed. Positional
eigenvectors are excluded because eigensolver sign conventions are arbitrary;
the signature covers the ordered six input features and exact directed edges.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.content_census import sha256


def candidates(directory, part):
    manifest = json.loads((directory / "manifest.json").read_text())
    for item in manifest["parts"][part]:
        path = directory / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Changed pack {path}")
        p = torch.load(path, map_location="cpu", weights_only=True)
        for i, (row, cluster) in enumerate(
            zip(p["row"].tolist(), p["cluster"].tolist())
        ):
            x = p["x"][p["node_ptr"][i] : p["node_ptr"][i + 1]].numpy().astype("<f4")
            edges = (
                p["edge_index"][:, p["edge_ptr"][i] : p["edge_ptr"][i + 1]]
                .numpy()
                .astype("<i8")
            )
            h = hashlib.sha256(b"student-ordered-inputs-v1")
            h.update(np.array(x.shape, dtype="<i8").tobytes())
            h.update(x.tobytes())
            h.update(edges.tobytes())
            yield row, cluster, int(p["y_cls"][i]), h.hexdigest()


def run(args):
    train = {v[-1] for v in candidates(args.prepared, "train")}
    test = list(candidates(args.prepared, "test"))
    membership = pd.DataFrame(test, columns=["row", "cluster", "true_y", "sha256"])
    membership["identical_training_candidate"] = membership.sha256.isin(train)
    summary = {
        "definition": __doc__,
        "inputs": {},
        "metrics": {},
        "n_test_candidates": len(test),
        "n_identical_training_candidate": int(
            membership.identical_training_candidate.sum()
        ),
    }
    for path in sorted(args.predictions.glob("*.csv")):
        df = pd.read_csv(path)
        merged = df.merge(
            membership, on=["row", "cluster", "true_y"], validate="one_to_one"
        )
        if len(merged) != len(membership):
            raise ValueError(f"Prediction membership mismatch: {path}")
        summary["inputs"][str(path)] = sha256(path)
        summary["metrics"][path.stem] = {}
        for label, mask in [
            ("all", np.ones(len(merged), dtype=bool)),
            ("seen_training_candidate", merged.identical_training_candidate),
            ("unseen_training_candidate", ~merged.identical_training_candidate),
        ]:
            a = merged[mask]
            summary["metrics"][path.stem][label] = {
                "n": len(a),
                "positive": int(a.true_y.sum()),
                "roc_auc": float(roc_auc_score(a.true_y, a.score))
                if a.true_y.nunique() == 2
                else None,
            }
    args.output.mkdir(parents=True, exist_ok=False)
    membership.to_csv(args.output / "membership.csv.gz", index=False)
    summary["membership_sha256"] = sha256(args.output / "membership.csv.gz")
    summary["prepared_manifest_sha256"] = sha256(args.prepared / "manifest.json")
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print({k: v for k, v in summary.items() if k.startswith("n_")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
