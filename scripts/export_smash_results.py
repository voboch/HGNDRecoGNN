"""Rebuild compact retrospective transfer tables from the verified seed ledger.

This exports the existing canonical result and checks its three-seed arithmetic.
It is not a second implementation of prediction scoring or a new test.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("results/smash_full_transfer_20261023/full_metrics_verified.json"),
    )
    p.add_argument("--output", type=Path, default=Path("output/reproduced_smash"))
    args = p.parse_args()
    x = json.loads(args.input.read_text())
    assert x["state"] == "verified" and not x["potential_sensitivity_measured"]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, data in x["tags"].items():
        assert sorted(data["per_seed"]) == ["123", "42", "456"]
        for metric, saved in data["summary"].items():
            values = [
                data["per_seed"][str(seed)]["metrics"][metric]
                for seed in (42, 123, 456)
            ]
            assert abs(mean(values) - saved["mean"]) < 1e-12
            assert abs(stdev(values) - saved["sample_sd"]) < 1e-12
            rows.append(
                {
                    "sample": tag,
                    "metric": metric,
                    "mean": saved["mean"],
                    "training_seed_sample_sd": saved["sample_sd"],
                    "clusters": data["clusters"],
                    "MC_positive_clusters": data["MC_positive_clusters"],
                }
            )
    with (args.output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "scope": x["scope"],
                "uncertainty": x["uncertainty"],
                "rows": len(rows),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Exported {len(rows)} metrics with checked seed arithmetic; {x['scope']}")


if __name__ == "__main__":
    main()
