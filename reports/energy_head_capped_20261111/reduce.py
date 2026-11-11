"""Reuse the exact matched-neutron and odd-file comparisons for the capped heads."""

import argparse
import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
old = importlib.import_module("HGNDRecoGNN.reports.energy_head_dombay_20261111.reduce")
ARMS = ("baseline_raw", "unit_mse", "capped_raw", "capped_calibrated")
PAIRED = {"capped_raw_minus_original": (2, 0), "capped_raw_minus_unit": (2, 1),
          "capped_calibrated_minus_raw": (3, 2), "capped_calibrated_minus_original": (3, 0)}


def main(args):
    plan_path = HERE / "plan.json"
    plan = old.load(plan_path)
    source = old.load(HERE / "source_package.json")
    for path, expected in plan["reduction_pins"].items():
        old.checked(old.ROOT / path, expected)
    runs, registry = old.compare_inputs(args.exports, plan, source["source_sums_sha256"], plan_path)
    validation = []
    for folder, info in runs.values():
        record = old.load(HERE / "acceptance" / f"task_{info['task']}.json")
        assert record["summary"] == info and record["calibration_band_decisions_exact"]
        assert record["calibration_replay"] == "within_2_ulps"
        old.checked(folder / "summary.json", record["receipt"]["summary_files"]["summary.json"])
        validation.append({
            "fold": info["fold"], "seed": info["seed"], **info["validation_choice"],
            "validation_choice_sha256": info["validation_choice_sha256"],
        })
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "matched_even_energy": old.relative_bins(runs, registry, ARMS, PAIRED),
        "odd_hardness": old.odd_contrast(runs, registry, args.output, ARMS, PAIRED),
        "validation": validation, "selection": plan["selection"],
        "plan_sha256": old.digest(plan_path), "source_sums_sha256": source["source_sums_sha256"],
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args())
