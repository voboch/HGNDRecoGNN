"""Join independently certified primary counters and recovered impact parameter.

Both sources need complete counter certificates with full particle identities.
The census fixes required run/member coverage. Equal keys and multiplicity alone
cannot establish that a b value belongs to the particles being counted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.analysis.truth_provenance import load_certified, merge_certified  # noqa: E402
from HGNDRecoGNN.scripts.content_census import sha256  # noqa: E402

SAMPLES = ("zeroSpot", "defaultSpot", "bigSpot")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--acc-dir",
        type=Path,
        required=True,
        help="certificate root with <sample>/proof.json for acceptance counters",
    )
    p.add_argument(
        "--b-dir",
        type=Path,
        required=True,
        help="certificate root with <sample>/proof.json for recovered b",
    )
    p.add_argument(
        "--census",
        type=Path,
        required=True,
        help="pinned complete primary membership and raw-input SHA256 inventory",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    census_sha = sha256(args.census)
    census = json.loads(args.census.read_text())
    if census.get("status") != "complete" or census.get("smoke") is not False:
        raise ValueError("A complete primary census is required")
    prepared, report = {}, {}
    # Validate every source before making an output directory or writing a result.
    for tag in SAMPLES:
        expected = {
            "/".join(Path(v["path"]).parts[-2:])
            for v in census["files"]
            if v["kind"] == "smash_primary" and v["tag"] == tag
        }
        acc, acc_id, acc_proof = load_certified(args.acc_dir / tag, "acceptance")
        bee, b_id, b_proof = load_certified(args.b_dir / tag, "impact")
        if acc_proof["tag"] != tag or b_proof["tag"] != tag:
            raise ValueError("Certificate potential differs from requested sample")
        if acc_proof["input_manifest_sha256"] != census_sha:
            raise ValueError("Acceptance certificate is not bound to the pinned census")
        if not expected or set(acc.index.get_level_values("member")) != expected:
            raise ValueError("Certificate omits or adds a declared primary member")
        result = merge_certified(acc, bee, acc_id, b_id)
        prepared[tag] = result
        report[tag] = {
            "files": len(expected),
            "events": len(result),
            "acceptance_proof_sha256": sha256(args.acc_dir / tag / "proof.json"),
            "impact_proof_sha256": sha256(args.b_dir / tag / "proof.json"),
            "mean_b": float(result.B.mean()),
            "particle_mismatches": 0,
        }
    if sha256(args.census) != census_sha:
        raise ValueError("Primary census changed during join")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for tag, result in prepared.items():
        plain = result.reset_index()
        members = sorted(plain.member.unique())
        member_indices = {member: i for i, member in enumerate(members)}
        plain["file_idx"] = plain.member.map(member_indices).astype(np.int16)
        output = args.out_dir / f"{tag}_events.pkl"
        plain.drop(columns="member").to_pickle(output)
        progress = {
            "done": {member: int((plain.member == member).sum()) for member in members}
        }
        progress_path = args.out_dir / f"{tag}_progress.json"
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
        report[tag]["events_sha256"] = sha256(output)
        report[tag]["progress_sha256"] = sha256(progress_path)
    receipt = {
        "schema": "certified-primary-b-join-v1",
        "status": "complete",
        "source_sha256": sha256(__file__),
        "census_sha256": census_sha,
        "samples": report,
        "scope": "Exact primary contents and counter replay; generator configuration remains external.",
    }
    (args.out_dir / "merge_report.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    print(
        "All three samples joined with exact primary-content equality; no files dropped."
    )


if __name__ == "__main__":
    main()
