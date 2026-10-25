"""B-free scientific path: counts, membership, spectra and execution gates."""

from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.bfree_truth_campaign import (
    TAGS,
    observables,
    summarize,
    validate_receipt,
)
from HGNDRecoGNN.scripts.certify_primary_counters import certify
from HGNDRecoGNN.scripts.content_census import sha256
from HGNDRecoGNN.scripts.reduce_prim_full import reduce_member
from HGNDRecoGNN.tests.test_truth_provenance import primary


def make_certificates(tmp_path, missing_b=-1):
    files, refs = [], {}
    for tag in TAGS:
        run = tmp_path / "raw" / tag / "run"
        run.mkdir(parents=True)
        frames, done = [], {}
        for i in range(2):
            base = primary()
            low_n, high_p = base.iloc[[0]].copy(), base.iloc[[1]].copy()
            low_n["Ekin"], low_n["Id"] = 0.4, 3
            high_p["Ekin"], high_p["Id"] = 2.2, 4
            d = pd.concat([base, low_n, high_p], ignore_index=True)
            d["B"] = missing_b
            raw = run / f"{i:04d}_prim.csv"
            d.to_csv(raw, index=False)
            f, _, _, _ = reduce_member(raw, i, histograms=False)
            frames.append(f)
            done["run/" + raw.name] = 1
            files.append(
                dict(
                    kind="smash_primary",
                    tag=tag,
                    path=str(raw),
                    sha256=sha256(raw),
                    events=1,
                )
            )
        ev, pr = tmp_path / (tag + ".pkl"), tmp_path / (tag + ".json")
        pd.concat(frames).to_pickle(ev)
        pr.write_text(json.dumps({"done": done}))
        refs[tag] = dict(
            events=str(ev),
            progress=str(pr),
            events_sha256=sha256(ev),
            progress_sha256=sha256(pr),
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    for tag in TAGS:
        certify(
            Namespace(
                input_manifest=manifest,
                reference_events=Path(refs[tag]["events"]),
                reference_progress=Path(refs[tag]["progress"]),
                tag=tag,
                role="acceptance",
                all=True,
                member=None,
                raw_copy=None,
                save_spectra=True,
                output=tmp_path / tag,
            )
        )
    return dict(manifest_path=manifest, references=refs)


def test_bfree_spectra_and_bootstrap_exact_counts_ignore_b(tmp_path, monkeypatch):
    # A physical-B histogram must never be called by this path.
    monkeypatch.setattr(
        np, "histogramdd", lambda *a, **kw: pytest.fail("B histogram called")
    )
    config = make_certificates(tmp_path)
    result = summarize(tmp_path, tmp_path, config=config)
    assert result["coverage"] == "full"
    for sample in result["samples"].values():
        assert sample["files"] == 2 and sample["events"] == 2
        assert set(sample["observables"].values()) == {1.0}
        assert set(sample["bootstrap_sd"].values()) == {0.0}
    assert result["endpoint_fraction_big_over_zero_minus_one"] == [0.0] * 4
    with np.load(tmp_path / "spectra.npz") as z:
        assert np.allclose(z["zeroSpot_per_event"].sum(axis=1), [2, 2])
        assert np.isnan(z["zeroSpot_np_ratio"][~z["zeroSpot_ratio_valid"]]).all()


def test_full_summary_rejects_wrong_census_and_smoke(tmp_path):
    config = make_certificates(tmp_path)
    manifest = json.loads(config["manifest_path"].read_text())
    manifest["files"][0]["events"] += 1
    config["manifest_path"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="another census"):
        summarize(tmp_path, tmp_path, config=config)


def test_observables_use_exact_threshold_counters_and_require_denominators():
    assert np.allclose(observables(np.array([10, 5, 3, 6, 2, 8])), [0.5, 0.25, 2, 2])
    with pytest.raises(ValueError, match="denominator"):
        observables(np.array([1, 0, 1, 1, 1, 1]))


def test_unverified_or_failed_smoke_cannot_launch_full(tmp_path):
    p = tmp_path / "receipt.json"
    for state, status in [("FAILED", "verified"), ("COMPLETED", "pending")]:
        p.write_text(
            json.dumps(
                dict(
                    status=status,
                    slurm_state=state,
                    exit_code="0:0",
                    mode="smoke",
                    source_digest="a",
                    environment={},
                )
            )
        )
        with pytest.raises(ValueError, match="Invalid external"):
            validate_receipt(p, "a", {}, mode="smoke")
