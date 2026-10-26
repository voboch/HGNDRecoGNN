"""Exact reach boundary and nondegenerate, direction-sensitive headline tests."""

from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.analysis.truth_reach import reach_counters, Y_CM
from HGNDRecoGNN.analysis.truth_statistics import replay
from HGNDRecoGNN.tests.test_truth_provenance import primary


def test_exact_floor_low_high_boundaries_and_disjoint_proxy():
    rows = []
    for code in [2112, 2212]:
        for e in [0.299999999, 0.3, 0.999999999, 1.0, 2.0]:
            r = primary().iloc[0].copy()
            r["PDG"] = code
            r["Ekin"] = e
            r["Rapid"] = Y_CM
            rows.append(r)
    d = pd.DataFrame(rows)
    c = reach_counters(d, 0.3, independent=True).iloc[0]
    assert c.acc_n_lo_reach == 2 and c.acc_p_lo_reach == 2
    assert c.acc_n_reach == 4 and c.acc_p_reach == 4 and c.ref_mult == 0
    d["Px"] = 0.0
    c = reach_counters(d, 0.3, independent=True).iloc[0]
    assert c.ref_mult == 10 and c.acc_n_reach == 0
    with pytest.raises(ValueError, match="exact 0.3"):
        reach_counters(d, 0.291)


def varied():
    rows = []
    for tag, k in [("zeroSpot", 1), ("defaultSpot", 2), ("bigSpot", 3)]:
        for i in range(3):
            counts = [100, 100, k * 10 + i, 20, 10, 20 + i]
            rows.append(dict(tag=tag, member=f"run/{i}.csv", events=100, counts=counts))
    return dict(columns=["n", "p", "nh", "nl", "ph", "pl"], rows=rows)


def test_nondegenerate_joint_replay_and_endpoint_direction():
    d = varied()
    a = replay(d)
    expected = (93 / 60) / (33 / 60) - 1
    assert a["endpoint_fraction_big_over_zero_minus_one"][0] == pytest.approx(expected)
    assert a["endpoint_bootstrap_sd"][0] > 0
    assert a == replay(d)
    for r in d["rows"]:
        r["tag"] = {"zeroSpot": "bigSpot", "bigSpot": "zeroSpot"}.get(
            r["tag"], r["tag"]
        )
    b = replay(d)
    assert b["endpoint_fraction_big_over_zero_minus_one"][0] < 0
    d["rows"].pop()
    assert replay(d)["samples"]["zeroSpot"]["files"] == 2


def test_replay_rejects_changed_bootstrap_and_duplicate_file():
    d = varied()
    with pytest.raises(ValueError, match="recipe"):
        replay(d, seed=1)
    d["rows"].append(d["rows"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        replay(d)
