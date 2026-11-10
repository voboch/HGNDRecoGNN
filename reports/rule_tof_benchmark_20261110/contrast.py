"""Reduce fixed candidate and geometric truth counts on identical file draws."""

import argparse
import json
from pathlib import Path

import numpy as np

from summarize import estimate, sha

HERE = Path(__file__).resolve().parent
REPORTS = HERE.parent
TAGS = {"0": "zeroSpot", "18": "defaultSpot", "90": "bigSpot"}
ARMS = ("mc", "rules", "gnn")
PAIRS = (("18", "0"), ("90", "0"), ("90", "18"))


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval(values):
    result = estimate(values)
    result["half_width_68"] = (result["q84"] - result["q16"]) / 2
    return result


def reduce_spectrum(weights, tags, soft, hard):
    # Sum the candidate counts over complete files before forming any ratios.
    settings = {}
    for setting, tag in TAGS.items():
        mask = tags == tag
        lo = np.einsum("df,fsa->dsa", weights[:, mask], soft[mask])
        hi = np.einsum("df,fsa->dsa", weights[:, mask], hard[mask])
        require((lo > 0).all() and (hi > 0).all(), f"Empty band in setting {setting}")
        settings[setting] = {
            "file_count": int(mask.sum()),
            "soft_by_seed": lo[0].tolist(),
            "hard_by_seed": hi[0].tolist(),
            "rn_by_seed": (hi[0] / lo[0]).tolist(),
            "min_soft_over_draws_and_seeds": lo[1:].min(axis=(0, 1)).tolist(),
        }
        settings[setting]["ratios"] = hi / lo

    result = {}
    for a, b in PAIRS:
        by_seed = 100 * (settings[a]["ratios"] / settings[b]["ratios"] - 1)
        # Apply the nonlinear contrast within each seed before equal averaging.
        mc, rules, gnn = by_seed.mean(axis=1).T
        result[f"{a}/{b}"] = {
            "mc": interval(mc),
            "rules": interval(rules),
            "gnn": interval(gnn),
            "rules_minus_mc": interval(rules - mc),
            "gnn_minus_mc": interval(gnn - mc),
            "gnn_minus_rules": interval(gnn - rules),
            "absolute_residual_gnn_minus_rules": interval(np.abs(gnn - mc) - np.abs(rules - mc)),
            "gnn_central_by_seed": by_seed[0, :, ARMS.index("gnn")].tolist(),
        }
    for setting in settings.values():
        del setting["ratios"]
    return {"settings": settings, "contrasts": result}


def analyze():
    counts_path = HERE / "counts.npz"
    benchmark_path = HERE / "results.json"
    benchmark = load(benchmark_path)
    prior = REPORTS / "response_diagnostics_20261106" / "counts.npz"
    pins = load(REPORTS / "primary_prediction_response_20261109" / "input_pins.json")
    plan_path = REPORTS / "first_hit_tof_20261110" / "plan.json"
    source = load(plan_path.with_name("source_package.json"))
    files = {row["content"]: row for row in load(plan_path)["files"]}
    require(sha(counts_path) == benchmark["counts_sha256"], "Benchmark count digest")
    require(sha(prior) == pins[str(prior.relative_to(REPORTS.parent))], "Historical ledger digest")
    require(sha(plan_path) == source["plan_sha256"], "Same-event plan digest")
    with np.load(counts_path) as a, np.load(prior) as historical:
        contents = a["content"]
        draws = a["weights"]
        require(len(contents) == len(files) == len(set(contents)) == 286
                and set(contents) == set(files), "Same-event inventory")
        require(np.array_equal(contents, historical["content"])
                and np.array_equal(draws, historical["weights"])
                and draws.shape == (400, 286) and (draws >= 0).all(),
                "Shared bootstrap draws")
        tags = np.array([files[cid]["tag"] for cid in contents])
        parts = np.array([files[cid]["part"] for cid in contents])
        require(set(tags) == set(TAGS.values()) and set(parts) == {0, 2, 4},
                "Potential and part strata")
        strata = {}
        for tag in TAGS.values():
            strata[tag] = {}
            for part in sorted(set(parts)):
                mask = (tags == tag) & (parts == part)
                require(mask.any() and (draws[:, mask].sum(axis=1) == mask.sum()).all(),
                        f"Draw changes stratum population {tag}/{part}")
                strata[tag][str(part)] = int(mask.sum())

        counts = a["counts"]
        keys = a["count_keys"].tolist()
        events = counts[:, 0, 0, keys.index("events")]
        require(events.tolist() == [files[cid]["processed_span"] for cid in contents]
                and events.sum() == benchmark["scopes"]["pooled"]["counts"]["events"]["rules"]["value"],
                "Event spans changed")
        truth = a["truth_bands"][..., 0, :]
        require((truth == truth[:, :1, :1]).all()
                and truth[:, 0, 0].sum() == 897_563, "Geometric truth inventory")
        require(np.array_equal(a["seeds"], ["42", "123", "456"]), "Frozen seed order")
        soft = np.concatenate([truth[:, :, :1, 1], counts[..., keys.index("soft_candidates")]], axis=2)
        hard = np.concatenate([truth[:, :, :1, 3], counts[..., keys.index("hard_candidates")]], axis=2)
        for band in (soft, hard):
            require((band[:, :, :2] == band[:, :1, :2]).all(), "Repeated MC or rules differ")
        rejection = a["rule_rejection_keys"].tolist().index("energy_ceiling")
        rules_over = np.broadcast_to(a["rule_rejections"][:, rejection, None], hard.shape[:2])
        overflow = np.stack([truth[:, :, 0, 4], rules_over, a["gnn_window_loss"][..., 1]], axis=2)
        old_gnn = historical["ledger"][:, :, 1, 3]
        require(np.array_equal(hard[..., 2] + overflow[..., 2], old_gnn[..., 0])
                and np.array_equal(soft[..., 2], old_gnn[..., 1]),
                "Open-band GNN must replay the historical frozen counts")
        seeds = a["seeds"].tolist()

    weights = np.vstack([np.ones(len(contents), dtype=np.int64), draws])
    open_hard = hard + overflow
    return {
        "file_count": len(contents), "event_count": int(events.sum()),
        "bootstrap_draws": len(draws), "seeds": seeds, "arms": ARMS,
        "strata": strata, "soft_GeV": [0.3, 1], "hard_min_GeV": 2,
        "candidate_lower_edge": "strict above 0.3 GeV; historical GNN soft counts replay exactly",
        "counts_sha256": benchmark["counts_sha256"], "historical_counts_sha256": sha(prior),
        "plan_sha256": source["plan_sha256"], "analysis_sha256": sha(Path(__file__)),
        "tail_counts_by_seed": overflow.sum(axis=0).tolist(),
        "tail_share_of_open_hard_percent_by_seed": (100 * overflow.sum(axis=0) / open_hard.sum(axis=0)).tolist(),
        "within_reach": reduce_spectrum(weights, tags, soft, hard),
        "open_upper": reduce_spectrum(weights, tags, soft, open_hard),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
