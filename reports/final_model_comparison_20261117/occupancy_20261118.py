"""Read frozen event-multiplicity ledgers; never access ROOT or model files."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "occupancy_results_20261118"
METHODS = ("rules", "dynamic_edgeconv")
SEEDS = ("42", "123", "456")
# These categorical bins are the complete resolution of the accepted ledger.
STATES = ("0", "1", "2", "3_or_more")
EXPECTED_LEDGER = "99baab7068803ebd080fe0d78cc6f39b4e3ad4925eb913dde0d112bd2d3638cd"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def estimate(values):
    """Do not assign finite intervals if a bootstrap ratio is undefined."""
    finite = np.isfinite(values)
    result = {"value": float(values[0]) if finite[0] else None,
              "invalid_draws": int((~finite[1:]).sum())}
    if finite[1:].all():
        q16, q84 = np.quantile(values[1:], (0.16, 0.84))
        result.update(file_sd=float(values[1:].std(ddof=1)),
                      q16=float(q16), q84=float(q84))
    else:
        result.update(file_sd=None, q16=None, q84=None)
    return result


def paired(values):
    # Per-seed ratios precede the equal-seed mean within every shared draw.
    means = values.mean(axis=1)
    return {name: estimate(means[:, i]) for i, name in enumerate(METHODS)} | {
        "gnn_minus_rules_pp": estimate(means[:, 1] - means[:, 0]),
        "by_seed_percent": {
            seed: dict(zip(METHODS, values[0, i].tolist(), strict=True))
            for i, seed in enumerate(SEEDS)
        },
    }


def reduce():
    summary_path = HERE / "even.json"
    summary = json.loads(summary_path.read_bytes())
    ledger = HERE / "even.npz"
    require(sha(ledger) == summary["counts_sha256"] == EXPECTED_LEDGER,
            "Accepted ledger SHA changed")
    with np.load(ledger, allow_pickle=False) as z:
        content, tags, draws = (z[k] for k in ("content", "tags", "draws"))
        matrix, counts = z["multiplicity"], z["counts"]
        keys = list(z["count_keys"])
    require(len(content) == len(set(content)) == 286, "File inventory")
    require(draws.shape == (400, 286) and np.all(draws >= 0), "Draw inventory")
    require(matrix.shape == (286, 3, 2, 4, 4), "Matrix axes")
    require(np.all(matrix >= 0), "Negative counts")
    np.testing.assert_array_equal(tags, [cid.split("/")[0] for cid in content])
    np.testing.assert_array_equal(matrix.sum(axis=(-2, -1)),
                                  counts[..., keys.index("events")])
    np.testing.assert_array_equal(matrix[..., 0].sum(axis=-1),
                                  counts[..., keys.index("zero_candidates")])
    truth = matrix.sum(axis=-1)
    np.testing.assert_array_equal(truth, np.broadcast_to(truth[:, :1, :1], truth.shape))
    np.testing.assert_array_equal(truth[..., 0], counts[..., keys.index("zero_truth_events")])
    # The rules procedure is identical across training seeds.
    np.testing.assert_array_equal(matrix[:, :, 0],
                                  np.broadcast_to(matrix[:, :1, 0], matrix[:, :, 0].shape))
    weights = np.vstack([np.ones(286, dtype=np.int64), draws])
    result = {
        "scope": "retrospective event-level response conditional on geometric MC multiplicity",
        "files": 286, "events": int(counts[:, 0, 0, keys.index("events")].sum()),
        "seeds": SEEDS, "methods": METHODS, "truth_states": STATES,
        "bootstrap_draws": 400, "uncertainty": "paired whole-file SD at fixed models",
        "source_sha256": {"even.npz": sha(ledger), "even.json": sha(summary_path),
                          "script": sha(Path(__file__)),
                          "protocol": sha(HERE / "occupancy_protocol_20261118.md")},
        "checks": ["SHA exact", "286 unique files", "400 shared draws",
                   "event and no-output conservation", "common truth marginals",
                   "rules identical across seeds", "accepted unconditional no-output replay"],
        "scopes": {},
    }
    require(result["events"] == summary["events"] == 1180595, "Processed events")
    for scope, use in [("pooled", np.ones(286, dtype=bool)),
                       *[(tag, tags == tag) for tag in sorted(set(tags))]]:
        total = np.tensordot(weights[:, use], matrix[use], axes=(1, 0))
        denominator = total.sum(axis=-1, keepdims=True)
        probability = np.divide(100.0 * total, denominator,
                                out=np.full(total.shape, np.nan), where=denominator > 0)
        unconditional = 100 * total[..., 0].sum(axis=-1) / total.sum(axis=(-2, -1))
        replay = paired(unconditional)
        accepted = summary["scopes"][scope]["rates"]["zero_candidate_events_percent"]
        for new_key, old_key in (("rules", "rules"), ("dynamic_edgeconv", "gnn"),
                                 ("gnn_minus_rules_pp", "gnn_minus_rules")):
            for field in ("value", "file_sd", "q16", "q84"):
                require(np.isclose(replay[new_key][field], accepted[old_key][field],
                                   atol=1e-12, rtol=0), "Unconditional replay")
        rows = {}
        for i, state in enumerate(STATES):
            rows[state] = {
                "events": int(denominator[0, 0, 0, i, 0]),
                "no_output_counts_by_seed_method": total[0, :, :, i, 0].tolist(),
                "no_output_percent": paired(probability[..., i, 0]),
                "candidate_multiplicity_percent": {
                    selected: paired(probability[..., i, j])
                    for j, selected in enumerate(STATES)
                },
            }
        result["scopes"][scope] = {"files": int(use.sum()), "by_truth_multiplicity": rows,
                                    "unconditional_no_output_percent": replay}
    return result


def plot(result):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": True,
                         "axes.spines.right": True, "xtick.direction": "in",
                         "ytick.direction": "in", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    x = np.arange(4)
    rows = result["scopes"]["pooled"]["by_truth_multiplicity"]
    for method, label, offset, marker, color in (
        ("rules", "Rules + ToF", -0.07, "s", "#333333"),
        ("dynamic_edgeconv", "DynamicEdgeConv", 0.07, "o", "#006AA7"),
    ):
        stats = [rows[state]["no_output_percent"][method] for state in STATES]
        axes[0].errorbar(x + offset, [v["value"] for v in stats],
                         yerr=[v["file_sd"] for v in stats], fmt=marker,
                         color=color, capsize=3, label=label)
    stats = [rows[state]["no_output_percent"]["gnn_minus_rules_pp"] for state in STATES]
    axes[1].errorbar(x, [v["value"] for v in stats],
                     yerr=[v["file_sd"] for v in stats], fmt="o",
                     color="#006AA7", capsize=3)
    axes[1].axhline(0, color="#777777", lw=0.8)
    for ax in axes:
        ax.set_xticks(x, ["0", "1", "2", r"$\geq3$"])
        ax.set_xlabel("Geometric MC primary-neutron multiplicity")
        ax.set_xlim(-0.4, 3.4)
        ax.tick_params(top=True, right=True)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Reconstructed events with no selected candidate (%)")
    axes[1].set_ylabel("Paired GNN minus rules difference (pp)")
    axes[0].set_title("Conditional event response")
    axes[1].set_title("Change in no-output probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=2, frameon=False)
    fig.suptitle("SMASH Xe+Cs, 2.87 GeV/nucleon | MC-conditioned reconstruction", y=1.02)
    counts = ", ".join(f"{state.replace('3_or_more', '>=3')}: {rows[state]['events']:,}" for state in STATES)
    fig.text(0.5, 0.105, f"Events by MC multiplicity — {counts}", ha="center", fontsize=10)
    fig.text(0.5, 0.015,
             "Frozen raw 70% purity target; 0.3 < T(reco) < 4 GeV; 3 seeds; bars: 400 paired file-draw SD.\n"
             "Previously inspected even files. Event no-output is not a per-neutron miss rate.",
             ha="center", va="bottom", fontsize=10)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.80, wspace=0.28)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"no_output_by_multiplicity.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    require(not OUTPUT.exists(), "Preserve an existing reduction")
    result = reduce()
    OUTPUT.mkdir()
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result)
    for state, row in result["scopes"]["pooled"]["by_truth_multiplicity"].items():
        print(state, row["events"], json.dumps(row["no_output_percent"]))


if __name__ == "__main__":
    main()
