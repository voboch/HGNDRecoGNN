"""Plot only the accepted local diagnostic JSON, without re-estimation."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ("rules", "dynamic_edgeconv")
COLORS = ("#333333", "#006AA7")
SEEDS = ("42", "123", "456")


def values(rows, field):
    return np.array([np.nan if r[field] is None else r[field] for r in rows])


def points(ax, x, rows, label, color, marker, offset=0):
    y, err = values(rows, "value"), values(rows, "file_sd")
    ax.errorbar(
        np.asarray(x) + offset,
        y,
        yerr=err,
        fmt=marker,
        ms=4,
        color=color,
        capsize=2,
        label=label,
        linestyle="none",
    )


def setup(axes):
    for ax in np.asarray(axes).flat:
        ax.tick_params(direction="in", top=True, right=True)


def save(fig, directory, name):
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def multiplicity(result, directory):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.7))
    setup(axes)
    columns = (
        ("miss_percent", "Direct misses / geometric MC neutrons (%)"),
        ("split_percent", "Split neutrons / geometric MC neutrons (%)"),
        ("merge_percent", "Merged / selected candidates (%)"),
    )
    rows = result["scopes"]["pooled"]["multiplicity"]
    for ci, (metric, title) in enumerate(columns):
        for mi, (method, color, marker) in enumerate(
            zip(METHODS, COLORS, ("s", "o"), strict=True)
        ):
            stats = [rows[state]["rates"][metric][method] for state in result["states"]]
            points(
                axes[0, ci],
                np.arange(4),
                stats,
                ("Rules + ToF", "DynamicEdgeConv")[mi],
                color,
                marker,
                (-0.06, 0.06)[mi],
            )
        paired = [
            rows[state]["rates"][metric]["gnn_minus_rules"]
            for state in result["states"]
        ]
        points(axes[1, ci], np.arange(4), paired, "Paired difference", COLORS[1], "o")
        axes[0, ci].set_title(title, fontsize=11)
        axes[1, ci].axhline(0, color="#777777", lw=0.8)
        axes[1, ci].set_ylabel("GNN minus rules (pp)")
        for ax in axes[:, ci]:
            ax.set_xticks(np.arange(4), ["0", "1", "2", r"$\geq3$"])
            ax.set_xlabel("Geometric MC neutron multiplicity")
            ax.set_xlim(-0.4, 3.4)
    axes[0, 0].set_ylim(0, 100)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
    )
    fig.suptitle(
        "SMASH Xe+Cs, 2.87 GeV/nucleon | MC-linked reconstructed candidates", y=1.0
    )
    fig.text(
        0.5,
        0.025,
        "286 inspected even files; 3 seeds; raw energy; 70% native-label validation-purity target.\n"
        "0.3 < T(reco) < 4 GeV; bars: 400 paired file-draw SD. All true energies in geometric MC denominator.\n"
        "Zero-truth primary rates undefined. Direct links do not resolve descendants.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.065, right=0.99, top=0.84, bottom=0.15, hspace=0.40, wspace=0.28
    )
    save(fig, directory, "multiplicity_links")


def energy(result, directory):
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 12))
    setup(axes)
    edges = np.asarray(result["y_edges"])
    x = (edges[1:] + edges[:-1]) / 2
    cells = result["scopes"]["pooled"]["energy"]["by_y_pT"]
    metrics = (
        ("mae_MeV", "Energy MAE (MeV)"),
        ("bias_MeV", "Mean energy error (MeV)"),
        ("central_width_percent", r"Relative central width $(Q_{84}-Q_{16})/2$ (%)"),
    )
    for pt in range(2):
        for ri, (metric, ylabel) in enumerate(metrics):
            for mi, method in enumerate(METHODS):
                rows = [cell[pt]["metrics"][metric][method] for cell in cells]
                points(
                    axes[ri, pt],
                    x,
                    rows,
                    ("Rules + ToF", "DynamicEdgeConv")[mi],
                    COLORS[mi],
                    ("s", "o")[mi],
                    (-0.012, 0.012)[mi],
                )
            axes[ri, pt].set_ylabel(ylabel, fontsize=10)
            if metric == "bias_MeV":
                axes[ri, pt].axhline(0, color="#777777", lw=0.8)
        stats = [cell[pt]["metrics"]["mae_MeV"]["gnn_minus_rules"] for cell in cells]
        points(axes[3, pt], x, stats, "Paired MAE difference", COLORS[1], "o")
        axes[3, pt].axhline(0, color="#777777", lw=0.8)
        axes[3, pt].set_ylabel("Paired MAE: GNN minus rules (MeV)", fontsize=10)
        for i, cell in enumerate(cells):
            counts = cell[pt]["n_by_seed"]
            if 0 < min(counts) < 25:
                value = cell[pt]["metrics"]["mae_MeV"]["gnn_minus_rules"]["value"]
                axes[3, pt].annotate(
                    f"N={min(counts)}–{max(counts)}/seed",
                    (x[i], value),
                    xytext=(-8, 12),
                    textcoords="offset points",
                    ha="right",
                    fontsize=9,
                )
        axes[0, pt].set_title(
            r"MC $p_T<0.5$ GeV/$c$" if pt == 0 else r"MC $p_T\geq0.5$ GeV/$c$"
        )
        for ax in axes[:, pt]:
            ax.set_xlim(edges[0], edges[-1])
            ax.set_xlabel(r"MC neutron rapidity $y-y_{\mathrm{NN,CM}}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Energy response on the same MC-linked neutrons selected by both methods",
        y=1.0,
        fontsize=13,
    )
    fig.text(
        0.5,
        0.018,
        "SMASH Xe+Cs, 2.87 GeV/nucleon; 286 inspected even files; 3 seeds; frozen 70% native-label validation-purity target.\n"
        "0.3 <= T(true) < 4 GeV and 0.3 < T(reco) < 4 GeV in both methods. Bars: 400 paired file-draw SD.\n"
        "No native qualification required. Width: per-seed quantiles, then mean; >=25 pairs per seed. Empty/undefined cells not drawn; all N in JSON.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.10, right=0.99, bottom=0.13, top=0.91, hspace=0.55, wspace=0.30
    )
    save(fig, directory, "energy_y_pT")


def seeds(result, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    setup(axes)
    energy = result["scopes"]["pooled"]["energy"]
    for cohort, label, offset, marker, color in (
        ("integrated", "Common within each seed", -0.07, "s", "#333333"),
        ("same_across_seeds", "Same primary keys across seeds", 0.07, "o", "#006AA7"),
    ):
        rows = energy[cohort]["metrics"]["mae_MeV"]["gnn_minus_rules"]["by_seed"]
        points(
            axes[0],
            np.arange(3),
            [rows[s] for s in SEEDS],
            label,
            color,
            marker,
            offset,
        )
    axes[0].axhline(0, color="#777777", lw=0.8)
    axes[0].set_xticks(np.arange(3), SEEDS)
    axes[0].set_xlabel("Training seed label (all held-out folds combined)")
    axes[0].set_ylabel("Paired energy MAE: GNN minus rules (MeV)")
    axes[0].set_title("Energy comparison controls sample selection")
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=10
    )
    cases = result["rn_seed_stability"]["cases"]
    for si, (seed, color, marker) in enumerate(
        zip(SEEDS, ("#006AA7", "#B33B24", "#33804B"), ("o", "s", "^"), strict=True)
    ):
        stats = [
            cases[f"{target}/open_upper"]["90/0"]["gnn_minus_mc_pp"]["by_seed"][seed]
            for target in ("raw70", "raw75_frozen")
        ]
        points(axes[1], [0, 1], stats, f"Seed {seed}", color, marker, (si - 1) * 0.09)
    axes[1].axhline(0, color="#777777", lw=0.8)
    axes[1].set_xticks([0, 1], ["GNN 70%", "GNN 75%"])
    axes[1].set_xlim(-0.4, 1.4)
    axes[1].set_ylabel(r"Paired $C_{90/0}$ shift from geometric MC (pp)")
    axes[1].set_title("Frozen odd-file hardness example")
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Seed variation and fixed-model file uncertainty are reported separately",
        y=1.02,
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        "SMASH Xe+Cs, 2.87 GeV/nucleon. Energy: 286 even files; 0.3 <= T(true) < 4 and 0.3 < T(reco) < 4 GeV.\n"
        "Raw energy, 70% native-label validation-purity target. Hardness: 282 odd files; frozen 70% / 75% targets.\n"
        r"$R_n=N_{\rm hard}/N_{\rm soft}$; $C_{90/0}=100[R_n(90)/R_n(0)-1]$. Hard T >= 2; soft MC [0.3,1), reco (0.3,1) GeV."
        "\n"
        "3 seed centers; bars: 400 paired file-draw SD. Previously inspected files. Seed spread is descriptive; independent closure is unresolved.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.35, top=0.86, wspace=0.30)
    save(fig, directory, "seed_stability")


def main(directory):
    plt.rcParams.update(
        {
            "font.size": 11,
            "pdf.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    result = json.loads((directory / "results.json").read_bytes())
    multiplicity(result, directory)
    energy(result, directory)
    seeds(result, directory)
    paths = [
        directory / f"{name}.{suffix}"
        for name in ("multiplicity_links", "energy_y_pT", "seed_stability")
        for suffix in ("png", "pdf")
    ]
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest["source"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["results.json"] = hashlib.sha256(
        (directory / "results.json").read_bytes()
    ).hexdigest()
    (directory / "figures.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=HERE / "results")
    main(p.parse_args().directory)
