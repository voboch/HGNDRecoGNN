"""Render the matched response and paired MC displacement from the verified reduction."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENERGY_ARMS = {
    "baseline_raw": ("Original raw", "#454650", "o"),
    "unit_mse": ("Unweighted fine-tune", "#00538a", "s"),
    "energy_weighted_mse": ("Energy-weighted fine-tune", "#a3392c", "D"),
}


def interval(row):
    return row["value"], row["q16"], row["q84"]


def save(fig, directory, name):
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def response(result, directory):
    rows = [r for r in result["matched_even_energy"]["bins"] if "linearity_percent" in r]
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.2), sharex=True, layout="constrained")
    coordinates = np.array([r["median_truth_GeV"] for r in rows])
    for ax, metric, label in zip(
        axes, ("linearity_percent", "resolution_percent"),
        ("Median relative residual (%)", "Central residual half-width (%)"), strict=True,
    ):
        for arm, (name, color, marker) in ENERGY_ARMS.items():
            values = [r[metric][arm] for r in rows]
            ax.errorbar(coordinates, [v["value"] for v in values],
                        yerr=[v["file_sd"] for v in values], label=name, color=color,
                        marker=marker, lw=1.35, markersize=4, capsize=2)
        ax.set_ylabel(label)
        ax.axvspan(5, 6.5, color="#f1f0ed", zorder=-2)
        ax.grid(axis="y", color="#dedede", linewidth=.55)
    axes[0].axhline(0, color="#777777", lw=.7, ls="--")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    axes[0].set_title("Reconstructed energy against primary-neutron MC truth")
    axes[1].set_xlabel("MC truth kinetic energy (GeV)")
    axes[1].set_xlim(.3, 6.5)
    axes[1].set_ylim(bottom=0)
    axes[1].text(5.75, .94, "Sparse tail", transform=axes[1].get_xaxis_transform(),
                 ha="center", va="top", color="#555555", fontsize=9)
    save(fig, directory, "energy_response")


def hardness(result, directory):
    pairs = result["odd_hardness"]["open_2_GeV"]["pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True)
    fig.subplots_adjust(left=.11, right=.98, bottom=.26, top=.87, wspace=.15)
    models = {**ENERGY_ARMS, "baseline_calibrated": ("Original calibrated", "#78628a", "^")}
    pair_order = ("18/0", "90/0", "90/18")
    offsets = np.linspace(-.27, .27, len(models) + 1)
    labels = ("Contrast (%)", "Paired displacement from MC truth (p.p.)")
    fields = ("contrast_percent", "paired_bias_vs_mc_pp")
    for i, (ax, field, label) in enumerate(zip(axes, fields, labels, strict=True)):
        for offset, (arm, (name, color, marker)) in zip(
            offsets, {"mc_truth": ("MC truth", "#222222", "x"), **models}.items(), strict=True
        ):
            if arm == "mc_truth" and i == 1:
                continue
            point = [interval(pairs[p][field][arm]) for p in pair_order]
            y = np.arange(len(pair_order)) + offset
            center, lo, hi = np.asarray(point).T
            ax.hlines(y, lo, hi, color=color, lw=1.4)
            ax.plot(center, y, linestyle="none", color=color, marker=marker,
                    markersize=4.8, label=name)
        ax.axvline(0, color="#777777", lw=.8, ls="--")
        ax.set_xlabel(label)
        ax.set_yticks(np.arange(len(pair_order)), pair_order)
        ax.grid(axis="x", color="#dedede", linewidth=.55)
    axes[0].set_ylabel("Symmetry-potential pair (MeV/MeV)")
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.5, .025),
               frameon=False, fontsize=9, ncols=3)
    fig.suptitle("Neutron hardness contrast on the same odd files")
    save(fig, directory, "hardness_contrast")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    output = parser.parse_args().directory
    result = json.loads((output / "results.json").read_text())
    response(result, output)
    hardness(result, output)
