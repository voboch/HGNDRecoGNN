"""Plot the two comparisons used in the direct-primary response report."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "results.json").read_bytes())
    survival = data["survival_seed_mean_percent"]
    energy = data["common_energy_seed_mean_MeV"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.1))
    x = np.arange(3)
    colors = ("#1864A3", "#C45016")
    labels = ("A1", "B1")
    for ai, (label, color) in enumerate(zip(labels, colors)):
        for bi, (band, marker) in enumerate(zip(("0.3–1 GeV", "≥2 GeV"), ("o", "^"))):
            left.errorbar(
                x + (ai - 0.5) * 0.19 + (bi - 0.5) * 0.075,
                np.asarray(survival["value"])[:, ai, bi],
                yerr=np.asarray(survival["file_sd"])[:, ai, bi],
                fmt=marker,
                color=color,
                capsize=3,
                label=f"{label}, {band}",
            )
        for ei, (energy_label, marker) in enumerate(
            zip(("raw", "calibrated"), ("o", "^"))
        ):
            right.errorbar(
                x + (ai - 0.5) * 0.19 + (ei - 0.5) * 0.075,
                np.asarray(energy["value"])[:, ai, ei, 1],
                yerr=np.asarray(energy["file_sd"])[:, ai, ei, 1],
                fmt=marker,
                color=color,
                capsize=3,
                label=f"{label}, {energy_label}",
            )
    left.set(
        title="Score survival of directly linked primaries",
        ylabel="Passing primaries / linked primaries (%)",
    )
    right.set(
        title="Energy error on the common isolated sample",
        ylabel="MAE against initial MC energy (MeV)",
    )
    for ax in (left, right):
        ax.set_xticks(x, data["potentials_MeV"])
        ax.set_xlabel(r"$U_{\mathrm{sym}}$ (MeV)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False
        )
    fig.text(
        0.5,
        0.965,
        "SMASH Xe+Cs, 2.87 GeV/nucleon • S286 geometric T1 • no impact-parameter selection",
        ha="center",
    )
    fig.text(
        0.5,
        0.045,
        "MC direct links + frozen TEST predictions • equal seed weights • bars show paired whole-file bootstrap SD",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.34, top=0.83, wspace=0.27)
    fig.savefig(HERE / "score_energy_response.png", dpi=180)
    fig.savefig(HERE / "score_energy_response.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
