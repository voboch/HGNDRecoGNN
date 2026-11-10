"""Render two complementary diagnostics on the common valid cohort."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = Path(__file__).resolve().parent
LABELS = [
    "First-hit ToF, original time",
    "First-hit ToF, jittered time",
    "DCM-trained, raw",
    "DCM-trained, calibrated",
    "SMASH-trained, raw",
    "SMASH-trained, calibrated",
]
COLORS = ["#444444"] * 2 + ["#a55022"] * 2 + ["#174d78"] * 2


def main():
    with np.load(H / "counts.npz") as data:
        weights = data["weights"]
        n = data["counts"][:, :, 1]
        moments = data["moments"][..., 1]
        bands = data["bands"]
        contents = data["content"]
    mae = 1000 * (moments.sum(0) / n.sum(0)[:, None]).mean(0)
    mae_draw = 1000 * (
        np.einsum("df,fsm->dsm", weights, moments) / (weights @ n)[..., None]
    ).mean(1)
    totals, draws = [], []
    for tag in ("bigSpot/", "zeroSpot/"):
        use = np.char.startswith(contents, tag)
        totals.append(bands[use].sum(0))
        draws.append(np.einsum("df,fsmb->dsmb", weights[:, use], bands[use]))
    ratio90, ratio0 = [x[..., 1] / x[..., 0] for x in totals]
    draw90, draw0 = [x[..., 1] / x[..., 0] for x in draws]
    contrast = 100 * (ratio90 / ratio0 - 1)
    contrast_draw = 100 * (draw90 / draw0 - 1)
    shift = (contrast[:, 1:] - contrast[:, :1]).mean(0)
    shift_draw = (contrast_draw[..., 1:] - contrast_draw[..., :1]).mean(1)

    paired = contrast[:, 5] - contrast[:, 1]
    paired_draw = contrast_draw[:, :, 5] - contrast_draw[:, :, 1]
    summary = {
        "methods": LABELS,
        "mean_over_training_repeats_MAE_MeV": mae.tolist(),
        "MAE_file_sd_MeV": mae_draw.std(0, ddof=1).tolist(),
        "mean_over_training_repeats_90_0_shift_pp": shift.tolist(),
        "shift_file_sd_pp": shift_draw.std(0, ddof=1).tolist(),
        "SMASH_raw_minus_original_ToF_90_0_pp": paired.tolist(),
        "paired_file_sd_pp": paired_draw.std(0, ddof=1).tolist(),
    }
    (H / "figure_data.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.size": 13, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), sharey=True)
    for ax, vals, sd, label in zip(
        axes,
        [mae, shift],
        [mae_draw.std(0, ddof=1), shift_draw.std(0, ddof=1)],
        [
            "Energy MAE against MC truth (MeV)",
            "90/0 contrast shift from matched truth (pp)",
        ],
    ):
        for i, (v, error, color) in enumerate(zip(vals, sd, COLORS)):
            ax.errorbar(v, i, xerr=error, marker="o", ms=5, color=color, capsize=3)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(range(len(LABELS)), LABELS)
    axes[0].invert_yaxis()
    axes[1].axvline(0, color="0.4", linestyle="--", linewidth=1)
    fig.text(
        0.5,
        0.95,
        "SMASH Xe+Cs, 2.87 GeV/nucleon • 286 files • isolated links passing both cuts",
        ha="center",
    )
    fig.text(
        0.5,
        0.055,
        "MC primary reference • both times physical • equal training-repeat weights • paired file-bootstrap SD",
        ha="center",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.225, right=0.98, bottom=0.25, top=0.87, wspace=0.19)
    fig.savefig(H / "first_hit_energy_comparison.png", dpi=180)
    fig.savefig(H / "first_hit_energy_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
