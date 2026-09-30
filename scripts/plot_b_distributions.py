#!/usr/bin/env python3
"""Plot the event-level impact-parameter distributions used in the centrality note.

The input CSVs contain one row per primary particle, so ``B`` must first be
reduced to one value per event.  The files used for the feasibility study are
byte-limited extracts; their final event is therefore dropped as incomplete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DATASETS = {
    "zeroSpot": {"label": r"$S_{pot}=0$ MeV", "color": "#2b2b2b", "marker": "o", "ls": "-"},
    "defaultSpot": {"label": r"$S_{pot}=18$ MeV", "color": "#B22222", "marker": "s", "ls": "--"},
    "bigSpot": {"label": r"$S_{pot}=90$ MeV", "color": "#1f77b4", "marker": "^", "ls": "-"},
}


def load_event_b(input_dir: Path, dataset: str) -> np.ndarray:
    """Read one B value per complete event, preserving file-local Row identity."""
    values: list[np.ndarray] = []
    paths = sorted(input_dir.glob(f"{dataset}_*_prim.csv"))
    if not paths:
        raise FileNotFoundError(f"no {dataset}_*_prim.csv files in {input_dir}")
    for path in paths:
        frame = pd.read_csv(path, usecols=["Row", "B"])
        if frame.empty:
            continue
        last_row = frame["Row"].iloc[-1]
        frame = frame.loc[frame["Row"] != last_row]
        event_b = frame.drop_duplicates("Row", keep="first")["B"].to_numpy(float)
        if not np.all(np.isfinite(event_b)) or np.any(event_b < 0):
            raise ValueError(f"invalid B values in {path}")
        values.append(event_b)
    if not values:
        raise ValueError(f"no complete events for {dataset}")
    return np.concatenate(values)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    return x, np.arange(1, x.size + 1, dtype=float) / x.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = {name: load_event_b(args.input_dir, name) for name in DATASETS}
    all_b = np.concatenate(list(samples.values()))
    upper = np.ceil(all_b.max() / args.bin_width) * args.bin_width
    edges = np.arange(0.0, upper + args.bin_width * 0.5, args.bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts = {name: np.histogram(values, bins=edges)[0] for name, values in samples.items()}
    density = {
        name: hist / (hist.sum() * np.diff(edges))
        for name, hist in counts.items()
    }
    reference = density["zeroSpot"]

    summary: dict[str, object] = {
        "sample_scope": "four sequential byte-limited CSV extracts per production",
        "last_event_dropped_per_file": True,
        "bin_width_fm": args.bin_width,
        "datasets": {},
        "ks_vs_zeroSpot": {},
    }
    for name, values in samples.items():
        summary["datasets"][name] = {
            "events": int(values.size),
            "min_b_fm": float(values.min()),
            "max_b_fm": float(values.max()),
            "mean_b_fm": float(values.mean()),
            "median_b_fm": float(np.median(values)),
            "standard_error_mean_b_fm": float(values.std(ddof=1) / np.sqrt(values.size)),
        }
        if name != "zeroSpot":
            result = stats.ks_2samp(samples["zeroSpot"], values)
            summary["ks_vs_zeroSpot"][name] = {
                "D": float(result.statistic),
                "p_value": float(result.pvalue),
            }

    summary["zero_to_big_mean_change_percent"] = float(
        100.0 * (samples["bigSpot"].mean() / samples["zeroSpot"].mean() - 1.0)
    )

    with (args.output_dir / "b_distribution_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    rows = []
    for i, center in enumerate(centers):
        for name in DATASETS:
            ratio = density[name][i] / reference[i] if reference[i] > 0 else np.nan
            rows.append({
                "dataset": name,
                "b_lo_fm": edges[i],
                "b_hi_fm": edges[i + 1],
                "b_center_fm": center,
                "events": int(counts[name][i]),
                "density_per_fm": density[name][i],
                "density_ratio_to_zeroSpot": ratio,
            })
    pd.DataFrame(rows).to_csv(args.output_dir / "b_distribution_binned.csv", index=False)

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })
    fig = plt.figure(figsize=(6.84, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.5, 1.0])
    ax_pdf = fig.add_subplot(grid[0, 0])
    ax_cdf = fig.add_subplot(grid[0, 1])
    ax_ratio = fig.add_subplot(grid[1, :])

    for name, style in DATASETS.items():
        values = samples[name]
        ax_pdf.stairs(
            density[name], edges,
            label=rf"{style['label']} ($N={values.size:,}$, $\langle b\rangle={values.mean():.3f}$ fm)",
            color=style["color"], linestyle=style["ls"], linewidth=1.35,
        )
        x, y = ecdf(values)
        ax_cdf.step(x, y, where="post", color=style["color"], linestyle=style["ls"], linewidth=1.35)

    for name, style in DATASETS.items():
        if name == "zeroSpot":
            continue
        hist = density[name]
        mask = (reference > 0) & (counts[name] > 0)
        ratio = hist[mask] / reference[mask]
        # Independent Poisson approximation for the binned shape ratio.
        error = ratio * np.sqrt(1.0 / counts[name][mask] + 1.0 / counts["zeroSpot"][mask])
        ax_ratio.errorbar(
            centers[mask], ratio, yerr=error, color=style["color"], marker=style["marker"],
            markerfacecolor="none" if name == "defaultSpot" else style["color"],
            linestyle=style["ls"], linewidth=1.0, markersize=3.4, capsize=1.8,
            label=style["label"],
        )

    ax_pdf.set(xlabel=r"impact parameter $b$ [fm]", ylabel=r"normalized density [fm$^{-1}$]", xlim=(0, upper))
    ax_pdf.legend(frameon=False, loc="upper left")
    ax_pdf.text(0.98, 0.04, "four sequential byte-limited extracts per production",
                transform=ax_pdf.transAxes, ha="right", va="bottom", fontsize=7)
    ax_pdf.text(0.02, 0.84, "(a)", transform=ax_pdf.transAxes, ha="left", va="top", fontweight="bold")

    ax_cdf.set(xlabel=r"impact parameter $b$ [fm]", ylabel="empirical cumulative probability", xlim=(0, upper), ylim=(0, 1))
    ax_cdf.text(0.02, 0.96, "(b)", transform=ax_cdf.transAxes, ha="left", va="top", fontweight="bold")
    ax_cdf.text(0.98, 0.04,
                "KS vs 0 MeV: "
                f"18 MeV $D={summary['ks_vs_zeroSpot']['defaultSpot']['D']:.3f}$, "
                f"$p={summary['ks_vs_zeroSpot']['defaultSpot']['p_value']:.1e}$;\n"
                f"90 MeV $D={summary['ks_vs_zeroSpot']['bigSpot']['D']:.3f}$, "
                f"$p={summary['ks_vs_zeroSpot']['bigSpot']['p_value']:.1e}$",
                transform=ax_cdf.transAxes, ha="right", va="bottom", fontsize=7)

    ax_ratio.axhline(1.0, color="0.35", linewidth=0.8, linestyle=":")
    ax_ratio.set(xlabel=r"impact parameter $b$ [fm]", ylabel="density / 0 MeV", xlim=(0, upper))
    ax_ratio.legend(frameon=False, ncol=2, loc="upper right")
    ax_ratio.text(0.01, 0.92, "(c)", transform=ax_ratio.transAxes, ha="left", va="top", fontweight="bold")
    ax_ratio.text(0.01, 0.08,
                  "normalized before division; bars are independent Poisson counting errors",
                  transform=ax_ratio.transAxes, ha="left", va="bottom", fontsize=7)

    for ax in (ax_pdf, ax_cdf, ax_ratio):
        ax.grid(False)
    fig.suptitle("Impact-parameter sampling in the sequential 1% feasibility subsets", fontsize=9)
    fig.savefig(args.output_dir / "b_distributions.pdf")
    fig.savefig(args.output_dir / "b_distributions.png", dpi=220)
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
