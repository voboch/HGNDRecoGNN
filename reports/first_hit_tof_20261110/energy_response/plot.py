"""Render linearity and core resolution with their common bin support."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from plot_b_full import apply_style, mark_provenance  # noqa: E402

SERIES = (
    ("Original first-hit ToF", "First-hit ToF", "#292929", "-", "o"),
    ("Jittered first-hit ToF", "First-hit ToF + jitter", "#a64b28", "--", "s"),
    ("SMASH-trained raw", "SMASH-trained, raw", "#135887", "-", "^"),
    ("SMASH-trained calibrated", "SMASH-trained, calibrated", "#135887", ":", "D"),
)
TRUTH_AXIS = r"Initial primary-neutron MC kinetic energy $E_{\mathrm{true}}$ (GeV)"


def panels(title, selected=False):
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(9.6, 6.0), gridspec_kw={"height_ratios": [2.5, 1]}
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.76, bottom=0.15, hspace=0)
    fig.text(0.13, 0.962, title, size=17, weight="medium", va="top")
    fig.text(
        0.13,
        0.908,
        "BM@N HGND simulation · SMASH Xe+Cs, 2.87 GeV/nucleon · 286 files · no impact-parameter selection",
        size=10,
        va="top",
    )
    fig.text(
        0.13,
        0.873,
        (
            "Even selection files · isolated links · contrast-selected SMASH and original DCM cuts · both times physical"
            if selected
            else "Nominal HGND window · isolated neutron links · both score cuts · physical speed for both first-hit times"
        ),
        size=10,
        va="top",
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                label=label,
                color=color,
                linestyle=style,
                marker=marker,
                mfc="white",
                ms=4,
            )
            for _, label, color, style, marker in SERIES
        ],
        loc="upper left",
        bbox_to_anchor=(0.12, 0.84),
        ncols=4,
        fontsize=10,
        handlelength=2.0,
        columnspacing=1.25,
    )
    fig.text(
        0.13,
        0.057,
        "Equal weight per training repeat · bars show SD from 400 paired whole-file resamples",
        fontsize=10,
    )
    fig.text(
        0.13,
        0.030,
        "MC matching includes ToF · nominal straight path, no separate event start · jitter SD 0.15 ns",
        fontsize=10,
    )
    for ax in (top, bottom):
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    bottom.set_xlabel(TRUTH_AXIS)
    return fig, top, bottom


def metric_values(bins, method, metric):
    return np.array(
        [
            (
                b["methods"][method][metric]["value"],
                b["methods"][method][metric]["file_sd"],
            )
            if method in b["methods"]
            else (np.nan, np.nan)
            for b in bins
        ]
    ).T


def draw_metric(ax, bins, edges, x, metric, *, stairs=True):
    for method, _, color, style, marker in SERIES:
        y, sd = metric_values(bins, method, metric)
        if stairs:
            ax.stairs(
                y, edges, baseline=None, color=color, linestyle=style, linewidth=1.6
            )
        ax.errorbar(
            x,
            y,
            yerr=sd,
            color=color,
            marker=marker,
            mfc="white",
            capsize=2.0,
            markersize=3.7,
            linestyle="none",
            elinewidth=0.8,
        )


def check_extent(ax, bins, x, metric):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    for method, *_ in SERIES:
        y, sd = metric_values(bins, method, metric)
        valid = np.isfinite(y)
        assert np.all((x[valid] >= x_min) & (x[valid] <= x_max)), metric
        assert np.all(
            (y[valid] - sd[valid] >= y_min) & (y[valid] + sd[valid] <= y_max)
        ), metric


def save(fig, stem, source):
    for ext in ("png", "pdf"):
        fig.savefig(source / f"{stem}.{ext}", dpi=200, bbox_inches=None)
    plt.close(fig)


def main(source=HERE):
    result = json.loads((source / "bins.json").read_text())
    bins = result["bins"]
    selected = result.get("smash_score_policy") == "contrast-selected"
    edges = np.array([bins[0]["low_GeV"], *(b["high_GeV"] for b in bins)])
    x = np.array([b.get("median_truth_GeV", np.nan) for b in bins])
    apply_style()
    plt.rcParams.update(
        {"axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10}
    )

    fig, top, bottom = panels("Neutron energy linearity", selected)
    top.plot(edges[[0, -1]], edges[[0, -1]], "--", color="0.5", lw=1, zorder=0)
    draw_metric(top, bins, edges, x, "median_reco_GeV", stairs=False)
    top.set_ylabel(r"Median reconstructed $E_{\mathrm{kin}}$ (GeV)")
    mark_provenance(top, "mixed", "upper left")
    bottom.axhline(0, linestyle="--", color="0.5", linewidth=1, zorder=0)
    draw_metric(bottom, bins, edges, x, "linearity_percent")
    bottom.set_ylabel("Median relative\nresidual (%)")
    lo, hi = bottom.get_ylim()
    bound = max(abs(lo), abs(hi))
    bottom.set_ylim(-bound, bound)
    bottom.set_xlim(edges[0], edges[-1])
    check_extent(top, bins, x, "median_reco_GeV")
    check_extent(bottom, bins, x, "linearity_percent")
    save(fig, "energy_linearity", source)

    fig, top, bottom = panels("Neutron energy resolution", selected)
    draw_metric(top, bins, edges, x, "resolution_percent")
    top.set_ylabel("Relative residual\ncentral 68% half-width (%)")
    mark_provenance(top, "mixed", "upper left")
    counts = np.array([b["counts"] for b in bins])
    low, high = counts.min(1), counts.max(1)
    bottom.stairs(high, edges, baseline=low, fill=True, color="0.78")
    bottom.stairs(low, edges, baseline=None, color="0.25", linewidth=1.1)
    bottom.stairs(high, edges, baseline=None, color="0.25", linewidth=1.1)
    bottom.set_yscale("log")
    bottom.set_ylabel("Matched objects")
    bottom.text(
        0.02,
        0.08,
        "Band spans training repeats",
        transform=bottom.transAxes,
        fontsize=9,
    )
    bottom.set_xlim(edges[0], edges[-1])
    check_extent(top, bins, x, "resolution_percent")
    save(fig, "energy_resolution", source)


if __name__ == "__main__":
    main()
