"""Show the validation trade-off and the paired physics outcomes."""

import argparse
import importlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
plots = importlib.import_module("HGNDRecoGNN.reports.energy_head_dombay_20261111.plot")
ARMS = {
    "baseline_raw": ("Original raw", "#454650", "o"),
    "unit_mse": ("Unweighted fine-tune", "#00538a", "s"),
    "capped_raw": ("Capped raw", "#a3392c", "D"),
    "capped_calibrated": ("Capped calibrated", "#267456", "v"),
}


def validation(result, directory):
    selection = result["selection"]
    caps = np.asarray(selection["caps"])
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), layout="constrained")
    thresholds = (None, selection["mae_ratio_limit"], selection["width_ratio_limit"])
    names = ("tail_bias_percent", "mae_MeV", "tail_half_width_percent")
    labels = ("Absolute 4–5 GeV bias (%)", "Overall MAE / control", "4–5 GeV width / control")
    offsets = np.linspace(-.26, .26, len(result["validation"]))
    for ax, threshold, name, label in zip(axes, thresholds, names, labels, strict=True):
        for offset, row in zip(offsets, result["validation"], strict=True):
            x = np.arange(len(caps)) + offset
            values = np.array([c["metrics"][name] for c in row["candidates"]])
            values = np.abs(values) if threshold is None else values / row["reference"][name]
            ax.plot(x, values, "o-", color="#7a858f", lw=.6, markersize=3, alpha=.7)
            selected, = np.flatnonzero(caps == row["selected_cap"])
            ax.plot(x[selected], values[selected], "D", color="#a3392c", markersize=4)
        if threshold is not None:
            ax.axhline(1, color="#454650", lw=.7)
            ax.axhline(threshold, color="#a3392c", ls="--", lw=.8)
        ax.set_xticks(np.arange(len(caps)), caps)
        ax.set_xlabel("Maximum training weight")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#dedede", linewidth=.55)
    fig.suptitle("Validation against MC-labelled clusters")
    plots.save(fig, directory, "validation_caps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    output = parser.parse_args().directory
    result = json.loads((output / "results.json").read_text())
    validation(result, output)
    plots.response(result, output, ARMS)
    plots.hardness(result, output, ARMS)
