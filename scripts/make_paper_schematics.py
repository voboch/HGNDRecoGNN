"""Generate compact vector schematics used by the HGND paper."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BLUE = "#2166AC"
RED = "#B2182B"
GRAY = "#4D4D4D"
LIGHT = "#E8EEF4"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 1,
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
    })


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{suffix}")
    plt.close(fig)


def detector_layout(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.set_xlim(-0.8, 8.2)
    ax.set_ylim(-3.2, 2.8)
    ax.axis("off")
    ax.annotate("", xy=(7.8, 0), xytext=(-0.65, 0),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=GRAY))
    ax.text(-0.72, 0.38, "beam", va="center", color=GRAY)
    ax.scatter([0], [0], s=90, marker="*", color=RED, zorder=4)
    ax.text(0, -0.42, "target", ha="center", va="top")

    for sign, label in ((1, "upper arm"), (-1, "lower arm")):
        y0 = sign * 1.32
        for layer in range(8):
            x = 6.75 + 0.095 * layer
            ax.add_patch(Rectangle((x, y0 - 0.58), 0.065, 1.16,
                                   facecolor=BLUE if layer % 2 == 0 else "#6BAED6",
                                   edgecolor="white", lw=0.25))
        ax.add_patch(Rectangle((6.68, y0 - 0.64), 0.88, 1.28,
                               fill=False, edgecolor=GRAY, lw=1.1))
        ax.plot([0, 6.68], [0, y0], color="0.60", ls="--", lw=0.9)
        label_y = y0 + 0.84 if sign > 0 else y0 - 0.72
        ax.text(7.12, label_y, label, ha="center",
                va="bottom" if sign > 0 else "top", fontweight="bold")
    ax.text(6.98, 0, "7 m", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7"))
    ax.text(2.25, 0.57, r"$10^\circ$", color=GRAY)
    ax.text(4.2, -2.85,
            r"2 arms $\times$ 8 active layers; each layer: $11\times11$ scintillator cells",
            ha="center", va="center", fontsize=8.5)
    _save(fig, out, "detector_layout")


def _box(ax, x: float, y: float, w: float, h: float, text: str,
         fc: str = LIGHT, ec: str = BLUE, fontsize: float = 8.5) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.025,rounding_size=0.04",
                               facecolor=fc, edgecolor=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize)


def _arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                mutation_scale=11, color=GRAY, lw=1.1))


def architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    _box(ax, 0.25, 4.25, 1.65, 0.85, "hit features\n$(x,y,z,t,E)$")
    _box(ax, 2.25, 4.25, 1.65, 0.85, "spatio-temporal\nradius graph")
    _box(ax, 4.25, 4.25, 2.05, 0.85,
         "hit message passing\nEdgeConv + SAGEConv")
    _box(ax, 6.65, 4.25, 1.45, 0.85, "cluster pooling")
    _box(ax, 8.45, 4.25, 1.3, 0.85, "cluster graph")
    for a, b in ((1.9, 2.25), (3.9, 4.25), (6.3, 6.65), (8.1, 8.45)):
        _arrow(ax, a, 4.675, b, 4.675)

    _box(ax, 6.55, 2.55, 1.65, 0.8, "neutron score\n$s\in[0,1]$",
         fc="#E5F5E0", ec="#238B45")
    _box(ax, 8.35, 2.55, 1.4, 0.8, "energy\n" + r"$E_{\rm pred}$",
         fc="#FEE0D2", ec=RED)
    _box(ax, 4.65, 2.55, 1.55, 0.8, "link score",
         fc="#F2F0F7", ec="#756BB1")
    _arrow(ax, 9.1, 4.25, 9.05, 3.35)
    _arrow(ax, 8.8, 4.25, 7.4, 3.35)
    _arrow(ax, 5.25, 4.25, 5.4, 3.35)

    _box(ax, 0.4, 0.55, 3.95, 0.9,
         "reference\nper-sample feature scaling",
         fc="#F7F7F7", ec=GRAY)
    _box(ax, 5.65, 0.55, 3.95, 0.9,
         "challenger\none scaler fitted on pooled training jobs",
         fc="#FFF2CC", ec="#B8860B")
    ax.text(5.0, 1.85,
            "same graph, architecture, loss, split, and working-point definition",
            ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(5.0, 5.7, "Heterogeneous graph-neural-network reconstruction",
            ha="center", va="center", fontsize=11, fontweight="bold")
    _save(fig, out, "gnn_architecture")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    detector_layout(args.out_dir)
    architecture(args.out_dir)
    print(f"wrote paper schematics to {args.out_dir}")


if __name__ == "__main__":
    main()
