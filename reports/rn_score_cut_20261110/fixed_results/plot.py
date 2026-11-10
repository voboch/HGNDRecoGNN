"""Render conditional response, contrast widths and selection stability."""

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

from summarize import totals  # noqa: E402

RESULTS = json.loads((HERE / "results.json").read_text())
ODD = RESULTS["scopes"]["odd_fixed_check"]
DARK, BASE, CUT = "#2b2b2b", "#a64b28", "#135887"
SYSTEM = "BM@N HGND simulation · SMASH Xe+Cs, 2.87 GeV/nucleon · no impact-parameter selection"


def header(fig, label, sub):
    fig.text(0.10, 0.966, label, fontsize=17, va="top")
    fig.text(0.10, 0.923, sub, fontsize=10, va="top")
    fig.text(0.10, 0.890, SYSTEM, fontsize=9.5, va="top")


def save(fig, name, note):
    fig.text(0.10, 0.035, note, fontsize=9.5)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def whisker(ax, x, row, color, marker):
    value, lo, hi = (row[k] for k in ("value", "q16", "q84"))
    ax.vlines(x, lo, hi, color=color, lw=1.3)
    ax.hlines([lo, hi], x - 0.025, x + 0.025, color=color, lw=1.3)
    ax.plot(x, value, marker, color=color, ms=5.5)


def contrast_figure():
    fig, axs = plt.subplots(
        2, 2, figsize=(9, 6.1), sharex="col", gridspec_kw={"height_ratios": [1.65, 1]}
    )
    fig.subplots_adjust(
        left=0.12, right=0.96, bottom=0.15, top=0.78, hspace=0.06, wspace=0.27
    )
    header(
        fig,
        "Hardness contrasts at the fixed score cut",
        "282 odd files · retrospective check · fixed bands [0.3, 1) GeV and ≥2 GeV",
    )
    legend = [
        Line2D([], [], marker=m, color=c, ls="none", label=label)
        for m, c, label in [
            ("o", DARK, "Primary MC truth"),
            ("s", BASE, "Original cut"),
            ("^", CUT, "Contrast-selected cut"),
        ]
    ]
    fig.legend(
        handles=legend,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.858),
        fontsize=10,
    )
    for j, (ax, change) in enumerate(zip(*axs)):
        ax.set_title(
            ("Neutron hardness", "Neutron hardness / ideal MC proton hardness")[j]
        )
        mark_provenance(ax, "mixed", "upper left")
        for i, row in enumerate(ODD["contrasts"].values()):
            reco = (
                row["calibrated_ladder"]["C4"] if j == 0 else row["hybrid_ideal_proton"]
            )
            mc = row["truth_neutron" if j == 0 else "truth_ideal_np"]
            for dx, info, color, marker in [
                (-0.17, mc, DARK, "o"),
                (0, reco["purity70"], BASE, "s"),
                (0.17, reco["purity90"], CUT, "^"),
            ]:
                whisker(ax, i + dx, info, color, marker)
            whisker(change, i, reco["paired_change"], CUT, "^")
        change.axhline(0, ls="--", color="0.45", lw=0.8)
        change.set_xticks(np.arange(3), list(ODD["contrasts"]))
        change.set_xlabel("Potential pair (MeV)")
        ax.set_xlim(-0.4, 2.4)
    axs[0, 0].set_ylabel("Contrast (%)")
    axs[1, 0].set_ylabel("Paired change (pp)")
    save(
        fig,
        "paired_contrasts",
        "Intervals are Q16–Q84 of 400 paired whole-file draws. Three training repeats carry equal weight.",
    )


def spectra_figure():
    with np.load(HERE / "files.npz") as z, np.load(HERE / "draw_weights.npz") as b:
        assert np.array_equal(z["ids"], b["ids"])
        use = z["parts"] % 2 == 1
        tags = z["tags"][use]
        w = np.vstack([np.ones(use.sum(), int), b["weights"][:, use]])
        edges = z["edges"][1:-1]
        counts = totals(w, tags, z["spectra"][use, :, :, 5, 1:-1]).mean(axis=2)
        ne = totals(w, tags, z["spans"][use])
    density = counts / ne[..., None, None] / np.diff(edges)
    # A bin with no original entries in a draw has no defined retention.
    supported = (counts[:, :, 0] > 0).all(axis=0)
    fraction = np.divide(
        counts[:, :, 1],
        counts[:, :, 0],
        out=np.full_like(counts[:, :, 1], np.nan),
        where=supported[None, ...],
    )
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(10.5, 5.6),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [1.8, 1]},
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, bottom=0.16, top=0.76, wspace=0.09, hspace=0.04
    )
    header(
        fig,
        "Reconstructed neutron-candidate energy spectra",
        "282 odd files · calibrated energies · normalization includes events with no selected candidates",
    )
    handles = [
        Line2D([], [], color=c, ls=ls, label=label)
        for c, ls, label in [
            (BASE, "--", "Original cut"),
            (CUT, "-", "Contrast-selected cut"),
        ]
    ]
    fig.legend(
        handles=handles,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.85),
        fontsize=10,
    )
    for i, (ax, sub) in enumerate(zip(*axs)):
        ax.set_title(f"Nominal potential {RESULTS['potentials_MeV'][i]} MeV")
        mark_provenance(ax, "reco", "lower left")
        for j, color, ls in [(0, BASE, "--"), (1, CUT, "-")]:
            val = density[0, i, j]
            qlo, qhi = np.quantile(density[1:, i, j], [0.16, 0.84], axis=0)
            ax.stairs(
                np.where(val > 0, val, np.nan),
                edges,
                baseline=None,
                color=color,
                ls=ls,
                lw=1.3,
            )
            ax.stairs(
                qhi,
                edges,
                baseline=qlo,
                fill=True,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        ax.set_yscale("log")
        qlo, qhi = np.quantile(fraction[1:, i], [0.16, 0.84], axis=0)
        sub.stairs(fraction[0, i], edges, baseline=None, color=CUT)
        sub.stairs(
            qhi, edges, baseline=qlo, fill=True, color=CUT, alpha=0.18, linewidth=0
        )
        sub.axhline(1, ls="--", lw=0.8, color="0.45")
        for edge in [0.3, 1, 2]:
            ax.axvline(edge, ls=":", lw=0.7, color=".7")
            sub.axvline(edge, ls=":", lw=0.7, color=".7")
        sub.set_xlim(edges[[0, -1]])
        sub.set_xlabel("Reconstructed energy (GeV)")
    axs[0, 0].set_ylabel("Candidates / event / GeV")
    axs[1, 0].set_ylabel("Selected / original")
    save(
        fig,
        "selected_spectra",
        "Bands show paired file-bootstrap Q16–Q84. Ratio omitted where a draw has no original entries. Tails retained in the data.",
    )


def plateau_figure():
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.7, 5.9), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.80, bottom=0.16, hspace=0.05)
    header(
        fig,
        "Width of the score-cut maximum",
        "286 even selection files · 90/0 contrast · calibrated reconstructed candidates",
    )
    rows = RESULTS["selection_plateau"]
    for row in rows:
        p = row["target"]
        color = CUT if row["eligible"] else ".55"
        marker = "D" if p == 90 else "o"
        whisker(a, p, row["contrast"], color, marker)
        whisker(b, p, row["best_minus_policy"], color, marker)
    for ax in (a, b):
        ax.axhline(0, ls="--", color=".45", lw=0.8)
        ax.axvline(90, ls=":", color=CUT, lw=0.8)
    a.set_ylabel("90/0 contrast (%)")
    b.set_ylabel("90% target minus\ncomparison (pp)")
    b.set_xticks([x["target"] for x in rows])
    b.set_xlabel("Target count purity on validation (%)")
    mark_provenance(a, "reco", "upper left")
    save(
        fig,
        "selection_width",
        "Q16–Q84 of 400 paired file draws. Grey exceeds the registered precision ceiling. Diamond marks the frozen cut.",
    )


def main():
    apply_style()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    contrast_figure()
    spectra_figure()
    plateau_figure()


if __name__ == "__main__":
    main()
