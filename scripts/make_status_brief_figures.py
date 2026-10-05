#!/usr/bin/env python3
"""Three presentation-ready figures for the 2026-10-01 HGND status brief.

The brief deliberately separates:
  1. algorithm performance of the final SMASH checkpoint;
  2. held-out physics-observable separation;
  3. detector and multiplicity losses that require response correction.

All inputs are existing frozen JSON/CSV products.  No fitting or window search
is performed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#2166AC"
RED = "#B2182B"
GOLD = "#B07D18"
GREY = "#777777"
LIGHT = "#E8EDF2"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "legend.frameon": False,
        "figure.dpi": 170,
        "savefig.dpi": 240,
    })


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def performance_figure(perf: dict, out: Path) -> None:
    scan = pd.DataFrame(perf["threshold_scan"])
    energy = pd.DataFrame(perf["energy_performance"])
    q = float(perf["working_threshold"])
    wp = scan.iloc[(scan.threshold - q).abs().argmin()]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

    ax = axes[0]
    ax.plot(scan.efficiency_energy, scan.purity_energy, "o-", ms=3.5,
            color=BLUE, label="final GNN")
    ax.plot(wp.efficiency_energy, wp.purity_energy, "*", ms=14,
            color="#222222", label=rf"working point $q>{q:.2f}$")
    ax.plot(0.80, 0.87, marker="X", ms=8, color=RED,
            label="reference target")
    ax.set(xlabel="energy-weighted efficiency", ylabel="energy-weighted purity",
           xlim=(0.25, 1.0), ylim=(0.48, 1.0))
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"classification: AUC = {perf['roc_auc']:.3f}")

    ax = axes[1]
    ax.axhspan(-10, 10, color=LIGHT, zorder=0, label=r"$\pm10\%$ target")
    ax.axhline(0, color="0.35", lw=0.9)
    ax.plot(energy.e_mid, 100 * energy.linearity, "o-", color=BLUE, ms=4)
    ax.set(xlabel=r"true neutron $E_{\rm kin}$ [GeV]",
           ylabel="median energy bias [%]", xlim=(0.25, 5.1), ylim=(-13, 13))
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("energy linearity")

    ax = axes[2]
    ax.axhspan(0, 10, color=LIGHT, zorder=0, label=r"$<10\%$ target")
    ax.plot(energy.e_mid, 100 * energy.resolution, "o-", color=RED, ms=4)
    ax.set(xlabel=r"true neutron $E_{\rm kin}$ [GeV]",
           ylabel="robust resolution [%]", xlim=(0.25, 5.1), ylim=(0, 12))
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("energy resolution")

    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle("Final HGND reconstruction performance on SMASH", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig01_final_reconstruction_performance")


def sensitivity_figure(summary: dict, out: Path) -> None:
    keep = ["truth_np", "reco_yield", "hardness_truth", "hardness_q060",
            "truth_dr", "hybrid_q060"]
    rows = {r["key"]: r for r in summary["rows"]}
    rows = [rows[k] for k in keep]
    labels = [
        r"truth $Y_n/Y_p$",
        r"reco $\widehat{Y}_n$",
        r"truth $R_n$",
        r"reco $R_n$",
        r"truth $R_n/R_p$",
        r"reco $R_n$/truth $R_p$",
    ]
    colors = [GREY, GREY, "#333333", BLUE, "#333333", BLUE]
    y = np.arange(len(rows))[::-1]
    response = 100 * np.array([r["rel"] for r in rows])
    error = 100 * np.array([r["err"] for r in rows])
    sigma = np.array([r["sigma"] for r in rows])
    ordered = np.array([r["ordered"] for r in rows])

    fig, (ax, az) = plt.subplots(1, 2, figsize=(9.5, 4.25),
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    for i in range(len(rows)):
        marker = "o" if ordered[i] else "s"
        mfc = colors[i] if ordered[i] else "white"
        ax.errorbar(response[i], y[i], xerr=error[i], fmt=marker,
                    color=colors[i], mfc=mfc, mec=colors[i], ms=6,
                    lw=1.5, capsize=2.5)
        az.barh(y[i], sigma[i], color=colors[i], alpha=0.88, height=0.58)
    ax.axvline(0, color="0.55", lw=0.9, ls="--")
    az.axvline(3, color=RED, lw=1.1, ls="--")
    ax.set_yticks(y, labels)
    az.set_yticks(y, [""] * len(y))
    ax.set_xlabel(r"held-out response $O_{90}/O_0-1$ [%]")
    az.set_xlabel(r"separation $|\Delta|/\sigma_\Delta$")
    ax.set_xlim(-3.5, 15.0)
    az.set_xlim(0, max(6.4, sigma.max() * 1.12))
    ax.grid(axis="x", alpha=0.15)
    az.grid(axis="x", alpha=0.15)
    ax.text(0.02, 0.02, "filled = ordered 0/18/90 MeV",
            transform=ax.transAxes, fontsize=8, color="0.3")
    az.text(3, len(rows) - 0.25, r"$3\sigma$", color=RED,
            ha="center", va="bottom", fontsize=9)
    fig.suptitle("Held-out observable separation with the final GNN", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig02_heldout_observable_separation")


def response_figure(eff: dict, ranks: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax = axes[0]
    db = np.asarray(eff["detection"]["bins"], float)
    de = np.asarray(eff["detection"]["eps"], float)
    ax.step(db, np.r_[de[0], de], where="pre", color="#222222", lw=1.6,
            label=r"detection: selected $\mid$ reaches HGND")
    eb = np.asarray(eff["e_bins"], float)
    ec = 0.5 * (eb[:-1] + eb[1:])
    values = np.stack([np.asarray(v["eff"], float)
                       for v in eff["selection"].values()])
    ax.fill_between(ec, values.min(axis=0), values.max(axis=0),
                    color=BLUE, alpha=0.18, lw=0)
    ax.plot(ec, values.mean(axis=0), "o-", color=BLUE, ms=4,
            label=r"selection: selected $\mid$ cluster exists")
    ax.axvspan(1.0, 3.0, color=GOLD, alpha=0.09, lw=0)
    ax.set(xlabel=r"true neutron $E_{\rm kin}$ [GeV]", ylabel="efficiency",
           xlim=(0, 5.2), ylim=(0, 1.02))
    ax.legend(fontsize=7.6, loc="lower right")
    ax.set_title("energy-dependent response")
    ax.grid(alpha=0.15)

    ax = axes[1]
    usable = ranks[(ranks["rank"] <= 4) & (ranks["eligible_events"] >= 20)]
    for field, label, color, marker in (
        ("rank_efficiency", r"$\varepsilon_k$: $k$th neutron selected", BLUE, "o"),
        ("cumulative_efficiency", r"$C_k$: at least $k$ selected", RED, "s"),
    ):
        pivot = usable.pivot(index="rank", columns="dataset", values=field)
        x = pivot.index.to_numpy(float)
        v = pivot.to_numpy(float)
        ax.fill_between(x, np.nanmin(v, axis=1), np.nanmax(v, axis=1),
                        color=color, alpha=0.16, lw=0)
        ax.plot(x, np.nanmean(v, axis=1), marker + "-", color=color,
                ms=5, label=label)
    ax.set(xlabel="neutron rank $k$ (descending true energy)",
           ylabel="conditional probability", xlim=(0.75, 4.25), ylim=(0, 0.75))
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(fontsize=7.6, loc="upper right")
    ax.set_title("multiplicity response")
    ax.grid(alpha=0.15)

    fig.suptitle("Response corrections required for neutron yields", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig03_efficiency_and_multiplicity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--performance", type=Path,
                    default=Path("results/final_eval/observables/performance_q060.json"))
    ap.add_argument("--sensitivity", type=Path,
                    default=Path("results/final_eval/figures/final_sensitivity.json"))
    ap.add_argument("--efficiency", type=Path,
                    default=Path("results/final_eval/figures/efficiency_vs_ekin.json"))
    ap.add_argument("--ranks", type=Path,
                    default=Path("results/final_eval/figures/conditional_rank_efficiency.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/status_brief_20261001"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()
    performance_figure(json.loads(args.performance.read_text()), args.out_dir)
    sensitivity_figure(json.loads(args.sensitivity.read_text()), args.out_dir)
    response_figure(json.loads(args.efficiency.read_text()),
                    pd.read_csv(args.ranks), args.out_dir)
    print(f"wrote status figures to {args.out_dir}")


if __name__ == "__main__":
    main()
