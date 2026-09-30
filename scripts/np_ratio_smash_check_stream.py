#!/usr/bin/env python3
"""Memory-bounded full-production reproduction of the n/p SMASH plots.

The primary CSV production is about 85 GB uncompressed.  This program reads one
CSV at a time and retains only sufficient statistics for the event-clustered
ratio uncertainty, so the result is exact without concatenating all particles
in memory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from np_ratio_smash_check import (
    DATASETS,
    PLOT_STYLE,
    REQUIRED_COLUMNS,
    SPOT_MEV,
    _footer,
    _style_axis,
    write_decision,
)


@dataclass(frozen=True)
class PanelSpec:
    figure: str
    panel: str
    variable: str
    edges: np.ndarray
    selector: str


EKIN_EDGES = np.r_[np.arange(0.0, 1.61, 0.1), 1.8, 2.0, 2.3, 2.8, 3.0]
Y_EDGES = np.arange(-1.5, 1.5001, 0.1)
PT_EDGES = np.r_[np.arange(0.0, 2.01, 0.1), 2.2, 2.4, 2.8, 3.0]
PANEL_SPECS = (
    PanelSpec("ekin", "abs_y_lt_0p5", "Ekin", EKIN_EDGES, "abs_y_lt"),
    PanelSpec("ekin", "abs_y_gt_0p5", "Ekin", EKIN_EDGES, "abs_y_ge"),
    PanelSpec("02_np_vs_y_ekin_gt_0p3", "pt_lt_0p5", "y_cm", Y_EDGES, "pt_lt_ecut"),
    PanelSpec("02_np_vs_y_ekin_gt_0p3", "pt_gt_0p5", "y_cm", Y_EDGES, "pt_ge_ecut"),
    PanelSpec("03_np_vs_pt_ekin_gt_0p3", "abs_y_lt_0p5", "Pt", PT_EDGES, "abs_y_lt_ecut"),
    PanelSpec("03_np_vs_pt_ekin_gt_0p3", "abs_y_gt_0p5", "Pt", PT_EDGES, "abs_y_ge_ecut"),
    PanelSpec("04_np_vs_y_all_ekin", "pt_lt_0p5", "y_cm", Y_EDGES, "pt_lt"),
    PanelSpec("04_np_vs_y_all_ekin", "pt_gt_0p5", "y_cm", Y_EDGES, "pt_ge"),
    PanelSpec("05_np_vs_pt_all_ekin", "abs_y_lt_0p5", "Pt", PT_EDGES, "abs_y_lt"),
    PanelSpec("05_np_vs_pt_all_ekin", "abs_y_gt_0p5", "Pt", PT_EDGES, "abs_y_ge"),
)


def discover_files(input_dir: Path, dataset: str) -> tuple[Path, ...]:
    """Find both the old flat export and the production's nested layout."""
    flat = sorted(input_dir.glob(f"{dataset}_*_prim.csv"))
    if flat:
        return tuple(flat)
    # Production archives extract one dataset directory directly below the
    # supplied root.  Do not recursively glob the root: on Lustre (and on the
    # local archive mount) that needlessly stats every hits/vacs file first.
    roots = sorted(
        path for path in input_dir.iterdir()
        if path.is_dir() and dataset in path.name
    )
    files: set[Path] = set()
    for root in roots:
        files.update(root.glob("*/*_prim.csv"))
    if not files and dataset in input_dir.name:
        files.update(input_dir.glob("*/*_prim.csv"))
    if not files:
        raise FileNotFoundError(f"No primary CSV files for {dataset} under {input_dir}")
    return tuple(sorted(files))


def _selection(name: str, ekin: np.ndarray, pt: np.ndarray, y_cm: np.ndarray) -> np.ndarray:
    masks = {
        "abs_y_lt": np.abs(y_cm) < 0.5,
        "abs_y_ge": np.abs(y_cm) >= 0.5,
        "pt_lt": pt < 0.5,
        "pt_ge": pt >= 0.5,
        "pt_lt_ecut": (pt < 0.5) & (ekin > 0.3),
        "pt_ge_ecut": (pt >= 0.5) & (ekin > 0.3),
        "abs_y_lt_ecut": (np.abs(y_cm) < 0.5) & (ekin > 0.3),
        "abs_y_ge_ecut": (np.abs(y_cm) >= 0.5) & (ekin > 0.3),
    }
    return masks[name]


def _empty_stats(nbins: int) -> dict[str, np.ndarray]:
    return {name: np.zeros(nbins, dtype=np.float64) for name in (
        "n", "p", "n_raw", "p_raw", "n2", "p2", "np",
    )}


def _accumulate_panel(
    stats: dict[str, np.ndarray],
    event_code: np.ndarray,
    pdg: np.ndarray,
    weight: np.ndarray,
    values: np.ndarray,
    selected: np.ndarray,
    edges: np.ndarray,
    nevents: int,
) -> None:
    nbins = len(edges) - 1
    bins = np.searchsorted(edges, values, side="right") - 1
    selected = selected & (bins >= 0) & (bins < nbins)
    keys = event_code[selected] * nbins + bins[selected]
    pdg_selected = pdg[selected]
    weights = weight[selected]
    n_mask = pdg_selected == 2112
    p_mask = pdg_selected == 2212
    size = nevents * nbins
    n_event = np.bincount(keys[n_mask], weights=weights[n_mask], minlength=size).reshape(nevents, nbins)
    p_event = np.bincount(keys[p_mask], weights=weights[p_mask], minlength=size).reshape(nevents, nbins)
    n_raw_event = np.bincount(keys[n_mask], minlength=size).reshape(nevents, nbins)
    p_raw_event = np.bincount(keys[p_mask], minlength=size).reshape(nevents, nbins)
    stats["n"] += n_event.sum(axis=0)
    stats["p"] += p_event.sum(axis=0)
    stats["n_raw"] += n_raw_event.sum(axis=0)
    stats["p_raw"] += p_raw_event.sum(axis=0)
    stats["n2"] += np.square(n_event).sum(axis=0)
    stats["p2"] += np.square(p_event).sum(axis=0)
    stats["np"] += (n_event * p_event).sum(axis=0)


def analyze_dataset(
    input_dir: Path,
    dataset: str,
    y_shift: float,
    min_count: int,
    max_files: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    files = discover_files(input_dir, dataset)
    if max_files is not None:
        files = files[:max_files]
    accumulators = {
        (spec.figure, spec.panel): _empty_stats(len(spec.edges) - 1)
        for spec in PANEL_SPECS
    }
    total_events = 0
    total_rows = 0
    all_columns: set[str] = set()
    b_valid = True
    npart_valid = True
    weights_present = True

    for file_number, path in enumerate(files, start=1):
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = REQUIRED_COLUMNS - header
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        all_columns.update(header)
        optional = [column for column in ("B", "NPrim", "Weight") if column in header]
        frame = pd.read_csv(path, usecols=sorted(REQUIRED_COLUMNS) + optional)
        row = frame["Row"].to_numpy(dtype=np.int64)
        _, event_code = np.unique(row, return_inverse=True)
        nevents = int(event_code.max()) + 1 if len(event_code) else 0
        total_events += nevents
        total_rows += len(frame)
        b_valid &= "B" in frame and np.isfinite(frame["B"]).all() and (frame["B"] >= 0).any()
        npart_valid &= "NPrim" in frame and np.isfinite(frame["NPrim"]).all() and (frame["NPrim"] >= 0).any()
        weights_present &= "Weight" in frame
        weight = frame["Weight"].to_numpy(float) if "Weight" in frame else np.ones(len(frame))
        pdg = frame["PDG"].to_numpy(np.int64)
        ekin = frame["Ekin"].to_numpy(float)
        pt = frame["Pt"].to_numpy(float)
        y_cm = frame["Rapid"].to_numpy(float) - y_shift
        values_by_name = {"Ekin": ekin, "Pt": pt, "y_cm": y_cm}
        for spec in PANEL_SPECS:
            _accumulate_panel(
                accumulators[(spec.figure, spec.panel)],
                event_code,
                pdg,
                weight,
                values_by_name[spec.variable],
                _selection(spec.selector, ekin, pt, y_cm),
                spec.edges,
                nevents,
            )
        print(
            f"{dataset}: {file_number}/{len(files)} files, "
            f"{total_events:,} events, {total_rows:,} rows",
            flush=True,
        )

    tables: list[pd.DataFrame] = []
    correction = total_events / max(total_events - 1, 1)
    for spec in PANEL_SPECS:
        stats = accumulators[(spec.figure, spec.panel)]
        valid = (stats["p"] > 0) & (stats["n_raw"] >= min_count) & (stats["p_raw"] >= min_count)
        ratio = np.divide(stats["n"], stats["p"], out=np.full_like(stats["n"], np.nan), where=valid)
        influence2 = stats["n2"] - 2 * ratio * stats["np"] + np.square(ratio) * stats["p2"]
        stat_error = np.divide(
            np.sqrt(correction * np.maximum(influence2, 0)),
            stats["p"],
            out=np.full_like(stats["p"], np.nan),
            where=valid,
        )
        table = pd.DataFrame({
            "figure": spec.figure,
            "panel": spec.panel,
            "dataset": dataset,
            "spot_mev": SPOT_MEV[dataset],
            "bin_lo": spec.edges[:-1],
            "bin_hi": spec.edges[1:],
            "bin_center": (spec.edges[:-1] + spec.edges[1:]) / 2,
            "n": stats["n"],
            "p": stats["p"],
            "n_raw": stats["n_raw"].astype(np.int64),
            "p_raw": stats["p_raw"].astype(np.int64),
            "ratio": ratio,
            "stat_error": stat_error,
            "valid": valid,
            "events_total": total_events,
        })
        tables.append(table)
    info = {
        "files": len(files),
        "events": total_events,
        "rows": total_rows,
        "columns": sorted(all_columns),
        "centrality_B_valid": bool(b_valid),
        "participant_count_valid": bool(npart_valid),
        "weights_present": bool(weights_present),
    }
    return pd.concat(tables, ignore_index=True), info


def _plot_table(ax: plt.Axes, table: pd.DataFrame, figure: str, panel: str) -> None:
    for dataset in DATASETS:
        selected = table[
            (table["figure"] == figure)
            & (table["panel"] == panel)
            & (table["dataset"] == dataset)
            & table["valid"]
        ]
        style = PLOT_STYLE[dataset]
        ax.errorbar(
            selected["bin_center"], selected["ratio"], yerr=selected["stat_error"],
            linestyle="none", marker=style["marker"], markersize=4.5,
            color=style["color"], markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"], markeredgewidth=0.8,
            elinewidth=1.0, capsize=0,
            label=rf"$S_{{\rm pot}}={SPOT_MEV[dataset]}$ MeV",
        )
    ax.legend(frameon=False, fontsize=8, loc="upper left", handletextpad=0.4)


def build_figures(
    table: pd.DataFrame,
    output_dir: Path,
    pdf_output: Path,
    beam_energy: float,
    centrality_valid: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    pages: list[tuple[str, plt.Figure]] = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.625))
    fig.suptitle(rf"Xe+Cs @ {beam_energy:g}A GeV: $E_{{\rm kin}}$-dependence", fontsize=21, y=0.96)
    for ax, panel, title, ylim in zip(
        axes,
        ("abs_y_lt_0p5", "abs_y_gt_0p5"),
        (r"$|y_{\rm cm}|<0.5$", r"$|y_{\rm cm}|>0.5$"),
        ((1.0, 1.52), (1.0, 1.80)),
    ):
        _plot_table(ax, table, "ekin", panel)
        _style_axis(ax, (0, 3), ylim)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel(r"$E_{\rm kin}$, GeV", fontweight="bold")
        ax.axvline(0.3, color="0.35", linewidth=2.0)
    fig.text(0.5, 0.105, r"Reference working cut: $E_{\rm kin}>300$ MeV", ha="center", fontsize=12, color="0.35")
    fig.tight_layout(rect=(0.03, 0.14, 0.99, 0.91), w_pad=4.0)
    _footer(fig, centrality_valid)
    pages.append(("01_np_vs_ekin_y", fig))

    page_specs = (
        ("02_np_vs_y_ekin_gt_0p3", "y", True),
        ("03_np_vs_pt_ekin_gt_0p3", "pt", True),
        ("04_np_vs_y_all_ekin", "y", False),
        ("05_np_vs_pt_all_ekin", "pt", False),
    )
    for page_id, kind, with_ecut in page_specs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.625))
        suffix = r" with $E_{\rm kin}>300$ MeV" if with_ecut else ""
        variable = r"y_{\rm cm}" if kind == "y" else r"p_T"
        fig.suptitle(rf"Xe+Cs @ {beam_energy:g}A GeV: ${variable}$-dependence{suffix}", fontsize=19, y=0.96)
        if kind == "y":
            panels = ("pt_lt_0p5", "pt_gt_0p5")
            titles = (r"$p_T<0.5$ GeV/c", r"$p_T>0.5$ GeV/c")
            xlabel, xlim, ylim = r"$y_{\rm cm}$", (-1.5, 1.5), (1.0, 2.0)
        else:
            panels = ("abs_y_lt_0p5", "abs_y_gt_0p5")
            titles = (r"$|y_{\rm cm}|<0.5$", r"$|y_{\rm cm}|>0.5$")
            xlabel, xlim, ylim = r"$p_T$, GeV/c", (0, 3), (1.0, 1.5)
        for ax, panel, title in zip(axes, panels, titles):
            _plot_table(ax, table, page_id, panel)
            _style_axis(ax, xlim, ylim)
            ax.set_title(title, fontsize=15, fontweight="bold")
            ax.set_xlabel(xlabel, fontweight="bold")
        fig.tight_layout(rect=(0.03, 0.14, 0.99, 0.91), w_pad=4.0)
        _footer(fig, centrality_valid)
        pages.append((page_id, fig))

    with PdfPages(pdf_output) as pdf:
        for page_id, fig in pages:
            fig.savefig(output_dir / f"{page_id}.png", dpi=180, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    parser.add_argument("--beam-energy", type=float, default=2.5)
    parser.add_argument("--y-shift", type=float, default=0.9863)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--max-files", type=int, default=None, help="Smoke-test limit per dataset")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    tables: list[pd.DataFrame] = []
    details: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        table, info = analyze_dataset(
            args.input_dir, dataset, args.y_shift, args.min_count, args.max_files
        )
        tables.append(table)
        details[dataset] = info
    combined = pd.concat(tables, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "np_ratio_binned.csv", index=False)
    validation = {
        "dataset_scope": "smoke" if args.max_files is not None else "whole_available_production",
        "file_counts": {name: info["files"] for name, info in details.items()},
        "event_counts": {name: info["events"] for name, info in details.items()},
        "row_counts": {name: info["rows"] for name, info in details.items()},
        "centrality_B_valid": all(bool(info["centrality_B_valid"]) for info in details.values()),
        "participant_count_valid": all(bool(info["participant_count_valid"]) for info in details.values()),
        "weights_present": all(bool(info["weights_present"]) for info in details.values()),
        "common_phase_space": True,
        "spot_reconstruction_from_primary_np": False,
        "decision": "DECLINE",
        "decision_reasons": [
            "impact parameter B is unavailable or invalid",
            "participant count NPrim is unavailable or invalid",
            "generator weights are absent and unit weighting is not documented",
            "HGND alone does not provide a common-acceptance proton measurement",
        ],
        "dataset_details": details,
    }
    build_figures(
        combined, args.output_dir, args.pdf_output, args.beam_energy,
        bool(validation["centrality_B_valid"]),
    )
    write_decision(args.output_dir, validation)
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2), flush=True)
    print(f"wrote plot deck to {args.pdf_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
