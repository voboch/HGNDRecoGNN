#!/usr/bin/env python3
"""Reproduce the plots from ``docs/n_p - smash check.pdf`` on local data.

The reference deck contains one 2.5A GeV Ekin plot and four 3.8A GeV
rapidity/pT plots.  The currently available primary-nucleon export is the
2.5A GeV production, so the latter four figures are faithful *plot-family*
reproductions at 2.5A GeV, never relabelled as 3.8A GeV.

Uncertainties are event-clustered delta-method standard errors.  Input ``Row``
is only unique within a CSV, so every event key includes the source file.

Example
-------
python scripts/np_ratio_smash_check.py \
  --input-dir /path/to/npsample \
  --output-dir results/np_ratio_smash_check \
  --pdf-output output/pdf/np_smash_check_reproduction.pdf
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


DATASETS = ("zeroSpot", "defaultSpot", "bigSpot")
SPOT_MEV = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
PLOT_STYLE = {
    "zeroSpot": dict(marker="^", color="black", markerfacecolor="black"),
    "defaultSpot": dict(marker="o", color="#0019c4", markerfacecolor="#0019c4"),
    "bigSpot": dict(marker="s", color="#9b1b1b", markerfacecolor="white"),
}
REQUIRED_COLUMNS = {"Row", "PDG", "Ekin", "Pt", "Pz", "Rapid"}


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    frame: pd.DataFrame
    files: tuple[Path, ...]
    columns: frozenset[str]

    @property
    def event_count(self) -> int:
        return int(self.frame["event_key"].nunique())


def discover_files(input_dir: Path, dataset: str) -> tuple[Path, ...]:
    files = tuple(sorted(input_dir.glob(f"{dataset}_*_prim.csv")))
    if not files:
        raise FileNotFoundError(
            f"No {dataset}_*_prim.csv files found under {input_dir}. "
            "Pass the directory containing the primary CSV exports."
        )
    return files


def load_dataset(input_dir: Path, dataset: str, y_shift: float) -> LoadedDataset:
    files = discover_files(input_dir, dataset)
    frames: list[pd.DataFrame] = []
    all_columns: set[str] = set()
    for file_index, path in enumerate(files):
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = REQUIRED_COLUMNS - header
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        all_columns.update(header)
        optional = [c for c in ("B", "NPrim", "Weight") if c in header]
        frame = pd.read_csv(path, usecols=sorted(REQUIRED_COLUMNS) + optional)
        # Row restarts in every file.  Never concatenate it without a file key.
        frame["event_key"] = (
            np.int64(file_index) * np.int64(10_000_000)
            + frame["Row"].to_numpy(dtype=np.int64)
        )
        frame["source_file"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["weight"] = (
        data["Weight"].astype(float) if "Weight" in data else np.ones(len(data))
    )
    data["y_cm"] = data["Rapid"].astype(float) - float(y_shift)
    return LoadedDataset(dataset, data, files, frozenset(all_columns))


def validate_inputs(datasets: dict[str, LoadedDataset]) -> dict[str, object]:
    event_counts = {name: ds.event_count for name, ds in datasets.items()}
    row_counts = {name: int(len(ds.frame)) for name, ds in datasets.items()}
    b_valid = all(
        "B" in ds.frame and np.isfinite(ds.frame["B"]).all() and (ds.frame["B"] >= 0).any()
        for ds in datasets.values()
    )
    npart_valid = all(
        "NPrim" in ds.frame
        and np.isfinite(ds.frame["NPrim"]).all()
        and (ds.frame["NPrim"] >= 0).any()
        for ds in datasets.values()
    )
    weights_present = all("Weight" in ds.columns for ds in datasets.values())
    return {
        "event_counts": event_counts,
        "row_counts": row_counts,
        "centrality_B_valid": b_valid,
        "participant_count_valid": npart_valid,
        "weights_present": weights_present,
        "common_phase_space": True,
        "spot_reconstruction_from_primary_np": False,
        "decision": "DECLINE",
        "decision_reasons": [
            "impact parameter B is unavailable or invalid" if not b_valid else "",
            "participant count NPrim is unavailable or invalid" if not npart_valid else "",
            "generator weights are absent and unit weighting is not documented"
            if not weights_present
            else "",
            "sample is a small sequential export rather than the complete/random production",
            "HGND alone does not provide a common-acceptance proton measurement",
        ],
    }


def event_clustered_ratio(
    frame: pd.DataFrame,
    selection: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
    min_count: int,
) -> pd.DataFrame:
    """Return n/p and an event-clustered standard error in each bin."""
    if len(selection) != len(frame) or len(values) != len(frame):
        raise ValueError("selection and values must align with frame")
    selected = np.asarray(selection, dtype=bool)
    bin_index = np.digitize(values, edges, right=False) - 1
    selected &= (bin_index >= 0) & (bin_index < len(edges) - 1)
    work = frame.loc[selected, ["event_key", "PDG", "weight"]].copy()
    work["bin"] = bin_index[selected]
    work["n"] = np.where(work["PDG"].to_numpy() == 2112, work["weight"], 0.0)
    work["p"] = np.where(work["PDG"].to_numpy() == 2212, work["weight"], 0.0)
    work["n_raw"] = (work["PDG"].to_numpy() == 2112).astype(int)
    work["p_raw"] = (work["PDG"].to_numpy() == 2212).astype(int)
    grouped = work.groupby(["bin", "event_key"], sort=False)[
        ["n", "p", "n_raw", "p_raw"]
    ].sum()
    total_events = int(frame["event_key"].nunique())

    rows: list[dict[str, float | int | bool]] = []
    for index in range(len(edges) - 1):
        if index in grouped.index.get_level_values("bin"):
            cell = grouped.xs(index, level="bin")
            n_weighted = float(cell["n"].sum())
            p_weighted = float(cell["p"].sum())
            n_raw = int(cell["n_raw"].sum())
            p_raw = int(cell["p_raw"].sum())
        else:
            cell = pd.DataFrame(columns=["n", "p", "n_raw", "p_raw"])
            n_weighted = p_weighted = 0.0
            n_raw = p_raw = 0

        valid = p_weighted > 0 and n_raw >= min_count and p_raw >= min_count
        ratio = n_weighted / p_weighted if valid else np.nan
        if valid:
            influence = cell["n"].to_numpy() - ratio * cell["p"].to_numpy()
            correction = total_events / max(total_events - 1, 1)
            stat_error = np.sqrt(correction * np.square(influence).sum()) / p_weighted
        else:
            stat_error = np.nan
        rows.append(
            {
                "bin_lo": float(edges[index]),
                "bin_hi": float(edges[index + 1]),
                "bin_center": float((edges[index] + edges[index + 1]) / 2),
                "n": n_weighted,
                "p": p_weighted,
                "n_raw": n_raw,
                "p_raw": p_raw,
                "ratio": ratio,
                "stat_error": stat_error,
                "valid": bool(valid),
                "events_total": total_events,
            }
        )
    return pd.DataFrame(rows)


def _style_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(which="both", direction="in", top=True, right=True, width=1.25)
    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=4)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    ax.grid(False)
    ax.set_ylabel("n/p", fontweight="bold")


def _plot_panel(
    ax: plt.Axes,
    datasets: dict[str, LoadedDataset],
    selection_fn: Callable[[pd.DataFrame], np.ndarray],
    value_column: str,
    edges: np.ndarray,
    min_count: int,
    figure_id: str,
    panel_id: str,
) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    for name in DATASETS:
        ds = datasets[name]
        table = event_clustered_ratio(
            ds.frame,
            selection_fn(ds.frame),
            ds.frame[value_column].to_numpy(dtype=float),
            edges,
            min_count,
        )
        table.insert(0, "spot_mev", SPOT_MEV[name])
        table.insert(0, "dataset", name)
        table.insert(0, "panel", panel_id)
        table.insert(0, "figure", figure_id)
        tables.append(table)
        valid = table[table["valid"]]
        style = PLOT_STYLE[name]
        ax.errorbar(
            valid["bin_center"],
            valid["ratio"],
            yerr=valid["stat_error"],
            linestyle="none",
            marker=style["marker"],
            markersize=4.5,
            color=style["color"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
            elinewidth=1.0,
            capsize=0,
            label=rf"$S_{{\rm pot}}={SPOT_MEV[name]}$ MeV",
        )
    ax.legend(frameon=False, fontsize=8, loc="upper left", handletextpad=0.4)
    return tables


def _footer(fig: plt.Figure, centrality_valid: bool) -> None:
    note = "Statistical uncertainties: event-clustered. All exported nucleons."
    if not centrality_valid:
        note += " Centrality unavailable - diagnostic only."
    fig.text(0.05, 0.035, note, fontsize=8.5, color="0.32")


def build_figures(
    datasets: dict[str, LoadedDataset],
    output_dir: Path,
    pdf_output: Path,
    beam_energy: float,
    min_count: int,
    validation: dict[str, object],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    records: list[pd.DataFrame] = []
    centrality_valid = bool(validation["centrality_B_valid"])

    ekin_edges = np.r_[np.arange(0.0, 1.61, 0.1), 1.8, 2.0, 2.3, 2.8, 3.0]
    y_edges = np.arange(-1.5, 1.5001, 0.1)
    pt_edges = np.r_[np.arange(0.0, 2.01, 0.1), 2.2, 2.4, 2.8, 3.0]

    pages: list[tuple[str, plt.Figure]] = []

    # Reference slide 3: Ekin dependence at 2.5A GeV.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.625))
    fig.suptitle(rf"Xe+Cs @ {beam_energy:g}A GeV: $E_{{\rm kin}}$-dependence", fontsize=21, y=0.96)
    panels = [
        ("abs_y_lt_0p5", r"$|y_{\rm cm}|<0.5$", lambda d: np.abs(d["y_cm"].to_numpy()) < 0.5, (1.0, 1.52)),
        ("abs_y_gt_0p5", r"$|y_{\rm cm}|>0.5$", lambda d: np.abs(d["y_cm"].to_numpy()) >= 0.5, (1.0, 1.80)),
    ]
    for ax, (panel_id, title, selection, ylim) in zip(axes, panels):
        records += _plot_panel(ax, datasets, selection, "Ekin", ekin_edges, min_count, "ekin", panel_id)
        _style_axis(ax, (0, 3), ylim)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel(r"$E_{\rm kin}$, GeV", fontweight="bold")
        ax.axvline(0.3, color="0.35", linewidth=2.0)
    fig.text(0.5, 0.105, r"Reference working cut: $E_{\rm kin}>300$ MeV", ha="center", fontsize=12, color="0.35")
    fig.tight_layout(rect=(0.03, 0.14, 0.99, 0.91), w_pad=4.0)
    _footer(fig, centrality_valid)
    pages.append(("01_np_vs_ekin_y", fig))

    def rapidity_page(with_ecut: bool, page_id: str) -> tuple[str, plt.Figure]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.625))
        suffix = r" with $E_{\rm kin}>300$ MeV" if with_ecut else ""
        fig.suptitle(rf"Xe+Cs @ {beam_energy:g}A GeV: $y_{{\rm cm}}$-dependence{suffix}", fontsize=19, y=0.96)
        specs = [
            ("pt_lt_0p5", r"$p_T<0.5$ GeV/c", lambda d: d["Pt"].to_numpy() < 0.5),
            ("pt_gt_0p5", r"$p_T>0.5$ GeV/c", lambda d: d["Pt"].to_numpy() >= 0.5),
        ]
        for ax, (panel_id, title, pt_selection) in zip(axes, specs):
            def selection(d: pd.DataFrame, fn=pt_selection) -> np.ndarray:
                mask = fn(d)
                if with_ecut:
                    mask &= d["Ekin"].to_numpy() > 0.3
                return mask
            records.extend(_plot_panel(ax, datasets, selection, "y_cm", y_edges, min_count, page_id, panel_id))
            _style_axis(ax, (-1.5, 1.5), (1.0, 2.0))
            ax.set_title(title, fontsize=15, fontweight="bold")
            ax.set_xlabel(r"$y_{\rm cm}$", fontweight="bold")
        fig.tight_layout(rect=(0.03, 0.14, 0.99, 0.91), w_pad=4.0)
        _footer(fig, centrality_valid)
        return page_id, fig

    def pt_page(with_ecut: bool, page_id: str) -> tuple[str, plt.Figure]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.625))
        suffix = r" with $E_{\rm kin}>300$ MeV" if with_ecut else ""
        fig.suptitle(rf"Xe+Cs @ {beam_energy:g}A GeV: $p_T$-dependence{suffix}", fontsize=19, y=0.96)
        specs = [
            ("abs_y_lt_0p5", r"$|y_{\rm cm}|<0.5$", lambda d: np.abs(d["y_cm"].to_numpy()) < 0.5),
            ("abs_y_gt_0p5", r"$|y_{\rm cm}|>0.5$", lambda d: np.abs(d["y_cm"].to_numpy()) >= 0.5),
        ]
        for ax, (panel_id, title, y_selection) in zip(axes, specs):
            def selection(d: pd.DataFrame, fn=y_selection) -> np.ndarray:
                mask = fn(d)
                if with_ecut:
                    mask &= d["Ekin"].to_numpy() > 0.3
                return mask
            records.extend(_plot_panel(ax, datasets, selection, "Pt", pt_edges, min_count, page_id, panel_id))
            _style_axis(ax, (0, 3), (1.0, 1.5))
            ax.set_title(title, fontsize=15, fontweight="bold")
            ax.set_xlabel(r"$p_T$, GeV/c", fontweight="bold")
        fig.tight_layout(rect=(0.03, 0.14, 0.99, 0.91), w_pad=4.0)
        _footer(fig, centrality_valid)
        return page_id, fig

    pages.append(rapidity_page(True, "02_np_vs_y_ekin_gt_0p3"))
    pages.append(pt_page(True, "03_np_vs_pt_ekin_gt_0p3"))
    pages.append(rapidity_page(False, "04_np_vs_y_all_ekin"))
    pages.append(pt_page(False, "05_np_vs_pt_all_ekin"))

    with PdfPages(pdf_output) as pdf:
        for page_id, fig in pages:
            fig.savefig(output_dir / f"{page_id}.png", dpi=180, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)

    combined = pd.concat(records, ignore_index=True)
    combined.to_csv(output_dir / "np_ratio_binned.csv", index=False)
    return combined


def write_decision(output_dir: Path, validation: dict[str, object]) -> None:
    reasons = [reason for reason in validation["decision_reasons"] if reason]
    validation["decision_reasons"] = reasons
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Existing-data decision for Spot reconstruction",
        "",
        "## Decision",
        "",
        "**DECLINE the current primary-nucleon sample for n/p-based Spot reconstruction.**",
        "",
        "The plots are valid diagnostics of the exported sample, but the sample fails the",
        "physics-use gates below:",
        "",
    ]
    lines.extend(f"- {reason}." for reason in reasons)
    lines += [
        "",
        "## What remains usable",
        "",
        "The existing full HGND reconstruction outputs may be used conditionally for a",
        "simulation-only, three-point neutron-spectrum sensitivity/ordinal-classification",
        "study. They are not sufficient for a calibrated continuous Spot estimator or an",
        "experimental n/p measurement. Such a study must use balanced event-level splits,",
        "match centrality, hold out independent productions, and use HGND observables only.",
        "",
    ]
    (output_dir / "USAGE_DECISION.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing *_prim.csv files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/np_ratio_smash_check"))
    parser.add_argument("--pdf-output", type=Path, default=Path("output/pdf/np_smash_check_reproduction.pdf"))
    parser.add_argument("--beam-energy", type=float, default=2.5, help="Beam kinetic energy in A GeV")
    parser.add_argument("--y-shift", type=float, default=0.9863, help="Convert y_lab to y_cm via y_cm=y_lab-shift")
    parser.add_argument("--min-count", type=int, default=20, help="Minimum raw n and p counts required per point")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = {name: load_dataset(args.input_dir, name, args.y_shift) for name in DATASETS}
    validation = validate_inputs(datasets)
    table = build_figures(
        datasets,
        args.output_dir,
        args.pdf_output,
        args.beam_energy,
        args.min_count,
        validation,
    )
    write_decision(args.output_dir, validation)
    print(json.dumps(validation, indent=2))
    print(f"wrote {len(table):,} binned rows to {args.output_dir}")
    print(f"wrote plot deck to {args.pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
