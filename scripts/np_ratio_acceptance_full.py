#!/usr/bin/env python3
"""Full-production HGND acceptance diagnostics for the n/p report.

The extracted primary-nucleon production contains hundreds of millions of rows.
This program reads one CSV at a time and retains event-level sufficient
statistics, giving exact event-clustered uncertainties with bounded memory.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from np_ratio_smash_check import DATASETS, SPOT_MEV
from np_ratio_smash_check_stream import discover_files


THETA_EDGES = np.array([0, 1, 2, 3, 4, 6, 8.9, 13.1, 18, 25, 40, 90, 180], dtype=float)
Y_BEAM = 1.9727
Y_SHIFT = 0.9863
SLIDE_CASES = (
    ("mid_ekin_1p3_1p7", True, 1.3, 1.7, 1.20),
    ("mid_ekin_0p9_1p3", True, 0.9, 1.3, 1.13),
    ("mid_ekin_0p3_0p7", True, 0.3, 0.7, 1.04),
    ("outer_ekin_0p3_0p7", False, 0.3, 0.7, 1.10),
    ("outer_ekin_0p9_1p3", False, 0.9, 1.3, 1.25),
)


@dataclass
class EventStats:
    n: np.ndarray
    p: np.ndarray
    n2: np.ndarray
    p2: np.ndarray
    np_: np.ndarray

    @classmethod
    def zeros(cls, nbins: int) -> "EventStats":
        z = lambda: np.zeros(nbins, dtype=np.float64)
        return cls(z(), z(), z(), z(), z())

    def add(
        self,
        event_code: np.ndarray,
        pdg: np.ndarray,
        selected: np.ndarray,
        bins: np.ndarray,
        nevents: int,
    ) -> None:
        nbins = len(self.n)
        selected = selected & (bins >= 0) & (bins < nbins)
        keys = event_code[selected] * nbins + bins[selected]
        species = pdg[selected]
        size = nevents * nbins
        n_event = np.bincount(keys[species == 2112], minlength=size).reshape(nevents, nbins)
        p_event = np.bincount(keys[species == 2212], minlength=size).reshape(nevents, nbins)
        self.n += n_event.sum(axis=0)
        self.p += p_event.sum(axis=0)
        self.n2 += np.square(n_event).sum(axis=0)
        self.p2 += np.square(p_event).sum(axis=0)
        self.np_ += (n_event * p_event).sum(axis=0)

    def finish(self, nevents: int) -> list[dict[str, float | int | None]]:
        correction = nevents / max(nevents - 1, 1)
        rows: list[dict[str, float | int | None]] = []
        for i in range(len(self.n)):
            n, p = float(self.n[i]), float(self.p[i])
            ratio = n / p if p else None
            if ratio is None:
                ratio_error = None
            else:
                influence2 = self.n2[i] - 2 * ratio * self.np_[i] + ratio * ratio * self.p2[i]
                ratio_error = float(np.sqrt(correction * max(influence2, 0.0)) / p)
            n_mean = n / nevents
            p_mean = p / nevents
            n_var_sum = correction * max(float(self.n2[i]) - n * n / nevents, 0.0)
            p_var_sum = correction * max(float(self.p2[i]) - p * p / nevents, 0.0)
            rows.append({
                "n": int(n),
                "p": int(p),
                "ratio": ratio,
                "ratio_error": ratio_error,
                "n_per_event": n_mean,
                "p_per_event": p_mean,
                "n_per_event_error": float(np.sqrt(n_var_sum) / nevents),
                "p_per_event_error": float(np.sqrt(p_var_sum) / nevents),
            })
        return rows


def _one_bin(size: int) -> np.ndarray:
    return np.zeros(size, dtype=np.int64)


def analyze_dataset(input_dir: Path, dataset: str, max_files: int | None) -> dict[str, object]:
    files = discover_files(input_dir, dataset)
    if max_files is not None:
        files = files[:max_files]
    theta_stats = EventStats.zeros(len(THETA_EDGES) - 1)
    total_stats = EventStats.zeros(1)
    spectator_stats = EventStats.zeros(1)
    remainder_stats = EventStats.zeros(1)
    slide_stats = {name: EventStats.zeros(1) for name, *_ in SLIDE_CASES}
    spectator_theta = np.zeros(len(THETA_EDGES) - 1, dtype=np.int64)
    nucleon_theta = np.zeros(len(THETA_EDGES) - 1, dtype=np.int64)
    total_events = 0
    total_rows = 0
    b_valid = True
    npart_valid = True
    weights_present = True

    for file_number, path in enumerate(files, start=1):
        header = set(pd.read_csv(path, nrows=0).columns)
        required = {"Row", "PDG", "Ekin", "Rapid", "Pt", "Pz"}
        missing = required - header
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        optional = [column for column in ("B", "NPrim", "Weight") if column in header]
        frame = pd.read_csv(path, usecols=sorted(required) + optional)
        row = frame["Row"].to_numpy(dtype=np.int64)
        _, event_code = np.unique(row, return_inverse=True)
        nevents = int(event_code.max()) + 1 if len(event_code) else 0
        total_events += nevents
        total_rows += len(frame)
        b_valid &= "B" in frame and np.isfinite(frame["B"]).all() and (frame["B"] >= 0).any()
        npart_valid &= "NPrim" in frame and np.isfinite(frame["NPrim"]).all() and (frame["NPrim"] >= 0).any()
        weights_present &= "Weight" in frame

        pdg = frame["PDG"].to_numpy(dtype=np.int64)
        nucleon = (pdg == 2112) | (pdg == 2212)
        pt = frame["Pt"].to_numpy(dtype=float)
        pz = frame["Pz"].to_numpy(dtype=float)
        rapid = frame["Rapid"].to_numpy(dtype=float)
        ekin = frame["Ekin"].to_numpy(dtype=float)
        theta = np.degrees(np.arctan2(pt, pz))
        theta_bins = np.searchsorted(THETA_EDGES, theta, side="right") - 1
        ones = _one_bin(len(frame))

        theta_stats.add(event_code, pdg, nucleon, theta_bins, nevents)
        total_stats.add(event_code, pdg, nucleon, ones, nevents)
        spectator = nucleon & (pt < 0.25) & (
            (np.abs(rapid - Y_BEAM) < 0.25) | (np.abs(rapid) < 0.25)
        )
        spectator_stats.add(event_code, pdg, spectator, ones, nevents)
        remainder_stats.add(event_code, pdg, nucleon & ~spectator, ones, nevents)
        valid_theta = nucleon & (theta_bins >= 0) & (theta_bins < len(spectator_theta))
        spectator_theta += np.bincount(
            theta_bins[spectator & valid_theta], minlength=len(spectator_theta)
        )
        nucleon_theta += np.bincount(
            theta_bins[valid_theta], minlength=len(nucleon_theta)
        )

        y_cm = rapid - Y_SHIFT
        for name, mid, e0, e1, _ in SLIDE_CASES:
            selected = nucleon & (ekin >= e0) & (ekin < e1)
            selected &= np.abs(y_cm) < 0.5 if mid else np.abs(y_cm) >= 0.5
            slide_stats[name].add(event_code, pdg, selected, ones, nevents)

        print(
            f"{dataset}: {file_number}/{len(files)} files, "
            f"{total_events:,} events, {total_rows:,} rows",
            flush=True,
        )

    theta_rows = theta_stats.finish(total_events)
    for i, row_out in enumerate(theta_rows):
        row_out["theta_lo"] = float(THETA_EDGES[i])
        row_out["theta_hi"] = float(THETA_EDGES[i + 1])
        row_out["spectator_fraction"] = (
            float(spectator_theta[i] / nucleon_theta[i]) if nucleon_theta[i] else None
        )
    return {
        "files": len(files),
        "events": total_events,
        "rows": total_rows,
        "nucleons_per_event": (total_stats.n[0] + total_stats.p[0]) / total_events,
        "centrality_B_valid": bool(b_valid),
        "participant_count_valid": bool(npart_valid),
        "weights_present": bool(weights_present),
        "theta": theta_rows,
        "total": total_stats.finish(total_events)[0],
        "spectator_proxy": spectator_stats.finish(total_events)[0],
        "remainder": remainder_stats.finish(total_events)[0],
        "slide_cases": {name: stats.finish(total_events)[0] for name, stats in slide_stats.items()},
    }


def _pair_summary(a: dict[str, object], b: dict[str, object]) -> dict[str, float | None]:
    r0, r9 = a["ratio"], b["ratio"]
    s0, s9 = a["ratio_error"], b["ratio_error"]
    if None in (r0, r9, s0, s9):
        return {"difference": None, "relative_change": None, "significance": None, "double_ratio": None, "double_ratio_error": None}
    difference = float(r9) - float(r0)
    error = float(np.hypot(float(s0), float(s9)))
    double_ratio = float(r9) / float(r0)
    double_error = double_ratio * float(np.hypot(float(s0) / float(r0), float(s9) / float(r9)))
    return {
        "difference": difference,
        "relative_change": difference / float(r0),
        "significance": abs(difference) / error if error else None,
        "double_ratio": double_ratio,
        "double_ratio_error": double_error,
    }


def _yield_pair(a: dict[str, object], b: dict[str, object], species: str) -> dict[str, float]:
    key = f"{species}_per_event"
    error_key = f"{species}_per_event_error"
    y0, y9 = float(a[key]), float(b[key])
    s0, s9 = float(a[error_key]), float(b[error_key])
    difference = y9 - y0
    error = float(np.hypot(s0, s9))
    return {
        "zero": y0,
        "big": y9,
        "relative_change": difference / y0,
        "significance": abs(difference) / error if error else float("nan"),
    }


def build_summary(details: dict[str, dict[str, object]], scope: str) -> dict[str, object]:
    zero, big = details["zeroSpot"], details["bigSpot"]
    theta_comparisons = []
    for index, zero_row in enumerate(zero["theta"]):
        rows = {name: detail["theta"][index] for name, detail in details.items()}
        theta_comparisons.append({
            "theta_lo": zero_row["theta_lo"],
            "theta_hi": zero_row["theta_hi"],
            "datasets": rows,
            "zero_to_big": _pair_summary(rows["zeroSpot"], rows["bigSpot"]),
            "ordered": (
                rows["zeroSpot"]["ratio"] < rows["defaultSpot"]["ratio"] < rows["bigSpot"]["ratio"]
                or rows["zeroSpot"]["ratio"] > rows["defaultSpot"]["ratio"] > rows["bigSpot"]["ratio"]
            ),
        })
    hgnd_index = next(i for i, row in enumerate(theta_comparisons) if row["theta_lo"] == 8.9)
    hgnd_rows = {name: detail["theta"][hgnd_index] for name, detail in details.items()}
    slide = []
    for name, mid, e0, e1, reference in SLIDE_CASES:
        rows = {dataset: details[dataset]["slide_cases"][name] for dataset in DATASETS}
        comparison = _pair_summary(rows["zeroSpot"], rows["bigSpot"])
        pull = None
        if comparison["double_ratio_error"]:
            pull = abs(float(comparison["double_ratio"]) - reference) / float(comparison["double_ratio_error"])
        slide.append({
            "name": name,
            "midrapidity": mid,
            "ekin_lo": e0,
            "ekin_hi": e1,
            "slide_double_ratio": reference,
            "datasets": rows,
            "zero_to_big": comparison,
            "local_pull_to_slide": pull,
        })
    return {
        "dataset_scope": scope,
        "file_counts": {name: detail["files"] for name, detail in details.items()},
        "event_counts": {name: detail["events"] for name, detail in details.items()},
        "row_counts": {name: detail["rows"] for name, detail in details.items()},
        "centrality_B_valid": all(detail["centrality_B_valid"] for detail in details.values()),
        "participant_count_valid": all(detail["participant_count_valid"] for detail in details.values()),
        "weights_present": all(detail["weights_present"] for detail in details.values()),
        "datasets": details,
        "theta_comparisons": theta_comparisons,
        "hgnd_band": {
            "datasets": hgnd_rows,
            "zero_to_big_ratio": _pair_summary(hgnd_rows["zeroSpot"], hgnd_rows["bigSpot"]),
            "zero_to_big_neutron_yield": _yield_pair(hgnd_rows["zeroSpot"], hgnd_rows["bigSpot"], "n"),
            "zero_to_big_proton_yield": _yield_pair(hgnd_rows["zeroSpot"], hgnd_rows["bigSpot"], "p"),
        },
        "total_zero_to_big": _pair_summary(zero["total"], big["total"]),
        "spectator_zero_to_big": _pair_summary(zero["spectator_proxy"], big["spectator_proxy"]),
        "remainder_zero_to_big": _pair_summary(zero["remainder"], big["remainder"]),
        "slide_crosscheck": slide,
    }


def _theta_arrays(summary: dict[str, object], dataset: str) -> tuple[np.ndarray, ...]:
    rows = summary["datasets"][dataset]["theta"]
    lo = np.array([row["theta_lo"] for row in rows], dtype=float)
    hi = np.array([row["theta_hi"] for row in rows], dtype=float)
    center = (lo + hi) / 2
    ratio = np.array([row["ratio"] for row in rows], dtype=float)
    error = np.array([row["ratio_error"] for row in rows], dtype=float)
    return lo, hi, center, ratio, error


def _style_axis(ax: plt.Axes, ylabel: str, xlim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"polar angle $\theta_{\rm lab}$ [deg]")
    ax.set_ylabel(ylabel)
    ax.axvspan(8.9, 13.1, color="#f2c14e", alpha=0.28, label="HGND 8.9°–13.1°")
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.tick_params(direction="in", top=True, right=True)


def build_plots(summary: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"zeroSpot": "black", "defaultSpot": "#003fcd", "bigSpot": "#bd170d"}
    markers = {"zeroSpot": "^", "defaultSpot": "o", "bigSpot": "s"}

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for dataset in DATASETS:
        lo, hi, center, ratio, error = _theta_arrays(summary, dataset)
        keep = hi <= 40
        ax.errorbar(
            center[keep], ratio[keep], yerr=error[keep],
            xerr=np.vstack((center[keep] - lo[keep], hi[keep] - center[keep])),
            linestyle="none", marker=markers[dataset], markersize=5,
            color=colors[dataset], capsize=2,
            label=rf"$S_{{\rm pot}}={SPOT_MEV[dataset]}$ MeV",
        )
    _style_axis(ax, "n/p", (0, 40))
    ax.set_title("Primary-nucleon n/p versus polar angle — full production")
    handles, labels = ax.get_legend_handles_labels()
    order = [len(handles) - 1] + list(range(len(handles) - 1))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "01_np_vs_theta_hgnd.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    zero_rows = summary["datasets"]["zeroSpot"]["theta"]
    lo = np.array([row["theta_lo"] for row in zero_rows], dtype=float)
    hi = np.array([row["theta_hi"] for row in zero_rows], dtype=float)
    center = (lo + hi) / 2
    keep = hi <= 40
    for dataset, linestyle in (("defaultSpot", "--"), ("bigSpot", "-")):
        rows = summary["datasets"][dataset]["theta"]
        for species, marker in (("n", "o"), ("p", "s")):
            y0 = np.array([row[f"{species}_per_event"] for row in zero_rows], dtype=float)
            sy0 = np.array([row[f"{species}_per_event_error"] for row in zero_rows], dtype=float)
            y = np.array([row[f"{species}_per_event"] for row in rows], dtype=float)
            sy = np.array([row[f"{species}_per_event_error"] for row in rows], dtype=float)
            change = 100 * (y / y0 - 1)
            error = 100 * (y / y0) * np.sqrt(np.square(sy / y) + np.square(sy0 / y0))
            ax.errorbar(
                center[keep], change[keep], yerr=error[keep], marker=marker,
                markersize=4, linewidth=1.1, linestyle=linestyle,
                color=colors[dataset], capsize=2,
                label=rf"{species}, $S_{{\rm pot}}={SPOT_MEV[dataset]}$ MeV",
            )
    _style_axis(ax, "yield change relative to 0 MeV [%]", (0, 40))
    ax.axhline(0, color="0.3", linewidth=0.9)
    ax.set_title("Species yields versus polar angle — full production")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "02_yield_change_vs_theta_hgnd.png", dpi=180, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for dataset in DATASETS:
        rows = summary["datasets"][dataset]["theta"]
        lo = np.array([row["theta_lo"] for row in rows], dtype=float)
        hi = np.array([row["theta_hi"] for row in rows], dtype=float)
        center = (lo + hi) / 2
        fraction = 100 * np.array([row["spectator_fraction"] for row in rows], dtype=float)
        keep = hi <= 40
        ax.plot(
            center[keep], fraction[keep], marker=markers[dataset], markersize=5,
            color=colors[dataset], linewidth=1.2,
            label=rf"$S_{{\rm pot}}={SPOT_MEV[dataset]}$ MeV",
        )
    _style_axis(ax, "spectator-proxy fraction [%]", (0, 40))
    ax.set_ylim(0, 102)
    ax.set_title("Spectator-proxy composition versus polar angle")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "03_spectator_fraction_vs_theta_hgnd.png", dpi=180, facecolor="white")
    plt.close(fig)


def write_theta_csv(summary: dict[str, object], path: Path) -> None:
    fields = [
        "dataset", "spot_mev", "theta_lo", "theta_hi", "n", "p", "ratio",
        "ratio_error", "n_per_event", "n_per_event_error", "p_per_event",
        "p_per_event_error", "spectator_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset in DATASETS:
            for row in summary["datasets"][dataset]["theta"]:
                writer.writerow({"dataset": dataset, "spot_mev": SPOT_MEV[dataset], **{
                    field: row[field] for field in fields if field not in {"dataset", "spot_mev"}
                }})


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args(argv)
    details = {name: analyze_dataset(args.input_dir, name, args.max_files) for name in DATASETS}
    summary = build_summary(
        details,
        "smoke" if args.max_files is not None else "whole_available_production",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_theta_csv(summary, args.output.parent / "acceptance_theta_binned.csv")
    build_plots(summary, args.output.parent)
    print(json.dumps({
        "dataset_scope": summary["dataset_scope"],
        "file_counts": summary["file_counts"],
        "event_counts": summary["event_counts"],
        "hgnd_band": summary["hgnd_band"],
    }, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
