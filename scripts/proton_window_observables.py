#!/usr/bin/env python3
"""Exact-acceptance MC-proton counterparts of reconstructed 1--2 GeV shapes.

The input unit for statistical resampling is one SMASH production CSV.  Every
bootstrap replicate resamples whole files and rebuilds the common impact-
parameter weights, matching the full-production truth analysis convention.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hgnd_acceptance import in_acceptance


SAMPLES = ("zeroSpot", "defaultSpot", "bigSpot")
U_MEV = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
USECOLS = ("Row", "PDG", "Ekin", "B", "Px", "Py", "Pz", "vX", "vY", "vZ")
COUNTERS = (
    "eq_lo", "eq_hi", "tail_lo", "tail_hi", "band_count", "band_esum",
    "conventional_lo", "conventional_hi",
)
OBSERVABLES = {
    "R_p_1p0_1p5_vs_1p5_2p0": ("eq_hi", "eq_lo"),
    "R_p_1p0_1p8_vs_1p8_2p0": ("tail_hi", "tail_lo"),
    "mean_E_p_1p0_2p0": ("band_esum", "band_count"),
    "R_p_conventional": ("conventional_hi", "conventional_lo"),
}
B_WIDTH = 0.10
MIN_COUNT = 20


def reduce_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_csv(path, usecols=USECOLS, engine="c")
    row = d["Row"].to_numpy()
    _, start = np.unique(row, return_index=True)
    start.sort()
    b_event = d["B"].to_numpy()[start].astype(float)
    e = d["Ekin"].to_numpy(float)
    proton = d["PDG"].to_numpy() == 2212
    acc = in_acceptance(
        d["Px"].to_numpy(float), d["Py"].to_numpy(float), d["Pz"].to_numpy(float),
        d["vX"].to_numpy(float), d["vY"].to_numpy(float), d["vZ"].to_numpy(float),
    )
    p = proton & acc
    masks = {
        "eq_lo": p & (e >= 1.0) & (e < 1.5),
        "eq_hi": p & (e >= 1.5) & (e < 2.0),
        "tail_lo": p & (e >= 1.0) & (e < 1.8),
        "tail_hi": p & (e >= 1.8) & (e < 2.0),
        "band_count": p & (e >= 1.0) & (e < 2.0),
        "conventional_lo": p & (e < 1.0),
        "conventional_hi": p & (e >= 2.0),
    }
    out = np.empty((len(start), len(COUNTERS)), dtype=float)
    for j, name in enumerate(COUNTERS):
        if name == "band_esum":
            values = e * masks["band_count"]
        else:
            values = masks[name].astype(float)
        out[:, j] = np.add.reduceat(values, start)
    return b_event, out


def reduce_sample(root: Path, max_files: int | None) -> dict:
    paths = []
    for attempt in range(3):
        try:
            paths = sorted(root.rglob("*_prim.csv"))
            break
        except PermissionError:
            if attempt == 2:
                raise
            print(f"  transient permission error while scanning {root}; retrying", flush=True)
            time.sleep(2.0)
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"no *_prim.csv below {root}")
    bmax = 15.0
    edges = np.arange(0.0, bmax + B_WIDTH, B_WIDTH)
    nb = len(edges) - 1
    n_files = np.zeros((len(paths), nb), dtype=float)
    bsum_files = np.zeros_like(n_files)
    counter_files = np.zeros((len(paths), nb, len(COUNTERS)), dtype=float)
    n_events = 0
    for i, path in enumerate(paths):
        b, counters = reduce_file(path)
        pos = np.clip(np.digitize(b, edges) - 1, 0, nb - 1)
        n_files[i] = np.bincount(pos, minlength=nb)
        bsum_files[i] = np.bincount(pos, weights=b, minlength=nb)
        for j in range(len(COUNTERS)):
            counter_files[i, :, j] = np.bincount(
                pos, weights=counters[:, j], minlength=nb
            )
        n_events += len(b)
        if (i + 1) % 10 == 0 or i + 1 == len(paths):
            print(f"  {root.name}: {i + 1}/{len(paths)} files, {n_events} events", flush=True)
    return {
        "paths": [str(p) for p in paths],
        "n_events": n_events,
        "n": n_files,
        "bsum": bsum_files,
        "counters": counter_files,
        "edges": edges,
    }


def aggregate(sample: dict, multiplicity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        multiplicity @ sample["n"],
        np.tensordot(multiplicity, sample["counters"], axes=(0, 0)),
        multiplicity @ sample["bsum"],
    )


def evaluate(samples: dict, multiplicities: dict, reweight: bool) -> tuple[dict, dict]:
    aggs = {tag: aggregate(samples[tag], multiplicities[tag]) for tag in SAMPLES}
    densities = [aggs[tag][0] / aggs[tag][0].sum() for tag in SAMPLES]
    reference = np.mean(densities, axis=0)
    values, meta = {}, {}
    for tag in SAMPLES:
        n, c, bsum = aggs[tag]
        if reweight:
            density = n / n.sum()
            ok = (n >= MIN_COUNT) & (reference > 0) & (density > 0)
            weights = np.zeros_like(reference)
            weights[ok] = reference[ok] / density[ok]
        else:
            weights = (n > 0).astype(float)
        totals = weights @ c
        values[tag] = {}
        for name, (num, den) in OBSERVABLES.items():
            numerator = totals[COUNTERS.index(num)]
            denominator = totals[COUNTERS.index(den)]
            values[tag][name] = float(numerator / denominator)
        wn = weights @ n
        meta[tag] = {
            "mean_b": float((weights * bsum).sum() / wn),
            "effective_events": float(wn * wn / ((weights * weights) @ n)),
        }
    return values, meta


def analyse(samples: dict, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    unit = {tag: np.ones(len(samples[tag]["paths"])) for tag in SAMPLES}
    output = {
        "definition": {
            "particle": "MC-truth primary protons intersecting the exact HGND front face",
            "uncertainty": "production-file bootstrap; common b weights rebuilt in every replicate",
            "observables": {k: f"{a}/{b}" for k, (a, b) in OBSERVABLES.items()},
            "b_bin_width_fm": B_WIDTH,
            "minimum_events_per_b_bin": MIN_COUNT,
        },
        "scope": {
            tag: {
                "U_MeV": U_MEV[tag], "n_files": len(samples[tag]["paths"]),
                "n_events": samples[tag]["n_events"],
            }
            for tag in SAMPLES
        },
        "n_boot": n_boot,
    }
    for mode, do_reweight in (("raw", False), ("b_reweighted", True)):
        central, meta = evaluate(samples, unit, do_reweight)
        boot_values = {obs: {tag: [] for tag in SAMPLES} for obs in OBSERVABLES}
        boot_response = {obs: [] for obs in OBSERVABLES}
        for _ in range(n_boot):
            mult = {}
            for tag in SAMPLES:
                nf = len(samples[tag]["paths"])
                pick = rng.integers(0, nf, nf)
                mult[tag] = np.bincount(pick, minlength=nf).astype(float)
            vals, _ = evaluate(samples, mult, do_reweight)
            for obs in OBSERVABLES:
                for tag in SAMPLES:
                    boot_values[obs][tag].append(vals[tag][obs])
                boot_response[obs].append(vals["bigSpot"][obs] / vals["zeroSpot"][obs] - 1.0)
        report = {"samples": meta, "observables": {}}
        for obs in OBSERVABLES:
            response = central["bigSpot"][obs] / central["zeroSpot"][obs] - 1.0
            response_error = float(np.std(boot_response[obs], ddof=1)) if n_boot > 1 else 0.0
            report["observables"][obs] = {
                "central": {tag: central[tag][obs] for tag in SAMPLES},
                "error": {
                    tag: float(np.std(boot_values[obs][tag], ddof=1)) if n_boot > 1 else 0.0
                    for tag in SAMPLES
                },
                "relative_change_90_over_0": response,
                "relative_error": response_error,
                "separation_sigma": abs(response) / response_error if response_error > 0 else None,
                "ordered_0_18_90": (
                    central["zeroSpot"][obs] < central["defaultSpot"][obs] < central["bigSpot"][obs]
                    or central["zeroSpot"][obs] > central["defaultSpot"][obs] > central["bigSpot"][obs]
                ),
            }
        output[mode] = report
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--max-files", type=int)
    args = ap.parse_args()
    samples = {}
    for tag in SAMPLES:
        root = args.input_dir / f"smash_xecs_2.87gev_hardSkyrme_{tag}"
        samples[tag] = reduce_sample(root, args.max_files)
    report = analyse(samples, args.n_boot, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
