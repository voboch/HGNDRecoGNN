"""Preparation helpers for the label-energy audit; production recipes stay frozen.

The associated MCTrack initial-energy floor (GeV), observed ToF half-width (ns), and
the analysis-energy bands are separate parameters. None removes the label
energy clause; zero retains the strict Ekin > 0 clause.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from HGNDRecoGNN.analysis.energy_bands import EnergyBands


def cluster_targets(energy, prompt, cluster, floor, n_clusters=None):
    """Aggregate labels and maximum qualifying-hit energy on an unchanged graph.

    `prompt` contains the original PDG, Side and ToF conditions, with no
    energy floor. The strict inequality reproduces the historical 0.3 rule.
    Increasing the floor trades soft-positive coverage for a narrower task;
    it does not change observations, clusters, scores or reconstructed energy.
    """
    energy = np.asarray(energy, dtype=float)
    prompt = np.asarray(prompt)
    cluster = np.asarray(cluster)
    if energy.ndim != 1 or energy.shape != prompt.shape or energy.shape != cluster.shape:
        raise ValueError("One energy, prompt flag and cluster index required per hit")
    if prompt.dtype != np.bool_ or cluster.dtype.kind not in "iu":
        raise ValueError("Prompt flags must be boolean and cluster indices integer")
    if not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("Hit kinetic energies must be finite and nonnegative")
    if floor is not None and (not np.isfinite(floor) or floor < 0):
        raise ValueError("Label floor must be None or finite and nonnegative")
    size = int(cluster.max()) + 1 if len(cluster) else 0
    nc = size if n_clusters is None else int(n_clusters)
    if nc < size or np.any(cluster < 0) or nc < 0:
        raise ValueError("Cluster index outside the declared inventory")
    selected = prompt if floor is None else prompt & (energy > floor)
    label = np.zeros(nc, dtype=np.int8)
    target = np.zeros(nc, dtype=np.float64)
    np.maximum.at(label, cluster, selected)
    np.maximum.at(target, cluster, np.where(selected, energy, 0.0))
    return label, target


def threshold_ledger(energy, prompt, cluster, floors):
    """Per-threshold counts and exact monotonicity guards, without model scores."""
    if len(set(floors)) != len(floors) or not floors:
        raise ValueError("A nonempty unique threshold grid is required")
    results = {floor: cluster_targets(energy, prompt, cluster, floor) for floor in floors}
    ordered = sorted(floors, key=lambda f: -np.inf if f is None else f)
    for lower, upper in zip(ordered, ordered[1:]):
        low_y, low_e = results[lower]
        up_y, up_e = results[upper]
        common = up_y == 1
        if np.any(up_y > low_y) or not np.array_equal(low_e[common], up_e[common]):
            raise ValueError("Relabelled targets violate threshold monotonicity")
    out = []
    for floor in floors:
        y, e = results[floor]
        selected = np.asarray(prompt) if floor is None else np.asarray(prompt) & (np.asarray(energy) > floor)
        out.append({"label_floor_GeV": floor, "signal_hits": int(selected.sum()),
                    "signal_clusters": int(y.sum()),
                    "positive_zero_energy_targets": int(((y == 1) & (e == 0)).sum())})
    return out, results


def origin_partition(cluster, origin, selected, n_clusters, cluster_subset=None):
    """Disjoint origin classes of qualifying hits; this is not ancestry recovery.

    Origin bits: 1=exact-bound direct primary, 2=nonprimary MCTrack flag,
    4=unresolved. Mother index zero belongs to the nonprimary class.
    """
    cluster, origin, selected = np.asarray(cluster), np.asarray(origin), np.asarray(selected)
    if cluster.shape != origin.shape or cluster.shape != selected.shape or selected.dtype != np.bool_:
        raise ValueError("Aligned cluster/origin/selection arrays required")
    if not np.isin(origin, [1, 2, 4]).all():
        raise ValueError("Unknown origin code")
    flags = np.zeros(n_clusters, dtype=np.int8)
    np.bitwise_or.at(flags, cluster[selected], origin[selected])
    subset = flags != 0 if cluster_subset is None else np.asarray(cluster_subset)
    if subset.dtype != np.bool_ or subset.shape != flags.shape or np.any(subset & (flags == 0)):
        raise ValueError("Origin subset must contain only positive clusters")
    result = {"direct_primary_only": int((subset & (flags == 1)).sum()),
              "nonprimary_track_only": int((subset & (flags == 2)).sum()),
              "unresolved_only": int((subset & (flags == 4)).sum()),
              "mixed_origins": int((subset & np.isin(flags, [3, 5, 6, 7])).sum())}
    if sum(result.values()) != int(subset.sum()):
        raise ValueError("Origin partition does not conserve positive clusters")
    return result


def window_grid(minima, low_stops, high_starts, maximum):
    """Finite, disjoint half-open bands; reject ordering rather than clip it.

    The proposed 3 GeV cap bounds the first study to the previously checked
    response range. The old open high tail remains a separate replay anchor.
    Grid spacings are exploratory conventions, not detector resolutions.
    """
    windows = []
    for minimum, low_stop, high_start in product(minima, low_stops, high_starts):
        if minimum < low_stop < high_start < maximum:
            windows.append(EnergyBands(minimum, low_stop, high_start, maximum))
    if not windows:
        raise ValueError("Grid has no ordered energy windows")
    if len(set(windows)) != len(windows):
        raise ValueError("Duplicate energy windows")
    return windows


def exact_primary_counts(energy, row, in_window, pdg, event_rows, processed_span, bands):
    """Exact primary-neutron counts and event moments, retaining empty events.

    Energies are generated primary energies, distinct from a secondary
    MCTrack's initial-energy target. `in_window` uses the existing rectangular-ray
    selection, not by a polar band or a claim of active detector acceptance.
    No histogram interpolation or pre-censoring at 0.3 GeV is allowed.
    """
    energy, row, pdg = np.asarray(energy), np.asarray(row), np.asarray(pdg)
    in_window, event_rows = np.asarray(in_window), np.asarray(event_rows)
    if energy.ndim != 1 or not (energy.shape == row.shape == pdg.shape == in_window.shape):
        raise ValueError("Primary arrays must align")
    if row.dtype.kind not in "iu" or event_rows.dtype.kind not in "iu" or in_window.dtype != np.bool_:
        raise ValueError("Integer event IDs and boolean window flags required")
    if (not np.isfinite(energy).all() or np.any(energy < 0) or np.any(row < 0)
            or np.any(event_rows < 0) or len(np.unique(event_rows)) != len(event_rows)
            or not np.isin(row, event_rows).all()):
        raise ValueError("Invalid primary energy or event inventory")
    if not isinstance(processed_span, (int, np.integer)) or processed_span <= 0:
        raise ValueError("A positive processed event span is required")
    inventory = event_rows[event_rows < processed_span]
    keep = in_window & (pdg == 2112) & (row < processed_span)
    low, high = bands.counts(energy[keep], row[keep], inventory)
    counts = np.column_stack((high, low)).astype(np.int64)
    return {"events": int(len(inventory)), "counts_high_low": counts.sum(axis=0),
            "event_second_moments": counts.T @ counts}
