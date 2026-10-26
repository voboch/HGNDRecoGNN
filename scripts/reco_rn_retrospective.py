"""Retrospective truth-to-reconstruction R_n ladder on the frozen corrected-SMASH transfer.

Protocol: reports/reco_rn_retrospective_protocol_20261026.md. Reads hash-pinned
inputs only; no training, inference, fitting or threshold choice. The result is a
retrospective feasibility check on reused data, never a holdout measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.analysis.truth_provenance import load_certified  # noqa: E402

TAGS = ("zeroSpot", "defaultSpot", "bigSpot")
U_MEV = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}
SEEDS = ("42", "123", "456")
# Manuscript hardness boundaries: an inherited convention, not optimised here.
E_LO, E_HI = 1.0, 2.0
# As in the B-free campaign; the Monte-Carlo error of a 400-replicate SD is
# about 1/sqrt(798) = 3.5 %, below two-significant-figure rounding.
N_BOOT = 400
# Arbitrary generator seeds, fixed before the run.
BOOT_SEED = 20261026
EVENT_BOOT_SEED = 20261027
# Injected weight on true-hard positive clusters: the size of the truth effect,
# a convention; it does not test linearity.
INJECT = 1.05
# Size of the differential energy response implied by review v9 M4 (percentage
# points); a G3 test whose 2-SD interval exceeds it cannot detect that response.
M4_SIZE = 1.4
# Inherited P1 convention: closure within 25 % of the truth effect, at 2 SD.
CLOSURE_FRACTION, N_SD = 0.25, 2.0
T_RUNGS = ("T0", "T1", "T2")
C_RUNGS = ("C1", "C2", "C3", "C4")
RUNGS = T_RUNGS + C_RUNGS
STEPS = (("T1", "T0"), ("T2", "T1"), ("C1", "T2"), ("C2", "C1"), ("C3", "C2"),
         ("C4", "C3"), ("C4", "T1"))
# Component steps of the reconstruction; the largest in absolute value, of either
# sign, names the component path B examines first (protocol, decision rule 2).
COMPONENT_STEPS = ("C2-C1", "C3-C2", "C4-C3")
TRANSFER = Path("results/smash_full_transfer_20261023")
BFREE = Path("results/bfree_truth_20261025")
PROTOCOL = Path("reports/reco_rn_retrospective_protocol_20261026.md")
COLUMNS = ["Row", "cl_score", "e_pred", "cl_label", "e_true"]
# Predictions committed in the protocol before the run (percent unless noted).
PREDICTED = [
    ("T0", "endpoint", 3.4, 5.4), ("T0 SD", "endpoint_sd", 0.5, 0.85),
    ("T1-T0", "step", -0.5, 0.5), ("T2-T1", "step", -0.5, 0.5),
    ("C1-T2", "step", -2.0, 1.5),
    ("C1", "endpoint", 3.0, 5.5), ("C1 SD", "endpoint_sd", 0.9, 1.5),
    ("C2-C1", "step", 1.0, 3.5), ("C3-C2", "step", -3.0, 1.0),
    ("C4-C3", "step", -3.5, -1.5),
    ("C4", "endpoint", 2.5, 4.5), ("C4 SD", "endpoint_sd", 1.2, 2.0),
    ("C4-T1", "step", -2.0, 0.5), ("C4-T1 SD", "step_sd", 1.0, 1.5),
    ("recovery_k", "k", 0.5, 0.9),
    ("C4 ordering fraction", "order", 0.5, 0.8),
    ("C4 file-over-event factor", "factor", 1.0, 1.6),
]
PREDICTED_GLOBAL = [("C4 seed spread", 0.0, 1.5), ("null max |z|", 0.0, 2.5)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def band_counts(energy, weight):
    """(high, low) counts; 1 <= E < 2 GeV enters neither band."""
    return np.array([weight[energy >= E_HI].sum(), weight[energy < E_LO].sum()], float)


def cluster_counts(frame, cut, weight=None):
    """Per-file (high, low) counts for rungs C1..C4.

    Selection is strict, score > cut, as in the transfer's measure().
    """
    pos = frame.cl_label.to_numpy() == 1
    sel = frame.cl_score.to_numpy() > cut
    e_true = frame.e_true.to_numpy()
    e_pred = frame.e_pred.to_numpy()
    w = np.ones(len(frame)) if weight is None else np.asarray(weight, float)
    return np.stack([
        band_counts(e_true[pos], w[pos]),
        band_counts(e_true[pos & sel], w[pos & sel]),
        band_counts(e_pred[pos & sel], w[pos & sel]),
        band_counts(e_pred[sel], w[sel]),
    ])


def injection_weight(frame):
    """Signal-path injection: background correlated with hard neutrons is not scaled."""
    return np.where((frame.cl_label.to_numpy() == 1) & (frame.e_true.to_numpy() >= E_HI),
                    INJECT, 1.0)


def truth_counts(events, span, candidate_rows):
    """(high, low) primary-neutron counts for T0, T1 and T2 in one primary file.

    `events` is indexed by Row. T1 keeps Row < span (processed range, assuming
    sequential processing); T2 keeps events with at least one detector candidate.
    """
    rows = events.index.to_numpy()
    hi = events.acc_n_hi.to_numpy(float)
    lo = events.acc_n_lo.to_numpy(float)
    keep = [np.ones(len(rows), bool), rows < span, np.isin(rows, candidate_rows)]
    return np.array([[hi[k].sum(), lo[k].sum()] for k in keep])


def event_counts(frame, cut):
    """Per-event (high, low) of rung C4, events with nonzero counts only."""
    sel = frame.cl_score.to_numpy() > cut
    e = frame.e_pred.to_numpy()
    df = pd.DataFrame({"Row": frame.Row.to_numpy()[sel], "hi": e[sel] >= E_HI, "lo": e[sel] < E_LO})
    g = df.groupby("Row")[["hi", "lo"]].sum()
    g = g[(g.hi > 0) | (g.lo > 0)]
    return g.to_numpy(float)


def ratio(counts):
    """R = high/low along the last axis; refuses an empty denominator."""
    counts = np.asarray(counts, float)
    if (counts[..., 1] <= 0).any():
        raise ValueError("Empty low-energy denominator")
    return counts[..., 0] / counts[..., 1]


def file_draws(n_files, rng):
    """Whole-file bootstrap weights, shape (N_BOOT, n_files)."""
    return rng.multinomial(n_files, np.full(n_files, 1.0 / n_files), size=N_BOOT)


def resampled(draws, per_file):
    """Sum per-file counts (files, ..., 2) under each replicate's file weights."""
    return np.tensordot(draws, per_file, axes=(1, 0))


def sd(x):
    return float(np.std(x, ddof=1))


def summarize_contrast(central, replicas):
    return {"central_percent": float(central), "bootstrap_sd_percent": sd(replicas)}


def gate_g2(diff, diff_sd, truth):
    """Inherited P1 closure gate: |diff| < 2 SD, and the 2-SD interval inside 25 % of |truth|."""
    if abs(diff) > 0 and abs(diff) >= N_SD * diff_sd:
        return "fail"
    if abs(diff) + N_SD * diff_sd < CLOSURE_FRACTION * abs(truth):
        return "pass"
    return "not resolvable"


def gate_g3(diff, diff_sd):
    """Energy-migration step consistent with zero; unresolvable if 2 SD exceed the M4 size."""
    if abs(diff) > 0 and abs(diff) >= N_SD * diff_sd:
        return "fail"
    return "not resolvable" if N_SD * diff_sd >= M4_SIZE else "pass"


def null_gate(z_values):
    """Twelve correlated null tests (T1 repeats per seed): judge the largest |z|.

    2.5 keeps the family-wise false-alarm rate near five per cent for about four
    effectively independent tests; |z| >= 3 (about one per cent) stops interpretation.
    """
    worst = max(abs(z) for z in z_values)
    return "stop" if worst >= 3 else ("pass" if worst < 2.5 else "marginal")


def combine_seeds(states):
    """A gate passes only if it passes for every seed; one failure fails it."""
    states = list(states)
    if all(s == "pass" for s in states):
        return "pass"
    return "fail" if "fail" in states else "not resolvable"


def files_needed(diff_sd, truth, n_zero, n_high):
    """Equal files per potential at which the 2-SD interval of a null closure difference
    equals the G2 threshold, scaling the measured SD as sqrt(1/n_zero + 1/n_high)."""
    half_threshold = CLOSURE_FRACTION * abs(truth) / N_SD
    return float(2 * diff_sd ** 2 / ((1 / n_zero + 1 / n_high) * half_threshold ** 2))


def load_inputs(root):
    """Hash-verify every input and return the pinned populations."""
    metrics = json.loads((root / TRANSFER / "full_metrics_verified.json").read_text())
    pinned = metrics["input_sha256"]
    for name in ("full_transfer_summary_bundle.json", "bound_inventory.json"):
        path = TRANSFER / name
        if sha256(root / path) != pinned[str(path)]:
            raise ValueError(f"Pinned transfer receipt changed: {path}")
    bundle = json.loads((root / TRANSFER / "full_transfer_summary_bundle.json").read_text())
    inventory = json.loads((root / TRANSFER / "bound_inventory.json").read_text())
    aggregate = json.loads((root / BFREE / "aggregate_4452505_verified.summary.json").read_text())
    cuts, checkpoints, tags = {}, {}, {}
    for tag in TAGS:
        base = root / TRANSFER / "full_transfer_inputs/transfer" / tag
        if sha256(base / "summary.json") != bundle["summary_sha256"][tag]:
            raise ValueError(f"Transfer summary changed: {tag}")
        per_seed = json.loads((base / "summary.json").read_text())["per_seed"]
        for seed in SEEDS:
            pair = (per_seed[seed]["threshold"], per_seed[seed]["checkpoint_sha256"])
            if cuts.setdefault(seed, pair[0]) != pair[0] or checkpoints.setdefault(seed, pair[1]) != pair[1]:
                raise ValueError("Score cut or checkpoint differs between potentials")
        bound = inventory["tags"][tag]
        members = [m for m in bound["analysis_pair_names"]
                   if m not in bound["empty_exports"] and bound["primary_member_map"][m] is not None]
        expected = {base / f"seed_{s}" / f"{m}.csv" for s in SEEDS for m in members}
        found = set(base.glob("seed_*/*/*.csv"))
        if found != expected:
            raise ValueError(f"Prediction files differ from the bound population: {tag}")
        for path in sorted(expected):
            if sha256(path) != pinned[str(path.relative_to(root))]:
                raise ValueError(f"Prediction file changed: {path}")
        cert = root / BFREE / "aggregate_4452505_verified/result" / tag
        if sha256(cert / "proof.json") != aggregate["samples"][tag]["proof_sha256"]:
            raise ValueError(f"Certified truth proof changed: {tag}")
        events, _, proof = load_certified(cert, "acceptance")
        if proof["counter_mismatches"] != 0 or proof["coverage"] != "full":
            raise ValueError("Truth certificate is not a complete exact replay")
        tags[tag] = {"base": base, "members": members,
                     "primary": {m: bound["primary_member_map"][m] for m in members},
                     "events": events}
    return {"cuts": cuts, "checkpoints": checkpoints, "tags": tags,
            "pins": {"full_metrics_verified": sha256(root / TRANSFER / "full_metrics_verified.json"),
                     "bound_inventory": pinned[str(TRANSFER / "bound_inventory.json")],
                     "bfree_aggregate_summary": sha256(root / BFREE / "aggregate_4452505_verified.summary.json")}}


def collect(inputs):
    """One pass over the predictions: per-file counts for every rung, seed and control."""
    out = {}
    for tag, spec in inputs["tags"].items():
        events = spec["events"]
        truth, clusters, injected, coverage = [], [], [], []
        per_event = {k: [] for k in ("T1",) + SEEDS}
        for member in spec["members"]:
            frames = {s: pd.read_csv(spec["base"] / f"seed_{s}" / f"{member}.csv", usecols=COLUMNS)
                      for s in SEEDS}
            reference = frames[SEEDS[0]][["Row", "cl_label", "e_true"]]
            if any(not f[["Row", "cl_label", "e_true"]].equals(reference) for f in frames.values()):
                raise ValueError(f"Seeds scored different clusters: {tag}/{member}")
            if not np.isfinite(reference.e_true.to_numpy()).all() or any(
                    not np.isfinite(f[["cl_score", "e_pred"]].to_numpy()).all() for f in frames.values()):
                raise ValueError(f"Nonfinite prediction: {tag}/{member}")
            primary = events.xs(spec["primary"][member] + "_prim.csv", level="member")
            rows = np.unique(reference.Row.to_numpy())
            if not np.isin(rows, primary.index.to_numpy()).all():
                raise ValueError(f"Detector Row absent from bound primary file: {tag}/{member}")
            span = int(rows.max()) + 1
            coverage.append({"detector_member": member, "primary_member": spec["primary"][member],
                             "events_with_candidates": int(len(rows)), "processed_span": span,
                             "primary_events": int(len(primary))})
            truth.append(truth_counts(primary, span, rows))
            clusters.append([cluster_counts(frames[s], inputs["cuts"][s]) for s in SEEDS])
            if tag == "zeroSpot":
                w = injection_weight(frames[SEEDS[0]])
                injected.append([cluster_counts(frames[s], inputs["cuts"][s], w) for s in SEEDS])
            for s in SEEDS:
                per_event[s].append(event_counts(frames[s], inputs["cuts"][s]))
            keep = primary.index.to_numpy() < span
            t1 = primary.loc[keep, ["acc_n_hi", "acc_n_lo"]].to_numpy(float)
            per_event["T1"].append(t1[(t1 > 0).any(axis=1)])
        out[tag] = {"truth": np.array(truth), "clusters": np.array(clusters),
                    "injected": np.array(injected) if injected else None,
                    "per_event": {k: np.concatenate(v) for k, v in per_event.items()},
                    "coverage": coverage}
    return out


def rung_values(tag_counts):
    """Stack truth (3 rungs, seed-independent) with clusters per seed: (files, seeds, 7, 2)."""
    truth = np.repeat(tag_counts["truth"][:, None], len(SEEDS), axis=1)
    return np.concatenate([truth, tag_counts["clusters"]], axis=2)


def analyse(counts):
    """Return the result dictionary and the replicate arrays it was computed from.

    One whole-file draw per potential and replicate is shared by every rung, seed
    and the injection, so their differences are paired. The null test draws its
    own replicates within half-samples after the main draws.
    """
    rng = np.random.default_rng(BOOT_SEED)
    per_file = {tag: rung_values(counts[tag]) for tag in TAGS}
    draws = {tag: file_draws(len(per_file[tag]), rng) for tag in TAGS}
    central = {tag: ratio(per_file[tag].sum(axis=0)) for tag in TAGS}              # (seeds, rungs)
    replicas = {tag: ratio(resampled(draws[tag], per_file[tag])) for tag in TAGS}  # (B, seeds, rungs)
    result = {"R": {}, "endpoint": {}, "midpoint": {}, "steps": {}, "ordering_C4": {}}
    endpoint_c = 100 * (central["bigSpot"] / central["zeroSpot"] - 1)
    endpoint_r = 100 * (replicas["bigSpot"] / replicas["zeroSpot"] - 1)
    midpoint_c = 100 * (central["defaultSpot"] / central["zeroSpot"] - 1)
    midpoint_r = 100 * (replicas["defaultSpot"] / replicas["zeroSpot"] - 1)
    for i, seed in enumerate(SEEDS):
        for j, rung in enumerate(RUNGS):
            key = f"{rung}/seed_{seed}"
            result["R"][key] = {tag: {"central": float(central[tag][i, j]),
                                      "bootstrap_sd": sd(replicas[tag][:, i, j])} for tag in TAGS}
            result["endpoint"][key] = summarize_contrast(endpoint_c[i, j], endpoint_r[:, i, j])
            result["midpoint"][key] = summarize_contrast(midpoint_c[i, j], midpoint_r[:, i, j])
        for a, b in STEPS:
            ja, jb = RUNGS.index(a), RUNGS.index(b)
            result["steps"][f"{a}-{b}/seed_{seed}"] = summarize_contrast(
                endpoint_c[i, ja] - endpoint_c[i, jb], endpoint_r[:, i, ja] - endpoint_r[:, i, jb])
        ordered = ((replicas["zeroSpot"][:, i, -1] < replicas["defaultSpot"][:, i, -1])
                   & (replicas["defaultSpot"][:, i, -1] < replicas["bigSpot"][:, i, -1]))
        result["ordering_C4"][f"seed_{seed}"] = {
            "central_ordered": bool(central["zeroSpot"][i, -1] < central["defaultSpot"][i, -1]
                                    < central["bigSpot"][i, -1]),
            "fraction_of_replicates_ordered": float(ordered.mean())}
    j4 = RUNGS.index("C4")
    result["seed_differences_C4"] = {
        f"seed_{a}-seed_{b}": summarize_contrast(
            endpoint_c[SEEDS.index(a), j4] - endpoint_c[SEEDS.index(b), j4],
            endpoint_r[:, SEEDS.index(a), j4] - endpoint_r[:, SEEDS.index(b), j4])
        for a, b in itertools.combinations(SEEDS, 2)}
    # Injection: zeroSpot only, on the same zeroSpot draws; C1 rises by exactly INJECT - 1.
    inj = counts["zeroSpot"]["injected"]
    base_c = ratio(counts["zeroSpot"]["clusters"].sum(axis=0))
    base_r = ratio(resampled(draws["zeroSpot"], counts["zeroSpot"]["clusters"]))
    shift_c = ratio(inj.sum(axis=0)) / base_c - 1
    shift_r = ratio(resampled(draws["zeroSpot"], inj)) / base_r - 1
    result["injection"] = {
        f"seed_{seed}": {
            "C1_shift_percent": float(100 * shift_c[i, 0]),
            "C4_shift_percent": float(100 * shift_c[i, 3]),
            "recovery_k": {"central": float(shift_c[i, 3] / shift_c[i, 0]),
                           "bootstrap_sd": sd(shift_r[:, i, 3] / shift_r[:, i, 0])}}
        for i, seed in enumerate(SEEDS)}
    # Same-sample null: index parity in the analysed file list, bootstrap within halves.
    result["null"] = {}
    for tag in ("zeroSpot", "defaultSpot"):
        odd, even = per_file[tag][1::2], per_file[tag][0::2]
        c = 100 * (ratio(odd.sum(axis=0)) / ratio(even.sum(axis=0)) - 1)
        r = 100 * (ratio(resampled(file_draws(len(odd), rng), odd))
                   / ratio(resampled(file_draws(len(even), rng), even)) - 1)
        for i, seed in enumerate(SEEDS):
            for rung in ("T1", "C4"):
                j = RUNGS.index(rung)
                s = sd(r[:, i, j])
                result["null"][f"{tag}/{rung}/seed_{seed}"] = {
                    "central_percent": float(c[i, j]), "bootstrap_sd_percent": s,
                    "z": float(c[i, j] / s)}
    # File-over-event factor: whole-file SD over an event-level Poisson-weight SD.
    # No precedent in the project; the quoted error is always the file SD.
    erng = np.random.default_rng(EVENT_BOOT_SEED)
    result["file_over_event_factor"] = {}
    for name in ("T1",) + SEEDS:
        reps = {}
        for tag in ("zeroSpot", "bigSpot"):
            ev = counts[tag]["per_event"][name]
            reps[tag] = np.array([ratio((erng.poisson(1.0, len(ev))[:, None] * ev).sum(axis=0))
                                  for _ in range(N_BOOT)])
        event_sd = sd(100 * (reps["bigSpot"] / reps["zeroSpot"] - 1))
        key = "T1" if name == "T1" else f"C4/seed_{name}"
        file_sd = result["endpoint"]["T1/seed_42" if name == "T1" else key]["bootstrap_sd_percent"]
        result["file_over_event_factor"][key] = {
            "file_sd_percent": file_sd, "event_sd_percent": event_sd, "factor": file_sd / event_sd}
    arrays = {"endpoint": endpoint_r, "midpoint": midpoint_r, "injection_shift": shift_r,
              **{f"R_{tag}": replicas[tag] for tag in TAGS}}
    return result, arrays


def gates(result, n_files):
    """Inherited P1 gates per seed, the all-seed rule and the decision outputs."""
    g5 = null_gate([v["z"] for v in result["null"].values()])
    c4 = [result["endpoint"][f"C4/seed_{s}"] for s in SEEDS]
    spread = max(x["central_percent"] for x in c4) - min(x["central_percent"] for x in c4)
    stat = float(np.mean([x["bootstrap_sd_percent"] for x in c4]))
    per_seed, decision = {}, {}
    for seed in SEEDS:
        t1 = result["endpoint"][f"T1/seed_{seed}"]["central_percent"]
        reco = result["endpoint"][f"C4/seed_{seed}"]["central_percent"]
        total = result["steps"][f"C4-T1/seed_{seed}"]
        energy = result["steps"][f"C3-C2/seed_{seed}"]
        per_seed[f"seed_{seed}"] = {
            "G1_same_sign": "pass" if np.sign(reco) == np.sign(t1) else "fail",
            "G2_closure": gate_g2(total["central_percent"], total["bootstrap_sd_percent"], t1),
            "G3_energy_migration": gate_g3(energy["central_percent"], energy["bootstrap_sd_percent"]),
        }
        losses = {k: result["steps"][f"{k}/seed_{seed}"] for k in COMPONENT_STEPS}
        largest = max(losses, key=lambda k: abs(losses[k]["central_percent"]))
        decision[f"seed_{seed}"] = {
            "files_per_potential_for_G2": files_needed(
                total["bootstrap_sd_percent"], t1, n_files["zeroSpot"], n_files["bigSpot"]),
            "largest_step": largest,
            "largest_step_beyond_2sd": bool(abs(losses[largest]["central_percent"])
                                            >= N_SD * losses[largest]["bootstrap_sd_percent"]),
            "recovery_k": result["injection"][f"seed_{seed}"]["recovery_k"]["central"],
        }
    combined = {g: combine_seeds(v[g] for v in per_seed.values())
                for g in ("G1_same_sign", "G2_closure", "G3_energy_migration")}
    return {"per_seed": per_seed, "all_seeds": combined,
            "G4_ordering_descriptive": result["ordering_C4"],
            "G5_null": g5,
            "G6_seed_stability": {"seed_spread_percent": spread, "mean_statistical_sd_percent": stat,
                                  "state": "pass" if spread < stat else "fail"},
            "decision": decision}


def prediction_check(result, gate_result):
    checks = {}

    def value(kind, key, seed):
        if kind == "endpoint":
            return result["endpoint"][f"{key}/seed_{seed}"]["central_percent"]
        if kind == "endpoint_sd":
            return result["endpoint"][f"{key.split()[0]}/seed_{seed}"]["bootstrap_sd_percent"]
        if kind == "step":
            return result["steps"][f"{key}/seed_{seed}"]["central_percent"]
        if kind == "step_sd":
            return result["steps"][f"{key.split()[0]}/seed_{seed}"]["bootstrap_sd_percent"]
        if kind == "k":
            return result["injection"][f"seed_{seed}"]["recovery_k"]["central"]
        if kind == "order":
            return result["ordering_C4"][f"seed_{seed}"]["fraction_of_replicates_ordered"]
        return result["file_over_event_factor"][f"C4/seed_{seed}"]["factor"]

    for seed in SEEDS:
        for key, kind, lo, hi in PREDICTED:
            v = value(kind, key, seed)
            checks[f"{key}/seed_{seed}"] = {"value": v, "predicted": [lo, hi], "inside": bool(lo <= v <= hi)}
    worst_z = max(abs(v["z"]) for v in result["null"].values())
    for (key, lo, hi), v in zip(PREDICTED_GLOBAL,
                                (gate_result["G6_seed_stability"]["seed_spread_percent"], worst_z)):
        checks[key] = {"value": v, "predicted": [lo, hi], "inside": bool(lo <= v <= hi)}
    return checks


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from HGNDRecoGNN.scripts.plot_b_full import mark_provenance
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8.5, "xtick.direction": "in",
                         "ytick.direction": "in"})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.84, 5.8),
                                      gridspec_kw={"height_ratios": [3, 2], "hspace": 0.55})
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.83)

    def headroom(ax, values, errors):
        # Free the top third of the panel for the provenance stamp.
        lo = min(min(v - e for v, e in zip(values, errors)), 0.0)
        hi = max(v + e for v, e in zip(values, errors))
        span = (hi - lo) or 1.0
        ax.set_ylim(lo - 0.08 * span, hi + 0.45 * span)
    colors = {"42": "#1f4e79", "123": "#b5651d", "456": "#4f7f3a"}
    x = np.arange(len(RUNGS))
    for k, seed in enumerate(SEEDS):
        y = [result["endpoint"][f"{r}/seed_{seed}"]["central_percent"] for r in RUNGS]
        e = [result["endpoint"][f"{r}/seed_{seed}"]["bootstrap_sd_percent"] for r in RUNGS]
        if k == 0:
            top.errorbar(x[:3], y[:3], yerr=e[:3], fmt="s", color="black", capsize=2, ms=4,
                         label="generator truth")
        top.errorbar(x[3:] + (k - 1) * 0.12, y[3:], yerr=e[3:], fmt="o", color=colors[seed],
                     capsize=2, ms=4, label=f"training seed {seed}")
    top.axhline(result["endpoint"]["T1/seed_42"]["central_percent"], color=".4", ls="--", lw=0.7,
                label="T1 central value")
    top.axhline(0, color=".7", lw=0.5)
    entries = [result["endpoint"][f"{r}/seed_{s}"] for r in RUNGS for s in SEEDS]
    headroom(top, [v["central_percent"] for v in entries], [v["bootstrap_sd_percent"] for v in entries])
    top.set_xticks(x)
    top.set_xticklabels(["T0\nall events", "T1\nprocessed", "T2\nwith cand.", "C1\nMC-pos.",
                         "C2\nselected", "C3\nreco E", "C4\n+ background"])
    top.set_ylabel(r"$100[R_n(90)/R_n(0)-1]$ (%)")
    top.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    mark_provenance(top, "mixed")
    steps = [f"{a}-{b}" for a, b in STEPS]
    xs = np.arange(len(steps))
    for k, seed in enumerate(SEEDS):
        y = [result["steps"][f"{s}/seed_{seed}"]["central_percent"] for s in steps]
        e = [result["steps"][f"{s}/seed_{seed}"]["bootstrap_sd_percent"] for s in steps]
        bottom.errorbar(xs + (k - 1) * 0.12, y, yerr=e, fmt="o", color=colors[seed], capsize=2, ms=3.5)
    bottom.axhline(0, color=".4", lw=0.6)
    entries = [result["steps"][f"{s}/seed_{seed}"] for s in steps for seed in SEEDS]
    headroom(bottom, [v["central_percent"] for v in entries], [v["bootstrap_sd_percent"] for v in entries])
    bottom.set_xticks(xs)
    bottom.set_xticklabels([s.replace("-", "−") for s in steps])
    bottom.set_ylabel("paired step (points)")
    mark_provenance(bottom, "mixed")
    header = [
        r"BM@N HGND simulation | SMASH Xe+Cs, $E_{\rm kin}/A=2.87$ GeV | $U_{\rm sym}=90$ vs 0 MeV",
        "No impact-parameter selection (B unavailable) | frozen DCM weights | retrospective, no holdout",
        "MC truth: T0–C3 (truth label; C3 with reconstructed energy) | reconstructed: C4",
        f"Bars: SD of {N_BOOT} whole-file bootstrap resamples; lower panel paired over the same resamples",
    ]
    for i, line in enumerate(header):
        fig.text(0.1, 0.975 - 0.03 * i, line, fontsize=7)
    fig.canvas.draw()
    renderer, canvas = fig.canvas.get_renderer(), fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.width and (box.x0 < canvas.x0 - 1 or box.x1 > canvas.x1 + 1
                              or box.y0 < canvas.y0 - 1 or box.y1 > canvas.y1 + 1):
                raise ValueError(f"Figure text outside the canvas: {text.get_text()!r}")
    (output / "figures").mkdir()
    for ext in ("pdf", "png"):
        fig.savefig(output / "figures" / f"ladder.{ext}", dpi=160,
                    metadata={"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
    plt.close(fig)


def main(args):
    root = ROOT
    started = time.monotonic()
    git = ["git", "-C", str(root)]
    commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = subprocess.run(git + ["status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise ValueError("Tracked files must be committed before the run:\n" + dirty)
    output = root / args.output
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise ValueError(f"Output already exists: {output}")
    inputs = load_inputs(root)
    counts = collect(inputs)
    result, arrays = analyse(counts)
    n_files = {tag: len(counts[tag]["coverage"]) for tag in TAGS}
    gate_result = gates(result, n_files)
    summary = {
        "schema": "reco-rn-retrospective-v1",
        "scope": "retrospective feasibility on reused corrected-SMASH data with frozen DCM-trained "
                 "weights; no holdout, no fitting; MC truth T0-C3 (C3 with reconstructed energy), "
                 "reconstructed C4",
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(root / PROTOCOL)},
        "code": {"path": "scripts/reco_rn_retrospective.py",
                 "sha256": sha256(root / "scripts/reco_rn_retrospective.py"),
                 "git_commit": commit, "tracked_tree_clean": True},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__},
        "inputs": inputs["pins"], "cuts": inputs["cuts"], "checkpoints": inputs["checkpoints"],
        "settings": {"E_low_GeV": E_LO, "E_high_GeV": E_HI, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                     "event_boot_seed": EVENT_BOOT_SEED, "inject_weight": INJECT, "m4_size": M4_SIZE,
                     "closure_fraction": CLOSURE_FRACTION, "n_sd": N_SD,
                     "selection": "cl_score > cut", "U_MeV": U_MEV},
        "population": {tag: {"files": n_files[tag],
                             "events_with_candidates": sum(c["events_with_candidates"]
                                                           for c in counts[tag]["coverage"]),
                             "processed_span_events": sum(c["processed_span"] for c in counts[tag]["coverage"]),
                             "primary_events": sum(c["primary_events"] for c in counts[tag]["coverage"])}
                       for tag in TAGS},
        "result": result, "gates": gate_result,
        "predictions": prediction_check(result, gate_result),
    }
    per_file = {tag: [{**c, "truth_hi_lo": counts[tag]["truth"][i].tolist(),
                       "clusters_hi_lo": {s: counts[tag]["clusters"][i, k].tolist()
                                          for k, s in enumerate(SEEDS)}}
                      for i, c in enumerate(counts[tag]["coverage"])] for tag in TAGS}
    # Everything is written to a partial directory and renamed only on success.
    partial.mkdir(parents=True)
    try:
        plot(result, partial)
        np.savez_compressed(partial / "replicas.npz", **arrays)
        (partial / "per_file_counts.json").write_text(json.dumps(per_file, indent=1) + "\n")
        summary["seconds"] = time.monotonic() - started
        (partial / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    except Exception:
        shutil.rmtree(partial)
        raise
    partial.rename(output)
    print(json.dumps({"status": "complete", "seconds": summary["seconds"], "output": str(output)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("results/reco_rn_retrospective_20261026"))
    main(p.parse_args())
