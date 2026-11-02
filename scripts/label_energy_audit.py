"""Read-only S-only label census on pinned observations and accepted native packs.

This preparation runner does not rebuild observations, graphs, scalers or
models. An explicit input manifest must bind all selection-set files first.
Protocol: reports/energy_label_optimization_protocol_20261029.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.analysis.label_energy_scan import origin_partition, threshold_ledger  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_path(record, base):
    path = Path(record["path"])
    path = path if path.is_absolute() else base / path
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Pinned input changed: {path}")
    return path


def check_manifest(manifest_path, config_path):
    """Check metadata/hashes before opening any observation or pack tensor."""
    base = Path(manifest_path).resolve().parent
    manifest = json.loads(Path(manifest_path).read_text())
    config = json.loads(Path(config_path).read_text())
    supported = {"tof_halfwidth_ns": .3, "label_reference_GeV": .3,
                 "graph_time_radius_ns": 1., "graph_spatial_radius_cells": 3.6,
                 "selection_parts": [0, 2, 4], "report_parts": [1, 3, 5]}
    if any(config[key] != value for key, value in supported.items()):
        raise ValueError("Configuration changes an unsupported frozen observation/split/reference rule")
    if manifest.get("schema") != "label-energy-audit-inputs-v1":
        raise ValueError("Unknown label-audit input schema")
    if manifest["config_sha256"] != sha256(config_path):
        raise ValueError("Input manifest binds another configuration")
    split_path = bound_path(manifest["split"], base)
    prepare_path = bound_path(manifest["prepare"], base)
    split = json.loads(split_path.read_text())
    prepare = json.loads(prepare_path.read_text())
    if prepare["state"] != "complete" or prepare["schema"] != "smash-train-prepare-v1":
        raise ValueError("A complete native preparation receipt is required")
    split_sha = sha256(split_path)
    if prepare["split_sha256"] != split_sha or config["split_sha256"] != split_sha:
        raise ValueError("Preparation/configuration was made from another split")
    view = prepare["graph_view"]
    if (view["radius_cells"] != config["graph_spatial_radius_cells"]
            or view["time_ns"] != config["graph_time_radius_ns"] or view["max_neighbors"] != 32):
        raise ValueError("Preparation uses another observed graph recipe")
    groups = {f"{g['tag']}/{g['kept']}": g for g in split["groups"]
              if g["truth_bound"] and g["part"] in config["selection_parts"]}
    if manifest.get("scope", "full") == "smoke":
        selected = {}
        for tag in ("zeroSpot", "defaultSpot", "bigSpot"):
            for part in config["selection_parts"]:
                candidates = sorted(cid for cid, g in groups.items() if g["tag"] == tag and g["part"] == part)
                if not candidates:
                    raise ValueError("Smoke requires every potential/selection-part stratum")
                selected[candidates[0]] = groups[candidates[0]]
        groups = selected
    elif manifest.get("scope", "full") != "full":
        raise ValueError("Unknown audit scope")
    if "census" in manifest:
        census_path = bound_path(manifest["census"], base)
        census_sha = sha256(census_path)
        if census_sha != split["census_sha256"]:
            raise ValueError("Origin census differs from the split proof")
        census = json.loads(census_path.read_text())
        if census["status"] != "complete" or census["coverage"] != "full":
            raise ValueError("A complete origin census is required")
        for cid, group in groups.items():
            tag = census["tags"][group["tag"]]
            record = tag["contents"][group["kept"]]
            binding = record["binding"]
            if (record["hits_sha256"] != group["hits_sha256"] or record["vacs_sha256"] != group["vacs_sha256"]
                    or binding["state"] != "exact_unique_file_candidate" or binding["primary"] != group["primary"]
                    or binding["matched_records"] != binding["eligible_records"] or binding["ambiguous_records"] != 0):
                raise ValueError(f"Direct-primary binding changed: {cid}")
            group["primary_binding"] = {"primary": group["primary"],
                                        "primary_sha256": tag["primary_files"][group["primary"]],
                                        "census_sha256": census_sha, "exact_direct_subset": True}
    records = manifest["files"]
    if set(records) != set(groups):
        raise ValueError("Inputs must cover exactly the truth-bound selection groups; no report files")
    paths = {}
    for cid, record in records.items():
        group = groups[cid]
        validation_fold = (group["part"] // 2 - 1) % 3
        source = prepare["files"][cid]
        if source["roles"].get(str(validation_fold)) != "validation":
            raise ValueError(f"Selected pack was not a validation pack: {cid}")
        pack_key = f"B/fold_{validation_fold}"
        if (record["observation"]["sha256"] != source["observation"]["sha256"]
                or record["pack"]["sha256"] != source["packs"][pack_key]["sha256"]):
            raise ValueError(f"Observation or validation pack differs from preparation: {cid}")
        if source["raw_sha256"] != {"hits": group["hits_sha256"], "vacs": group["vacs_sha256"]}:
            raise ValueError(f"Preparation differs from split raw pins: {cid}")
        paths[cid] = {key: bound_path(record[key], base) for key in ("observation", "pack")}
    return config, groups, paths


def audit_frame(frame, pack, floors, hist_edges, tof_halfwidth_ns=.3, primary_binding=None):
    """Use the accepted pack's cluster IDs; never construct another graph."""
    from HGNDRecoGNN.data.graph_dataset import _dfToF
    from HGNDRecoGNN.data.smash_transfer import _ordered
    from HGNDRecoGNN.scripts.audit_smash_sources import matching

    if tof_halfwidth_ns != .3:
        raise ValueError("This audit holds the native ToF half-width fixed at 0.3 ns")
    frame, rows, sides = _ordered(frame)
    array = {key: value.detach().cpu().numpy() for key, value in pack.items()}
    lengths = np.diff(array["node_ptr"])
    if (int(lengths.sum()) != len(frame) or np.any(lengths <= 0)
            or not np.array_equal(np.repeat(array["row"], lengths), rows)
            or not np.array_equal(np.repeat(array["side"], lengths), sides)):
        raise ValueError("Observation hit order differs from the accepted native pack")
    local = array["cluster"]
    cluster_sizes = np.diff(array["cluster_ptr"])
    if (local.dtype.kind not in "iu" or len(local) != len(frame) or np.any(local < 0)
            or np.any(local >= np.repeat(cluster_sizes, lengths))):
        raise ValueError("Native cluster membership differs from graph pointers")
    cluster = local + np.repeat(array["cluster_ptr"][:-1], lengths)
    if len(np.unique(cluster)) != len(array["label"]):
        raise ValueError("Native pack has an unrepresented cluster")
    energy = frame.Ekin.to_numpy(float)
    distance = np.linalg.norm(frame[["fX", "fY", "fZ"]].to_numpy(float) / 100, axis=1)
    t = frame.fTime.to_numpy(float)
    # 0.3 ns is the unchanged ToF condition, not the energy floor being tested.
    prompt = ((np.abs(frame.PDG.to_numpy()) == 2112) & frame.Side.isin([5, 6]).to_numpy()
              & (energy >= _dfToF(distance, t + tof_halfwidth_ns))
              & (energy <= _dfToF(distance, t - tof_halfwidth_ns)))
    ledger, targets = threshold_ledger(energy, prompt, cluster, floors)
    if .3 not in targets:
        raise ValueError("Native 0.3 reference must be present")
    native_y, native_e = targets[.3]
    if (not np.array_equal(native_y, array["label"])
            or not np.array_equal(native_e.astype(np.float32), array["energy"])):
        raise ValueError("Native labels or float32 targets fail exact baseline replay")
    edges = np.asarray(hist_edges, float)
    if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0) or edges[0] != 0:
        raise ValueError("Energy histogram must have increasing finite edges beginning at zero")
    sizes = np.bincount(cluster, minlength=len(native_y))
    origin = np.full(len(frame), 4, dtype=np.int8)
    if "fMotherId" in frame:
        mother = frame.fMotherId.to_numpy(float)
        valid = np.isfinite(mother) & (mother == np.floor(mother))
        origin[valid & (mother >= 0)] = 2
        if primary_binding and primary_binding.get("exact_direct_subset"):
            origin[valid & (mother == -1)] = 1
        # Conflicting metadata for one physical object remain unresolved.
        columns = [c for c in ("PDG", "fMotherId", "vX", "vY", "vZ") if c in frame]
        ambiguity = frame.groupby(["Row", "Id"], sort=False)[columns].transform("nunique", dropna=False).max(axis=1) > 1
        origin[ambiguity.to_numpy()] = 4
    for record in ledger:
        floor = record["label_floor_GeV"]
        y, e = targets[floor]
        added = (y == 1) & (native_y == 0)
        removed = (y == 0) & (native_y == 1)
        selected = prompt if floor is None else prompt & (energy > floor)
        record.update({"added_vs_native": int(added.sum()), "removed_vs_native": int(removed.sum()),
                       "added_singletons": int((added & (sizes == 1)).sum()),
                       "target_histogram": np.histogram(e[(y == 1) & (e < edges[-1])], edges)[0].tolist(),
                       "target_overflow": int(((y == 1) & (e >= edges[-1])).sum()),
                       "matching": matching(cluster, rows, frame.Id.to_numpy(), selected, len(y)),
                       "qualified_origin_clusters": origin_partition(cluster, origin, selected, len(y)),
                       "added_origin_clusters": origin_partition(cluster, origin, selected, len(y), added),
                       "direct_primary_matching": matching(cluster, rows, frame.Id.to_numpy(), selected & (origin == 1), len(y))})
        if sum(record["target_histogram"]) + record["target_overflow"] != record["signal_clusters"]:
            raise ValueError("Energy-target histogram does not conserve positive count")
    hit_hist = np.histogram(energy[prompt & (energy < edges[-1])], edges)[0].tolist()
    hit_overflow = int((prompt & (energy >= edges[-1])).sum())
    if sum(hit_hist) + hit_overflow != int(prompt.sum()):
        raise ValueError("Prompt-hit histogram does not conserve count")
    return {"hits": len(frame), "clusters": len(native_y), "ledger": ledger,
            "prompt_hit_energy_histogram": hit_hist,
            "prompt_hit_energy_overflow": hit_overflow,
            "observed_hits_after_jitter_t_ge_35ns": int((t >= 35).sum()),
            "observed_time_min_max_ns": [float(t.min()), float(t.max())],
            "native_replay_exact": True, "ancestry_checked": False,
            "primary_binding": primary_binding,
            "energy_definition": "Initial kinetic energy of associated MCTrack, from Convert2Csv.C; not energy at the cell",
            "provenance": "MC truth label diagnostics; observed candidate graphs frozen"}


def plot_counts(records, groups, config, out):
    """Shared whole-group draws for threshold contrasts; one bounded final phase."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.grid": False,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    floors = config["label_floor_GeV"]
    baseline = floors.index(config["label_reference_GeV"])
    rng = np.random.default_rng(config["bootstrap_proposal"]["seed"])
    repeats = config["bootstrap_proposal"]["resamples"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, layout="constrained")
    bootstrap = {}
    for column, tag in enumerate(("zeroSpot", "defaultSpot", "bigSpot")):
        members = sorted(cid for cid in records if groups[cid]["tag"] == tag)
        n = np.array([records[cid]["clusters"] for cid in members], float)
        positive = np.array([[r["signal_clusters"] for r in records[cid]["ledger"]] for cid in members])
        draws = np.zeros((repeats, len(members)))
        for part in config["selection_parts"]:
            ix = [i for i, cid in enumerate(members) if groups[cid]["part"] == part]
            if not ix:
                raise ValueError("Empty potential/part resampling stratum")
            draws[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=repeats)
        central = 100 * positive.sum(axis=0) / n.sum()
        replicas = 100 * (draws @ positive) / (draws @ n)[:, None]
        delta = central - central[baseline]
        delta_replicas = replicas - replicas[:, baseline, None]
        bootstrap[tag] = {"positive_fraction_percent": central.tolist(),
                          "bootstrap_sd_percent": replicas.std(axis=0, ddof=1).tolist(),
                          "paired_delta_vs_native_points": delta.tolist(),
                          "paired_delta_sd_points": delta_replicas.std(axis=0, ddof=1).tolist()}
        for ax, value, replica in ((axes[0, column], central, replicas),
                                   (axes[1, column], delta, delta_replicas)):
            ax.errorbar(np.arange(len(floors)), value, yerr=replica.std(axis=0, ddof=1), fmt="o", capsize=2)
            ax.set_xticks(np.arange(len(floors)), ["off" if f is None else str(f) for f in floors])
            ax.set_xlabel("Label energy floor [GeV]; off = no clause")
        potential = {"zeroSpot": 0, "defaultSpot": 18, "bigSpot": 90}[tag]
        axes[0, column].set_title(f"U_sym = {potential} MeV; {len(members)} groups")
        axes[1, column].axhline(0, color=".3", ls="--", lw=.7)
    axes[0, 0].set_ylabel("Positive / all observed clusters [%]")
    axes[1, 0].set_ylabel("Paired change from 0.3 [points]")
    fig.suptitle("MC truth label diagnostics | Xe+Cs, E_kin/A = 2.87 GeV\n"
                 f"S only; unchanged graphs; {repeats} shared group draws; bars = bootstrap SD; no B selection")
    fig.savefig(out / "label_counts.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return bootstrap


def run(manifest_path, config_path, output):
    import torch
    from HGNDRecoGNN.scripts.smash_train_prepare import load_observation

    config, groups, paths = check_manifest(manifest_path, config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    records = {}
    for cid, path in sorted(paths.items()):
        frame = load_observation(path["observation"])
        pack = torch.load(path["pack"], weights_only=True, map_location="cpu")
        records[cid] = audit_frame(frame, pack, config["label_floor_GeV"], config["target_hist_edges_GeV"],
                                   config["tof_halfwidth_ns"], groups[cid].get("primary_binding"))
    bootstrap = plot_counts(records, groups, config, output)
    summary = {"schema": "label-energy-audit-v1", "state": "complete", "files": records,
               "config": config, "bootstrap": bootstrap, "elapsed_seconds": time.monotonic() - start,
               "input_manifest_sha256": sha256(manifest_path), "config_sha256": sha256(config_path),
               "source_sha256": {p: sha256(ROOT / p) for p in (
                   "scripts/label_energy_audit.py", "analysis/label_energy_scan.py",
                   "scripts/smash_train_prepare.py", "data/graph_dataset.py",
                   "data/smash_transfer.py", "scripts/audit_smash_sources.py")},
               "audit_scope": json.loads(Path(manifest_path).read_text()).get("scope", "full"),
               "scope": "S-only native-graph label census; direct-source proof if bound; no full ancestry, model calls, T1 or optimizer result"}
    with (output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "reports/energy_label_optimization_20261029/config.json")
    parser.add_argument("--check-only", action="store_true", help="Check pins and membership without opening tensors")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.check_only:
        _, groups, _ = check_manifest(args.inputs, args.config)
        print(json.dumps({"state": "inputs_verified", "selection_groups": len(groups)}))
    elif args.output is None:
        parser.error("--output is required for an audit")
    else:
        result = run(args.inputs, args.config, args.output)
        print(json.dumps({"state": result["state"], "files": len(result["files"])}))


if __name__ == "__main__":
    main()
