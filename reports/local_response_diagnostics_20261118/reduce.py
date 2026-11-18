"""Frozen local neutron-response diagnostics; no ROOT or model invocation."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile
from time import monotonic

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAPIDITY = ROOT / "reports/rapidity_export_20261117"
FINAL = ROOT / "reports/final_model_comparison_20261117"
sys.path.insert(0, str(RAPIDITY))
import scan

FIELDS = (
    "events",
    "truth",
    "direct",
    "miss",
    "split",
    "split_excess_edges",
    "candidates",
    "merge",
    "mixed_merge",
)
STATES = ("0", "1", "2", "3_or_more")
METHODS = ("rules", "dynamic_edgeconv")
SAMPLE_FIELDS = (
    "true_y",
    "true_pT",
    "true_energy",
    "rules_energy",
    "gnn_energy",
    "rules_y",
    "gnn_y",
)


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path, digest=None):
    raw = Path(path).read_bytes()
    scan.arch.require(
        digest is None or sha256(raw).hexdigest() == digest, f"SHA: {path}"
    )
    return json.loads(raw)


def ledger(path, digest):
    raw = Path(path).read_bytes()
    scan.arch.require(sha256(raw).hexdigest() == digest, f"Ledger SHA: {path}")
    with np.load(BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sources():
    even = read(FINAL / "even.json")
    old = ledger(FINAL / "even.npz", even["counts_sha256"])
    rapid = read(RAPIDITY / "scan_results/scan.json")
    scan.arch.require(
        sha(RAPIDITY / "scan.py") == rapid["source_sha256"], "Scan dependency"
    )
    y = ledger(RAPIDITY / "scan_results/counts.npz", rapid["ledger_sha256"])
    joint = read(RAPIDITY / "joint_results_20261118/joint.json")
    j = ledger(
        RAPIDITY / "joint_results_20261118/joint_counts.npz", joint["ledger_sha256"]
    )
    for z in (y, j):
        for key in ("content", "tags"):
            np.testing.assert_array_equal(old[key], z[key])
        np.testing.assert_array_equal(old["draws"], z["weights"])
    return old, even, y, rapid, j, joint


def assemble(input_dir, output):
    start = monotonic()
    old, _, previous_y, rapid_summary, _, _ = sources()
    scan.arch.require(
        sha(input_dir / "acceptance.json") == rapid_summary["export_acceptance_sha256"],
        "Sidecar acceptance differs from the accepted rapidity scan",
    )
    acceptance = read(input_dir / "acceptance.json")
    info = read(input_dir / "summary.json", acceptance["summary_sha256"])
    scan.arch.require(
        acceptance["state"] == "accepted" and len(info["files"]) == 286,
        "Sidecar acceptance",
    )
    sidecars = {r["content"]: r for r in info["files"]}
    primary_acceptance = read(scan.arch.PRIMARY / "acceptance.json")
    primary_summary = read(
        scan.arch.PRIMARY / "result/summary.json", primary_acceptance["summary_sha256"]
    )
    records = {r["content"]: r for r in primary_summary["files"]}
    launch = read(RAPIDITY / "launch.json")
    plan = read(ROOT / launch["historical_plan"], launch["historical_plan_sha256"])[
        "files"
    ]
    rule_tar, entries = scan.export_inputs(scan.arch.BENCHMARK / "export_4502291", plan)
    scan.arch.require(set(old["content"]) == set(sidecars), "Sidecar inventory")
    keys = list(old["count_keys"])
    per_file, samples, sample_keys, natives, stable = (
        [],
        [[] for _ in range(3)],
        [[] for _ in range(3)],
        [[] for _ in range(3)],
        [[] for _ in range(3)],
    )
    pins = {}
    with tarfile.open(rule_tar, "r:") as archive:
        for fi, cid in enumerate(old["content"]):
            path = input_dir / cid / "kinematics.npz"
            raw_k = scan.arch.accepted.checked(path, sidecars[cid]["output_sha256"])
            record = records[cid]
            span = record["processed_span"]
            raw_n = scan.arch.accepted.checked(
                scan.arch.PRIMARY / "result" / cid / "neutrons.npz",
                record["files"]["neutrons.npz"],
            )
            raw_r = archive.extractfile(entries[cid]["member"]).read()
            scan.arch.require(
                sha256(raw_r).hexdigest() == entries[cid]["sha256"], "Rule member SHA"
            )
            pins[cid] = {
                "kinematics": sha256(raw_k).hexdigest(),
                "neutrons": sha256(raw_n).hexdigest(),
                "rules": sha256(raw_r).hexdigest(),
                "processed_span": span,
            }
            with (
                np.load(BytesIO(raw_k)) as z,
                np.load(BytesIO(raw_n)) as n,
                np.load(BytesIO(raw_r)) as r,
            ):
                row, ident, energy = n["Row"], n["Id"], n["Ekin"]
                inside = n["in_rectangle"] & (row < span)
                objects = scan.arch.pd.MultiIndex.from_arrays([row, ident])
                scan.arch.require(objects.is_unique, "Primary key uniqueness")
                gkeys = np.column_stack(
                    [n["cluster_row"], n["cluster_local_id"], n["cluster_side"]]
                )
                rkeys = np.column_stack(
                    [z["rules_keys"], np.zeros(len(r["rows"]), dtype=np.int64)]
                )
                for actual, expected in (
                    (z["neutron_Row"], row),
                    (z["neutron_Id"], ident),
                    (z["neutron_Ekin"], energy),
                    (z["gnn_keys"], gkeys),
                    (z["rules_keys"][:, 0], r["rows"]),
                    (z["rules_selection"], r["selected"]),
                    (z["event_Row"][z["event_in_span"]], np.arange(span)),
                ):
                    np.testing.assert_array_equal(actual, expected)
                graphs = (
                    scan.arch.accepted.edge_indices(r["edges"], objects, rkeys),
                    scan.arch.accepted.edge_indices(
                        n["observed_edges"], objects, gkeys
                    ),
                )
                isolation = [
                    scan.arch.isolation(o, c, len(row), len(k))
                    for (o, c), k in zip(graphs, (rkeys, gkeys), strict=True)
                ]
                event_mult = np.bincount(row[inside], minlength=span).clip(max=3)
                common = np.flatnonzero(
                    inside
                    & (energy >= 0.3)
                    & (energy < 4)
                    & (isolation[0] >= 0)
                    & (isolation[1] >= 0)
                )
                rule_keep = z["rules_selection"]
                selected = []
                for seed in scan.arch.SEEDS:
                    e = z[f"dynamic_seed_{seed}_energy_GeV"]
                    selected.append(
                        z[f"dynamic_seed_{seed}_keep70"] & (e > 0.3) & (e < 4)
                    )
                pairs_by_seed = [
                    common[rule_keep[isolation[0][common]] & keep[isolation[1][common]]]
                    for keep in selected
                ]
                common3 = np.logical_and.reduce(
                    [np.isin(common, p) for p in pairs_by_seed]
                )
                stable_objects = common[common3]
                counters = np.zeros((3, 2, 4, len(FIELDS)), dtype=np.int64)
                for si, seed in enumerate(scan.arch.SEEDS):
                    for mi, ((obj, cl), candidate_keys, keep) in enumerate(
                        zip(
                            graphs,
                            (rkeys, gkeys),
                            (rule_keep, selected[si]),
                            strict=True,
                        )
                    ):
                        cr = candidate_keys[:, 0]
                        selected_mult = np.bincount(cr[keep], minlength=span).clip(
                            max=3
                        )
                        event_matrix = np.zeros((4, 4), dtype=np.int64)
                        np.add.at(event_matrix, (event_mult, selected_mult), 1)
                        np.testing.assert_array_equal(
                            event_matrix, old["multiplicity"][fi, si, mi]
                        )
                        scan.arch.require(
                            (cr < span).all()
                            and (cr >= 0).all()
                            and np.array_equal(row[obj], cr[cl]),
                            "Cross-event graph",
                        )
                        active = keep[cl]
                        td = np.bincount(obj[active], minlength=len(row))
                        cd = np.bincount(cl[active], minlength=len(cr))
                        ci = np.bincount(cl[active & inside[obj]], minlength=len(cr))
                        for category in range(4):
                            truth_mask = inside & (
                                event_mult[row.clip(max=span - 1)] == category
                            )
                            cand_mask = keep & (event_mult[cr] == category)
                            values = (
                                np.count_nonzero(event_mult == category),
                                truth_mask.sum(),
                                np.count_nonzero(truth_mask & (td > 0)),
                                np.count_nonzero(truth_mask & (td == 0)),
                                np.count_nonzero(truth_mask & (td > 1)),
                                np.maximum(td[truth_mask] - 1, 0).sum(),
                                cand_mask.sum(),
                                np.count_nonzero(cand_mask & (cd > 1)),
                                np.count_nonzero(cand_mask & (ci > 0) & (cd > ci)),
                            )
                            counters[si, mi, category] = values
                        for field, reference in (
                            ("truth", "truth"),
                            ("direct", "truth_with_edge"),
                            ("miss", "truth_missed"),
                            ("split", "truth_split"),
                            ("candidates", "candidates"),
                            ("merge", "candidate_merge"),
                            ("mixed_merge", "candidate_mixed_merge"),
                        ):
                            scan.arch.require(
                                counters[si, mi, :, FIELDS.index(field)].sum()
                                == old["counts"][fi, si, mi, keys.index(reference)],
                                f"Multiplicity marginal {cid}/{seed}/{field}",
                            )
                        np.testing.assert_array_equal(
                            counters[si, mi, :, 0],
                            old["multiplicity"][fi, si, mi].sum(axis=-1),
                        )
                    p = pairs_by_seed[si]
                    # The old energy comparison used strict >0.3; prove membership.
                    scan.arch.require(
                        not np.any(energy[p] == 0.3), "True lower-edge cohort changed"
                    )
                    er = z["rules_energy_GeV"][isolation[0][p]]
                    eg = z[f"dynamic_seed_{seed}_energy_GeV"][isolation[1][p]]
                    yr = z["rules_y_nn_cm"][isolation[0][p]]
                    yg = z[f"dynamic_seed_{seed}_y_nn_cm"][isolation[1][p]]
                    sample = np.column_stack(
                        [
                            z["neutron_y_nn_cm"][p],
                            z["neutron_pT_GeV"][p],
                            energy[p],
                            er,
                            eg,
                            yr,
                            yg,
                        ]
                    )
                    errors = sample[:, 3:5] - sample[:, 2:3]
                    moments = np.stack(
                        [
                            np.full(2, len(p)),
                            errors.sum(axis=0),
                            abs(errors).sum(axis=0),
                        ],
                        axis=-1,
                    )
                    np.testing.assert_allclose(
                        moments, old["rule_energy_moments"][fi, si], rtol=0, atol=1e-10
                    )
                    for mi in range(2):
                        yerr = sample[:, 5 + mi] - sample[:, 0]
                        np.testing.assert_allclose(
                            [len(p), abs(yerr).sum(), yerr.sum()],
                            previous_y["errors"][fi, si, mi, 0].sum(axis=-1),
                            rtol=0,
                            atol=1e-10,
                        )
                    samples[si].append(sample)
                    sample_keys[si].append(
                        np.column_stack([np.full(len(p), fi), row[p], ident[p]])
                    )
                    natives[si].append(n["qualifying_native"][p])
                    stable[si].append(np.isin(p, stable_objects))
            per_file.append(counters)
            if (fi + 1) % 50 == 0:
                print(f"Assembled/replayed {fi + 1}/286", flush=True)
    output.mkdir(exist_ok=False)
    data = {
        "content": old["content"],
        "tags": old["tags"],
        "draws": old["draws"],
        "counts": np.asarray(per_file),
        "count_fields": FIELDS,
        "sample_fields": SAMPLE_FIELDS,
    }
    for si in range(3):
        for name, value in (
            ("sample", samples),
            ("keys", sample_keys),
            ("native", natives),
            ("stable", stable),
        ):
            data[f"{name}_{si}"] = np.concatenate(value[si])
    for si in (1, 2):
        np.testing.assert_array_equal(
            data["keys_0"][data["stable_0"]], data[f"keys_{si}"][data[f"stable_{si}"]]
        )
    np.savez_compressed(output / "data.npz", **data)
    meta = {
        "state": "assembled_and_reconciled",
        "elapsed_seconds": monotonic() - start,
        "files": 286,
        "seeds": scan.arch.SEEDS,
        "sample_counts": [len(data[f"sample_{i}"]) for i in range(3)],
        "stable_counts": [int(data[f"stable_{i}"].sum()) for i in range(3)],
        "inputs": pins,
        "data_sha256": sha(output / "data.npz"),
        "sidecar_acceptance_sha256": sha(input_dir / "acceptance.json"),
        "source_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(HERE / "protocol.md"),
        "helper_sources_sha256": {
            str(Path(p).relative_to(ROOT)): sha(p)
            for p in (
                scan.__file__,
                scan.arch.__file__,
                scan.arch.accepted.__file__,
                sys.modules[scan.export_inputs.__module__].__file__,
            )
        },
    }
    (output / "assembly.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Assembly complete in {meta['elapsed_seconds']:.1f}s", flush=True)


def describe(values):
    """Shape draw x seed. Preserve every undefined seed/draw in the mean."""
    out = scan.arch.interval(values.mean(axis=1))
    out["by_seed"] = {
        s: scan.arch.interval(values[:, i]) for i, s in enumerate(scan.arch.SEEDS)
    }
    central = values[0]
    finite = np.isfinite(central).all()
    out["seed_center_sd"] = float(central.std(ddof=1)) if finite else None
    out["seed_min"] = float(central.min()) if finite else None
    out["seed_max"] = float(central.max()) if finite else None
    out["same_nonzero_sign_across_seeds"] = (
        bool((central > 0).all() or (central < 0).all()) if finite else None
    )
    return out


def paired(values):
    return {m: describe(values[:, :, i]) for i, m in enumerate(METHODS)} | {
        "gnn_minus_rules": describe(values[:, :, 1] - values[:, :, 0])
    }


def energy_row(samples, masks, weights):
    shape = (len(weights), 3, 2)
    values = {
        k: np.full(shape, np.nan)
        for k in (
            "bias_MeV",
            "mae_MeV",
            "median_relative_percent",
            "central_width_percent",
        )
    }
    counts = []
    for si, (sample, fi, use) in enumerate(
        zip(samples[0], samples[1], masks, strict=True)
    ):
        target, indices = sample[use, 2], fi[use]
        n = weights @ np.bincount(indices, minlength=weights.shape[1])
        counts.append(int(n[0]))
        for mi in range(2):
            error = sample[use, 3 + mi] - target
            for name, val in (("bias_MeV", error), ("mae_MeV", abs(error))):
                sums = weights @ np.bincount(
                    indices, weights=val, minlength=weights.shape[1]
                )
                values[name][:, si, mi] = 1000 * scan.ratio(sums, n)
            # Inherited minimum for central quantiles, not for mean errors.
            if len(target) >= 25:
                q16, q50, q84 = scan.arch.weighted_quantiles(
                    error / target, indices, weights
                ).T
                values["median_relative_percent"][:, si, mi] = 100 * q50
                values["central_width_percent"][:, si, mi] = 50 * (q84 - q16)
    return {
        "n_by_seed": counts,
        "metrics": {name: paired(v) for name, v in values.items()},
    }


def rn_diagnostics():
    deep_root = ROOT / "reports/smash_architecture_20261112/odd_result"
    rule_root = ROOT / "reports/rule_tof_odd_20261110"
    ds, rs = read(deep_root / "summary.json"), read(rule_root / "results.json")
    d = ledger(deep_root / "file_counts.npz", ds["file_counts_sha256"])
    r = ledger(rule_root / "counts.npz", rs["counts_sha256"])
    for key in ("content", "tags", "draws", "seeds"):
        np.testing.assert_array_equal(d[key], r[key])
    weights = np.vstack([np.ones(282, dtype=np.int64), d["draws"]])
    mi = list(d["models"]).index("student_dynedge")
    arm = [list(r["arms"]).index(name) for name in ("mc", "rules")]
    result = {}
    accepted = read(FINAL / "odd.json")
    for target, field in (
        ("raw70", "candidate_counts"),
        ("raw75_frozen", "alternative_candidate_counts"),
    ):
        bands = np.concatenate(
            [
                np.stack(
                    [r[k][..., arm] for k in ("soft", "hard", "overflow")], axis=-1
                ),
                d[field][:, :, mi : mi + 1, :3],
            ],
            axis=2,
        )
        for window in ("open_upper", "within_reach"):
            ratios = {}
            for tag in set(d["tags"]):
                use = d["tags"] == tag
                soft = np.tensordot(weights[:, use], bands[use, ..., 0], axes=1)
                hard_values = bands[..., 1] + (
                    bands[..., 2] if window == "open_upper" else 0
                )
                hard = np.tensordot(weights[:, use], hard_values[use], axes=1)
                ratios[tag] = scan.ratio(hard, soft)
            rows = {}
            for a, b in (("18", "0"), ("90", "0"), ("90", "18")):
                c = 100 * (
                    scan.ratio(ratios[scan.arch.TAGS[a]], ratios[scan.arch.TAGS[b]]) - 1
                )
                row = {
                    name: describe(c[:, :, i])
                    for i, name in enumerate(("mc", "rules", "gnn"))
                }
                row["gnn_minus_mc_pp"] = describe(c[:, :, 2] - c[:, :, 0])
                row["gnn_minus_rules_pp"] = describe(c[:, :, 2] - c[:, :, 1])
                old = accepted["cases"][target][window]["contrasts"][f"{a}/{b}"]
                for new, previous in (
                    ("mc", "mc"),
                    ("rules", "rules"),
                    ("gnn", "gnn"),
                    ("gnn_minus_mc_pp", "gnn_minus_mc"),
                    ("gnn_minus_rules_pp", "gnn_minus_rules"),
                ):
                    for field_name in ("value", "file_sd", "q16", "q84"):
                        scan.arch.require(
                            np.isclose(
                                row[new][field_name],
                                old[previous][field_name],
                                rtol=0,
                                atol=1e-12,
                            ),
                            "Odd Rn replay",
                        )
                rows[f"{a}/{b}"] = row
            result[f"{target}/{window}"] = rows
    return {
        "files": 282,
        "retrospective": True,
        "cases": result,
        "inputs_sha256": {
            "architecture_counts": ds["file_counts_sha256"],
            "rules_counts": rs["counts_sha256"],
            "accepted_odd": sha(FINAL / "odd.json"),
        },
    }


def summarize(output):
    start = monotonic()
    meta = read(output / "assembly.json")
    scan.arch.require(
        meta["source_sha256"] == sha(Path(__file__))
        and meta["protocol_sha256"] == sha(HERE / "protocol.md"),
        "Assembly source changed",
    )
    data = ledger(output / "data.npz", meta["data_sha256"])
    for helper, digest in meta["helper_sources_sha256"].items():
        scan.arch.require(sha(ROOT / helper) == digest, "Assembly helper changed")
    old, old_json, rapid, rapid_json, joint, joint_json = sources()
    np.testing.assert_array_equal(data["content"], old["content"])
    np.testing.assert_array_equal(data["draws"], old["draws"])
    weights = np.vstack([np.ones(286, dtype=np.int64), data["draws"]])
    samples = [data[f"sample_{i}"] for i in range(3)]
    file_index = [data[f"keys_{i}"][:, 0] for i in range(3)]
    results = {
        "scope": "retrospective geometric MC diagnostic at frozen raw70",
        "seeds": scan.arch.SEEDS,
        "methods": METHODS,
        "states": STATES,
        "y_edges": scan.EDGES.tolist(),
        "pT_break_GeV": scan.PT_BREAK,
        "files": 286,
        "events": 1180595,
        "bootstrap_draws": 400,
        "counts_sha256": meta["data_sha256"],
        "assembly_sha256": sha(output / "assembly.json"),
        "source_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(HERE / "protocol.md"),
        "previous_sha256": {
            "even": sha(FINAL / "even.json"),
            "rapidity": sha(RAPIDITY / "scan_results/scan.json"),
            "joint": sha(RAPIDITY / "joint_results_20261118/joint.json"),
        },
        "scopes": {},
    }
    for scope in ("pooled", *sorted(set(data["tags"]))):
        use = np.ones(286, bool) if scope == "pooled" else data["tags"] == scope
        total = np.tensordot(weights[:, use], data["counts"][use], axes=1)
        field = {name: total[..., i] for i, name in enumerate(FIELDS)}
        rates = {
            name: 100 * scan.ratio(field[num], field[den])
            for name, num, den in (
                ("direct_percent", "direct", "truth"),
                ("miss_percent", "miss", "truth"),
                ("split_percent", "split", "truth"),
                ("split_excess_edges_per_100_truth", "split_excess_edges", "truth"),
                ("merge_percent", "merge", "candidates"),
                ("mixed_merge_percent", "mixed_merge", "candidates"),
            )
        }
        multiplicity = {
            state: {
                "counts_by_seed_method": total[0, :, :, i].tolist(),
                "rates": {name: paired(v[..., i]) for name, v in rates.items()},
            }
            for i, state in enumerate(STATES)
        }
        sum_field = {name: v.sum(axis=-1) for name, v in field.items()}
        integrated = {
            name: paired(100 * scan.ratio(sum_field[num], sum_field[den]))
            for name, num, den in (
                ("direct_percent", "direct", "truth"),
                ("miss_percent", "miss", "truth"),
                ("split_percent", "split", "truth"),
                ("merge_percent", "merge", "candidates"),
            )
        }
        for new, previous in (
            ("direct_percent", "direct_coverage_percent"),
            ("miss_percent", "direct_miss_percent"),
            ("split_percent", "split_percent_of_truth"),
            ("merge_percent", "merge_percent_of_selected"),
        ):
            for method, key in (
                ("rules", "rules"),
                ("dynamic_edgeconv", "gnn"),
                ("gnn_minus_rules", "gnn_minus_rules"),
            ):
                for stat in ("value", "file_sd", "q16", "q84"):
                    scan.arch.require(
                        np.isclose(
                            integrated[new][method][stat],
                            old_json["scopes"][scope]["rates"][previous][key][stat],
                            atol=1e-12,
                            rtol=0,
                        ),
                        "Integrated multiplicity rate replay",
                    )
        base_masks = [use[fi] for fi in file_index]
        energy = {
            "integrated": energy_row((samples, file_index), base_masks, weights),
            "native_qualified": energy_row(
                (samples, file_index),
                [u & data[f"native_{i}"] for i, u in enumerate(base_masks)],
                weights,
            ),
            "same_across_seeds": energy_row(
                (samples, file_index),
                [u & data[f"stable_{i}"] for i, u in enumerate(base_masks)],
                weights,
            ),
        }
        energy["by_pT"] = [
            energy_row(
                (samples, file_index),
                [
                    u & ((s[:, 1] >= scan.PT_BREAK) == bool(pt))
                    for u, s in zip(base_masks, samples, strict=True)
                ],
                weights,
            )
            for pt in range(2)
        ]
        energy["by_y"] = []
        energy["by_y_pT"] = []
        for lo, hi in zip(scan.EDGES[:-1], scan.EDGES[1:], strict=True):
            y_masks = [
                u & (s[:, 0] >= lo) & (s[:, 0] < hi)
                for u, s in zip(base_masks, samples, strict=True)
            ]
            energy["by_y"].append(energy_row((samples, file_index), y_masks, weights))
            energy["by_y_pT"].append(
                [
                    energy_row(
                        (samples, file_index),
                        [
                            u & ((s[:, 1] >= scan.PT_BREAK) == bool(pt))
                            for u, s in zip(y_masks, samples, strict=True)
                        ],
                        weights,
                    )
                    for pt in range(2)
                ]
            )
        for projected in (
            energy["by_pT"],
            energy["by_y"],
            [cell for row in energy["by_y_pT"] for cell in row],
        ):
            np.testing.assert_array_equal(
                np.sum([row["n_by_seed"] for row in projected], axis=0),
                energy["integrated"]["n_by_seed"],
            )
        # Reproduce accepted integrated energy estimators and uncertainty.
        for metric in energy["integrated"]["metrics"]:
            old_name = {
                "median_relative_percent": "median_relative_error_percent",
                "central_width_percent": "central_half_width_percent",
            }.get(metric, metric)
            before = old_json["rules_energy"]["primary"][old_name]
            for name, key in (("rules", "rules"), ("dynamic_edgeconv", "gnn")):
                for stat in ("value", "file_sd", "q16", "q84"):
                    scan.arch.require(
                        np.isclose(
                            energy["integrated"]["metrics"][metric][name][stat],
                            before[key][stat],
                            atol=1e-9,
                            rtol=0,
                        ),
                        "Energy replay",
                    ) if scope == "pooled" else None
        yerr = np.tensordot(
            weights[:, use], rapid["errors"][use, :, :, 0].sum(axis=-1), axes=1
        )
        y_mae = paired(scan.ratio(yerr[..., 1], yerr[..., 0]))
        matrix = joint["joint_migration"][use, :, :, 0]
        n = np.tensordot(weights[:, use], matrix.sum(axis=(-4, -3, -2, -1)), axes=1)
        diagonal = np.trace(matrix[..., 0, 0, :, :], axis1=-2, axis2=-1) + np.trace(
            matrix[..., 1, 1, :, :], axis1=-2, axis2=-1
        )
        same_joint = paired(
            100 * scan.ratio(np.tensordot(weights[:, use], diagonal, axes=1), n)
        )
        for metric, old_metric, old_result in (
            (
                y_mae,
                "common_isolated_y_absolute_error_and_bias",
                rapid_json["scopes"][scope]["integrated"],
            ),
            (same_joint, "same_joint_bin_percent", joint_json["scopes"][scope]),
        ):
            for mi, name in enumerate(METHODS):
                for stat in ("value", "file_sd", "q16", "q84"):
                    expected = (
                        old_result[old_metric][stat][mi][0]
                        if old_metric.startswith("same_")
                        else old_result[old_metric][stat][mi][0][0]
                    )
                    scan.arch.require(
                        np.isclose(metric[name][stat], expected, rtol=0, atol=1e-12),
                        "Y/joint replay",
                    )
        results["scopes"][scope] = {
            "multiplicity": multiplicity,
            "integrated_rates": integrated,
            "energy": energy,
            "rapidity_mae": y_mae,
            "same_joint_bin_percent": same_joint,
        }
        print(f"Summarized {scope}", flush=True)
    results["rn_seed_stability"] = rn_diagnostics()
    results["elapsed_seconds"] = monotonic() - start
    results["checks"] = [
        "286 SHA-bound sidecars and association graphs",
        "exact occupancy count marginals",
        "common stable primary keys across seeds",
        "strict true-energy boundary reconciled",
        "per-file energy and y-error moments replay",
        "integrated rates and energy estimators replay",
        "y and joint-bin estimates replay",
        "all frozen odd Rn center/SD/quantile fields replay",
    ]
    (output / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    print(f"Summary complete in {results['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("assemble", "summarize"))
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results/rapidity_export_20261117/s286_4525207/accepted",
    )
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if args.stage == "assemble":
        assemble(args.input, args.output)
    else:
        summarize(args.output)
