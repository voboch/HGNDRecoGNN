"""Join accepted primary edges to the frozen S286 prediction tables."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIMARY = ROOT / "reports/primary_association_20261106"
RESPONSE = ROOT / "reports/response_diagnostics_20261106"
SEEDS = ("42", "123", "456")
ARMS = ("A1", "B1")
KEYS = ["Row", "ClusterID", "cl_istop"]
COLUMNS = KEYS + ["cl_label", "e_true", "cluster_size", "cl_score", "e_pred"]
ENERGY_BOUNDS = [0, 0.3, 1, 2]
BAND_STATES = [2, 4]  # inherited [0.3,1) and [2,infinity)


def digest(array, dtype):
    return sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def checked(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Input hash differs: {path}")
    return raw


def load(path):
    return json.loads(path.read_bytes())


def registry(contents, split, pins):
    records = {}
    for job in ("4466339", "4467086", "4467091"):
        base = ROOT / f"results/smash_train_20261028/evaluate_{job}"
        path = base / "evaluation_summary.json"
        summary = json.loads(checked(path, pins[str(path.relative_to(ROOT))]))
        assert summary["state"] == "complete" and not summary["test_threshold_tuning"]
        for content in set(summary["test_files"]) & contents.keys():
            assert content not in records
            assert (
                contents[content]["part"]
                in split["folds"][str(summary["fold"])]["test_parts"]
            )
            records[content] = (base / "extracted/evaluation", summary)
    assert set(records) == set(contents)
    return records


def edge_indices(edges, objects, cluster_keys):
    primary = objects.get_indexer(pd.MultiIndex.from_arrays(edges[:, :2].T))
    cluster = edges[:, 2]
    assert (primary >= 0).all() and (
        (0 <= cluster) & (cluster < len(cluster_keys))
    ).all()
    assert np.array_equal(edges[:, 0], cluster_keys[cluster, 0])
    assert len(np.unique(edges, axis=0)) == len(edges)
    return primary, cluster


def read_geometry(raw, record, native_reference):
    with np.load(BytesIO(raw)) as data:
        cluster_keys = np.column_stack(
            [data["cluster_row"], data["cluster_local_id"], data["cluster_side"]]
        )
        keys = pd.MultiIndex.from_arrays(cluster_keys.T)
        assert (
            keys.is_unique
            and digest(cluster_keys, "<i8") == native_reference["keys_sha256"]
        )
        row = data["Row"]
        objects = pd.MultiIndex.from_arrays([row, data["Id"]])
        assert objects.is_unique
        energy = data["Ekin"]
        assert np.isfinite(energy).all() and (energy >= 0).all()
        states = np.digitize(energy, ENERGY_BOUNDS)
        inside = data["in_rectangle"] & (row < record["processed_span"])
        observed_primary, observed_cluster = edge_indices(
            data["observed_edges"], objects, cluster_keys
        )
        native_primary, native_cluster = edge_indices(
            data["native_edges"], objects, cluster_keys
        )

        # Full observed direct graph, including exported out-of-window objects.
        primary_degree = np.bincount(observed_primary, minlength=len(objects))
        cluster_degree = np.bincount(observed_cluster, minlength=len(keys))
        observed_pairs = observed_primary * len(keys) + observed_cluster
        native_pairs = native_primary * len(keys) + native_cluster
        assert np.isin(native_pairs, observed_pairs).all()
        native_linked = np.bincount(native_primary, minlength=len(objects)) > 0
        assert np.array_equal(native_linked, data["qualifying_native"])
        assert np.all(energy[native_linked] > 0.3)

    eligible = inside & native_linked
    split = primary_degree > 1
    merge = np.zeros(len(objects), dtype=bool)
    np.logical_or.at(merge, native_primary, cluster_degree[native_cluster] > 1)
    unique_edge = eligible[native_primary] & (primary_degree[native_primary] == 1)
    unique_edge &= cluster_degree[native_cluster] == 1
    unique_primary = native_primary[unique_edge]
    unique_cluster = native_cluster[unique_edge]
    assert len(np.unique(unique_primary)) == len(unique_primary)
    assert len(np.unique(unique_cluster)) == len(unique_cluster)
    assert len(unique_primary) == np.count_nonzero(eligible & ~split & ~merge)
    coverage = np.stack(
        [
            np.bincount(states[mask], minlength=5)
            for mask in (
                inside,
                eligible,
                eligible & split & ~merge,
                eligible & ~split & merge,
                eligible & split & merge,
            )
        ]
    )
    isolated_counts = np.bincount(states[unique_primary], minlength=5)
    assert np.array_equal(coverage[1], coverage[2:].sum(axis=0) + isolated_counts)
    assert coverage[0].sum() == sum(record["hist"][1][1][0])
    assert (
        coverage[1].sum()
        == record["cardinality"]["processed_span"]["True_native"]["linked_objects"]
    )
    return (
        keys,
        states,
        energy,
        eligible,
        native_primary,
        native_cluster,
        unique_primary,
        unique_cluster,
        coverage,
    )


def predictions(base, summary, content, seed, arm, sources, keys, reference):
    entry = summary["predictions"][arm][seed][content]
    path = base / entry["path"]
    relative = str(path.relative_to(ROOT))
    assert entry["sha256"] == sources[relative]
    frame = pd.read_csv(BytesIO(checked(path, entry["sha256"])), usecols=COLUMNS)
    file_keys = pd.MultiIndex.from_frame(frame[KEYS])
    assert file_keys.is_unique and len(file_keys) == len(keys)
    order = file_keys.get_indexer(keys)
    assert (order >= 0).all(), (content, seed, arm)
    frame = frame.iloc[order]
    assert digest(frame[["cl_label", "e_true"]], "<f4") == reference["targets_sha256"]
    assert digest(frame["cluster_size"], "<i8") == reference["sizes_sha256"]
    assert frame.cl_label.isin([0, 1]).all() and frame.cl_score.between(0, 1).all()
    assert np.isfinite(frame[["e_true", "e_pred", "cl_score"]]).all(axis=None)
    return frame


def band_counts(energy, mask):
    return np.bincount(np.digitize(energy[mask], ENERGY_BOUNDS), minlength=5)[
        BAND_STATES
    ]


def main():
    start = monotonic()
    if (HERE / "counts.npz").exists():
        raise FileExistsError("Completed reduction already exists")
    pins = load(HERE / "input_pins.json")
    for path, expected in pins.items():
        checked(ROOT / path, expected)
    upstream = load(RESPONSE / "input_pins.json")
    split = json.loads(
        checked(ROOT / upstream["split"], upstream["files"][upstream["split"]])
    )
    contents = {
        group["tag"] + "/" + group["kept"]: group
        for group in split["groups"]
        if group["truth_bound"] and group["part"] in (0, 2, 4)
    }
    records = registry(contents, split, upstream["files"])
    reference = {
        r["content"]: r
        for r in load(ROOT / "reports/native_features_20261106/plan.json")["files"]
    }
    primary = {r["content"]: r for r in load(PRIMARY / "result/summary.json")["files"]}
    members = {
        r["member"]: r["sha256"]
        for r in load(PRIMARY / "export_4491592/manifest.json")["files"]
    }
    sources = load(RESPONSE / "results.json")["sources"]
    with np.load(RESPONSE / "counts.npz") as accepted:
        content_order = accepted["content"]
        weights = accepted["weights"]
        previous = accepted["ledger"]
    assert len(contents) == len(primary) == len(reference) == 286
    assert set(content_order) == set(contents) == set(primary) == set(reference)
    assert weights.shape == (400, 286)
    shape = (len(contents), len(SEEDS), len(ARMS))
    coverage = np.zeros((len(contents), 5, 5), dtype=np.int64)
    survival = np.zeros((*shape, 5), dtype=np.int64)
    # Same isolated primary population before score, then after score with true,
    # raw predicted and calibrated predicted energies.
    flow = np.zeros((*shape, 4, 5), dtype=np.int64)
    migration = np.zeros((*shape, 2, 5, 5), dtype=np.int64)
    # Residual sums by selection (own, common), energy (raw, calibrated),
    # and statistic (count, signed sum, absolute sum).
    errors = np.zeros((*shape, 2, 2, 3))
    candidates = 0
    for i, content in enumerate(content_order):
        member = f"{content}/neutrons.npz"
        raw = checked(PRIMARY / "result" / member, members[member])
        assert members[member] == primary[content]["files"]["neutrons.npz"]
        (
            keys,
            states,
            energy,
            eligible,
            native_primary,
            native_cluster,
            unique_primary,
            unique_cluster,
            coverage[i],
        ) = read_geometry(raw, primary[content], reference[content])
        candidates += len(keys)
        base, summary = records[content]
        assert summary["fold"] == reference[content]["fold"]
        true_state = states[unique_primary]
        for si, seed in enumerate(SEEDS):
            arms = []
            for ai, arm in enumerate(ARMS):
                raw_arm = arm[0] + "0"
                cal = predictions(
                    base, summary, content, seed, arm, sources, keys, reference[content]
                )
                raw = predictions(
                    base,
                    summary,
                    content,
                    seed,
                    raw_arm,
                    sources,
                    keys,
                    reference[content],
                )
                assert np.array_equal(raw.cl_score, cal.cl_score)
                score = cal.cl_score.to_numpy()
                cut = summary["training"][seed]["cuts"][arm]
                selected = score > cut
                positive = cal.cl_label.to_numpy() == 1
                assert positive[native_cluster].all()
                target = cal.e_true.to_numpy()
                raw_energy = raw.e_pred.to_numpy()
                cal_energy = cal.e_pred.to_numpy()
                replay = [
                    band_counts(target, positive),
                    band_counts(target, selected & positive),
                    band_counts(cal_energy, selected & positive),
                    band_counts(cal_energy, selected),
                    band_counts(raw_energy, selected & positive),
                    band_counts(raw_energy, selected),
                ]
                # Previous ledger orders its bands high then low.
                assert np.array_equal(np.asarray(replay)[:, ::-1], previous[i, si, ai])
                passed = np.zeros(len(energy), dtype=bool)
                np.logical_or.at(passed, native_primary, selected[native_cluster])
                survival[i, si, ai] = np.bincount(
                    states[eligible & passed], minlength=5
                )
                keep = selected[unique_cluster]
                raw_unique, cal_unique = (
                    raw_energy[unique_cluster],
                    cal_energy[unique_cluster],
                )
                flow[i, si, ai, 0] = np.bincount(true_state, minlength=5)
                flow[i, si, ai, 1] = np.bincount(true_state[keep], minlength=5)
                for ei, estimate in enumerate((raw_unique, cal_unique)):
                    predicted_state = np.digitize(estimate[keep], ENERGY_BOUNDS)
                    matrix = migration[i, si, ai, ei]
                    np.add.at(matrix, (true_state[keep], predicted_state), 1)
                    assert np.array_equal(matrix.sum(axis=1), flow[i, si, ai, 1])
                    flow[i, si, ai, 2 + ei] = matrix.sum(axis=0)
                arms.append((keep, raw_unique, cal_unique))
            common = arms[0][0] & arms[1][0]
            for ai, (keep, raw_unique, cal_unique) in enumerate(arms):
                for ci, mask in enumerate((keep, common)):
                    for ei, estimate in enumerate((raw_unique, cal_unique)):
                        residual = estimate[mask] - energy[unique_primary[mask]]
                        errors[i, si, ai, ci, ei] = [
                            len(residual),
                            residual.sum(),
                            np.abs(residual).sum(),
                        ]
        if (i + 1) % 50 == 0 or i + 1 == len(contents):
            print(f"Verified and joined {i + 1}/{len(contents)} files", flush=True)
    assert candidates == 1_798_225
    assert coverage[:, 0].sum() == 897_563 and coverage[:, 1].sum() == 190_002
    assert (survival <= coverage[:, None, None, 1, :]).all()
    assert np.array_equal(flow[..., 1, :], migration.sum(axis=-1)[..., 0, :])
    assert np.array_equal(errors[:, :, 0, 1, :, 0], errors[:, :, 1, 1, :, 0])
    np.savez_compressed(
        HERE / "counts.npz",
        content=content_order,
        weights=weights,
        coverage=coverage,
        survival=survival,
        flow=flow,
        migration=migration,
        errors=errors,
    )
    execution = {
        "state": "complete",
        "date": "2026-11-09",
        "files": len(contents),
        "candidates": candidates,
        "prediction_files": len(contents) * len(SEEDS) * 4,
        "elapsed_seconds": monotonic() - start,
        "new_jobs": [],
        "new_model_calls": 0,
        "input_pins_sha256": sha256(
            (HERE / "input_pins.json").read_bytes()
        ).hexdigest(),
        "protocol_sha256": sha256((HERE / "protocol.md").read_bytes()).hexdigest(),
        "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts_sha256": sha256((HERE / "counts.npz").read_bytes()).hexdigest(),
    }
    (HERE / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")


if __name__ == "__main__":
    main()
