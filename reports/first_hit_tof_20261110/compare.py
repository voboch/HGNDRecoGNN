"""Compare first-hit ToF and saved network energies on identical primary links."""

from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "reports/primary_prediction_response_20261109"
sys.path.insert(0, str(PREVIOUS))
from analyze import (  # noqa: E402
    ARMS,
    PRIMARY,
    RESPONSE,
    SEEDS,
    checked,
    digest,
    load,
    predictions,
    read_geometry,
    registry,
    band_counts,
)

METHODS = [
    "Original first-hit ToF",
    "Jittered first-hit ToF",
    "DCM-trained raw",
    "DCM-trained calibrated",
    "SMASH-trained raw",
    "SMASH-trained calibrated",
]
TAGS = ["zeroSpot", "defaultSpot", "bigSpot"]
PAIR_INDEX = {"18/0": (1, 0), "90/0": (2, 0), "90/18": (2, 1)}


def tof(times, length):
    positive = times > 0
    beta = np.divide(
        length, 0.299792458 * times, out=np.zeros_like(times), where=positive
    )
    valid = positive & (beta > 0) & (beta < 1)
    energy = np.full(len(times), np.nan)
    # Stable evaluation of gamma-1 with the production neutron mass in GeV.
    gamma = 1 / np.sqrt((1 - beta[valid]) * (1 + beta[valid]))
    energy[valid] = 0.93955 * gamma**2 * beta[valid] ** 2 / (gamma + 1)
    assert np.isfinite(energy[valid]).all()
    return energy, valid, ~positive, positive & ~valid


def stats(value, replicas):
    return {"value": value.tolist(), "file_sd": replicas.std(axis=0, ddof=1).tolist()}


def main():
    if (HERE / "results.json").exists():
        raise FileExistsError("Existing result")
    primary_acceptance = load(PRIMARY / "acceptance.json")
    pins = load(PREVIOUS / "input_pins.json")
    for path, expected in pins.items():
        checked(ROOT / path, expected)
    upstream = load(RESPONSE / "input_pins.json")
    split = json.loads(
        checked(ROOT / upstream["split"], upstream["files"][upstream["split"]])
    )
    contents = {
        g["tag"] + "/" + g["kept"]: g
        for g in split["groups"]
        if g["truth_bound"] and g["part"] in (0, 2, 4)
    }
    pred_registry = registry(contents, split, upstream["files"])
    source = load(RESPONSE / "results.json")["sources"]
    summary = json.loads(
        checked(PRIMARY / "result/summary.json", primary_acceptance["summary_sha256"])
    )
    records = {r["content"]: r for r in summary["files"]}
    native = {
        r["content"]: r
        for r in load(ROOT / "reports/native_features_20261106/plan.json")["files"]
    }
    plan = {r["content"]: r for r in load(HERE / "plan.json")["files"]}
    export = load(HERE / "export/summary.json")
    assert (
        export["state"] == "complete"
        and export["files"] == 286
        and export["isolated"] == 142_912
    )
    assert export["native_replay"] == "exact"
    raw = checked(HERE / "export/first_hits.npz", export["first_hits_sha256"])
    selection = checked(HERE / "selection.npz", export["selection_sha256"])
    checked(HERE / "extract.py", export["extractor_sha256"])
    checked(HERE / "plan.json", export["plan_sha256"])
    earlier = load(PREVIOUS / "execution.json")
    previous = checked(PREVIOUS / "counts.npz", earlier["counts_sha256"])
    with np.load(BytesIO(previous)) as p:
        ids, weights, old = p["content"], p["weights"], p["errors"]
    counts = np.zeros((len(ids), 3, 8), dtype=np.int64)
    moments = np.zeros((len(ids), 3, 6, 2))
    bands = np.zeros((len(ids), 3, 7, 2), dtype=np.int64)
    med_inputs = [[[] for _ in SEEDS] for _ in TAGS]
    with np.load(BytesIO(raw)) as hit, np.load(BytesIO(selection)) as selected_clusters:
        assert (
            set(ids)
            == set(hit.files)
            == set(selected_clusters.files)
            == set(plan)
            == set(contents)
        )
        for i, cid in enumerate(ids):
            r = records[cid]
            data = checked(
                PRIMARY / "result" / cid / "neutrons.npz", r["files"]["neutrons.npz"]
            )
            keys, _, truth, _, _, _, p, cluster, _ = read_geometry(data, r, native[cid])
            assert np.array_equal(cluster, selected_clusters[cid])
            target = truth[p]
            assert digest(target, "<f8") == plan[cid]["isolated_truth_sha256"]
            t, tj, length = hit[cid].T
            assert len(t) == len(cluster) and (length > 0).all()
            raw_tof, raw_ok, raw_bad_t, raw_bad_beta = tof(t, length)
            jitter, jit_ok, jit_bad_t, jit_bad_beta = tof(tj, length)
            base, evaluation = pred_registry[cid]
            tag_idx = TAGS.index(cid.split("/")[0])
            for si, seed in enumerate(SEEDS):
                common = np.ones(len(cluster), dtype=bool)
                energies = [raw_tof, jitter]
                for ai, arm in enumerate(ARMS):
                    cal = predictions(
                        base, evaluation, cid, seed, arm, source, keys, native[cid]
                    )
                    raw_pred = predictions(
                        base,
                        evaluation,
                        cid,
                        seed,
                        arm[0] + "0",
                        source,
                        keys,
                        native[cid],
                    )
                    assert np.array_equal(raw_pred.cl_score, cal.cl_score)
                    cut = evaluation["training"][seed]["cuts"][arm]
                    common &= cal.cl_score.to_numpy()[cluster] > cut
                    energies.extend(
                        [
                            raw_pred.e_pred.to_numpy()[cluster],
                            cal.e_pred.to_numpy()[cluster],
                        ]
                    )
                assert common.sum() == old[i, si, 0, 1, 0, 0] == old[i, si, 1, 1, 0, 0]
                for ai in range(2):
                    for j in range(2):
                        residual = energies[2 + ai * 2 + j][common] - target[common]
                        assert np.allclose(
                            [len(residual), residual.sum(), np.abs(residual).sum()],
                            old[i, si, ai, 1, j],
                            rtol=1e-12,
                            atol=1e-12,
                        )
                valid = common & raw_ok & jit_ok
                counts[i, si] = [
                    common.sum(),
                    valid.sum(),
                    (common & raw_bad_t).sum(),
                    (common & raw_bad_beta).sum(),
                    (common & jit_bad_t).sum(),
                    (common & jit_bad_beta).sum(),
                    (valid & (raw_tof > 10)).sum(),
                    (valid & (jitter > 10)).sum(),
                ]
                energies = np.stack(energies)
                residual = energies[:, valid] - target[valid]
                moments[i, si, :, 0] = residual.sum(axis=1)
                moments[i, si, :, 1] = np.abs(residual).sum(axis=1)
                med_inputs[tag_idx][si].append(np.abs(residual))
                bands[i, si, 0] = band_counts(target, valid)
                for m, estimate in enumerate(energies):
                    bands[i, si, m + 1] = band_counts(estimate, valid)
    assert (counts[..., 1] <= counts[..., 0]).all()
    groups = [np.char.startswith(ids, tag + "/") for tag in TAGS]
    stats_by_group, all_bands, all_band_draws = {}, [], []
    for tag, mask in zip(TAGS + ["pooled"], groups + [np.ones(len(ids), bool)]):
        c = counts[mask].sum(axis=0)
        draws = weights[:, mask] @ counts[mask, :, 1]
        assert (c[:, 1] > 0).all() and (draws > 0).all()
        metrics = 1000 * moments[mask].sum(axis=0) / c[:, 1, None, None]
        replicas = (
            1000
            * np.einsum("df,fsmt->dsmt", weights[:, mask], moments[mask])
            / draws[..., None, None]
        )
        tag_idx = TAGS.index(tag) if tag in TAGS else None
        if tag_idx is None:
            abs_err = [
                sum([med_inputs[j][si] for j in range(3)], []) for si in range(3)
            ]
        else:
            abs_err = med_inputs[tag_idx]
        median = np.stack(
            [
                1000 * np.median(np.concatenate(parts, axis=1), axis=1)
                for parts in abs_err
            ]
        )
        stats_by_group[tag] = {
            "counts": c.tolist(),
            "bias_MAE_MeV": stats(metrics, replicas),
            "median_absolute_error_MeV": median.tolist(),
            "MAE_minus_original_ToF_MeV": stats(
                metrics[:, :, 1] - metrics[:, :1, 1],
                replicas[..., 1] - replicas[:, :, :1, 1],
            ),
        }
        if tag_idx is not None:
            all_bands.append(bands[mask].sum(axis=0))
            all_band_draws.append(
                np.einsum("df,fsmb->dsmb", weights[:, mask], bands[mask])
            )
    b, bd = np.stack(all_bands), np.stack(all_band_draws, axis=1)
    assert (b[..., 0] > 0).all() and (bd[..., 0] > 0).all()
    ratios, ratio_draws = b[..., 1] / b[..., 0], bd[..., 1] / bd[..., 0]
    if not (ratios > 0).all() or not (ratio_draws > 0).all():
        raise ValueError("Undefined contrast against a zero hardness ratio")
    contrasts = {}
    for pair, (a, c) in PAIR_INDEX.items():
        k = 100 * (ratios[a] / ratios[c] - 1)
        kd = 100 * (ratio_draws[:, a] / ratio_draws[:, c] - 1)
        contrasts[pair] = {
            "contrast_percent": stats(k, kd),
            "shift_from_matched_truth_pp": stats(k - k[:, :1], kd - kd[:, :, :1]),
        }
    result = {
        "methods": METHODS,
        "seeds": SEEDS,
        "count_order": [
            "both score cuts",
            "both ToF times valid",
            "original t <= 0",
            "original nonphysical beta at t > 0",
            "jittered t <= 0",
            "jittered nonphysical beta at t > 0",
            "valid original E > 10",
            "valid jittered E > 10",
        ],
        "stats": stats_by_group,
        "band_order": ["low", "high"],
        "contrast_method_order": ["MC primary truth", *METHODS],
        "band_counts": b.tolist(),
        "contrasts": contrasts,
        "n_files": len(ids),
        "shared_file_draws": len(weights),
        "source_job": load(HERE / "submission.json")["job_id"],
        "scope": "same isolated links, selected by both original cuts, with both ToF times physical",
    }
    np.savez_compressed(
        HERE / "counts.npz",
        content=ids,
        weights=weights,
        counts=counts,
        moments=moments,
        bands=bands,
    )
    (HERE / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
