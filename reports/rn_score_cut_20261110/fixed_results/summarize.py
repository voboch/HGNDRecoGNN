"""Summarize the fixed cut with paired widths and the even-only plateau."""

from io import BytesIO
import sys

import numpy as np

from reduce import CHOICE, HERE, SCAN, bands

sys.path.insert(0, str(HERE.parent))
from analyze import DRAWS, SEEDS, TAGS, load, read, summarize  # noqa: E402

FIXED_CHECK = "903126f85d1c06018fd05524c114eaa7b93d5ad0b192c4037b822835c8efa81c"
PAIRS = {"18/0": (1, 0), "90/0": (2, 0), "90/18": (2, 1)}
STAGES = ("T0", "T1", "T2", "C1", "C2", "C3", "C4")
STEPS = ("T2-T1", "C1-T2", "C2-C1", "C3-C2", "C4-C3")


def weights_for(parts, tags, split_replay=False):
    weights = np.zeros((DRAWS + 1, len(parts)), dtype=int)
    weights[0] = 1
    # The separate S/F results replay their original draw streams.
    # For the pooled result use one stream so strata do not share RNG blocks.
    strata = (
        (parts[parts % 2 == 0], parts[parts % 2 == 1]) if split_replay else (parts,)
    )
    for strata_parts in strata:
        rng = np.random.default_rng(20261106)
        for tag in TAGS:
            for part in sorted(set(strata_parts)):
                ix = np.flatnonzero((tags == tag) & (parts == part))
                assert len(ix)
                weights[1:, ix] = rng.multinomial(
                    len(ix), np.full(len(ix), 1 / len(ix)), DRAWS
                )
    return weights


def totals(weights, tags, x):
    # Each tag gets a separate sum before forming any ratio.
    return np.stack(
        [np.tensordot(weights[:, tags == tag], x[tags == tag], axes=1) for tag in TAGS],
        axis=1,
    )


def ratio(x):
    assert (x > 0).all()
    return x[..., 0] / x[..., 1]


def stat(draws):
    lo, hi = np.quantile(draws[1:], [0.16, 0.84])
    return {
        "value": float(draws[0]),
        "file_sd": float(draws[1:].std(ddof=1)),
        "q16": float(lo),
        "q84": float(hi),
        "half_width_68": float((hi - lo) / 2),
    }


def repeat_stat(draws):
    assert draws.shape == (DRAWS + 1, 3)
    mean = stat(draws.mean(axis=1))
    mean["repeat_range"] = [float(draws[0].min()), float(draws[0].max())]
    mean["by_repeat"] = {s: stat(v) for s, v in zip(SEEDS, draws.T)}
    return mean


def policy_stats(draws, reference=None):
    out = {
        "purity70": repeat_stat(draws[..., 0]),
        "purity90": repeat_stat(draws[..., 1]),
        "paired_change": repeat_stat(draws[..., 1] - draws[..., 0]),
    }
    if reference is not None:
        out["purity70_minus_truth"] = repeat_stat(draws[..., 0] - reference)
        out["purity90_minus_truth"] = repeat_stat(draws[..., 1] - reference)
    return out


def cluster_factor(tagcounts, gram, n_events, n_candidates):
    # Delta-method variance for log(H/L), treating events or candidates as units.
    # The resulting file bootstrap below retains both event and file correlations.
    counts = tagcounts.reshape(len(TAGS), 3, 4)
    terms = np.zeros((len(TAGS), 3, 2))
    for t in range(len(TAGS)):
        for s in range(3):
            for j in range(2):
                h, low = counts[t, s, 2 * j : 2 * j + 2]
                g = np.array([1 / h, -1 / low])
                m = gram[t, s, :, 2 * j : 2 * j + 2, 2 * j : 2 * j + 2]
                n = np.array([n_events[t], n_candidates[t, s]])
                variance = (m @ g @ g) * n / (n - 1)
                terms[t, s, j] = np.sqrt(variance[0] / variance[1])
    return terms.tolist()


def describe(weights, tags, t, counts, obj, dcm, gram, spans):
    truth = totals(weights, tags, t)
    h = totals(weights, tags, counts)
    objects = totals(weights, tags, obj)
    dcm = ratio(totals(weights, tags, dcm))
    ne = totals(weights, tags, spans)
    rn = ratio(h)
    mc = ratio(truth)
    nref = np.broadcast_to(mc[:, :, None, 1], rn.shape[:3])
    pref = np.broadcast_to(mc[:, :, None, 3], rn.shape[:3])
    by_scope = {
        "n_files_by_potential": [int((tags == tag).sum()) for tag in TAGS],
        "processed_events_by_potential": ne[0].tolist(),
        "counts_by_potential_repeat_policy_hi_low": h[0].tolist(),
        "truth_counts_by_potential": truth[0].tolist(),
        "objects_by_potential_repeat_policy": objects[0].tolist(),
        "hardness_by_potential": {},
        "contrasts": {},
    }
    by_scope["event_to_candidate_sd_ratio"] = cluster_factor(
        h[0, :, :, :, 5],
        totals(weights[:1], tags, gram)[0],
        ne[0],
        objects[0, :, :, 0, 0],
    )
    for i, tag in enumerate(TAGS):
        c = objects[:, i]
        assert (c[..., [1, 2]] > 0).all()
        by_scope["hardness_by_potential"][tag] = {
            "truth_neutron": stat(mc[:, i, 1]),
            "truth_proton": stat(mc[:, i, 3]),
            "calibrated_reconstructed": policy_stats(rn[:, i, :, :, 5], nref[:, i]),
            "raw_reconstructed": policy_stats(rn[:, i, :, :, 4], nref[:, i]),
            "hybrid_ideal_proton": policy_stats(
                rn[:, i, :, :, 5] / pref[:, i, :, None]
            ),
            "purity_percent": policy_stats(100 * c[..., 3] / c[..., 2]),
            "recall_percent": policy_stats(100 * c[..., 3] / c[..., 1]),
        }
    for pair, (hi, lo) in PAIRS.items():
        delta = lambda x: 100 * (x[:, hi] / x[:, lo] - 1)  # noqa: E731
        nt = delta(nref)
        ntg = delta(mc)
        ideal = delta(nref / pref)
        nd = delta(rn)
        hybrid = delta(rn[..., 5] / pref[..., None])
        assert np.allclose(
            (1 + hybrid / 100) / (1 + ideal[..., None] / 100),
            (1 + nd[..., 5] / 100) / (1 + nt[..., None] / 100),
            rtol=1e-14,
        )
        row = {
            "truth_neutron": stat(nt[:, 0]),
            "truth_proton": stat(ntg[:, 3]),
            "truth_ideal_np": stat(ideal[:, 0]),
            "hybrid_ideal_proton": policy_stats(hybrid, ideal),
            "dcm_original": repeat_stat(delta(dcm)[..., 3]),
            "calibrated_ladder": {},
            "raw_ladder": {},
            "calibrated_steps": {},
        }
        for view, indices in (
            ("calibrated_ladder", (0, 1, 3, 5)),
            ("raw_ladder", (0, 1, 2, 4)),
        ):
            ladder = {
                name: np.broadcast_to(ntg[:, None, None, i], rn.shape[:1] + (3, 2))
                for i, name in enumerate(STAGES[:3])
            }
            ladder.update({name: nd[..., i] for name, i in zip(STAGES[3:], indices)})
            row[view] = {name: policy_stats(d, nt) for name, d in ladder.items()}
            if view == "calibrated_ladder":
                for step in STEPS:
                    a, b = step.split("-")
                    row["calibrated_steps"][step] = policy_stats(ladder[a] - ladder[b])
        by_scope["contrasts"][pair] = row
    return by_scope


def plateau(ids, parts, tags, truth):
    groups = {k: {"part": int(p), "tag": t} for k, p, t in zip(ids, parts, tags)}
    with np.load(BytesIO(read(HERE.parent / "counts.npz", SCAN))) as saved:
        even_ids = saved["selection_ids"]
        even_data = saved["selection"]
    use = np.isin(ids, even_ids)
    assert np.array_equal(ids[use], even_ids)
    delta, _, _ = summarize(even_ids, even_data, truth[use, 1], groups)
    choice = load(HERE.parent / "choice.json", CHOICE)
    mean = delta.mean(axis=1)
    best = next(
        i for i, r in enumerate(choice["scan"]) if r["purity_target_percent"] == 90
    )
    rows = []
    for i, r in enumerate(choice["scan"]):
        d = stat(mean[:, i])
        assert np.isclose(r["contrast"]["value"], d["value"], atol=1e-12, rtol=0)
        assert np.isclose(r["contrast"]["file_sd"], d["file_sd"], atol=1e-12, rtol=0)
        gap = stat(mean[:, best] - mean[:, i])
        rows.append(
            {
                "target": r["purity_target_percent"],
                "eligible": r["eligible"],
                "contrast": d,
                "best_minus_policy": gap,
                "contrast_over_file_sd": d["value"] / d["file_sd"],
                "one_sd_plateau": bool(
                    r["eligible"] and gap["value"] <= gap["file_sd"]
                ),
            }
        )
    return rows


def main():
    with np.load(HERE / "files.npz") as z:
        data = dict(z)
    parts, tags, folds = (data[k] for k in ["parts", "tags", "folds"])
    weights = weights_for(parts, tags, split_replay=True)
    pooled_weights = weights_for(parts, tags)
    inputs = [
        data["truth"],
        bands(data["spectra"]),
        data["objects"],
        data["dcm"],
        data["gram"],
        data["spans"],
    ]
    scopes = {
        "odd_fixed_check": parts % 2 == 1,
        "even_selection": parts % 2 == 0,
        "pooled_descriptive": np.ones(len(parts), bool),
    }
    scopes.update(
        {
            f"{sample}_fold{fold}": use & (folds == fold)
            for sample, use in list(scopes.items())
            for fold in range(3)
        }
    )
    out = {}
    for label, use in scopes.items():
        draws = pooled_weights if label.startswith("pooled_") else weights
        out[label] = describe(draws[:, use], tags[use], *(x[use] for x in inputs))
    choice = load(HERE.parent / "choice.json", CHOICE)
    fixed = load(HERE.parent / "fixed_check.json", FIXED_CHECK)
    for scope, old in (
        ("odd_fixed_check", fixed["points"]),
        ("even_selection", [choice["scan"][2], choice["scan"][6]]),
    ):
        c4 = out[scope]["contrasts"]["90/0"]["calibrated_ladder"]["C4"]
        for ref, key in zip(old, ("purity70", "purity90")):
            for key_stat in ("value", "file_sd"):
                assert np.isclose(
                    c4[key][key_stat], ref["contrast"][key_stat], rtol=0, atol=1e-12
                )
                assert np.isclose(
                    c4["paired_change"][key_stat],
                    old[-1]["change"][key_stat],
                    rtol=0,
                    atol=1e-12,
                )
    # Keep the source report directory intact.
    import json

    result = {
        "seeds": SEEDS,
        "potentials_MeV": [0, 18, 90],
        "draws": DRAWS,
        "selection_plateau": plateau(data["ids"], parts, tags, data["truth"]),
        "scopes": out,
    }
    (HERE / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(
        HERE / "draw_weights.npz",
        ids=data["ids"],
        weights=weights[1:],
        pooled_weights=pooled_weights[1:],
    )


if __name__ == "__main__":
    main()
