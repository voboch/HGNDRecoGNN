"""Post-hoc composition controls on certified exact-0.3-GeV counters.

Five equal-potential-weight quantile classes, with integer ties intact. The
estimator is MC-truth midrapidity nucleon multiplicity strictly outside the
rectangle. It is not experimental centrality. Weights and quantiles are rebuilt
inside every whole-file bootstrap replicate; no threshold is optimised.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.analysis.truth_provenance import load_certified
from HGNDRecoGNN.analysis.truth_reach import COLUMNS, Y_CM
from HGNDRecoGNN.analysis.truth_statistics import TAGS, values
from HGNDRecoGNN.scripts.content_census import sha256


def classes(hist):
    density = hist[:, :, 0] / hist[:, :, 0].sum(axis=1, keepdims=True)
    cumulative = density.mean(axis=0).cumsum()
    # Five is a descriptive resolution/cost convention, not an optimised cut.
    cuts = np.searchsorted(cumulative, [0.2, 0.4, 0.6, 0.8]) + 1
    k = 4 - np.searchsorted(cuts, np.arange(hist.shape[1]), side="right")
    if len(set(cuts)) != 4:
        raise ValueError("Proxy ties leave empty quantile classes")
    out = np.stack([hist[:, k == i].sum(axis=1) for i in range(5)], axis=1)
    if (out[:, :, 0] <= 0).any():
        raise ValueError("Empty proxy class")
    return out, cuts.tolist()


def reweighted(per_class):
    density = per_class[..., 0] / per_class[..., 0].sum(axis=-1, keepdims=True)
    if (density <= 0).any():
        raise ValueError("Empty class in composition matching")
    weights = density.mean(axis=-2, keepdims=True) / density
    return (per_class[..., 1:] * weights[..., None]).sum(axis=-2)


def run(root, output):
    output.mkdir(exist_ok=False, parents=True)
    summary = json.loads((root / "summary.json").read_text())
    all_events = {}
    names = {}
    bins = 0
    autocorr = {}
    filedata = {}
    for tag in TAGS:
        ev, _, proof = load_certified(root / tag, "acceptance")
        if (
            sha256(root / tag / "proof.json") != summary["samples"][tag]["proof_sha256"]
            or proof["effective_recipe"]["reach_floor_GeV"] != 0.3
        ):
            raise ValueError("Controls require the bound exact reach aggregate")
        all_events[tag] = ev
        names[tag] = sorted(ev.index.get_level_values("member").unique())
        bins = max(bins, int(ev.ref_mult.max()) + 1)
        autocorr[tag] = {}
        for lag in (1, 5, 10):
            num = den = 0.0
            for _, g in ev.groupby(level="member"):
                x = g.ref_mult.to_numpy(dtype=float)
                x -= x.mean()
                num += float(x[lag:] @ x[:-lag])
                den += float(x @ x)
            autocorr[tag][str(lag)] = num / den
    hists = {}
    boots = {}
    rng = np.random.default_rng(20261025)
    for tag in TAGS:
        ev = all_events[tag]
        h = np.zeros((len(names[tag]), bins, 7), dtype=np.int64)
        for i, member in enumerate(names[tag]):
            g = ev.xs(member, level="member")
            proxy = g.ref_mult.to_numpy(dtype=np.int64)
            h[i, :, 0] = np.bincount(proxy, minlength=bins)
            for j, c in enumerate(COLUMNS):
                h[i, :, j + 1] = np.bincount(
                    proxy, weights=g[c], minlength=bins
                ).astype(np.int64)
        hists[tag] = h
        w = rng.multinomial(len(h), np.full(len(h), 1 / len(h)), 400)
        boots[tag] = np.tensordot(w, h, axes=(1, 0))
    central = np.stack([hists[t].sum(axis=0) for t in TAGS])
    cls, cuts = classes(central)
    bootclasses = np.array(
        [classes(np.stack([boots[t][i] for t in TAGS]))[0] for i in range(400)]
    )
    byclass = values(cls[:, :, 1:])
    byclassboot = values(bootclasses[:, :, :, 1:])
    ec = byclass[2] / byclass[0] - 1
    eb = byclassboot[:, 2] / byclassboot[:, 0] - 1
    raw = values(central[:, :, 1:].sum(axis=1))
    rawboot = values(np.stack([boots[t][:, :, 1:].sum(axis=1) for t in TAGS], axis=1))
    wc = values(reweighted(cls))
    wb = values(reweighted(bootclasses))
    shift = (wc[2] / wc[0] - raw[2] / raw[0]) * 100
    shiftboot = (wb[:, 2] / wb[:, 0] - rawboot[:, 2] / rawboot[:, 0]) * 100
    fractions = cls[:, :, 0] / cls[:, :, 0].sum(axis=1, keepdims=True)
    result = {
        "schema": "truth-reach-controls-v1",
        "aggregate_summary_sha256": sha256(root / "summary.json"),
        "code_sha256": sha256(__file__),
        "estimator": {
            "definition": "N(n,p), |y-y_beam/2|<0.5, strictly outside rectangle",
            "cm_rapidity": float(Y_CM),
            "window_overlap": 0,
            "quantile_cuts_increasing": cuts,
            "class_fractions": dict(zip(TAGS, fractions.tolist())),
            "autocorrelation_within_file": autocorr,
        },
        "classes": {
            "labels": ["0–20", "20–40", "40–60", "60–80", "80–100"],
            "events": cls[:, :, 0].astype(int).tolist(),
            "endpoint_percent": (ec * 100).tolist(),
            "sd_percent": (eb.std(axis=0, ddof=1) * 100).tolist(),
        },
        "reweighting": {
            "n_boot": 400,
            "seed": 20261025,
            "shift_percent": shift.tolist(),
            "shift_paired_sd_percent": shiftboot.std(axis=0, ddof=1).tolist(),
            "policy": "Class thresholds and common class composition refitted in every replicate",
        },
        "central_classes_counts": cls.tolist(),
        "bootstrap_class_counts": bootclasses.tolist(),
        "bootstrap_weighted_counts": reweighted(bootclasses).tolist(),
        "bootstrap_unweighted_counts": np.stack(
            [boots[t][:, :, 1:].sum(axis=1) for t in TAGS], axis=1
        ).tolist(),
        "scope": "Post-hoc MC truth diagnostic. Not experimental centrality; matching this proxy cannot exclude other composition differences.",
    }
    (output / "controls.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    # Compact per-file class arrays use the nominal shared class boundaries.
    kval = 4 - np.searchsorted(cuts, np.arange(bins), side="right")
    for tag in TAGS:
        h = hists[tag]
        filedata[tag] = {
            "members": names[tag],
            "classes": np.stack(
                [h[:, kval == i].sum(axis=1) for i in range(5)], axis=1
            ).tolist(),
        }
    (output / "class_files.json").write_text(
        json.dumps(filedata, separators=(",", ":")) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.output)
