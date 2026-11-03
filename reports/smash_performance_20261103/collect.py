"""Reduce accepted prediction CSVs to report counts; no model, fit or cut choice."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAGS = ("zeroSpot", "defaultSpot", "bigSpot")
SEEDS = ("42", "123", "456")
ARMS = ("A1", "B0", "B1")
E = np.array([.1, .3, .5, .7, 1., 1.4, 1.8, 2.2, 2.6, 3., 3.5, 4., 5., 6.5])
SPECTRUM_EDGES = np.r_[-np.inf, 0., E, np.inf]
# Same 401-grid as the accepted validation protocol; only displayed, not optimised.
THRESHOLDS = np.linspace(.001, .999, 401)
SCORE_EDGES = np.linspace(0, 1, 51)  # 0.02 display bins, not a new working-point scan.
SIZE = np.array([0, 1, 3, 7, np.inf])  # Accepted cluster-size categories.
COLUMNS = ["Row", "ClusterID", "cl_istop", "cl_label", "e_true", "cluster_size", "cl_score", "e_pred"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histogram(values, edges=SPECTRUM_EDGES):
    assert np.isfinite(values).all()
    h = np.histogram(values, edges)[0]
    assert h.sum() == len(values)
    return h


def collect():
    pins = json.loads((HERE / "input_pins.json").read_text())
    for p, h in pins["files"].items():
        assert sha(ROOT / p) == h, p
    analysis = json.loads((ROOT / "results/smash_train_20261028/analysis/summary.json").read_text())
    sources, records = {}, []
    for job in ("4466339", "4467086", "4467091"):
        directory = ROOT / f"results/smash_train_20261028/evaluate_{job}"
        summary = json.loads((directory / "evaluation_summary.json").read_text())
        assert summary["state"] == "complete" and not summary["test_threshold_tuning"]
        for cid in summary["test_files"]:
            assert cid not in summary["empty_after_cuts"]
            records.append((cid, summary["fold"], directory, summary))
    records.sort(key=lambda x: x[0])
    assert len(records) == 568 and len({x[0] for x in records}) == 568
    data = {f"{a}__{s}": {} for a in ARMS for s in SEEDS}
    metadata = {"source_sha256": pins["files"], "content": [x[0] for x in records],
                "tag": [x[0].split('/')[0] for x in records], "fold": [x[1] for x in records],
                "spectra_edges_GeV": [None, 0., *E.tolist(), None], "energy_edges_GeV": E.tolist(),
                "spectrum_tail_convention": "first bin E<0, last bin E>=6.5; never discarded",
                "score_edges": SCORE_EDGES.tolist(), "thresholds": THRESHOLDS.tolist(),
                "csv_hashes": {}, "cuts": {}, "kind": "accepted-test-prediction-report-export"}
    for pos, (cid, fold, directory, summary) in enumerate(records):
        reference = None
        for arm in ARMS:
            for seed in SEEDS:
                entry = summary["predictions"][arm][seed][cid]
                p = directory / "extracted/evaluation" / entry["path"]
                raw = p.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                assert digest == entry["sha256"], p
                metadata["csv_hashes"][str(p.relative_to(ROOT))] = digest
                f = pd.read_csv(BytesIO(raw), usecols=COLUMNS)
                keys = f[COLUMNS[:6]].to_numpy()
                if reference is None:
                    reference = keys
                else:
                    assert np.array_equal(reference, keys), (cid, arm, seed)
                label = f.cl_label.to_numpy() == 1
                et, ep, score = (f[k].to_numpy(float) for k in ("e_true", "e_pred", "cl_score"))
                assert np.isfinite(et).all() and np.isfinite(ep).all() and np.isfinite(score).all()
                assert ((score >= 0) & (score <= 1)).all()
                cut = summary["training"][seed]["cuts"][arm]
                sel = score > cut
                d = data[f"{arm}__{seed}"]
                metadata["cuts"].setdefault(arm, {}).setdefault(seed, {})[str(fold)] = cut
                row = {"n": len(f), "n_pos": int(label.sum()), "n_sel": int(sel.sum()),
                       "n_selpos": int((sel & label).sum()), "represented_events": f.Row.nunique(),
                       "T": histogram(et[label]), "P": histogram(ep[label]),
                       "C2": histogram(et[sel & label]), "C3": histogram(ep[sel & label]),
                       "C4": histogram(ep[sel]),
                       "score_pos": histogram(score[label], SCORE_EDGES),
                       "score_neg": histogram(score[~label], SCORE_EDGES),
                       "TP": len(score[label]) - np.searchsorted(np.sort(score[label]), THRESHOLDS, side="right"),
                       "FP": len(score[~label]) - np.searchsorted(np.sort(score[~label]), THRESHOLDS, side="right")}
                # Histograms include negative and overflow predictions; a background
                # e_true=0 is never interpreted as physical zero energy.
                assert row["T"].sum() == row["P"].sum() == row["n_pos"]
                assert row["C2"].sum() == row["C3"].sum() == row["n_selpos"]
                assert row["C4"].sum() == row["n_sel"]
                m = label & (et >= E[0]) & (et < E[-1]) & (ep >= E[0]) & (ep < E[-1])
                row["migration"] = np.histogram2d(et[m], ep[m], bins=(E, E))[0].astype(np.int64)
                row["true_energy_n"] = histogram(et[label])[2:-1]
                assert row["true_energy_n"].shape == (len(E) - 1,)
                size_rows = []
                for lo, hi in zip(SIZE[:-1], SIZE[1:]):
                    m = (f.cluster_size.to_numpy() > lo) & (f.cluster_size.to_numpy() <= hi)
                    size_rows.append([int(m.sum()), int((m & label).sum()), int((m & sel).sum()), int((m & sel & label).sum())])
                row["size_counts"] = np.array(size_rows)
                for k, v in row.items():
                    d.setdefault(k, []).append(v)
        if pos % 50 == 0 or pos == len(records)-1:
            print(f"Accepted CSVs reduced: {pos+1}/{len(records)} content groups", flush=True)
    arrays = {f"{key}__{field}": np.asarray(values) for key, d in data.items() for field, values in d.items()}
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}__{seed}"
            for fold in range(3):
                ix = np.array(metadata["fold"]) == fold
                old = analysis["arms"][arm][seed]["classification"]["pooled"][f"fold_{fold}"]["all"]
                counts = {field: int(arrays[f"{key}__{name}"][ix].sum()) for field, name in
                          (("n", "n"), ("signal", "n_pos"), ("selected", "n_sel"), ("selected_signal", "n_selpos"))}
                assert all(old[k] == v for k, v in counts.items()), (arm, seed, fold)
    np.savez_compressed(HERE / "counts.npz", **arrays)
    metadata["files_verified"] = len(metadata["csv_hashes"])
    metadata["parity"] = "candidate keys/truth exact; all accepted per-fold classification counts reproduced"
    metadata["counts_sha256"] = sha(HERE / "counts.npz")
    (HERE / "data.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def primary_spectra():
    base = ROOT / "results/truth_reach_20261026/aggregate_4453896_verified/result"
    old = np.load(base / "spectra.npz")
    summary = json.loads((base / "summary.json").read_text())
    rng = np.random.default_rng(summary["bootstrap"]["seed"])
    out = {"edges": old["e_edges"]}
    for tag in TAGS:
        p = base / tag / "spectra.npz"
        proof = json.loads((base / tag / "proof.json").read_text())
        assert sha(p) == proof["artifacts_sha256"]["spectra.npz"]
        z = np.load(p)
        counts, nevent = z["counts"], z["events"]
        assert np.array_equal(z["e_edges"], old["e_edges"])
        weights = rng.multinomial(len(nevent), np.full(len(nevent), 1 / len(nevent)), size=400)
        norm = counts.sum(axis=0) / nevent.sum()
        reps = np.einsum("rf,fse->rse", weights, counts) / (weights @ nevent)[:, None, None]
        assert np.allclose(norm, old[tag+"_per_event"], rtol=0, atol=1e-14)
        assert np.allclose(reps.std(axis=0, ddof=1), old[tag+"_sd"], rtol=0, atol=1e-14)
        out[tag+"__central"] = norm
        out[tag+"__replicas"] = reps
        out[tag+"__events"] = nevent.sum()
    np.savez_compressed(HERE / "primary_spectra.npz", **out)


if __name__ == "__main__":
    primary_spectra()
    collect()
