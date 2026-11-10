"""Replay both frozen score policies and retain per-file response counts."""

from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analyze import BASE, ROOT, SEEDS, TAGS, inputs, load, read  # noqa: E402

CHOICE = "22c81979ca8064aed4248cb3d11bb4379048c8b2ea22a236688054e15c1a0880"
SCAN = "df0e602347bed3bf6316abeb9014490f077247d9c5dc8fcad2dba4bb02ba7695"
LEDGER = "2f2f7ac1342651631d105701012776c539145166c4ae6826ad3522a239785218"
HYBRID = "1c10ff1245a5fb7abc453d51b1865e49e4d493df79d9b0ff15380b558b576bad"
EVALUATIONS = {
    4466339: "30e456e22b83d78b1d947247f338da771353b7918bbf3e5f3c3ebb7eecdcd6a7",
    4467086: "6306bc293bdce9011ac5800929e4fd373ee63a86de3089bfbcde57e7254f984a",
    4467091: "2b4fcbb19376874674a0b9fb7ef675a14780266f9755f1e734ce131e09b7154c",
}
# Accepted performance-report edges, plus the fixed 2 GeV boundary for exact closure.
EDGES = np.array(
    [
        -np.inf,
        0,
        0.1,
        0.3,
        0.5,
        0.7,
        1,
        1.4,
        1.8,
        2,
        2.2,
        2.6,
        3,
        3.5,
        4,
        5,
        6.5,
        np.inf,
    ]
)
COLUMNS = ["Row", "ClusterID", "cl_istop", "cl_label", "e_true", "cl_score", "e_pred"]


def hist(energy):
    out = np.histogram(energy, EDGES)[0]
    assert out.sum() == len(energy)
    return out


def bands(h):
    return np.stack(
        [
            h[..., EDGES[:-1] >= 2].sum(-1),
            h[..., (EDGES[:-1] >= 0.3) & (EDGES[:-1] < 1)].sum(-1),
        ],
        axis=-1,
    )


def records(grouped):
    return {r["content"]: r for tag in TAGS for r in grouped[tag]}


def main():
    if (HERE / "files.npz").exists():
        raise FileExistsError("Existing fixed-policy reduction")
    choice = load(HERE.parent / "choice.json", CHOICE)
    with np.load(BytesIO(read(HERE.parent / "counts.npz", SCAN))) as frozen:
        scanned = {}
        for scope, indices in (("selection", [2, 6]), ("check", [0, 1])):
            scanned.update(zip(frozen[f"{scope}_ids"], frozen[scope][:, :, indices]))
    for job, digest in EVALUATIONS.items():
        read(BASE / f"evaluate_{job}/evaluation_summary.json", digest)
    groups, registry, _ = inputs()
    assert choice["chosen_target_percent"] == 90 and set(scanned) == set(groups)
    accepted = load(BASE / "heldout_rn/per_file_counts.json", LEDGER)["primary"]
    b0, b1, a1 = (records(accepted[key]) for key in ("B0", "B1", "A1"))
    hybrid = load(ROOT / "reports/hybrid_np_20261105/results.json", HYBRID)
    hybrid = {r["content"]: r for r in hybrid["records"]}
    ids = sorted(groups)
    n = len(ids)
    # Stages: labelled truth, selected truth, selected positive raw/cal,
    # all selected raw/cal, and selected negatives with calibrated energy.
    spectra = np.empty((n, 3, 2, 7, len(EDGES) - 1), dtype=np.int64)
    # Candidates, label positives, selected, selected positives.
    objects = np.empty((n, 3, 2, 4), dtype=np.int64)
    migration = np.empty((n, 3, 2, len(EDGES) - 1, len(EDGES) - 1), dtype=np.int64)
    # Paired high/low event and candidate cross-products for the clustering check.
    gram = np.empty((n, 3, 2, 4, 4), dtype=np.int64)
    truth = np.empty((n, 4, 2), dtype=np.int64)  # T0/T1/T2 plus ideal proton T1.
    dcm = np.empty((n, 3, 4, 2), dtype=np.int64)
    spans = np.empty(n, dtype=np.int64)
    folds = np.empty(n, dtype=np.int64)
    for i, cid in enumerate(ids):
        base, meta = registry[cid]
        fold = folds[i] = meta["fold"]
        old = b1[cid]
        span = spans[i] = old["processed_span"]
        assert old["fold"] == b0[cid]["fold"] == a1[cid]["fold"] == fold
        h = hybrid[cid]
        assert (
            h["processed_span"] == span and h["primary_member"] == old["primary_member"]
        )
        assert b0[cid]["truth_hi_lo"] == a1[cid]["truth_hi_lo"] == old["truth_hi_lo"]
        truth[i, :3] = old["truth_hi_lo"]
        proton = np.asarray(h["counts"])[:, 1]
        assert np.array_equal(proton, np.broadcast_to(proton[0], proton.shape))
        assert np.array_equal(
            np.asarray(h["counts"])[:, 0], np.broadcast_to(truth[i, 1], proton.shape)
        )
        truth[i, 3] = proton[0]
        dcm[i] = a1[cid]["clusters_hi_lo"]
        assert np.array_equal(dcm[i], np.asarray(h["counts"])[:, 2:6])
        assert np.array_equal(old["clusters_hi_lo"], np.asarray(h["counts"])[:, 6:])
        reference = None
        for s, seed in enumerate(SEEDS):
            frames = []
            for arm in ("B1", "B0"):
                receipt = meta["predictions"][arm][seed][cid]
                frame = pd.read_csv(
                    BytesIO(read(base / receipt["path"], receipt["sha256"])),
                    usecols=COLUMNS,
                )
                assert not frame.duplicated(COLUMNS[:3]).any()
                identity = frame[COLUMNS[:5]].to_numpy()
                if reference is None:
                    reference = identity
                assert np.array_equal(identity, reference), (cid, seed, arm)
                frames.append(frame)
            cal, raw = frames
            score, et, ec = (
                cal[k].to_numpy() for k in ("cl_score", "e_true", "e_pred")
            )
            er = raw.e_pred.to_numpy()
            assert np.array_equal(score, raw.cl_score)
            assert np.isfinite(np.stack([score, et, ec, er])).all()
            assert ((score >= 0) & (score <= 1)).all() and cal.cl_label.isin(
                [0, 1]
            ).all()
            assert int(cal.Row.max()) + 1 == span and (cal.Row >= 0).all()
            pos = cal.cl_label.to_numpy() == 1
            row = cal.Row.to_numpy(dtype=np.int64)
            band_masks = []
            train_cuts = meta["training"][seed]["cuts"]
            assert train_cuts["B0"] == train_cuts["B1"]
            for j, cut in enumerate(
                (train_cuts["B1"], choice["per_fold_seed_score_cuts"][f"{fold}/{seed}"])
            ):
                keep = score > cut
                signal = keep & pos
                background = keep & ~pos
                spectra[i, s, j] = [
                    hist(et[pos]),
                    hist(et[signal]),
                    hist(er[signal]),
                    hist(ec[signal]),
                    hist(er[keep]),
                    hist(ec[keep]),
                    hist(ec[background]),
                ]
                hs = spectra[i, s, j]
                migration[i, s, j] = np.histogram2d(
                    et[signal], ec[signal], bins=(EDGES, EDGES)
                )[0]
                assert np.array_equal(migration[i, s, j].sum(axis=1), hs[1])
                assert np.array_equal(migration[i, s, j].sum(axis=0), hs[3])
                band_masks.extend([keep & (ec >= 2), keep & (ec >= 0.3) & (ec < 1)])
                assert np.array_equal(hs[3] + hs[6], hs[5])
                count = objects[i, s, j] = [
                    len(pos),
                    pos.sum(),
                    keep.sum(),
                    signal.sum(),
                ]
                assert np.array_equal([*bands(hs[5]), *count[2:]], scanned[cid][s, j])
                if j == 0:
                    assert np.array_equal(
                        bands(hs[[0, 1, 3, 5]]), old["clusters_hi_lo"][s]
                    )
                    assert np.array_equal(
                        bands(hs[[0, 1, 2, 4]]), b0[cid]["clusters_hi_lo"][s]
                    )
            event = np.stack([np.bincount(row[m], minlength=span) for m in band_masks])
            candidate = np.stack(band_masks).astype(np.int64)
            gram[i, s] = [event @ event.T, candidate @ candidate.T]
    assert (objects[:, :, 1, 2:] <= objects[:, :, 0, 2:]).all()
    np.savez_compressed(
        HERE / "files.npz",
        ids=ids,
        parts=[groups[k]["part"] for k in ids],
        tags=[groups[k]["tag"] for k in ids],
        folds=folds,
        truth=truth,
        spans=spans,
        edges=EDGES,
        spectra=spectra,
        objects=objects,
        dcm=dcm,
        migration=migration,
        gram=gram,
    )


if __name__ == "__main__":
    main()
