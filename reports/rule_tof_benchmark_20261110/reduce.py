"""Paired rule-ToF and frozen GNN accounting on the accepted SMASH events."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile
from time import monotonic

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT.parent))
from rule import RULE_REJECTIONS

PRIOR = ROOT / "reports/primary_prediction_response_20261109"
spec = spec_from_file_location("accepted_prediction_reader", PRIOR / "analyze.py")
accepted = module_from_spec(spec)
spec.loader.exec_module(accepted)
BANDS = np.array([0.3, 1.0, 2.0, 4.0])
SEEDS = accepted.SEEDS
COUNT_KEYS = (
    "events", "zero_truth_events", "zero_candidates", "truth", "candidates",
    "truth_with_edge", "truth_missed", "truth_split", "candidate_merge",
    "candidate_mixed_merge", "candidate_in", "candidate_out_only", "candidate_no_edge",
    "exclusive_matches", "exclusive_truth_deficit", "exclusive_candidate_excess",
    "soft_candidates", "hard_candidates",
)
MULTIPLICITY_STATES = 4


def load(path):
    return json.loads(path.read_bytes())


def require(condition, context):
    if not condition:
        raise ValueError(context)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def metric(rows, truth_row, inside, primary, candidate, selection, energy, span):
    require(len(rows) == len(selection) == len(energy), "Candidate length mismatch")
    require((rows >= 0).all() and (rows < span).all(), "Candidate row outside span")
    truth_mult = np.bincount(truth_row[inside], minlength=span)
    selected_mult = np.bincount(rows[selection], minlength=span)
    multiplicity = np.zeros((MULTIPLICITY_STATES, MULTIPLICITY_STATES), np.int64)
    np.add.at(multiplicity, (truth_mult.clip(max=3), selected_mult.clip(max=3)), 1)
    active = selection[candidate]
    obj, cl = primary[active], candidate[active]
    require(np.array_equal(truth_row[obj], rows[cl]), "Cross-event direct link")
    td = np.bincount(obj, minlength=len(inside))
    cd = np.bincount(cl, minlength=len(rows))
    linked_in = inside[obj]
    cid = cl[linked_in]
    pid = obj[linked_in]
    ci = np.bincount(cid, minlength=len(rows))
    co = cd - ci
    # The global graph is block-diagonal in Row, so its cardinality sums
    # the event maxima without choosing a physics energy assignment.
    adj = csr_matrix((np.ones(len(cid), dtype=bool), (cid, pid)),
                     shape=(len(rows), len(inside)))
    matches = int((maximum_bipartite_matching(adj, perm_type="column") >= 0).sum())
    ntruth, nc = int(inside.sum()), int(selection.sum())
    measures = (
        span, np.count_nonzero(truth_mult == 0), np.count_nonzero(selected_mult == 0),
        ntruth, nc, np.count_nonzero(inside & (td > 0)),
        np.count_nonzero(inside & (td == 0)), np.count_nonzero(inside & (td > 1)),
        np.count_nonzero(cd > 1), np.count_nonzero((ci > 0) & (co > 0)),
        np.count_nonzero(ci), np.count_nonzero((ci == 0) & (co > 0)),
        np.count_nonzero(selection & (cd == 0)), matches, ntruth - matches, nc - matches,
        np.count_nonzero(selection & (energy >= .3) & (energy < 1)),
        np.count_nonzero(selection & (energy >= 2) & (energy < 4)),
    )
    require(sum(measures[10:13]) == nc and measures[5] + measures[6] == ntruth,
            "Direct category conservation")
    require(multiplicity.sum() == span, "Multiplicity conservation")
    return measures, multiplicity, td


def isolation(primary, candidate, n_primary, n_candidates):
    primary_degree = np.bincount(primary, minlength=n_primary)
    candidate_degree = np.bincount(candidate, minlength=n_candidates)
    unique = (primary_degree[primary] == 1) & (candidate_degree[candidate] == 1)
    result = np.full(n_primary, -1, dtype=np.int64)
    result[primary[unique]] = candidate[unique]
    return result


def compare_file(record, raw_rule, state):
    cid, span = record["content"], record["processed_span"]
    meta = state["primary"][cid]
    root = accepted.PRIMARY / "result" / cid
    blobs = {name: accepted.checked(root / name, meta["files"][name])
             for name in ("events.csv.gz", "neutrons.npz")}
    events = pd.read_csv(BytesIO(blobs["events.csv.gz"]), compression="gzip")
    event_span = events.loc[events.Row < span]
    require(np.array_equal(event_span.Row, np.arange(span)), f"{cid} zero-event inventory")
    with np.load(BytesIO(blobs["neutrons.npz"])) as z:
        row = z["Row"]
        mc_energy = z["Ekin"]
        inside = z["in_rectangle"] & (row < span)
        objects = pd.MultiIndex.from_arrays([row, z["Id"]])
        gnn_keys = np.column_stack([z["cluster_row"], z["cluster_local_id"], z["cluster_side"]])
        gnn_obj, gnn_cl = accepted.edge_indices(z["observed_edges"], objects, gnn_keys)
        observed = z["observed_hit"]
    require(np.array_equal(np.bincount(row[inside], minlength=span), event_span.n_all),
            f"{cid} truth event inventory")
    require(accepted.digest(gnn_keys, "<i8") == record["keys_sha256"], f"{cid} GNN keys")
    require(len(gnn_keys) == meta["clusters"], f"{cid} GNN cluster count")

    with np.load(BytesIO(raw_rule)) as z:
        rule_rows = z["rows"]
        rule_energy = z["energy"]
        rule_selection = z["selected"]
        literal = z["literal"]
        require(not (literal & ~rule_selection).any() and not (literal & (rule_energy >= 2)).any(),
                f"{cid} nested beta control")
        literal_counts = [int(literal.sum()), int((literal & (rule_energy < 1)).sum())]
        rejected_rules = z["rule_rejections"]
        keys = np.column_stack([rule_rows, np.arange(len(rule_rows)), np.zeros(len(rule_rows))])
        rule_obj, rule_cl = accepted.edge_indices(z["edges"], objects, keys)
    require(np.array_equal(np.bincount(rule_obj, minlength=len(objects)) > 0, observed),
            f"{cid} rule and accepted direct hit identities differ")
    require(rejected_rules[0] == len(rule_rows) and
            rejected_rules[1:].sum() == len(rule_rows) and
            rejected_rules[-1] == rule_selection.sum(), f"{cid} rule ledger")
    rule_counts, rule_mult, rule_degree = metric(
        rule_rows, row, inside, rule_obj, rule_cl, rule_selection, rule_energy, span
    )
    rule_isolated = isolation(rule_obj, rule_cl, len(objects), len(rule_rows))
    gnn_isolated = isolation(gnn_obj, gnn_cl, len(objects), len(gnn_keys))
    truth_band = np.digitize(mc_energy, BANDS)
    cohort = inside & (mc_energy > .3) & (mc_energy < 4)
    cohort &= (rule_isolated >= 0) & (gnn_isolated >= 0)
    bin_counts = np.stack([
        np.bincount(truth_band[inside], minlength=len(BANDS) + 1),
        np.bincount(truth_band[inside & (rule_degree > 0)], minlength=len(BANDS) + 1),
        np.bincount(truth_band[inside & (rule_degree > 1)], minlength=len(BANDS) + 1),
    ])
    base, summary = state["predictions"][cid]
    keys = pd.MultiIndex.from_arrays(gnn_keys.T)
    counts, mult, by_energy, errors = [], [], [], []
    gnn_baseline, gnn_window_loss = [], []
    for seed in SEEDS:
        prediction = accepted.predictions(base, summary, cid, seed, "B1", state["sources"],
                                          keys, state["native"][cid])
        score = prediction.cl_score.to_numpy()
        energy = prediction.e_pred.to_numpy()
        score_selection = score > summary["training"][seed]["cuts"]["B1"]
        gnn_baseline.append([
            np.count_nonzero(score_selection & (energy >= 2)),
            np.count_nonzero(score_selection & (energy >= .3) & (energy < 1)),
        ])
        gnn_window_loss.append([
            np.count_nonzero(score_selection & (energy <= .3)),
            np.count_nonzero(score_selection & (energy >= 4)),
        ])
        selection = score_selection & (energy > .3) & (energy < 4)
        gnn_counts, gnn_mult, gnn_degree = metric(
            gnn_keys[:, 0], row, inside, gnn_obj, gnn_cl, selection, energy, span
        )
        counts.append([rule_counts, gnn_counts])
        mult.append([rule_mult, gnn_mult])
        by_energy.append([bin_counts, np.stack([
            bin_counts[0],
            np.bincount(truth_band[inside & (gnn_degree > 0)], minlength=len(BANDS) + 1),
            np.bincount(truth_band[inside & (gnn_degree > 1)], minlength=len(BANDS) + 1),
        ])])
        pair = np.flatnonzero(cohort)
        rc, gc = rule_isolated[pair], gnn_isolated[pair]
        good = rule_selection[rc] & selection[gc] & (energy[gc] >= 0)
        pair = pair[good]
        er = rule_energy[rc[good]] - mc_energy[pair]
        eg = energy[gc[good]] - mc_energy[pair]
        errors.append([[len(pair), er.sum(), np.abs(er).sum()],
                       [len(pair), eg.sum(), np.abs(eg).sum()]])
    return {"content": cid, "counts": np.asarray(counts), "multiplicity": np.asarray(mult),
            "truth_bands": np.asarray(by_energy), "error": np.asarray(errors),
            "rule_rejections": rejected_rules, "literal_counts": literal_counts,
            "gnn_baseline": gnn_baseline, "gnn_window_loss": gnn_window_loss, "hits": record["observation"]["observed_hits"], "gnn_clusters": len(gnn_keys)}


def reduce_tag(task):
    archive_path, records, entries, state = task
    outputs = []
    with tarfile.open(archive_path, "r:") as archive:
        for record in records:
            entry = entries[record["content"]]
            with archive.extractfile(entry["member"]) as member:
                raw = member.read()
            require(sha(raw) == entry["sha256"], f"Export member {entry['member']}")
            outputs.append(compare_file(record, raw, state))
            if len(outputs) % 25 == 0:
                print(f"{record['tag']}: {len(outputs)}/{len(records)} files", flush=True)
    return outputs


def export_inputs(directory, records):
    receipt = load(directory / "receipt.json")
    manifest = json.loads(accepted.checked(directory / "manifest.json", receipt["manifest_sha256"]))
    archive_path = directory / "inputs.tar"
    with archive_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(receipt["state"] == "complete" and
            digest == manifest["archive_sha256"] == receipt["archive_sha256"] and
            archive_path.stat().st_size == receipt["archive_bytes"] == manifest["archive_bytes"],
            "Rule export archive receipt")
    expected = {entry["member"]: entry["sha256"] for entry in manifest["files"]}
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        require(len(members) == len(expected) and
                all(m.isfile() and m.name in expected for m in members),
                "Rule export member inventory")
        with archive.extractfile("summary.json") as member:
            summary_raw = member.read()
    require(sha(summary_raw) == expected["summary.json"] == receipt["summary_files"]["summary.json"],
            "Rule export summary digest")
    summary = json.loads(summary_raw)
    plan_path = ROOT / "reports/first_hit_tof_20261110/plan.json"
    require(summary["state"] == "complete" and summary["native_observation_replay"] == "exact"
            and summary["plan_sha256"] == sha(plan_path.read_bytes())
            and summary["extractor_sha256"] == sha((HERE / "extract.py").read_bytes())
            and summary["rule_sha256"] == sha((HERE / "rule.py").read_bytes())
            and tuple(summary["rule_rejection_keys"]) == RULE_REJECTIONS,
            "Rule source and plan binding")
    files = {entry["content"]: entry for entry in summary["files"]}
    require(len(summary["files"]) == len(files) == len(records) and
            set(files) == {record["content"] for record in records}, "Rule cohort inventory")
    require(len(expected) == 1 + len(files), "Unexpected rule export entries")
    for record in records:
        entry = files[record["content"]]
        require(entry["member"] == record["content"] + ".npz"
                and entry["sha256"] == expected[entry["member"]]
                and entry["observation_sha256"] == record["observation"]["sha256"]
                and entry["hits"] == record["observation"]["observed_hits"]
                and entry["processed_span"] == record["processed_span"],
                f"Rule observation binding for {record['content']}")
    return archive_path, files


def inputs():
    pins = load(PRIOR / "input_pins.json")
    for relative, expected in pins.items():
        accepted.checked(ROOT / relative, expected)
    response = load(accepted.RESPONSE / "input_pins.json")
    split = json.loads(accepted.checked(ROOT / response["split"], response["files"][response["split"]]))
    contents = {f"{g['tag']}/{g['kept']}": g for g in split["groups"]
                if g["truth_bound"] and g["part"] in (0, 2, 4)}
    plan_path = ROOT / "reports/first_hit_tof_20261110/plan.json"
    record_pin = load(ROOT / "reports/first_hit_tof_20261110/source_package.json")["plan_sha256"]
    records = json.loads(accepted.checked(plan_path, record_pin))["files"]
    primary = {r["content"]: r for r in load(accepted.PRIMARY / "result/summary.json")["files"]}
    native = {r["content"]: r for r in load(ROOT / "reports/native_features_20261106/plan.json")["files"]}
    require(len(contents) == 286 and {r["content"] for r in records} == set(contents)
            == set(primary) == set(native), "Content cohort changed")
    for record in records:
        require(record["processed_span"] == primary[record["content"]]["processed_span"],
                "Historical span changed")
    state = {"primary": primary, "native": native,
             "predictions": accepted.registry(contents, split, response["files"]),
             "sources": load(accepted.RESPONSE / "results.json")["sources"]}
    return records, state


def main(args):
    start = monotonic()
    require(not args.output.exists(), "Output already exists")
    records, state = inputs()
    archive_path, entries = export_inputs(args.rule_export, records)
    tasks = [(archive_path, [r for r in records if r["tag"] == tag], entries, state)
             for tag in ("zeroSpot", "defaultSpot", "bigSpot")]
    with ProcessPoolExecutor(max_workers=3) as pool:
        files = [row for result in pool.map(reduce_tag, tasks) for row in result]
    with np.load(accepted.RESPONSE / "counts.npz") as previous:
        lookup = {r["content"]: r for r in files}
        order = previous["content"].tolist()
        require(set(order) == set(lookup), "File order lost")
        files = [lookup[cid] for cid in order]
        weights = previous["weights"]
        prior_gnn_bands = previous["ledger"][:, :, 1, 3]
    counts = np.stack([r["counts"] for r in files])
    require(counts[:, 0, 0, 0].sum() == 1180595 and counts[:, 0, 0, 3].sum() == 897563,
            "Accepted event or geometric count changed")
    require(sum(r["hits"] for r in files) == 5833940 and
            sum(r["gnn_clusters"] for r in files) == 1798225, "Accepted detector count changed")
    require(np.array_equal(np.stack([r["gnn_baseline"] for r in files]), prior_gnn_bands),
            "Frozen GNN band count replay before the range mask")
    require(counts[:, :, 0, COUNT_KEYS.index("hard_candidates")].sum() > 0,
            "Physical-beta rule arm has no candidates above 2 GeV")
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, content=np.asarray(order), weights=weights,
                            counts=counts, count_keys=COUNT_KEYS, seeds=SEEDS,
                            multiplicity=np.stack([r["multiplicity"] for r in files]),
                            truth_bands=np.stack([r["truth_bands"] for r in files]),
                            error=np.stack([r["error"] for r in files]),
                            rule_rejections=np.stack([r["rule_rejections"] for r in files]),
                            rule_rejection_keys=RULE_REJECTIONS,
                            literal_counts=np.stack([r["literal_counts"] for r in files]),
                            gnn_window_loss=np.stack([r["gnn_window_loss"] for r in files]))
    print(f"Completed {len(files)} files in {monotonic() - start:.1f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule-export", type=Path, required=True)
    main(parser.parse_args())
