"""Mechanism checks for label floors and exact uncensored primary bands."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest

from HGNDRecoGNN.analysis.energy_bands import EnergyBands
from HGNDRecoGNN.analysis.label_energy_scan import (
    cluster_targets, exact_primary_counts, origin_partition, threshold_ledger, window_grid,
)
from HGNDRecoGNN.scripts import label_energy_audit as audit


def test_lower_floor_adds_soft_only_clusters_without_changing_old_targets():
    energy = np.array([0., .1, .2, .3, np.nextafter(.3, 1), .7, .15, .9])
    cluster = np.array([0, 1, 1, 2, 3, 3, 4, 4])
    prompt = np.array([True, True, True, True, True, True, True, False])
    floors = [None, 0., .1, .2, .3, .4, .5]
    ledger, targets = threshold_ledger(energy, prompt, cluster, floors)
    native_y, native_e = targets[.3]
    assert native_y.tolist() == [0, 0, 0, 1, 0]
    assert native_e.tolist() == [0, 0, 0, .7, 0]
    assert np.array_equal(native_e[native_y == 1], targets[None][1][native_y == 1])
    assert targets[None][0][0] == 1 and targets[0.][0][0] == 0
    assert ledger[0]["positive_zero_energy_targets"] == 1
    assert targets[.2][0][2] == 1 and targets[.3][0][2] == 0
    assert targets[.3][0][3] == 1  # nextafter(.3, 1) is strictly above the floor
    assert targets[None][1][4] == .15  # non-prompt hard hit cannot define the target


@pytest.mark.parametrize("seed", [2, 7, 31])
def test_general_grid_matches_direct_historical_native_formula(seed):
    rng = np.random.default_rng(seed)
    energy = rng.uniform(0, 2, 300)
    prompt = rng.random(300) > .4
    cluster = rng.integers(0, 30, 300)
    y, e = cluster_targets(energy, prompt, cluster, .3, 30)
    original_y, original_e = np.zeros(30), np.zeros(30)
    np.maximum.at(original_y, cluster, prompt & (energy > .3))
    np.maximum.at(original_e, cluster, np.where(prompt & (energy > .3), energy, 0))
    assert np.array_equal(y, original_y) and np.array_equal(e, original_e)
    assert len(threshold_ledger(energy, prompt, cluster, [None, .1, .2, .3, .4, .5])[0]) == 6


@pytest.mark.parametrize("change", ["nan", "negative", "bad_index", "flag_type"])
def test_bad_hit_inputs_are_refused(change):
    e, p, c = np.array([.2, .8]), np.array([True, False]), np.array([0, 1])
    if change == "nan":
        e[0] = np.nan
    elif change == "negative":
        e[0] = -.1
    elif change == "bad_index":
        c[0] = -1
    else:
        p = p.astype(int)
    with pytest.raises(ValueError):
        cluster_targets(e, p, c, .3)


def test_exact_primary_counts_do_not_censor_soft_particles_or_use_full_file_span():
    e = np.array([.2, .3, 1., 2., 3., .5, 2.1])
    row = np.array([0, 0, 0, 1, 1, 2, 4])
    inventory = np.arange(6)  # events 3 and 5 have zero selected neutrons
    acc = np.array([True, True, True, True, True, False, True])
    pdg = np.full(len(e), 2112)
    result = exact_primary_counts(e, row, acc, pdg, inventory, 4, EnergyBands(.1, 1., 2., 3.))
    assert result["events"] == 4
    assert result["counts_high_low"].tolist() == [1, 2]
    assert result["event_second_moments"].tolist() == [[1, 0], [0, 4]]
    old = exact_primary_counts(e, row, acc, pdg, inventory, 4, EnergyBands(.3, 1., 2., 3.))
    assert old["counts_high_low"].tolist() == [1, 1]
    assert e[4] == 3.  # finite upper edge is excluded, as in every half-open band
    with pytest.raises(ValueError, match="inventory"):
        exact_primary_counts(e, row, acc, pdg, inventory[:4], 4, EnergyBands(.1, 1., 2., 3.))


def test_committed_window_grid_is_ordered_and_contains_exact_0p3_reference():
    path = Path(__file__).resolve().parents[1] / "reports/energy_label_optimization_20261029/config.json"
    config = json.loads(path.read_text())
    grid = config["analysis_windows_GeV"]
    windows = window_grid(grid["minimum"], grid["low_stop"], grid["high_start"], grid["maximum"])
    assert len(windows) == 148
    assert EnergyBands(*grid["reference"]) in windows
    assert EnergyBands(.1, 1, 2, 3) in windows
    assert config["tof_halfwidth_ns"] == .3
    assert config["label_floor_GeV"][0] is None


def observation_pack():
    """Fixed synthetic graph with exact floor/last-bin edges and a non-neutron."""
    import pandas as pd
    import torch

    energy = np.array([0., .1, .3, np.nextafter(.3, 1), .8, 3., 4., 1.])
    xyz = np.tile([0., -20., 700.], (len(energy), 1))
    distance = np.linalg.norm(xyz / 100, axis=1)
    # Inverse of the existing _dfToF convention; this is a test fixture,
    # not a new physical mass choice or a timing model.
    beta = np.sqrt(1 - 1 / (1 + np.maximum(energy, 1e-8) / .93955) ** 2)
    times = distance / (299792458. * beta) * 1e9
    times[0] = 1e10  # gives exactly zero in the clipped historical _dfToF
    frame = pd.DataFrame({"Ekin": energy, "PDG": [2112] * 7 + [2212], "Side": 5,
                          "Id": np.arange(len(energy)), "Row": 0, "fTime": times,
                          "fX": xyz[:, 0], "fY": xyz[:, 1], "fZ": xyz[:, 2]})
    native = (energy > .3) & (frame.PDG.to_numpy() == 2112)
    pack = {"row": torch.tensor([0]), "side": torch.tensor([0]),
            "node_ptr": torch.tensor([0, len(energy)]),
            "cluster_ptr": torch.tensor([0, len(energy)]),
            "cluster": torch.arange(len(energy)), "label": torch.tensor(native, dtype=torch.float32),
            "energy": torch.tensor(np.where(native, energy, 0), dtype=torch.float32)}
    return frame, pack


def test_real_audit_adapter_preserves_counts_and_does_not_modify_the_native_pack():
    import torch

    frame, pack = observation_pack()
    before = {k: v.clone() for k, v in pack.items()}
    result = audit.audit_frame(frame, pack, [None, .1, .2, .3, .4, .5], [0, .1, .2, .3, 1, 2, 3])
    assert result["native_replay_exact"]
    assert all(torch.equal(pack[k], v) for k, v in before.items())
    assert result["prompt_hit_energy_overflow"] == 2  # exact3 and4, counted once each
    assert sum(result["prompt_hit_energy_histogram"]) == 5
    for r in result["ledger"]:
        assert sum(r["target_histogram"]) + r["target_overflow"] == r["signal_clusters"]
    assert result["ledger"][0]["positive_zero_energy_targets"] == 1
    with pytest.raises(ValueError, match="ToF"):
        audit.audit_frame(frame, pack, [None, .3], [0, .3, 3], tof_halfwidth_ns=.2)
    pack["label"][1] = 1
    with pytest.raises(ValueError, match="baseline replay"):
        audit.audit_frame(frame, pack, [None, .3], [0, .3, 3])


def input_world(tmp_path):
    """A small pinned input world; paths are checked before tensors are loaded."""
    config_path = tmp_path / "config.json"
    config = json.loads((Path(__file__).resolve().parents[1] /
                         "reports/energy_label_optimization_20261029/config.json").read_text())
    observation, pack = tmp_path / "observation.pt", tmp_path / "pack.pt"
    observation.write_bytes(b"observation placeholder")
    pack.write_bytes(b"native pack placeholder")
    cid = "zeroSpot/10943245/0001"
    def pin(p):
        return {"path": p.name, "sha256": audit.sha256(p)}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"groups": [{"tag": "zeroSpot", "kept": "10943245/0001",
                                                "truth_bound": True, "part": 0,
                                                "hits_sha256": "a" * 64, "vacs_sha256": "b" * 64}]}))
    config["split_sha256"] = audit.sha256(split_path)
    config_path.write_text(json.dumps(config))
    source = {"observation": pin(observation), "packs": {"B/fold_2": pin(pack)},
              "roles": {"2": "validation"}, "raw_sha256": {"hits": "a" * 64, "vacs": "b" * 64}}
    prep_path = tmp_path / "prepare.json"
    prep = {"state": "complete", "schema": "smash-train-prepare-v1", "files": {cid: source},
            "split_sha256": audit.sha256(split_path),
            "graph_view": {"radius_cells": 3.6, "time_ns": 1., "max_neighbors": 32}}
    prep_path.write_text(json.dumps(prep))
    manifest = {"schema": "label-energy-audit-inputs-v1", "config_sha256": audit.sha256(config_path),
                "split": pin(split_path), "prepare": pin(prep_path),
                "files": {cid: {"observation": pin(observation), "pack": pin(pack)}}}
    manifest_path = tmp_path / "inputs.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, config_path, prep_path, prep, manifest, cid


@pytest.mark.parametrize("corruption", ["split", "role", "graph", "config", "input", "membership"])
def test_manifest_rejects_changed_split_train_packs_graphs_config_and_missing_selection(tmp_path, corruption):
    manifest_path, config_path, prep_path, prep, manifest, cid = input_world(tmp_path)
    assert len(audit.check_manifest(manifest_path, config_path)[1]) == 1
    if corruption == "split":
        prep["split_sha256"] = "0" * 64
    elif corruption == "role":
        prep["files"][cid]["roles"]["2"] = "train"
    elif corruption == "graph":
        prep["graph_view"]["time_ns"] = 1.5
    elif corruption == "config":
        config = json.loads(config_path.read_text())
        config["tof_halfwidth_ns"] = .2
        config_path.write_text(json.dumps(config))
        manifest["config_sha256"] = audit.sha256(config_path)
    elif corruption == "input":
        (tmp_path / "observation.pt").write_bytes(b"corrupt")
    else:
        manifest["files"]["bigSpot/report_file"] = manifest["files"][cid]
    prep_path.write_text(json.dumps(prep))
    manifest["prepare"]["sha256"] = audit.sha256(prep_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        audit.check_manifest(manifest_path, config_path)


def test_full_census_path_on_synthetic_files_includes_shared_bootstrap_and_plot(tmp_path):
    import torch

    manifest_path, config_path, prep_path, prep, manifest, _ = input_world(tmp_path)
    frame, pack = observation_pack()
    torch.save({"columns": frame.columns.tolist(),
                "values": torch.tensor(frame.to_numpy(float))}, tmp_path / "observation.pt")
    torch.save(pack, tmp_path / "pack.pt")
    def pin(p):
        return {"path": p.name, "sha256": audit.sha256(p)}
    groups, sources, inputs = [], {}, {}
    for tag in ("zeroSpot", "defaultSpot", "bigSpot"):
        for part in (0, 2, 4):
            cid = f"{tag}/10943245/{part:04d}"
            fold = (part // 2 - 1) % 3
            groups.append({"tag": tag, "kept": f"10943245/{part:04d}", "part": part,
                           "truth_bound": True, "hits_sha256": "a" * 64, "vacs_sha256": "b" * 64})
            record = {"observation": pin(tmp_path / "observation.pt"), "pack": pin(tmp_path / "pack.pt")}
            inputs[cid] = record
            sources[cid] = {"observation": record["observation"], "packs": {f"B/fold_{fold}": record["pack"]},
                            "roles": {str(fold): "validation"},
                            "raw_sha256": {"hits": "a" * 64, "vacs": "b" * 64}}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"groups": groups}))
    config = json.loads(config_path.read_text())
    config["split_sha256"] = audit.sha256(split_path)
    config_path.write_text(json.dumps(config))
    prep.update(files=sources, split_sha256=audit.sha256(split_path))
    prep_path.write_text(json.dumps(prep))
    manifest.update(files=inputs, split=pin(split_path), prepare=pin(prep_path),
                    config_sha256=audit.sha256(config_path))
    manifest_path.write_text(json.dumps(manifest))
    result = audit.run(manifest_path, config_path, tmp_path / "out")
    assert result["state"] == "complete" and len(result["files"]) == 9
    assert (tmp_path / "out/label_counts.pdf").stat().st_size > 1000
    assert json.loads((tmp_path / "out/summary.json").read_text())["input_manifest_sha256"] == audit.sha256(manifest_path)
    for tag in ("zeroSpot", "defaultSpot", "bigSpot"):
        baseline = config["label_floor_GeV"].index(.3)
        assert result["bootstrap"][tag]["paired_delta_vs_native_points"][baseline] == 0
        assert result["bootstrap"][tag]["paired_delta_sd_points"][baseline] == 0
    with pytest.raises(FileExistsError):
        audit.run(manifest_path, config_path, tmp_path / "out")


def test_origin_partition_is_disjoint_and_applies_to_added_clusters():
    cluster = np.array([0, 1, 2, 3, 3, 4, 4])
    origin = np.array([1, 2, 4, 1, 2, 1, 4])
    selected = np.ones(7, dtype=bool)
    result = origin_partition(cluster, origin, selected, 5)
    assert result == {"direct_primary_only": 1, "nonprimary_track_only": 1,
                      "unresolved_only": 1, "mixed_origins": 2}
    added = np.array([True, False, False, True, False])
    result = origin_partition(cluster, origin, selected, 5, added)
    assert sum(result.values()) == 2 and result["mixed_origins"] == 1


def test_real_adapter_keeps_mother_zero_secondary_and_unbound_primary_unresolved():
    frame, pack = observation_pack()
    frame["fMotherId"] = [-1, 0, -1, -1, 0, -2, -1, -1]
    result = audit.audit_frame(frame, pack, [None, .3], [0, .3, 3],
                               primary_binding={"exact_direct_subset": True})
    native = result["ledger"][1]["qualified_origin_clusters"]
    assert native["direct_primary_only"] == 2 and native["nonprimary_track_only"] == 1
    assert native["unresolved_only"] == 1
    result = audit.audit_frame(frame, pack, [None, .3], [0, .3, 3])
    assert result["ledger"][1]["qualified_origin_clusters"]["direct_primary_only"] == 0


def test_conflicting_object_origin_is_unresolved_instead_of_counted_as_direct():
    frame, pack = observation_pack()
    frame["fMotherId"] = -1
    frame.loc[4, "Id"] = frame.loc[3, "Id"]
    frame.loc[4, "fMotherId"] = 0
    result = audit.audit_frame(frame, pack, [None, .3], [0, .3, 3],
                               primary_binding={"exact_direct_subset": True})
    assert result["ledger"][1]["qualified_origin_clusters"]["unresolved_only"] == 2


def test_smoke_selection_has_one_fixed_file_from_every_stratum_and_no_report_files():
    from HGNDRecoGNN.scripts.label_energy_campaign import select_groups
    groups = [{"tag": tag, "kept": f"run/{part}_{index}", "truth_bound": True, "part": part}
              for tag in ("zeroSpot", "defaultSpot", "bigSpot") for part in range(6) for index in [0, 1]]
    chosen = select_groups({"groups": groups}, "smoke")
    assert len(chosen) == 9
    assert all(g["part"] in [0, 2, 4] and g["kept"].endswith("_0") for g in chosen.values())
    with pytest.raises(ValueError, match="S286"):
        select_groups({"groups": groups}, "full")
