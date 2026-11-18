"""Reduce joint rapidity/pT migration on the accepted common isolated pairs."""
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
import tarfile

import numpy as np
import scan

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def reduce(input_dir, output):
    accepted = scan.arch.read_json(input_dir / 'acceptance.json')
    scan.arch.require(accepted['state'] == 'accepted', 'Unaccepted input')
    summary = scan.arch.read_json(input_dir / 'summary.json', accepted['summary_sha256'])
    entries = {r['content']: r for r in summary['files']}
    old_summary = scan.arch.read_json(HERE / 'scan_results/scan.json')
    scan.arch.require(scan.sha(Path(scan.__file__)) == old_summary['source_sha256'], 'Prior reducer dependency changed')
    old_path = HERE / 'scan_results/counts.npz'
    scan.arch.require(scan.sha(old_path) == old_summary['ledger_sha256'], 'Prior ledger changed')
    records = {r['content']: r for r in scan.arch.read_json(scan.arch.PRIMARY / 'result/summary.json')['files']}
    launch = scan.arch.read_json(HERE / 'launch.json')
    plan = scan.arch.read_json(ROOT / launch['historical_plan'], launch['historical_plan_sha256'])['files']
    rule_tar, rule_entries = scan.export_inputs(scan.arch.BENCHMARK / 'export_4502291', plan)
    output_rows = []
    with np.load(old_path, allow_pickle=False) as old, tarfile.open(rule_tar, 'r:') as archive:
        ids, weights, tags, events = (old[k] for k in ('content', 'weights', 'tags', 'events'))
        old_migration = old['migration']
        for i, cid in enumerate(ids):
            path = input_dir / cid / 'kinematics.npz'
            scan.arch.require(scan.sha(path) == entries[cid]['output_sha256'], 'Kinematics changed')
            source = scan.arch.PRIMARY / 'result' / cid / 'neutrons.npz'
            raw = scan.arch.accepted.checked(source, records[cid]['files']['neutrons.npz'])
            rule_raw = archive.extractfile(rule_entries[cid]['member']).read()
            scan.arch.require(scan.sha256(rule_raw).hexdigest() == rule_entries[cid]['sha256'], 'Rule member changed')
            with np.load(path, allow_pickle=False) as z, np.load(BytesIO(raw), allow_pickle=False) as n, np.load(BytesIO(rule_raw), allow_pickle=False) as r:
                objects = scan.arch.pd.MultiIndex.from_arrays([n['Row'], n['Id']])
                gnn_keys = z['gnn_keys']
                rule_keys = np.column_stack([z['rules_keys'], np.zeros(len(r['rows']), dtype=np.int64)])
                graphs = (scan.arch.accepted.edge_indices(r['edges'], objects, rule_keys),
                          scan.arch.accepted.edge_indices(n['observed_edges'], objects, gnn_keys))
                isolated = [scan.arch.isolation(p, c, len(n['Row']), len(keys))
                            for (p, c), keys in zip(graphs, (rule_keys, gnn_keys), strict=True)]
                target_y, target_pt, target_e = z['neutron_y_nn_cm'], z['neutron_pT_GeV'], n['Ekin']
                common = np.flatnonzero(n['in_rectangle'] & (n['Row'] < events[i]) & (target_e >= .3) & (target_e < 4)
                                        & (isolated[0] >= 0) & (isolated[1] >= 0))
                by_seed = []
                for seed in scan.arch.SEEDS:
                    prefix = f'dynamic_seed_{seed}_'
                    energy = z[prefix + 'energy_GeV']
                    keep = (z['rules_selection'], z[prefix + 'keep70'] & (energy > .3) & (energy < 4))
                    pairs = common[keep[0][isolated[0][common]] & keep[1][isolated[1][common]]]
                    methods = []
                    for method, name in enumerate(('rules_', prefix)):
                        cohorts = []
                        for objects_in_cohort in (pairs, pairs[n['qualifying_native'][pairs]]):
                            cl = isolated[method][objects_in_cohort]
                            y = z[name + 'y_nn_cm'][cl]
                            pt = z[name + 'pT_GeV'][cl]
                            matrix = np.zeros((2, 2, len(scan.EDGES) - 1, len(scan.EDGES) - 1), dtype=np.int64)
                            for true_pt in range(2):
                                for reco_pt in range(2):
                                    select = ((target_pt[objects_in_cohort] >= scan.PT_BREAK) == bool(true_pt)) & ((pt >= scan.PT_BREAK) == bool(reco_pt))
                                    matrix[true_pt, reco_pt] = np.histogram2d(target_y[objects_in_cohort][select], y[select], (scan.EDGES, scan.EDGES))[0]
                            scan.arch.require(matrix.sum() == len(objects_in_cohort), 'Joint overflow')
                            cohorts.append(matrix)
                        methods.append(cohorts)
                    by_seed.append(methods)
                value = np.asarray(by_seed)
                np.testing.assert_array_equal(value.sum(axis=(3, 4)), old_migration[i])
                output_rows.append(value)
            if (i + 1) % 50 == 0:
                print(f'Joint migration checked {i+1}/286', flush=True)
    data = np.asarray(output_rows)
    draws = np.vstack([np.ones(len(ids), dtype=np.int64), weights])
    same_joint = np.trace(data[..., 0, 0, :, :], axis1=-2, axis2=-1) + np.trace(data[..., 1, 1, :, :], axis1=-2, axis2=-1)
    total = data.sum(axis=(-4, -3, -2, -1))
    pt_counts = data.sum(axis=(-2, -1))
    scopes = {}
    for tag in ('pooled', *scan.arch.TAGS.values()):
        mask = np.ones(len(ids), bool) if tag == 'pooled' else tags == tag
        counts = np.tensordot(draws[:, mask], pt_counts[mask], axes=1)
        n = np.tensordot(draws[:, mask], total[mask], axes=1)
        joint = 100 * scan.ratio(np.tensordot(draws[:, mask], same_joint[mask], axes=1), n)
        pt = 100 * scan.ratio(np.trace(counts, axis1=-2, axis2=-1), n)
        scopes[tag] = {
            'common_counts_by_seed_and_cohort': n[0, :, 0].tolist(),
            'pT_transition_counts_by_seed': counts[0].tolist(),
            'pT_transition_percent': scan.interval_fields((100 * scan.ratio(counts, counts.sum(axis=-1, keepdims=True))).mean(axis=1)),
            'same_joint_bin_percent': scan.interval_fields(joint.mean(axis=1)),
            'paired_dynamic_minus_rules_same_joint_bin_pp': scan.interval_fields((joint[:, :, 1] - joint[:, :, 0]).mean(axis=1)),
            'same_pT_band_percent': scan.interval_fields(pt.mean(axis=1)),
            'paired_dynamic_minus_rules_same_pT_band_pp': scan.interval_fields((pt[:, :, 1] - pt[:, :, 0]).mean(axis=1)),
        }
    output.mkdir(exist_ok=False)
    ledger = output / 'joint_counts.npz'
    np.savez_compressed(ledger, content=ids, tags=tags, weights=weights, events=events, joint_migration=data)
    scan.arch.write_json(output / 'joint.json', {
        'state': 'complete', 'scope': 'S286 retrospective common selected isolated pairs', 'files': len(ids),
        'seeds': scan.arch.SEEDS, 'methods': scan.METHODS, 'cohorts': old_summary['common_isolated_cohorts'],
        'y_nn_cm_edges': scan.EDGES.tolist(), 'pT_break_GeV': scan.PT_BREAK,
        'pT_axis_order': ['true_below_or_at_least_break', 'reconstructed_below_or_at_least_break'],
        'scopes': scopes, 'marginal_reconciliation_files': len(ids), 'ledger_sha256': scan.sha(ledger),
        'prior_scan_sha256': scan.sha(HERE / 'scan_results/scan.json'), 'export_acceptance_sha256': scan.sha(input_dir / 'acceptance.json'),
        'protocol_sha256': scan.sha(HERE / 'joint_protocol_20261118.md'), 'source_sha256': scan.sha(Path(__file__)),
    })


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    reduce(a.input, a.output)
