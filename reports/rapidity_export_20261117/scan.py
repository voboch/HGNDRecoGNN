"""Reduce accepted S286 rapidity sidecars without fitting or changing cuts."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import smash_architecture_reduce as arch
from reduce import export_inputs
from accept_s286 import sha

# Diagnostic grid fixed before opening the output. See scan_protocol.md.
EDGES = np.arange(-11, 15, dtype=float) / 10
PT_BREAK = .5
PT_SCOPES = ('all', 'below_0p5', 'at_least_0p5')
METHODS = ('rules', 'dynamic_edgeconv')
TRUE_FIELDS = ('direct', 'miss', 'split')
RECO_FIELDS = ('all_candidates', 'inside_link', 'outside_only', 'no_direct_link', 'merge')
SAMPLE_FIELDS = ('count', 'absolute_rapidity_error', 'rapidity_error')
ENERGY_FIELDS = ('inside_below_0p3', 'inside_0p3_to_4', 'inside_at_least_4')


def masks(pt):
    return np.ones(len(pt), bool), pt < PT_BREAK, pt >= PT_BREAK


def histogram(y, pt, selected):
    out = np.stack([np.histogram(y[selected & mask], EDGES)[0] for mask in masks(pt)])
    arch.require(out[0].sum() == selected.sum() and (out[0] == out[1:].sum(axis=0)).all(), 'Rapidity or pT overflow')
    return out


def error_sums(target, reconstructed):
    return np.stack([np.histogram(target, EDGES, weights=w)[0] for w in (
        np.ones(len(target)), np.abs(reconstructed - target), reconstructed - target)])


def pairs_by_event(npz, rules, previous, record):
    cid = record['content']
    span = record['processed_span']
    old = arch.PRIMARY / 'result' / cid / 'neutrons.npz'
    raw = arch.accepted.checked(old, record['files']['neutrons.npz'])
    with np.load(BytesIO(raw), allow_pickle=False) as z:
        row, ident, inside, e = (z[k] for k in ('Row', 'Id', 'in_rectangle', 'Ekin'))
        objects = arch.pd.MultiIndex.from_arrays([row, ident])
        keys = np.column_stack([z['cluster_row'], z['cluster_local_id'], z['cluster_side']])
        direct = arch.accepted.edge_indices(z['observed_edges'], objects, keys)
        native = z['qualifying_native']
    for actual, expected in ((npz['neutron_Row'], row), (npz['neutron_Id'], ident),
                             (npz['neutron_in_rectangle'], inside), (npz['neutron_Ekin'], e), (npz['gnn_keys'], keys)):
        np.testing.assert_array_equal(actual, expected)
    events = arch.PRIMARY / 'result' / cid / 'events.csv.gz'
    expected_events = arch.pd.read_csv(BytesIO(arch.accepted.checked(events, record['files']['events.csv.gz'])), compression='gzip')
    np.testing.assert_array_equal(npz['event_Row'], expected_events.Row)
    np.testing.assert_array_equal(npz['event_in_span'], expected_events.Row < span)
    n_scope = inside & (row < span) & (e >= .3) & (e < 4)
    p_scope = npz['proton_in_rectangle'] & (npz['proton_Row'] < span) & (npz['proton_Ekin'] >= .3) & (npz['proton_Ekin'] < 4)
    for species, field, filt in (('neutron', 'n_all', inside), ('proton', 'p_all', npz['proton_in_rectangle'])):
        np.testing.assert_array_equal(np.bincount(npz[f'{species}_Row'][filt], minlength=len(expected_events)), expected_events[field])
    np.testing.assert_array_equal(npz['rules_keys'][:, 0], rules['rows'])
    np.testing.assert_array_equal(npz['rules_selection'], rules['selected'])
    rule_keys = np.column_stack([npz['rules_keys'], np.zeros(len(rules['rows']), dtype=np.int64)])
    linked_rules = arch.accepted.edge_indices(rules['edges'], objects, rule_keys)
    graphs = (linked_rules, direct)
    lengths = (len(rule_keys), len(keys))
    selected_keys = [arch.isolation(p, c, len(row), size) for (p, c), size in zip(graphs, lengths, strict=True)]
    ymc, ptmc = npz['neutron_y_nn_cm'], npz['neutron_pT_GeV']
    denominators = np.stack([histogram(ymc, ptmc, n_scope), histogram(npz['proton_y_nn_cm'], npz['proton_pT_GeV'], p_scope)])
    truth_counts, candidate_counts, migration, sample_error, link_error, energy_origin = [], [], [], [], [], []
    common = np.flatnonzero(n_scope & (selected_keys[0] >= 0) & (selected_keys[1] >= 0))
    for s, seed in enumerate(arch.SEEDS):
        prefix = f'dynamic_seed_{seed}_'
        energy = npz[prefix + 'energy_GeV']
        keep = [npz['rules_selection'], npz[prefix + 'keep70'] & (energy > .3) & (energy < 4)]
        observed_y = [npz['rules_y_nn_cm'], npz[prefix + 'y_nn_cm']]
        observed_pt = [npz['rules_pT_GeV'], npz[prefix + 'pT_GeV']]
        truth, reco = [], []
        pairs = common[keep[0][selected_keys[0][common]] & keep[1][selected_keys[1][common]]]
        arrays_migration, arrays_error, arrays_link_error, arrays_origin = [], [], [], []
        for m, ((obj, cl), k, y, pt) in enumerate(zip(graphs, keep, observed_y, observed_pt, strict=True)):
            size = lengths[m]
            candidate_rows = (rule_keys, keys)[m][:, 0]
            active = k[cl]
            arch.require((candidate_rows >= 0).all() and (candidate_rows < span).all()
                         and np.array_equal(row[obj[active]], candidate_rows[cl[active]]), f'{cid}: candidate span or edge mismatch')
            t_degree = np.bincount(obj[active], minlength=len(inside))
            c_degree = np.bincount(cl[active], minlength=size)
            inside_edges = active & inside[obj] & (row[obj] < span)
            inside_degree = np.bincount(cl[inside_edges], minlength=size)
            truth.append(np.stack([histogram(ymc, ptmc, n_scope & mask) for mask in (t_degree > 0, t_degree == 0, t_degree > 1)]))
            reco.append(np.stack([histogram(y, pt, k & mask) for mask in (
                np.ones(size, bool), inside_degree > 0, (inside_degree == 0) & (c_degree > 0), c_degree == 0, c_degree > 1)]))
            np.testing.assert_array_equal(reco[-1][0], reco[-1][1:4].sum(axis=0))
            np.testing.assert_array_equal(truth[-1][:2].sum(axis=0), denominators[0])
            np.testing.assert_array_equal(reco[-1][:, 0].sum(axis=-1), previous['candidate_counts'][s, m],
                                          err_msg=f'{cid}/{seed}/{METHODS[m]} candidate accounting')
            np.testing.assert_array_equal(denominators[0, 0].sum(), previous['truth_bands'][s, m, 0], err_msg=cid)
            np.testing.assert_array_equal(truth[-1][[0, 2], 0].sum(axis=-1), previous['truth_bands'][s, m, 1:], err_msg=cid)
            arch.require(np.count_nonzero(e[pairs] > .3) == previous['common_pairs'][s, m], f'{cid}: old common pair count')
            origin = []
            for mask in (e < .3, (e >= .3) & (e < 4), e >= 4):
                from_band = np.bincount(cl[inside_edges & mask[obj]], minlength=size) > 0
                origin.append(histogram(y, pt, from_band))
            arrays_origin.append(origin)
            edge_scope = active & n_scope[obj]
            arrays_link_error.append([error_sums(ymc[obj[edges]], y[cl[edges]]) for edges in (
                edge_scope, edge_scope & native[obj])])
            matrices, errors = [], []
            for selection in (pairs, pairs[native[pairs]]):
                yy = y[selected_keys[m][selection]]
                target = ymc[selection]
                mat = np.histogram2d(target, yy, bins=(EDGES, EDGES))[0].astype(np.int64)
                arch.require(mat.sum() == len(selection), 'Migration overflow')
                matrices.append(mat)
                errors.append(error_sums(target, yy))
            arrays_migration.append(matrices)
            arrays_error.append(errors)
        truth_counts.append(truth)
        candidate_counts.append(reco)
        migration.append(arrays_migration)
        sample_error.append(arrays_error)
        link_error.append(arrays_link_error)
        energy_origin.append(arrays_origin)
    return denominators, truth_counts, candidate_counts, migration, sample_error, link_error, energy_origin


def ratio(numerator, denominator):
    expected = np.broadcast_shapes(numerator.shape, denominator.shape)
    return np.divide(numerator, denominator, out=np.full(expected, np.nan), where=denominator > 0)


def interval_fields(replicas):
    # No conditioning on nonzero resampled denominators. Mark these bins explicitly.
    valid = np.isfinite(replicas).all(axis=0)
    sd = np.full(replicas.shape[1:], np.nan)
    bounds = np.full((2, *replicas.shape[1:]), np.nan)
    sd[valid] = replicas[1:, valid].std(axis=0, ddof=1)
    bounds[:, valid] = np.quantile(replicas[1:, valid], [.16, .84], axis=0) if valid.any() else np.empty((2, 0))
    clean = lambda a: np.where(np.isfinite(a), a, None).tolist()
    return {'value': clean(replicas[0]), 'file_sd': clean(sd), 'q16': clean(bounds[0]), 'q84': clean(bounds[1]),
            'invalid_draws': (~np.isfinite(replicas[1:])).sum(axis=0).tolist()}


def summarize(data, weights, ev):
    primary, observed, candidates, error, links = (np.tensordot(weights, data[i], axes=1) for i in (0, 1, 2, 4, 5))
    migration = np.tensordot(weights[0], data[3], axes=1)
    origin = np.tensordot(weights[0], data[6], axes=1)
    counts = weights @ ev
    coverage = 100 * ratio(observed, primary[:, None, None, None, 0, :, :])
    rates = 100 * ratio(candidates, candidates[..., :1, :, :])
    bias_width = ratio(error[..., 1:, :], error[..., :1, :])
    paired_error = bias_width[:, :, 1] - bias_width[:, :, 0]
    truth_totals = observed[..., 0, :].sum(axis=-1)
    truth_rates = 100 * ratio(truth_totals, primary[:, 0, 0, :].sum(axis=-1)[:, None, None, None])
    candidate_totals = candidates[..., 0, :].sum(axis=-1)
    candidate_rates = 100 * ratio(candidate_totals, candidate_totals[..., :1])
    error_totals = error.sum(axis=-1)
    metrics = ratio(error_totals[..., 1:], error_totals[..., :1])
    primary_totals = primary[:, :, 0, :].sum(axis=-1)
    diagonal = np.tensordot(weights, np.trace(data[3], axis1=-2, axis2=-1), axes=1)
    same_bin = 100 * ratio(diagonal, error_totals[..., 0])
    return {
        'events': int(counts[0]), 'bin_counts_mc_n_p': primary[0].tolist(),
        'mc_yield_per_event_per_rapidity': interval_fields(primary / counts[:, None, None, None] / np.diff(EDGES)),
        'mc_n_over_p': interval_fields(ratio(primary[:, 0], primary[:, 1])),
        'truth_counts_by_seed': observed[0].tolist(), 'candidate_counts_by_seed': candidates[0].tolist(),
        'geometric_rates_percent': interval_fields(coverage.mean(axis=1)),
        'paired_dynamic_minus_rules_geometric_pp': interval_fields((coverage[:, :, 1] - coverage[:, :, 0]).mean(axis=1)),
        'candidate_composition_percent': interval_fields(rates.mean(axis=1)),
        'candidate_yield_per_event_per_rapidity': interval_fields((candidates[..., 0, :, :] / counts[:, None, None, None, None] / np.diff(EDGES)).mean(axis=1)),
        'candidate_energy_origin_counts_by_seed': origin.tolist(),
        'common_isolated_y_counts_and_errors': error[0].tolist(),
        'common_isolated_y_absolute_error_and_bias': interval_fields(bias_width.mean(axis=1)),
        'paired_dynamic_minus_rules_y_error': interval_fields(paired_error.mean(axis=1)),
        'common_isolated_y_migration': migration.tolist(),
        'all_selected_direct_link_counts_and_errors': links[0].tolist(),
        'all_selected_direct_link_y_absolute_error_and_bias': interval_fields(ratio(links[..., 1:, :], links[..., :1, :]).mean(axis=1)),
        'integrated': {
            'mc_n_p_counts': primary_totals[0].tolist(),
            'mc_n_over_p': interval_fields(ratio(primary_totals[:, 0], primary_totals[:, 1])),
            'geometric_rates_percent': interval_fields(truth_rates.mean(axis=1)),
            'paired_dynamic_minus_rules_geometric_pp': interval_fields((truth_rates[:, :, 1] - truth_rates[:, :, 0]).mean(axis=1)),
            'candidate_composition_percent': interval_fields(candidate_rates.mean(axis=1)),
            'paired_dynamic_minus_rules_composition_pp': interval_fields((candidate_rates[:, :, 1] - candidate_rates[:, :, 0]).mean(axis=1)),
            'common_isolated_counts_by_seed': error_totals[0, :, 0, :, 0].astype(np.int64).tolist(),
            'common_isolated_y_absolute_error_and_bias': interval_fields(metrics.mean(axis=1)),
            'paired_dynamic_minus_rules_y_error': interval_fields((metrics[:, :, 1] - metrics[:, :, 0]).mean(axis=1)),
            'common_isolated_same_rapidity_bin_percent': interval_fields(same_bin.mean(axis=1)),
            'paired_dynamic_minus_rules_same_bin_pp': interval_fields((same_bin[:, :, 1] - same_bin[:, :, 0]).mean(axis=1)),
        },
    }


def run(input_dir, output):
    acceptance = arch.read_json(input_dir / 'acceptance.json')
    info = arch.read_json(input_dir / 'summary.json', acceptance['summary_sha256'])
    arch.require(acceptance['state'] == 'accepted' and info['checks']['files'] == 286, 'Export not accepted')
    files = {r['content']: r for r in info['files']}
    reference = arch.read_json(arch.PRIMARY / 'acceptance.json')
    records = {r['content']: r for r in arch.read_json(arch.PRIMARY / 'result/summary.json', reference['summary_sha256'])['files']}
    plan = arch.read_json(HERE / 'launch.json')
    oldplan = arch.read_json(ROOT / plan['historical_plan'], plan['historical_plan_sha256'])['files']
    rule_tar, entries = export_inputs(arch.BENCHMARK / 'export_4502291', oldplan)
    prior_path = ROOT / 'reports/final_model_comparison_20261117/even.npz'
    prior_sha = arch.read_json(prior_path.with_suffix('.json'))['counts_sha256']
    arch.require(sha(prior_path) == prior_sha, 'Even comparison count ledger changed')
    with np.load(prior_path, allow_pickle=False) as prior, tarfile.open(rule_tar, 'r:') as tar:
        ids, weights, tags = prior['content'], prior['draws'], prior['tags']
        count_keys = list(prior['count_keys'])
        old = {
            'candidate_counts': prior['counts'][..., [count_keys.index(name) for name in (
                'candidates', 'candidate_in', 'candidate_out_only', 'candidate_no_edge', 'candidate_merge')]],
            'truth_bands': prior['truth_bands'][..., 1:4].sum(axis=-1),
            'common_pairs': prior['rule_energy_moments'][..., 0],
        }
        arch.require(len(ids) == len(set(ids)) == len(files) == 286 and set(ids) == set(files), 'File inventory')
        output_rows = []
        for i, cid in enumerate(ids):
            path = input_dir / cid / 'kinematics.npz'
            arch.require(sha(path) == files[cid]['output_sha256'], 'Kinematic sidecar changed after acceptance')
            entry = entries[cid]
            with tar.extractfile(entry['member']) as f:
                raw = f.read()
            arch.require(sha256(raw).hexdigest() == entry['sha256'], 'Rules link member differs')
            with np.load(path, allow_pickle=False) as kinematics, np.load(BytesIO(raw), allow_pickle=False) as rules:
                output_rows.append(pairs_by_event(kinematics, rules, {k: v[i] for k, v in old.items()}, records[cid]))
            if (i + 1) % 50 == 0:
                print(f'Checked and reduced {i+1}/286', flush=True)
    data = [np.asarray(items) for items in zip(*output_rows, strict=True)]
    ev = np.array([records[c]['processed_span'] for c in ids])
    arch.require(ev.sum() == 1180595 and weights.shape == (400, 286), 'Frozen coverage or draws changed')
    all_weights = np.vstack([np.ones(len(ids), dtype=np.int64), weights])
    scopes = {}
    for tag in ('pooled', *arch.TAGS.values()):
        mask = np.ones(len(ids), bool) if tag == 'pooled' else tags == tag
        scopes[tag] = summarize([d[mask] for d in data], all_weights[:, mask], ev[mask])
    output.mkdir(exist_ok=False)
    ledgers = output / 'counts.npz'
    np.savez_compressed(ledgers, content=ids, weights=weights, tags=tags, events=ev,
                        primary=data[0], geometric=data[1], candidates=data[2], migration=data[3], errors=data[4],
                        all_direct_link_errors=data[5], candidate_energy_origins=data[6])
    result = {'state': 'complete', 'scope': 'S286 retrospective even diagnostic', 'files': len(ids), 'draws': 400,
              'y_nn_cm_edges': EDGES.tolist(), 'pT_break_GeV': PT_BREAK, 'pT_scopes': PT_SCOPES, 'methods': METHODS,
              'truth_fields': TRUE_FIELDS, 'candidate_fields': RECO_FIELDS,
              'candidate_energy_origin_fields': ENERGY_FIELDS,
              'common_isolated_cohorts': ['without_native_requirement', 'same_selection_native_qualified'],
              'isolation_fields': SAMPLE_FIELDS, 'seeds': arch.SEEDS, 'scopes': scopes,
              'ledger_sha256': sha(ledgers), 'export_acceptance_sha256': sha(input_dir / 'acceptance.json'),
              'even_reference_counts_sha256': prior_sha, 'protocol_sha256': sha(HERE / 'scan_protocol.md'),
              'source_sha256': sha(Path(__file__))}
    arch.write_json(output / 'scan.json', result)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    run(a.input, a.output)
