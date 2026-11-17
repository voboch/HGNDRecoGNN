"""Compare the selected raw DynamicEdgeConv with fixed ToF controls on saved events."""

import argparse
from io import BytesIO
from pathlib import Path
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import smash_architecture_reduce as arch
from contrast import reduce_spectrum
from reduce import export_inputs, BANDS, COUNT_KEYS
from summarize import scope

sys.path.insert(0, str(ROOT / 'reports/first_hit_tof_20261110'))
from compare import tof

FINAL = 'student_dynedge'
CAMPAIGN = arch.REPORTS / 'smash_architecture_20261112'
RULE_ODD = arch.REPORTS / 'rule_tof_odd_20261110'
FIRST = arch.REPORTS / 'first_hit_tof_20261110'
RULE_EXPORT = arch.BENCHMARK / 'export_4502291'
SCORE = ROOT / 'results/smash_architecture_20261112/score_even_accepted'


def checked_ledger(directory, file_name, summary_name, digest_key):
    summary = arch.read_json(directory / summary_name)
    raw = arch.accepted.checked(directory / file_name, summary[digest_key])
    return np.load(BytesIO(raw), allow_pickle=False), summary


def same_centers(new, old, label):
    for key in ('value', 'file_sd', 'q16', 'q84'):
        if not np.isclose(new[key], old[key], atol=1e-12, rtol=0):
            raise ValueError(f'{label} {key} replay differs: {new[key]} versus {old[key]}')


def odd():
    with_deep, deep_summary = checked_ledger(
        CAMPAIGN / 'odd_result', 'file_counts.npz', 'summary.json', 'file_counts_sha256')
    with_rules, rule_summary = checked_ledger(RULE_ODD, 'counts.npz', 'results.json', 'counts_sha256')
    with with_deep as deep, with_rules as rules:
        for key in ('content', 'tags', 'draws', 'seeds'):
            np.testing.assert_array_equal(deep[key], rules[key], err_msg=key)
        arch.require(len(set(deep['content'])) == 282 and deep['draws'].shape == (400, 282)
                     and deep_summary['events'] == rule_summary['event_count'] == 1165216,
                     'Odd scope')
        arch.require(tuple(deep['candidate_count_keys'][:3]) == ('soft', 'hard', 'overflow'),
                     'Architecture band order')
        for index, key in enumerate(('soft', 'hard', 'overflow')):
            mc = rules[key][..., list(rules['arms']).index('mc')]
            np.testing.assert_array_equal(mc, np.broadcast_to(deep['truth'][:, index, None], mc.shape))
        policy_path = CAMPAIGN / 'even_result/frozen_policy.json'
        policy = arch.read_json(policy_path, deep_summary['frozen_policy_sha256'])
        arch.require(str(policy['selected_targets'][FINAL]) == '75', 'Frozen DynamicEdgeConv target')
        model = list(deep['models']).index(FINAL)
        arm = [list(rules['arms']).index(name) for name in ('mc', 'rules')]
        reference = np.stack([rules[key][..., arm] for key in ('soft', 'hard', 'overflow')], axis=-1)
        weights = np.vstack([np.ones(282, dtype=np.int64), deep['draws']])
        cases = {}
        for title, key, historical in (
            ('raw70', 'candidate_counts', 'hardness_raw70'),
            ('raw75_frozen', 'alternative_candidate_counts', 'hardness_frozen_even_policy'),
        ):
            data = np.concatenate([reference, deep[key][:, :, model:model+1, :3]], axis=2)
            soft, hard, tail = np.moveaxis(data, -1, 0)
            cases[title] = {}
            for window, hi in (('open_upper', hard + tail), ('within_reach', hard)):
                result = reduce_spectrum(weights, deep['tags'], soft, hi)
                for pair, new in result['contrasts'].items():
                    old = deep_summary[historical][window]['contrasts'][pair]
                    for name, original in (
                        ('mc', old['geometric_mc_contrast_percent']),
                        ('rules', rule_summary[window]['contrasts'][pair]['rules']),
                        ('gnn', old['reconstructed_contrast_percent']['by_model'][FINAL]),
                        ('gnn_minus_mc', old['paired_shift_from_mc_pp'][FINAL]),
                    ):
                        same_centers(new[name], original, f'{title}/{window}/{pair}/{name}')
                    new['contrast_file_sd_change_pp'] = new['gnn']['file_sd'] - new['rules']['file_sd']
                cases[title][window] = result
        return {'files': 282, 'events': 1165216, 'arms': ['mc', 'rules', FINAL],
                'retrospective': True, 'shared_file_draws': 400, 'cases': cases,
                'source_sha256': [deep_summary['file_counts_sha256'], rule_summary['counts_sha256'],
                                  deep_summary['frozen_policy_sha256']]}


def energy_response(samples, weights, methods):
    """Every row compares energies on the same primary link within a seed."""
    bands = [('primary', .3, 4)] + [(f'{lo:g}_to_{hi:g}', lo, hi) for lo, hi in
                                    zip(arch.ENERGY_EDGES[:-1], arch.ENERGY_EDGES[1:], strict=True)]
    merged = []
    for by_file in zip(*samples, strict=True):
        indices = np.repeat(np.arange(len(by_file)), [len(values) for values in by_file])
        merged.append((np.concatenate(by_file), indices))
    report = {}
    for label, lo, hi in bands:
        shape = (len(weights), len(arch.SEEDS), len(methods))
        metrics = {name: np.full(shape, np.nan) for name in ('mae_MeV', 'bias_MeV',
                             'median_relative_error_percent', 'central_half_width_percent')}
        counts, median_truth = [], []
        for seed_idx, (data, indices) in enumerate(merged):
            use = (data[:, 0] >= lo) & (data[:, 0] < hi)
            target, file_index = data[use, 0], indices[use]
            n = weights @ np.bincount(file_index, minlength=weights.shape[1])
            counts.append(int(n[0]))
            median_truth.append(float(np.median(target)) if len(target) else None)
            if not len(target):
                continue
            for method in range(len(methods)):
                error = data[use, method + 1] - target
                for name, values in (('mae_MeV', abs(error)), ('bias_MeV', error)):
                    sums = weights @ np.bincount(file_index, weights=values, minlength=weights.shape[1])
                    metrics[name][:, seed_idx, method] = arch.count_ratio(sums, n, 1000)
                if len(target) >= 25:  # Existing matched-energy quantile rule.
                    q16, q50, q84 = arch.weighted_quantiles(error / target, file_index, weights).T
                    metrics['median_relative_error_percent'][:, seed_idx, method] = 100 * q50
                    metrics['central_half_width_percent'][:, seed_idx, method] = 50 * (q84 - q16)
        row = {'true_range_GeV': [lo, hi], 'n_by_seed': counts, 'median_truth_GeV_by_seed': median_truth}
        for name, value in metrics.items():
            replicas = value.mean(axis=1)
            row[name] = {m: arch.interval(v) | {'by_seed': arch.clean_values(value[0, :, i])}
                         for i, (m, v) in enumerate(zip(methods, replicas.T, strict=True))}
            row[name]['paired_gnn_minus_reference'] = {
                m: arch.interval(replicas[:, -1] - replicas[:, i]) for i, m in enumerate(methods[:-1])}
        report[label] = row
    return report


def even():
    acceptance = arch.read_json(SCORE / 'acceptance.json')
    arch.require(acceptance['state'] == 'accepted' and acceptance['scope'] == 'even', 'Even score acceptance')
    bundles = [SCORE / Path(task['bundle']).name for task in acceptance['tasks'].values()]
    for bundle, task in zip(bundles, acceptance['tasks'].values(), strict=True):
        arch.require(arch.sha(bundle / 'summary.json') == task['summary_sha256'], 'Score summary pin')
    ids, tags, weights, records, folds, _, models = arch.inputs('even', bundles, False, None, None)
    prior, old = checked_ledger(CAMPAIGN / 'even_result', 'file_counts.npz', 'summary.json', 'file_counts_sha256')
    historical, _ = checked_ledger(arch.BENCHMARK, 'counts.npz', 'results.json', 'counts_sha256')
    plan_path = FIRST / 'plan.json'
    export = arch.read_json(FIRST / 'export/summary.json')
    plan = {r['content']: r for r in arch.read_json(plan_path, export['plan_sha256'])['files']}
    first = np.load(BytesIO(arch.accepted.checked(FIRST / 'export/first_hits.npz', export['first_hits_sha256'])))
    selection = np.load(BytesIO(arch.accepted.checked(FIRST / 'selection.npz', export['selection_sha256'])))
    arch.require(set(ids) == set(plan) == set(first.files) == set(selection.files), 'First-hit file cohort')
    arch.require(export['state'] == 'complete' and export['native_replay'] == 'exact', 'First-hit export')
    arch.accepted.checked(FIRST / 'extract.py', export['extractor_sha256'])
    native = {r['content']: r for r in arch.read_json(arch.REPORTS / 'native_features_20261106/plan.json',
        'f0c33dab8b2e90ce4563323de19af75ac79650ac998c761a752047e4e38a639a')['files']}
    manifest = {r['member']: r['sha256'] for r in arch.read_json(arch.PRIMARY / 'export_4491592/manifest.json',
        'e32d7ea8f4629354864d8a63e76a674c8dfaa4d63d2f941889399c885063ccba')['files']}
    tar_path, entries = export_inputs(RULE_EXPORT, list(plan.values()))
    counts, bands, multiplicity, moments, rule_samples, tof_samples, first_counts = [], [], [], [], [], [], []
    arch.require(np.array_equal(prior['content'], ids) and np.array_equal(prior['draws'], weights[1:])
                 and np.array_equal(historical['content'], ids), 'Even shared draw ordering')
    model_index = list(prior['models']).index(FINAL)
    previous_candidates = prior['candidate_counts'][:, :, model_index]
    with tarfile.open(tar_path, 'r:') as archive:
        for i, cid in enumerate(ids):
            record = records[cid]
            span = record['processed_span']
            arch.require(span == plan[cid]['processed_span'], f'{cid} span changed')
            keys, rows, truth_rows, truth, inside, gobj, gcl, gunique = arch.geometry(record, native[cid], manifest)
            entry = entries[cid]
            raw_rules = archive.extractfile(entry['member']).read()
            arch.require(arch.sha256(raw_rules).hexdigest() == entry['sha256'], f'{cid} rule member')
            with np.load(BytesIO(raw_rules)) as z:
                rule_rows, rule_energy, rule_keep = z['rows'], z['energy'], z['selected']
                graw = arch.accepted.checked(arch.PRIMARY / 'result' / cid / 'neutrons.npz',
                                             record['files']['neutrons.npz'])
                with np.load(BytesIO(graw)) as geometry:
                    objects = arch.pd.MultiIndex.from_arrays([truth_rows, geometry['Id']])
                    rule_keys = np.column_stack([rule_rows, np.arange(len(rule_rows)), np.zeros(len(rule_rows))])
                    robj, rcl = arch.accepted.edge_indices(z['edges'], objects, rule_keys)
                    np.testing.assert_array_equal(np.bincount(robj, minlength=len(truth)) > 0,
                                                  geometry['observed_hit'])
            np.testing.assert_array_equal(rule_keep, rule_keep & (rule_energy > .3) & (rule_energy < 4))
            rc, rm, rd = arch.metric(rule_rows, truth_rows, inside, robj, rcl, rule_keep, rule_energy, span)
            np.testing.assert_array_equal(rc, historical['counts'][i, 0, 0])
            np.testing.assert_array_equal(rm, historical['multiplicity'][i, 0, 0])
            runique = arch.isolation(robj, rcl, len(truth), len(rule_rows))
            common = np.flatnonzero(inside & (truth > .3) & (truth < 4) & (runique >= 0) & (gunique >= 0))
            band_index = np.digitize(truth, BANDS)
            def truth_bands(degree):
                return np.stack([np.bincount(band_index[mask], minlength=len(BANDS) + 1)
                                 for mask in (inside, inside & (degree > 0), inside & (degree > 1))])
            raw_keys, _, energies, _, _, _, obj, cluster, _ = arch.accepted.read_geometry(graw, record, native[cid])
            np.testing.assert_array_equal(cluster, selection[cid])
            arch.require(raw_keys.equals(keys) and arch.accepted.digest(energies[obj], '<f8') ==
                         plan[cid]['isolated_truth_sha256'], f'{cid} first-hit key binding')
            t, tj, length = first[cid].T
            arch.require(len(t) == len(cluster) and np.all(length > 0), f'{cid} first-hit lengths')
            original, valid_original, _, _ = tof(t, length)
            jittered, valid_jittered, _, _ = tof(tj, length)
            file_counts, file_bands, file_mult, file_moments, file_rules, file_tof, file_first = ([] for _ in range(7))
            for si, seed in enumerate(arch.SEEDS):
                model = models[(FINAL, folds[cid], seed)]
                pin = old['identities'][f'{FINAL}/{folds[cid]}/{seed}']
                arch.require(pin == model.provenance | {'selection': model.selection, 'cuts': model.cuts},
                             'Model receipt must match accepted architecture reduction')
                frame = arch.align(model.frame(cid, span), keys, None, cid)
                native_target = frame[['cl_label', 'e_true']].to_numpy(np.float32)
                arch.require(arch.accepted.digest(native_target, '<f4') == native[cid]['targets_sha256']
                             and arch.accepted.digest(frame.cluster_size, '<i8') == native[cid]['sizes_sha256'],
                             f'{cid} native targets/sizes')
                raw = frame.e_pred_raw.to_numpy()
                score_keep = frame.cl_score.to_numpy() > model.cuts['70']
                np.testing.assert_array_equal(arch.candidate_bands(raw, score_keep, frame.cl_label.to_numpy(bool)),
                                              previous_candidates[i, si])
                keep = score_keep & (raw > .3) & (raw < 4)
                gc, gm, gd = arch.metric(rows, truth_rows, inside, gobj, gcl, keep, raw, span)
                pair = common[rule_keep[runique[common]] & keep[gunique[common]]]
                rule_sample = np.column_stack([truth[pair], rule_energy[runique[pair]], raw[gunique[pair]]])
                residual = rule_sample[:, 1:] - rule_sample[:, :1]
                file_counts.append([rc, gc]); file_bands.append([truth_bands(rd), truth_bands(gd)])
                file_mult.append([rm, gm]); file_rules.append(rule_sample)
                file_moments.append(np.column_stack([np.full(2, len(pair)), residual.sum(0), abs(residual).sum(0)]))
                valid = score_keep[cluster] & valid_original & valid_jittered
                file_tof.append(np.column_stack([truth[obj][valid], original[valid], jittered[valid], raw[cluster][valid]]))
                file_first.append([len(cluster), score_keep[cluster].sum(), valid.sum()])
            for buffer, item in zip((counts, bands, multiplicity, moments, rule_samples, tof_samples, first_counts),
                                    (file_counts, file_bands, file_mult, file_moments, file_rules, file_tof, file_first), strict=True):
                buffer.append(item)
            if (i + 1) % 50 == 0:
                print(f'Joined {i + 1}/{len(ids)} even files', flush=True)
    arrays = dict(content=ids, tags=tags, draws=weights[1:], counts=np.asarray(counts), count_keys=COUNT_KEYS,
                  truth_bands=np.asarray(bands), multiplicity=np.asarray(multiplicity),
                  rule_energy_moments=np.asarray(moments), first_hit_counts=np.asarray(first_counts))
    arch.require((arrays['first_hit_counts'][:, :, 0].sum(0) == 142912).all(), 'First-hit total cohort')
    summaries = {name: scope(weights[:, mask], arrays['counts'][mask], arrays['truth_bands'][mask],
                    arrays['multiplicity'][mask], arrays['rule_energy_moments'][mask], COUNT_KEYS)
                 for name, mask in [('pooled', np.ones(len(ids), bool)), *[(tag, tags == tag) for tag in arch.TAGS.values()]]}
    return {'files': len(ids), 'events': int(arrays['counts'][:, 0, 0, 0].sum()),
            'arms': ['rules', FINAL], 'cut_target': 70, 'scopes': summaries,
            'rules_energy': energy_response(rule_samples, weights, ('rules', 'gnn')),
            'first_hit_energy': energy_response(tof_samples, weights, ('original_tof', 'jittered_tof', 'gnn')),
            'first_hit_flow_by_seed': arrays['first_hit_counts'].sum(0).tolist(),
            'first_hit_flow_columns': ['native_isolated_before_cut', 'passing_score_cut', 'passing_both_physical_ToF_masks'],
            'source_sha256': {str(p.relative_to(ROOT)): arch.sha(p) for p in (
                SCORE / 'acceptance.json', CAMPAIGN / 'even_result/file_counts.npz',
                arch.BENCHMARK / 'counts.npz', RULE_EXPORT / 'inputs.tar',
                FIRST / 'export/first_hits.npz', FIRST / 'selection.npz', FIRST / 'plan.json')}}, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope', choices=('even', 'odd'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    arch.require(not args.output.exists(), 'Output exists')
    if args.scope == 'odd':
        result = odd()
    else:
        result, arrays = even()
        counts_path = args.output.with_suffix('.npz')
        with counts_path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        result['counts_sha256'] = arch.sha(counts_path)
    arch.write_json(args.output, result | {'model': FINAL, 'energy': 'raw', 'state': 'complete',
                    'seeds': arch.SEEDS, 'reducer_sha256': arch.sha(Path(__file__))})


if __name__ == '__main__':
    main()
