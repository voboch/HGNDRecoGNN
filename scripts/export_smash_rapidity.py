#!/usr/bin/env python3
"""Add primary momenta and detector-only candidate directions to a pinned S286 file."""

from argparse import ArgumentParser
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.data.graph_dataset import _dfToF
from HGNDRecoGNN.data.smash_transfer import _ordered
from HGNDRecoGNN.analysis.truth_reach import BEAM_KINETIC, NUCLEON_MASS, Y_CM
from HGNDRecoGNN.scripts.audit_smash_sources import HITS, VACS, observe, read_csv
from HGNDRecoGNN.scripts.hgnd_acceptance import in_acceptance
from HGNDRecoGNN.scripts.map_smash_primary import KEYS, targets_from_vacs

RULES = ROOT / 'reports/rule_tof_benchmark_20261110'
sys.path.insert(0, str(RULES))
from rule import reconstruct

PRIMARY = ROOT / 'reports/primary_association_20261106'
FIRST = ROOT / 'reports/first_hit_tof_20261110'
NAMES = ('Row', 'Id')
KINEMATICS = ('PDG', 'Px', 'Py', 'Pz', 'E', 'Ekin', 'Mass', 'Rapid', 'vX', 'vY', 'vZ')
NEUTRON_MASS_GEV = .93955  # Same mass as both accepted ToF comparisons.
LIGHT_SPEED_M_PER_NS = .299792458


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(path, expected):
    require(sha(path) == expected, f'SHA-256 mismatch: {path}')
    return path


def read_json(path):
    return json.loads(path.read_text())


def key_digest(values):
    return sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()


def primary_arrays(path, expected, accepted_path, events_path, vacs):
    with np.load(accepted_path, allow_pickle=False) as z:
        n_keys = pd.MultiIndex.from_arrays([z['Row'], z['Id']], names=NAMES)
        old_energy, old_geometry = z['Ekin'], z['in_rectangle']
    events = pd.read_csv(events_path).set_index('Row')
    counters = pd.DataFrame(0, index=events.index, columns=['exported_particles', 'n_all', 'p_all'])
    require(n_keys.is_unique and events.index.is_unique, 'Repeated accepted key')
    targets = targets_from_vacs(vacs, expected['content'])
    require(len(targets) == expected['binding']['eligible_records'], 'Direct binding census changed')
    neutron_parts, proton_parts, direct_parts = [], [], []
    boundary_ids, last_row = set(), -1
    for chunk in pd.read_csv(path, usecols=[*NAMES, *KINEMATICS], chunksize=100_000):
        row, ident = chunk.Row.to_numpy(), chunk.Id.to_numpy()
        require(np.isfinite(chunk.to_numpy()).all() and np.equal(row, np.floor(row)).all()
                and np.equal(ident, np.floor(ident)).all() and (row >= 0).all() and (ident >= 0).all()
                and row[0] >= last_row and (np.diff(row) >= 0).all()
                and not chunk.duplicated(list(NAMES)).any(), 'Invalid primary key or kinematics')
        require(not boundary_ids.intersection(ident[row == last_row]), 'Duplicate key across CSV chunks')
        final_ids = set(ident[row == row[-1]])
        boundary_ids = boundary_ids | final_ids if row[-1] == last_row else final_ids
        last_row = row[-1]
        geometry = in_acceptance(chunk.Px, chunk.Py, chunk.Pz, chunk.vX, chunk.vY, chunk.vZ)
        n_inside = geometry & (chunk.PDG == 2112)
        p_inside = geometry & (chunk.PDG == 2212)
        counts = pd.DataFrame({'Row': row, 'exported_particles': 1, 'n_all': n_inside, 'p_all': p_inside})
        counts = counts.groupby('Row', sort=False).sum()
        require(counts.index.isin(events.index).all(), 'Primary event absent from accepted inventory')
        counters.loc[counts.index] += counts
        n_keep = n_inside | pd.MultiIndex.from_frame(chunk[list(NAMES)]).isin(n_keys)
        neutron_parts.append(chunk.loc[n_keep].assign(in_rectangle=geometry[n_keep]))
        proton_parts.append(chunk.loc[p_inside])
        direct = targets.merge(chunk, on=list(NAMES), suffixes=('_det', '_prim'), validate='one_to_one')
        require(all(np.array_equal(direct[f'{name}_det'], direct[f'{name}_prim']) for name in KEYS),
                'Direct primary MCTrack fields changed')
        direct_parts.append(direct[list(NAMES)])
    pd.testing.assert_frame_equal(counters, events[counters.columns], check_dtype=False)
    n = pd.concat(neutron_parts, ignore_index=True)
    p = pd.concat(proton_parts, ignore_index=True)
    direct = pd.concat(direct_parts)
    require(len(direct) == len(targets) and not direct.duplicated(list(NAMES)).any(), 'Incomplete direct MCTrack proof')
    np.testing.assert_array_equal(n[list(NAMES)], n_keys.to_frame(index=False))
    np.testing.assert_array_equal(n.Ekin, old_energy)
    np.testing.assert_array_equal(n.in_rectangle, old_geometry)
    require(n.PDG.eq(2112).all() and p.PDG.eq(2212).all(), 'Incorrect species')
    arrays = {'event_Row': events.index.to_numpy(np.int64), 'event_in_span': events.index < expected['processed_span']}
    diagnostics = {}
    for name, frame in (('neutron', n), ('proton', p)):
        require((frame.E > frame.Pz.abs()).all() and (frame.Ekin >= 0).all() and (frame.Mass > 0).all(),
                f'Invalid selected {name} four-vector')
        calculated = np.arctanh(frame.Pz.to_numpy() / frame.E.to_numpy())
        arrays.update({f'{name}_{col}': frame[col].to_numpy() for col in (*NAMES, *KINEMATICS)})
        arrays[f'{name}_in_rectangle'] = n.in_rectangle.to_numpy() if name == 'neutron' else np.ones(len(p), bool)
        arrays[f'{name}_y_lab'] = calculated
        arrays[f'{name}_y_nn_cm'] = calculated - Y_CM
        arrays[f'{name}_pT_GeV'] = np.hypot(frame.Px.to_numpy(), frame.Py.to_numpy())
        diagnostic = np.abs(calculated - frame.Rapid.to_numpy())
        diagnostics[name] = {'rows': len(frame), 'in_span': int((frame.Row < expected['processed_span']).sum()),
                             'max_abs_y_minus_exported': float(diagnostic.max(initial=0))}
    return arrays, diagnostics


def head_arrays(frame, head, length, raw_time):
    coordinates = frame[['fX', 'fY', 'fZ']].to_numpy(np.float64)[head]
    unit = coordinates / (100 * length[head, None])
    require(np.isfinite(unit).all(), 'Invalid detector-based direction')
    return {'head_instance': frame.Instance.to_numpy(np.int64)[head], 'head_xyz_cm': coordinates,
            'direction': unit, 'length_m': length[head], 'time_original_ns': raw_time[head],
            'time_input_ns': frame.fTime.to_numpy()[head]}


def candidate_momentum(kinetic_GeV, directions):
    """Convert reconstructed kinetic energy and observed directions to p, pT and y."""
    valid = np.isfinite(kinetic_GeV) & (kinetic_GeV >= 0)
    size = np.full(len(kinetic_GeV), np.nan)
    size[valid] = np.sqrt(kinetic_GeV[valid] * (kinetic_GeV[valid] + 2 * NEUTRON_MASS_GEV))
    momentum = size[:, None] * directions
    pt = np.hypot(momentum[:, 0], momentum[:, 1])
    y_lab = np.arcsinh(momentum[:, 2] / np.hypot(NEUTRON_MASS_GEV, pt))
    return {'momentum_GeV': momentum, 'pT_GeV': pt, 'y_lab': y_lab,
            'y_nn_cm': y_lab - Y_CM, 'valid_energy': valid}


def candidate_arrays(hits_path, pack_path, vacs, expected, accepted_path, observed_path=None):
    cid = expected['content']
    hits = read_csv(hits_path, HITS)
    _, obs, _ = observe(hits, vacs, expected['member'] + '_hits.csv')
    columns = [column for column in obs if column != '_merge']
    values = obs[columns].to_numpy(np.float64)
    with TemporaryDirectory() as tmp:
        replay = Path(tmp) / (Path(expected['member']).name + '.pt')
        torch.save({'columns': columns, 'values': torch.tensor(values)}, replay)
        observation_sha = sha(replay)
    if observed_path is None:
        require(observation_sha == expected['observation']['sha256'],
                'Observation serialization differs; use the pinned original observation for value parity')
    else:
        check(observed_path, expected['observation']['sha256'])
        original = torch.load(observed_path, map_location='cpu', weights_only=True)
        require(columns == original['columns'], 'Observation columns changed')
        np.testing.assert_array_equal(values, original['values'].numpy())
    frame, row, side = _ordered(obs)
    pack = torch.load(pack_path, map_location='cpu', weights_only=True)
    a = {k: v.numpy() for k, v in pack.items()}
    nodes, clusters = np.diff(a['node_ptr']), np.diff(a['cluster_ptr'])
    np.testing.assert_array_equal(row, np.repeat(a['row'], nodes))
    np.testing.assert_array_equal(side, np.repeat(a['side'], nodes))
    cluster = a['cluster'] + np.repeat(a['cluster_ptr'][:-1], nodes)
    ncl = len(a['label'])
    keys = np.column_stack([np.repeat(a['row'], clusters), np.arange(ncl) - np.repeat(a['cluster_ptr'][:-1], clusters),
                            np.repeat(a['side'], clusters)])
    require(key_digest(keys) == expected['keys_sha256'], 'Candidate keys changed')
    with np.load(accepted_path, allow_pickle=False) as old:
        np.testing.assert_array_equal(keys, np.column_stack([old['cluster_row'], old['cluster_local_id'], old['cluster_side']]))
    index = pd.MultiIndex.from_frame(hits[['Row', 'Instance']]).get_indexer(pd.MultiIndex.from_frame(frame[['Row', 'Instance']]))
    require((index >= 0).all(), 'Lost original times')
    raw_time = hits.fTime.to_numpy()[index]
    length = np.linalg.norm(frame[['fX', 'fY', 'fZ']].to_numpy() / 100, axis=1)
    target = frame.Ekin.to_numpy()
    timing = frame.fTime.to_numpy()
    prompt = ((np.abs(frame.PDG) == 2112) & frame.Side.isin([5, 6]) & (target >= _dfToF(length, timing + .3))
              & (target <= _dfToF(length, timing - .3)))
    native = prompt & (target > .3)
    label, energy = np.zeros_like(a['label']), np.zeros(ncl)
    np.maximum.at(label, cluster, native.to_numpy().astype(label.dtype))
    np.maximum.at(energy, cluster, np.where(native, target, 0))
    require(label.tobytes() == a['label'].tobytes() and energy.astype(a['energy'].dtype).tobytes() == a['energy'].tobytes(),
            'Native label or energy target bytes differ')
    ordered = np.lexsort((frame.Instance.to_numpy(), raw_time, cluster))
    unique, starts = np.unique(cluster[ordered], return_index=True)
    np.testing.assert_array_equal(unique, np.arange(ncl))
    gnn = head_arrays(frame, ordered[starts], length, raw_time)
    first_meta = read_json(FIRST / 'export/summary.json')
    old_hits = check(FIRST / 'export/first_hits.npz', first_meta['first_hits_sha256'])
    old_selection = check(FIRST / 'selection.npz', first_meta['selection_sha256'])
    with np.load(old_hits, allow_pickle=False) as old, np.load(old_selection, allow_pickle=False) as mask:
        first = mask[cid]
        require(key_digest(keys[first]) == expected['isolated_keys_sha256'], 'First-hit keys changed')
        np.testing.assert_array_equal(np.column_stack([gnn[k][first] for k in ('time_original_ns', 'time_input_ns', 'length_m')]), old[cid])
    gnn['keys'] = keys
    gnn['native_label'] = a['label']
    gnn['native_target_GeV'] = a['energy']
    gnn['cluster_size'] = np.bincount(cluster, minlength=ncl)
    # Rule benchmark clusters were constructed on the observation order, before native-pack sorting.
    rule_cluster, rule_rows, rule_energy, keep, literal, ledger = reconstruct(obs)
    r_xyz = obs[['fX', 'fY', 'fZ']].to_numpy() / 100
    r_length = np.linalg.norm(r_xyz, axis=1)
    speed = r_length / (LIGHT_SPEED_M_PER_NS * obs.fTime.to_numpy())
    ordered = np.lexsort((obs.Instance.to_numpy(), -speed, rule_cluster))
    ids, starts = np.unique(rule_cluster[ordered], return_index=True)
    np.testing.assert_array_equal(ids, np.arange(len(rule_rows)))
    r_index = pd.MultiIndex.from_frame(hits[['Row', 'Instance']]).get_indexer(pd.MultiIndex.from_frame(obs[['Row', 'Instance']]))
    require((r_index >= 0).all(), 'Missing rule hit identity')
    rules = head_arrays(obs, ordered[starts], r_length, hits.fTime.to_numpy()[r_index])
    np.testing.assert_array_equal(obs.Row.to_numpy()[ordered[starts]], rule_rows)
    rules.update(keys=np.column_stack([rule_rows, ids]), energy_GeV=rule_energy, selection=keep)
    rules.update(candidate_momentum(rule_energy, rules['direction']))
    rdir = RULES / 'export_4502291'
    receipt = read_json(rdir / 'receipt.json')
    manifest = read_json(check(rdir / 'manifest.json', receipt['manifest_sha256']))
    entry = next(x for x in manifest['files'] if x['member'] == cid + '.npz')
    with tarfile.open(rdir / 'inputs.tar', 'r:') as archive:
        with archive.extractfile(entry['member']) as stream:
            raw = stream.read()
    require(sha256(raw).hexdigest() == entry['sha256'], 'Rules comparison member changed')
    with np.load(BytesIO(raw), allow_pickle=False) as old:
        for name, values in [('rows', rule_rows), ('energy', rule_energy), ('selected', keep),
                             ('literal', literal), ('rule_rejections', ledger)]:
            np.testing.assert_array_equal(values, old[name], err_msg=f'Rule replay {name}')
    arrays = {f'{method}_{name}': value for method, fields in (('gnn', gnn), ('rules', rules))
              for name, value in fields.items()}
    return arrays, {'gnn_candidates': ncl, 'rule_candidates': len(rule_rows), 'isolated_first_hits_replayed': len(first),
                    'observation_sha256': expected['observation']['sha256'],
                    'observation_serialization_exact': observation_sha == expected['observation']['sha256'],
                    'observation_values_exact': True, 'native_target_bytes_exact': True,
                    'rule_member_sha256': entry['sha256']}


def dynamic_momenta(score_root, expected, candidates):
    """Attach each accepted fold/seed's raw DynamicEdgeConv energy by candidate key."""
    from HGNDRecoGNN.scripts.smash_architecture_reduce import (
        MODELS, SEEDS, PredictionSet, align,
    )

    cid = expected['content']
    acceptance_path = score_root / 'acceptance.json'
    accepted = read_json(acceptance_path)
    require(accepted['state'] == 'accepted' and accepted['scope'] == 'even', 'Unaccepted score summaries')
    split_path = check(ROOT / 'results/smash_train_20261028/split/split_manifest.json', accepted['split_sha256'])
    split = read_json(split_path)
    # The pack in the earlier observation plan can be from another fold.
    # The score must come from the split's held-out role for this content part.
    test_fold, = [int(fold) for fold, roles in split['folds'].items() if expected['part'] in roles['test_parts']]
    model = MODELS[1]
    sources = {}
    for task in accepted['tasks'].values():
        bundle = score_root / Path(task['bundle']).name
        meta = read_json(check(bundle / 'summary.json', task['summary_sha256']))
        if meta['model'] != model or cid not in meta['test_files']:
            continue
        seed = str(meta['seed'])
        require(seed not in sources and meta['state'] == 'complete' and meta['fold'] == test_fold
                and meta['test_scope'] == 'even'
                and meta['split_sha256'] == accepted['split_sha256'], 'DynamicEdgeConv score identity')
        sources[seed] = (bundle, task, meta)
    require(set(sources) == set(SEEDS), 'Expected all three accepted DynamicEdgeConv seeds')
    arrays, evidence = {}, {}
    index = pd.MultiIndex.from_arrays(candidates['gnn_keys'].T)
    native = (np.column_stack([candidates['gnn_native_label'], candidates['gnn_native_target_GeV']]).astype(np.float32),
              candidates['gnn_cluster_size'])
    for seed in SEEDS:
        bundle, task, meta = sources[seed]
        predictions = PredictionSet(bundle, meta['predictions'], meta['cuts'],
                                    frozenset(meta['empty_after_cuts']), meta['selection'], {})
        frame = align(predictions.frame(cid, expected['processed_span']), index, native, f'{cid}/{seed}')
        energy = frame.e_pred_raw.to_numpy()
        score = frame.cl_score.to_numpy()
        cut = meta['cuts']['70']
        quantities = candidate_momentum(energy, candidates['gnn_direction'])
        arrays[f'dynamic_seed_{seed}_energy_GeV'] = energy
        arrays[f'dynamic_seed_{seed}_score'] = score
        arrays[f'dynamic_seed_{seed}_keep70'] = score > cut
        arrays.update({f'dynamic_seed_{seed}_{k}': v for k, v in quantities.items()})
        evidence[seed] = {'fold': meta['fold'], 'threshold70': cut, 'summary_sha256': task['summary_sha256'],
                          'prediction_sha256': meta['predictions'][cid]['sha256'], 'e_en': meta['selection']['e_en']}
    return arrays, {'acceptance_sha256': sha(acceptance_path), 'split_sha256': sha(split_path),
                    'observation_pack_fold': expected['fold'], 'evaluation_fold': test_fold, 'seeds': evidence}


def run(args, *, verbose=True):
    start = monotonic()
    first_meta = read_json(FIRST / 'export/summary.json')
    plan_path = check(FIRST / 'plan.json', first_meta['plan_sha256'])
    expected, = [r for r in read_json(plan_path)['files'] if r['content'] == args.content]
    require(expected['part'] in (0, 2, 4), 'Only accepted even contents are supported')
    accept = read_json(PRIMARY / 'acceptance.json')
    primary_summary = check(PRIMARY / 'result/summary.json', accept['summary_sha256'])
    record, = [r for r in read_json(primary_summary)['files'] if r['content'] == args.content]
    accepted_path = check(PRIMARY / 'result' / args.content / 'neutrons.npz', record['files']['neutrons.npz'])
    events_path = check(PRIMARY / 'result' / args.content / 'events.csv.gz', record['files']['events.csv.gz'])
    check(args.primary, expected['primary_sha256'])
    check(args.vacs, expected['sha256']['vacs'])
    require((args.hits is None) == (args.pack is None), 'Pass hits and native pack together')
    require(args.hits is not None or args.observation is None and args.dynamic_scores is None,
            'Candidate inputs require hits and native pack')
    if args.hits is not None:
        check(args.hits, expected['sha256']['hits'])
        check(args.pack, expected['pack']['sha256'])
    require(not args.output.exists(), 'Output already exists')
    vacs = read_csv(args.vacs, VACS)
    arrays, primary = primary_arrays(args.primary, expected, accepted_path, events_path, vacs)
    candidate = None
    if args.hits is not None:
        extra, candidate = candidate_arrays(args.hits, args.pack, vacs, expected, accepted_path, args.observation)
        arrays.update(extra)
        if args.dynamic_scores is not None:
            extra, candidate['dynamic_scores'] = dynamic_momenta(
                args.dynamic_scores, expected, arrays)
            arrays.update(extra)
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.output / 'kinematics.npz'
    np.savez_compressed(data, **arrays)
    info = {'schema': 'smash-rapidity-kinematics-v1', 'state': 'complete', 'content': args.content,
            'processed_span': expected['processed_span'], 'primary': primary, 'candidates': candidate,
            'units': {'momentum': 'GeV/c', 'energy_mass': 'GeV with c=1', 'vertex_hit': 'cm',
                      'time': 'ns', 'length': 'm', 'rapidity_direction': 'dimensionless'},
            'nn_cm': {'beam_kinetic_GeV_per_nucleon': BEAM_KINETIC, 'equal_nucleon_mass_GeV': NUCLEON_MASS,
                      'lab_rapidity_offset': float(Y_CM), 'convention': 'fixed-target NN center of mass'},
            'weighting': 'unweighted object inventory; the primary CSV does not supply generator weights here',
            'phase_space': 'accepted geometric neutrons plus linked out-of-window neutrons; geometric protons only',
            'direction': 'nominal origin to detector hit, original-time earliest for GNN, input-time fastest beta for rules',
            'source_sha256': {'plan': sha(plan_path), 'primary': expected['primary_sha256'],
                              'vacs': expected['sha256']['vacs'], 'hits': expected['sha256']['hits'] if candidate else None,
                              'pack': expected['pack']['sha256'] if candidate else None,
                              'associations': sha(accepted_path), 'events': sha(events_path)},
            'output_sha256': sha(data), 'script_sha256': sha(Path(__file__)), 'elapsed_seconds': monotonic() - start}
    (args.output / 'receipt.json').write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
    if verbose:
        print(json.dumps(info, indent=2, allow_nan=False))
    return info


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--content', required=True)
    for name in ('primary', 'vacs', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('hits', 'pack', 'observation', 'dynamic-scores'):
        parser.add_argument('--' + name, type=Path)
    run(parser.parse_args())
