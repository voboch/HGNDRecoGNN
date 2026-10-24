"""Frozen GraphSAGE DCM-to-SMASH development diagnostic, with no fitting."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN import models, device  # noqa: E402
from HGNDRecoGNN.data.smash_transfer import transfer_pack  # noqa: E402
from HGNDRecoGNN.data.student_pipeline import NativePackDataset  # noqa: E402
from HGNDRecoGNN.scripts.audit_smash_sources import TAGS, matching  # noqa: E402
from HGNDRecoGNN.scripts.diagnose_student_pipeline import clean, measure  # noqa: E402
from HGNDRecoGNN.training.checkpoint import load_checkpoint  # noqa: E402
from HGNDRecoGNN.training.eval import predict  # noqa: E402

NATIVE_SOURCE_SHA = '56bcc48206186e32e182a801410d42865b7f25df656d85166e7a64f67a67fe96'
SPLIT_SHA = '1ab9285e3315dc0f043b7d5116255f1459978477dc9a9b4d95e046649163278b'
INVENTORY_SHA = '9ca9df3edc0fafdc9d2e1720f37136443d41d96424966ec4fd43c35be786e110'
SEEDS = (42, 123, 456)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def frozen_dcm(args):
    manifest_path = args.prepared/'manifest.json'
    if sha(manifest_path) != SPLIT_SHA:
        raise ValueError('Approved DCM preprocessing changed')
    manifest = json.loads(manifest_path.read_text())
    for name, h in manifest['sha256'].items():
        if sha(args.prepared/name) != h:
            raise ValueError('Frozen train-only scaler/remap changed')
    old_root = Path(manifest['data_root'])/'processed'
    old = [pd.read_pickle(old_root/f'scaler_{side}.pkl') for side in ['bot', 'top']]
    new = [pd.read_pickle(args.prepared/f'scaler_{side}.pkl') for side in ['bot', 'top']]
    arrays = np.load(args.prepared/'indices_and_scaling.npz')
    a, b = arrays['a'], arrays['b']
    for side in (0, 1):
        assert np.array_equal(old[side].scale_/new[side].scale_, a[side])
        assert np.array_equal((old[side].mean_-new[side].mean_)/new[side].scale_, b[side])
    prepared = json.loads((args.native_campaign/'prepared.json').read_text())
    assert prepared['source_sha256'] == NATIVE_SOURCE_SHA
    train_path = args.native_campaign/'packs/train.pt'
    assert sha(train_path) == prepared['files']['train.pt']
    train = torch.load(train_path, weights_only=True, map_location='cpu')
    x = train['x'].numpy()
    sides = np.repeat(train['side'].numpy(), np.diff(train['node_ptr'].numpy()))
    cells = np.empty((len(x), 3), dtype=np.int64)
    # The original features encode integer cell coordinates. Rounding the
    # inverse float32 remap recovers the discrete cell IDs; no cut is fitted.
    for side in (0, 1):
        m = sides == side
        raw = (x[m, 3:6].astype(np.float64)-b[side, 3:6])/a[side, 3:6]
        raw = raw*old[side].scale_[3:6]+old[side].mean_[3:6]
        cells[m] = np.rint(raw).astype(np.int64)
    geometry = {name: [int(cells[:, i].min()), int(cells[:, i].max())]
                for i, name in enumerate(['LayerId', 'RowId', 'ColumnId'])}
    surface = np.zeros(len(x), dtype=bool)
    for i, (lo, hi) in enumerate(geometry.values()):
        surface |= (cells[:, i] == lo) | (cells[:, i] == hi)
    if not np.array_equal(surface.astype(np.float32), x[:, -1]):
        raise ValueError('DCM SurfaceHit does not match recovered geometry')
    del train, x, cells
    provenance = {'split_sha256': SPLIT_SHA, 'geometry_from': 'verified DCM training inputs only',
                  'geometry_bounds': geometry, 'train_pack_sha256': prepared['files']['train.pt'],
                  'scalers_fit_on_smash': False,
                  'old_scalers_sha256': {s: sha(old_root/f'scaler_{s}.pkl') for s in ['bot','top']},
                  'prepared_files_sha256': manifest['sha256']}
    return old, a, b, geometry, provenance


def observation(path, expected):
    if sha(path) != expected:
        raise ValueError('Audited observed hits changed')
    p = torch.load(path, weights_only=True, map_location='cpu')
    return pd.DataFrame(p['values'].numpy(), columns=p['columns'])


def validate_audit_summary(summary, frozen, expected, inventory_sha):
    """Keep header-only exports in coverage without inventing source evidence."""
    members = [f"{f['run']}/{f['stem']}" for f in summary['files']]
    if summary['state'] != 'complete' or summary['inventory_sha256'] != inventory_sha or members != expected:
        raise ValueError('Incomplete or changed audited population')
    empty, aliases = [], []
    for member, file in zip(members, summary['files']):
        alias = file.get('duplicate_of')
        if alias is not None:
            first = next((f for f in summary['files'] if f"{f['run']}/{f['stem']}" == alias), None)
            if (frozen.get('duplicate_exports', {}).get(member) != alias
                    or first is None or first.get('duplicate_of') or first.get('header_only_export')
                    or first['sha256'] != file['sha256']
                    or first['primary_source_member'] != file['primary_source_member']
                    or file.get('observations') is not None):
                raise ValueError('Invalid exact-byte alias accounting')
            aliases.append(member)
        elif file.get('header_only_export', False):
            declaration = frozen.get('empty_exports', {}).get(member)
            if (declaration != {'sha256': file['sha256'], 'exported_hits': 0, 'exported_truth_records': 0}
                    or file.get('exported_hits') != 0 or file.get('exported_truth_records') != 0
                    or file.get('observed_export_events') != 0
                    or file.get('primary_source_member') is not None or file.get('primary_probe') is not None):
                raise ValueError('Invalid header-only accounting')
            empty.append(member)
        elif file.get('primary_probe') is None:
            raise ValueError('Nonempty detector export lacks binding evidence')
        if 'input_sha256' in frozen and file['sha256'] != frozen['input_sha256'].get(member):
            raise ValueError('Audited input hashes differ from frozen mapping')
    binding = summary['primary_binding_probe']
    if (summary.get('primary_probe_files') + len(empty) + len(aliases) != len(expected)
            or not (binding['eligible_records'] == binding['matched_ids'] == binding['pdg_and_vertex_agree'])):
        raise ValueError('Primary binding gate failed')
    if frozen.get('binding_policy') == 'exact_initial_mctrack_all_files':
        if binding['eligible_records'] != binding.get('pdg_vertex_kinematics_agree'):
            raise ValueError('Exact primary kinematics gate failed')
        for member, file in zip(members, summary['files']):
            if member in empty:
                continue
            mapped = frozen['primary_member_map'][member]
            if (file['primary_source_member'] != mapped
                    or file['primary_probe_sha256'] != frozen['primary_sha256'][mapped]):
                raise ValueError('Audited primary source differs from exact map')
    hashes = [(f['sha256']['hits'], f['sha256']['vacs']) for f in summary['files']
              if not f.get('header_only_export', False) and not f.get('duplicate_of')]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate nonempty detector exports')
    if empty or aliases:
        accounting = summary['file_accounting']
        if (accounting['header_only_exports'] != len(empty)
                or accounting['bound_nonempty_files'] != len(expected)-len(empty)-len(aliases)
                or accounting.get('duplicate_byte_aliases', 0) != len(aliases)
                or accounting['processed_root_events'] is not None
                or accounting['generated_event_count'] is not None):
            raise ValueError('Unknown processed-event accounting was changed')


def verify_audit(args, inventory):
    source = args.audit_campaign/('smoke' if args.smoke else 'audit')/args.tag
    summary_path = source/'summary.json'
    summary = json.loads(summary_path.read_text())
    expected = inventory['tags'][args.tag]['common_primary_pair_names']
    if args.smoke:
        expected = expected[:1]
    validate_audit_summary(summary, inventory['tags'][args.tag], expected, args.inventory_sha256)
    if not args.smoke:
        # Require complete audits of all potentials before any full inference.
        for tag in TAGS:
            other = json.loads((args.audit_campaign/'audit'/tag/'summary.json').read_text())
            validate_audit_summary(other, inventory['tags'][tag],
                                   inventory['tags'][tag]['common_primary_pair_names'], args.inventory_sha256)
    return source, summary, sha(summary_path)


def check_prediction(pack, frame):
    lengths = np.diff(pack['cluster_ptr'].numpy())
    rows = np.repeat(pack['row'].numpy(), lengths)
    sides = np.repeat(pack['side'].numpy(), lengths)
    cid = np.arange(lengths.sum())-np.repeat(pack['cluster_ptr'][:-1].numpy(), lengths)
    assert np.array_equal(rows, frame.Row)
    assert np.array_equal(sides, frame.cl_istop)
    assert np.array_equal(cid, frame.ClusterID)
    assert np.array_equal(pack['label'].numpy(), frame.cl_label)
    assert np.array_equal(pack['energy'].numpy(), frame.e_true)
    assert np.isfinite(frame[['cl_score','e_pred','e_true']]).all(axis=None)


def primary_associations(frame, pack, predictions, threshold):
    rows = frame.Row.to_numpy(np.int64)
    sides = (frame.fY.to_numpy() > 0).astype(np.int64)
    order = np.lexsort((np.arange(len(rows)), rows, sides))
    f = frame.iloc[order]
    cluster = pack['cluster'].numpy() + np.repeat(pack['cluster_ptr'][:-1].numpy(), np.diff(pack['node_ptr'].numpy()))
    direct = ((f.PDG.abs() == 2112) & (f.fMotherId == -1) & f.Side.isin([5, 6])).to_numpy()
    selected = predictions.cl_score.to_numpy() > threshold
    n = len(predictions)
    before = matching(cluster, f.Row.to_numpy(), f.Id.to_numpy(), direct, n)
    after = matching(cluster, f.Row.to_numpy(), f.Id.to_numpy(), direct & selected[cluster], n)
    with_primary = np.zeros(n, dtype=bool)
    np.maximum.at(with_primary, cluster, direct)
    return {'scope': 'direct primary-front-face neutron associations after hit cut; no ancestry closure',
            'all_candidates': before, 'selected_candidates': after,
            'selected_without_direct_primary_neutron': int((selected & ~with_primary).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-campaign', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-sha256', default=INVENTORY_SHA,
                        help='Explicitly pinned inventory hash; changing population requires a new protocol')
    parser.add_argument('--native-campaign', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tag', choices=TAGS, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--probe-first', action='store_true',
                        help='Validate every full audit but infer only the first file; CUDA engineering smoke')
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    assert sha(args.inventory) == args.inventory_sha256
    inventory = json.loads(args.inventory.read_text())
    source, audit, audit_sha = verify_audit(args, inventory)
    args.output.mkdir(parents=True, exist_ok=False)
    scalers, a, b, geometry, dcm_receipt = frozen_dcm(args)
    if not args.smoke:
        for tag in TAGS:
            data = json.loads((args.audit_campaign/'audit'/tag/'summary.json').read_text())
            for col, bound in geometry.items():
                observed = [f['geometry_bounds'][col] for f in data['files'] if f['geometry_bounds']]
                if [min(x[0] for x in observed), max(x[1] for x in observed)] != bound:
                    raise ValueError('Common SMASH geometry differs from DCM: '+col)
    write(args.output/'dcm_inputs.json', dcm_receipt)
    started = time.monotonic()
    model_objects, collected, receipts = {}, {s: [] for s in SEEDS}, []
    association_totals = {s: [] for s in SEEDS}
    files = audit['files'][:1] if args.probe_first else audit['files']
    for file in files:
        member = file['run']+'/'+file['stem']
        if file.get('duplicate_of'):
            receipts.append({'member': member, 'duplicate_of': file['duplicate_of'],
                             'excluded_exact_byte_alias': True})
            continue
        f = observation(source/file['observations']['path'], file['observations']['sha256'])
        pack = transfer_pack(f, scalers, a, b, geometry)
        if pack is None:
            receipts.append({'member': member, 'empty_after_cuts': True,
                             'header_only_export': file.get('header_only_export', False)})
            continue
        dataset = NativePackDataset(pack)
        if not model_objects:
            for seed in SEEDS:
                folder = args.native_campaign/f'student_sage_{seed}'
                complete = json.loads((folder/'complete.json').read_text())
                assert complete['config']['source_sha256'] == NATIVE_SOURCE_SHA
                assert complete['config']['prepared_sha256'] == sha(args.native_campaign/'prepared.json')
                path = folder/'checkpoints/model.pt'
                assert sha(path) == complete['files']['checkpoints/model.pt']
                checkpoint = load_checkpoint(str(path))
                assert checkpoint.epoch == complete['selected_epoch'] and checkpoint.arch_name == 'student_sage'
                model, spec = models.get('student_sage', dataset, **complete['config']['architecture'])
                model.load_state_dict(checkpoint.state_dict)
                plan = device.plan_for_spec(model, args.device, spec.default_cpu_pinned)
                device.to_device(model, plan)
                if args.device == 'cuda' and next(model.parameters()).device.type != 'cuda':
                    raise ValueError('Frozen GraphSAGE did not move to CUDA')
                model_objects[seed] = (model, spec, plan, complete['validation_threshold'], sha(path))
        item = {'member': member, 'observation_sha256': file['observations']['sha256'], 'graphs': len(dataset),
                'clusters': len(pack['label']), 'signal': int(pack['label'].sum()), 'outputs': {}}
        for seed in SEEDS:
            model, spec, plan, threshold, checkpoint_sha = model_objects[seed]
            predictions = predict(model, DataLoader(dataset, batch_size=256, shuffle=False),
                                  plan=plan, forward_fn=spec.forward_fn, verbose=False)['clusters']
            check_prediction(pack, predictions)
            # Observable candidate size is exported for fixed-bin diagnostics;
            # it does not change the frozen model, labels, scores or cut.
            global_cluster = pack['cluster'].numpy()+np.repeat(pack['cluster_ptr'][:-1].numpy(), np.diff(pack['node_ptr'].numpy()))
            predictions['cluster_size'] = np.bincount(global_cluster, minlength=len(predictions))
            folder = args.output/f'seed_{seed}'/file['run']
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/(file['stem']+'.csv')
            predictions.to_csv(path, index=False, float_format='%.9g')
            collected[seed].append(predictions[['cl_score','cl_label','e_pred','e_true']])
            association = primary_associations(f, pack, predictions, threshold)
            association_totals[seed].append(association)
            item['outputs'][str(seed)] = {'path': str(path.relative_to(args.output)), 'sha256': sha(path),
                                        'threshold': threshold, 'checkpoint_sha256': checkpoint_sha,
                                        'metrics': measure(predictions, threshold), 'direct_primary': association}
        receipts.append(item)
        write(args.output/'progress.json', {'state': 'running', 'files_done': len(receipts),
                                           'files_expected': len(files), 'elapsed_seconds': time.monotonic()-started})
        print(args.tag, len(receipts), '/', len(files), 'elapsed', round(time.monotonic()-started, 1), flush=True)
    result = {'state': 'complete', 'scope': 'retrospective common-file SMASH development; not independent test',
              'tag': args.tag, 'smoke': args.smoke or args.probe_first, 'audit_summary_sha256': audit_sha,
              'inventory_sha256': args.inventory_sha256, 'native_source_sha256': NATIVE_SOURCE_SHA,
              'graph_recipe': {'radius_cells': 3.6, 'time_ns': 1.0, 'scalers': 'frozen DCM training'},
              'no_training_or_calibration': True, 'files': receipts, 'per_seed': {},
              'device_requested': args.device, 'model_device': str(next(model_objects[SEEDS[0]][0].parameters()).device),
              'elapsed_seconds': time.monotonic()-started}
    for seed in SEEDS:
        frame = pd.concat(collected[seed], ignore_index=True)
        result['per_seed'][str(seed)] = {'threshold': model_objects[seed][3],
                                        'checkpoint_sha256': model_objects[seed][4],
                                        'metrics': measure(frame, model_objects[seed][3])}
    write(args.output/'summary.json', result)


if __name__ == '__main__':
    main()
