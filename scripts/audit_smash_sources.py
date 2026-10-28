"""Read-only corrected-SMASH inventory, label and candidate/object ledger.

No model is trained or evaluated. New noise is a reproducible audit view,
shared by both label definitions; historical cache noise is not reconstructed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_bipartite_matching
import torch
from torch_geometric.nn import radius_graph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.data.graph_dataset import _dfToF  # noqa: E402

HITS = ['Row', 'Instance', 'fX', 'fY', 'fZ', 'DetectorId', 'LayerId', 'RowId', 'ColumnId', 'fTime', 'fELoss']
VACS = ['Row', 'Instance', 'Id', 'PDG', 'Ekin', 'Rapid', 'fMotherId', 'Weight', 'DetectorID', 'Side', 'X', 'Y', 'Z', 'vX', 'vY', 'vZ', 'Px', 'Py', 'Pz']
TAGS = {'zeroSpot': 0, 'defaultSpot': 18, 'bigSpot': 90}
# Existing production conventions, not optimised here. The same realization
# and graph are used for both targets to isolate removal of the Ekin floor.
POLICY = {'edep_GeV': .003, 'sigma_time_ns': .15, 'noise_seed': 42,
          'radius_cells': 3.6, 'edge_time_ns': 1.5, 'max_neighbors': 32,
          'native_floor_GeV': .3, 'prompt_half_width_ns': .3}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def read_csv(path, names):
    with path.open() as f:
        header = [s.strip() for s in f.readline().strip().split(',') if s.strip()]
    if header != names:
        raise ValueError(f'Unexpected CSV schema: {path}')
    return pd.read_csv(path, names=names, skiprows=1, usecols=range(len(names)), dtype=float)


def native_components(frame, time_window=None):
    """Batched native radius intersection; includes singleton components."""
    if not len(frame):
        return np.empty(0, dtype=np.int64)
    keys = pd.MultiIndex.from_arrays([frame.Row, frame.fY > 0])
    batch, _ = pd.factorize(keys, sort=False)
    order = np.argsort(batch, kind='stable')
    batch_t = torch.tensor(batch[order], dtype=torch.long)
    pos = torch.tensor(frame[['LayerId', 'RowId', 'ColumnId']].to_numpy(np.float32)[order])
    t = torch.tensor(frame.fTime.to_numpy(np.float32)[order, None])
    spatial = radius_graph(pos, r=POLICY['radius_cells'], batch=batch_t, loop=True, max_num_neighbors=32)
    temporal = radius_graph(t, r=POLICY['edge_time_ns'] if time_window is None else time_window,
                            batch=batch_t, loop=True, max_num_neighbors=32)
    stride = len(frame)+1
    a = spatial[0].numpy()*stride+spatial[1].numpy()
    b = temporal[0].numpy()*stride+temporal[1].numpy()
    edges = spatial.numpy()[:, np.isin(a, b)]
    _, lab = connected_components(csr_matrix((np.ones(edges.shape[1]), edges), shape=(len(frame), len(frame))), directed=False)
    restored = np.empty(len(frame), dtype=np.int64)
    restored[order] = lab
    return np.unique(restored, return_inverse=True)[1]


def matching(cluster, rows, ids, mask, n_clusters):
    if not mask.any():
        return {'eligible_objects': 0, 'signal_clusters': 0, 'matched_objects': 0,
                'merged_clusters': 0, 'split_objects': 0}
    objects, inverse = np.unique(np.column_stack([rows[mask], ids[mask]]), axis=0, return_inverse=True)
    pairs = np.unique(np.column_stack([cluster[mask], inverse]), axis=0)
    adj = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_clusters, len(objects)))
    n_obj = np.diff(adj.indptr)
    n_cl = np.diff(adj.tocsc().indptr)
    match = maximum_bipartite_matching(adj, perm_type='column')
    return {'eligible_objects': len(objects), 'signal_clusters': int((n_obj > 0).sum()),
            'matched_objects': int((match >= 0).sum()),
            'merged_clusters': int((n_obj > 1).sum()), 'split_objects': int((n_cl > 1).sum())}


def ledger(frame, cluster):
    nc = int(cluster.max())+1 if len(cluster) else 0
    ek = frame.Ekin.to_numpy(float)
    pdg = np.abs(frame.PDG.to_numpy())
    distance = np.linalg.norm(frame[['fX', 'fY', 'fZ']].to_numpy().T/100., axis=0)
    t = frame.fTime.to_numpy()
    lo, hi = _dfToF(distance, t+.3), _dfToF(distance, t-.3)
    relaxed = (pdg == 2112) & frame.Side.isin([5, 6]).to_numpy() & (ek >= lo) & (ek <= hi)
    native = relaxed & (ek > .3)
    energy_old, energy_new = np.zeros(nc), np.zeros(nc)
    label_old, label_new = np.zeros(nc, dtype=np.int8), np.zeros(nc, dtype=np.int8)
    np.maximum.at(label_old, cluster, native)
    np.maximum.at(label_new, cluster, relaxed)
    np.maximum.at(energy_old, cluster, np.where(native, ek, 0.))
    np.maximum.at(energy_new, cluster, np.where(relaxed, ek, 0.))
    assert not np.any((label_old == 1) & (label_new == 0))
    assert np.array_equal(energy_old[label_old == 1], energy_new[label_old == 1])
    changed = (label_new == 1) & (label_old == 0)
    assert np.all(energy_new[changed] <= .3)
    sizes = np.bincount(cluster, minlength=nc)
    return {'observed_hits': len(frame), 'clusters': nc,
            'native_signal_hits': int(native.sum()), 'relaxed_signal_hits': int(relaxed.sum()),
            'added_soft_hits': int((relaxed & ~native).sum()),
            'native_signal_clusters': int(label_old.sum()), 'relaxed_signal_clusters': int(label_new.sum()),
            'added_soft_clusters': int(changed.sum()),
            'added_soft_singletons': int((changed & (sizes == 1)).sum()),
            'native_target_changes_above_floor': 0,
            'native_matching': matching(cluster, frame.Row.to_numpy(), frame.Id.to_numpy(), native, nc),
            'relaxed_matching': matching(cluster, frame.Row.to_numpy(), frame.Id.to_numpy(), relaxed, nc)}


def noise_seed(noise_key):
    """Path-seeded audit jitter; the key is the hits path relative to the tag root."""
    return int.from_bytes(hashlib.sha256((noise_key+':audit-noise-v1:42').encode()).digest()[:8], 'little')


def observe(hf, vf, noise_key, hp=None, vp=None):
    """Inherited observation recipe of the audit, shared with SMASH training packs.

    Edep > 3 MeV and finite hit fields, one-to-one MC association, path-seeded
    0.15 ns time jitter, fY != 0, and the production half-event PDG==0
    rejection. Returns (threshold-pass hits, kept observed hits, rejected count).
    `hp`/`vp` only label error messages.
    """
    if vf.duplicated(['Row', 'Instance']).any() or hf.duplicated(['Row', 'Instance']).any():
        raise ValueError(f'Duplicate local Row/Instance in {hp}')
    obs = hf.loc[(hf.fELoss > .003) & np.isfinite(hf[HITS]).all(axis=1)].copy()
    truth_cols = ['Row', 'Instance', 'Id', 'PDG', 'Ekin', 'Side', 'fMotherId', 'vX', 'vY', 'vZ']
    merged = obs.merge(vf[truth_cols], on=['Row', 'Instance'], how='left', validate='one_to_one', indicator=True)
    missing = merged._merge != 'both'
    if missing.any():
        raise ValueError(f'{int(missing.sum())} observed hits lack MC association in {hp}')
    if not np.isfinite(merged[['Id', 'PDG', 'Ekin', 'Side']]).all(axis=None):
        raise ValueError(f'Invalid truth fields in {vp}')
    merged['fTime'] += np.random.default_rng(noise_seed(noise_key)).normal(0, .15, len(merged))
    merged = merged.loc[merged.fY != 0].copy()
    # Preserve the production half-event PDG==0 rejection, and record it.
    keys = pd.MultiIndex.from_arrays([merged.Row, merged.fY > 0])
    bad_keys = keys[merged.PDG.to_numpy() == 0].unique()
    rejected = keys.isin(bad_keys)
    dropped = int(rejected.sum())
    kept = merged.loc[~rejected].copy()
    return obs, kept, dropped


def add(total, values):
    for k, v in values.items():
        if isinstance(v, dict):
            add(total.setdefault(k, {}), v)
        else:
            total[k] = total.get(k, 0)+int(v)


def primary_probe(vacs, primary, exact_kinematics=False):
    """Engineering binding check, restricted to primary-front-face records.

    Compare IDs, PDG and production vertices, not transported kinetic energy.
    2e-6 cm is two units of the exported six-decimal representation.
    """
    # The primary inventory exports neutrons/protons only. Missing mesons in
    # this inventory do not imply a failed nucleon association.
    use = vacs[(vacs.fMotherId == -1) & vacs.Side.isin([5, 6])
               & vacs.PDG.abs().isin([2112, 2212])].drop_duplicates(['Row', 'Id'])
    if not len(use):
        return {'eligible_records': 0, 'matched_ids': 0, 'pdg_and_vertex_agree': 0}
    cols = ['Row', 'Id', 'PDG', 'vX', 'vY', 'vZ']
    if exact_kinematics:
        cols += ['Ekin', 'Px', 'Py', 'Pz']
    prim = pd.read_csv(primary, usecols=cols)
    if prim.duplicated(['Row', 'Id']).any():
        raise ValueError('Primary Row/Id is not unique')
    joined = use[cols].merge(prim, on=['Row', 'Id'], suffixes=('_det', '_prim'), how='inner')
    agree = joined.PDG_det.to_numpy() == joined.PDG_prim.to_numpy()
    for c in ['vX', 'vY', 'vZ']:
        agree &= np.isclose(joined[c+'_det'], joined[c+'_prim'], rtol=0, atol=2e-6)
    neutrons = joined.PDG_det.abs().to_numpy() == 2112
    result = {'eligible_records': len(use), 'matched_ids': len(joined), 'pdg_and_vertex_agree': int(agree.sum()),
            'eligible_neutrons': int((use.PDG.abs() == 2112).sum()),
            'matched_neutrons': int(neutrons.sum()),
            'neutron_pdg_and_vertex_agree': int((neutrons & agree).sum())}
    if exact_kinematics:
        exact = np.ones(len(joined), dtype=bool)
        for col in cols[2:]:
            exact &= joined[col+'_det'].to_numpy() == joined[col+'_prim'].to_numpy()
        result['pdg_vertex_kinematics_agree'] = int(exact.sum())
    return result


def run(args):
    tag = args.tag
    root = args.raw_root/('smash_xecs_2.87gev_hardSkyrme_'+tag)
    primary_root = args.primary_root/root.name
    args.output.mkdir(parents=True, exist_ok=False)
    files = []
    inventory_path = getattr(args, 'inventory', None)
    frozen = json.loads(inventory_path.read_text())['tags'][tag] if inventory_path else {}
    staged = getattr(args, 'staged_root', None)
    if staged is not None and (not inventory_path or not (staged/'bundle_verified.json').is_file()):
        raise ValueError('Staged inputs require a frozen inventory and verified bundle')
    # Compare directory names before opening explicit frozen members. Calling
    # stat on every unrelated file made a one-file smoke traverse the full
    # source and fail before its intended engineering probe.
    if staged is None:
        hits = {str(p.relative_to(root)).removesuffix('_hits.csv') for p in root.glob('*/*_hits.csv')}
        vacs = {str(p.relative_to(root)).removesuffix('_vacs.csv') for p in root.glob('*/*_vacs.csv')}
        if hits != vacs:
            raise ValueError(f'{len(hits ^ vacs)} unpaired CSV names')
        members = sorted(hits)
        if inventory_path and frozen['all_pair_names'] != members:
            raise ValueError('Corrected source membership changed')
    if inventory_path:
        # With verified copies, read only the frozen members. Per-file hashes
        # below bind every input; listing unrelated Lustre paths is unnecessary.
        members = frozen['common_primary_pair_names']
        primary_map = frozen.get('primary_member_map')
        if primary_map is not None and set(primary_map) != set(members):
            raise ValueError('Explicit primary map does not cover the frozen members')
    else:
        primary_map = None
    for member in members:
        files.append((root/(member+'_hits.csv'), root/(member+'_vacs.csv')))
    if not files:
        raise ValueError('No detector export pairs found')
    files_total = len(files)
    if args.limit_files is not None:
        files = files[:args.limit_files]
    result = {'tag': tag, 'U_sym_MeV': TAGS[tag], 'state': 'running', 'policy': POLICY,
              'raw_root': str(root), 'primary_root': str(primary_root), 'files_expected': len(files),
              'files_processed': 0, 'files_total_in_source': files_total,
              'limited_engineering_probe': args.limit_files is not None,
              'population': 'frozen corrected/primary common-file development population' if inventory_path else 'corrected detector exports',
              'inventory_sha256': sha(inventory_path) if inventory_path else None,
              'staged_inputs_verified': staged is not None,
              'original_directory_listing_rechecked': staged is None,
              'totals': {}, 'primary_binding_probe': {}, 'primary_probe_files': 0, 'files': [],
              'file_accounting': {'bound_nonempty_files': 0, 'header_only_exports': 0,
                                  'duplicate_byte_aliases': 0,
                                  'nonempty_exports': 0, 'nonempty_exports_empty_after_cuts': 0,
                                  'processed_root_events': None, 'generated_event_count': None},
              'scope': 'corrected detector-export observable view; no generated-neutron closure or model inference',
              'noise': 'new deterministic 0.15ns audit view; identical for both label definitions',
              'generator_job_identity_established': False, 'fresh_blind_holdout_established': False}
    started = time.monotonic()
    seen_content = {}
    for i, (hp, vp) in enumerate(files):
        member = str(hp.relative_to(root)).removesuffix('_hits.csv')
        original_hp = hp
        if staged is not None:
            hp, vp = [staged/'raw'/tag/(member+s) for s in ['_hits.csv', '_vacs.csv']]
        hashes = {'hits': sha(hp), 'vacs': sha(vp)}
        pinned = frozen.get('input_sha256', {}).get(member)
        if 'input_sha256' in frozen and pinned != hashes:
            raise ValueError('Frozen detector content changed: '+member)
        alias = frozen.get('duplicate_exports', {}).get(member)
        if alias is not None:
            previous = next((f for f in result['files'] if f"{f['run']}/{f['stem']}" == alias), None)
            if (previous is None or previous.get('duplicate_of') is not None
                    or previous.get('header_only_export') or previous['sha256'] != hashes
                    or primary_map[member] != previous['primary_source_member']):
                raise ValueError('Invalid exact-byte alias: '+member)
            result['files'].append({**previous, 'run': hp.parent.name,
                                    'stem': hp.name.removesuffix('_hits.csv'),
                                    'duplicate_of': alias, 'observations': None,
                                    'primary_probe': None, 'primary_binding_state': 'verified_byte_alias'})
            result['file_accounting']['duplicate_byte_aliases'] += 1
            result['files_processed'] = i+1
            continue
        hf, vf = read_csv(hp, HITS), read_csv(vp, VACS)
        header_only = len(hf) == len(vf) == 0
        empty_declared = frozen.get('empty_exports', {}).get(member)
        if header_only and inventory_path:
            if empty_declared != {'sha256': hashes, 'exported_hits': 0, 'exported_truth_records': 0}:
                raise ValueError('Header-only export lacks explicit accounting: '+member)
        elif empty_declared is not None:
            raise ValueError('Declared empty export contains rows: '+member)
        pair_hash = (hashes['hits'], hashes['vacs'])
        if len(hf) or len(vf):
            if pair_hash in seen_content:
                raise ValueError(f'Duplicate nonempty export pair: {hp} and {seen_content[pair_hash]}')
            seen_content[pair_hash] = str(hp)
        obs, kept, dropped = observe(hf, vf, str(original_hp.relative_to(root)), hp, vp)
        inventory = np.union1d(hf.Row.unique(), vf.Row.unique())
        arrival = vf.loc[(vf.PDG.abs() == 2112) & vf.Side.isin([5, 6])]
        # The same transported object may have multiple hit associations.
        # This is detector-export coverage, not the generated inventory.
        arrival_ids = arrival[['Row', 'Id']].drop_duplicates()
        hard_arrival_ids = arrival.loc[arrival.Ekin > .3, ['Row', 'Id']].drop_duplicates()
        cluster = native_components(kept)
        counts = ledger(kept, cluster)
        counts.update(export_events=len(inventory), exported_hits=len(hf), exported_truth_records=len(vf),
                      detector_export_neutron_objects=len(arrival_ids),
                      detector_export_neutron_objects_above_floor=len(hard_arrival_ids),
                      observed_hit_threshold_pass=int(len(obs)), native_rejected_hits_pdg0=dropped)
        add(result['totals'], counts)
        probe = None
        primary_member = primary_map[member] if primary_map is not None else member
        primary = None
        if header_only:
            if primary_member is not None:
                raise ValueError('Header-only export cannot establish primary binding: '+member)
            result['file_accounting']['header_only_exports'] += 1
        else:
            result['file_accounting']['nonempty_exports'] += 1
            if not len(kept):
                result['file_accounting']['nonempty_exports_empty_after_cuts'] += 1
            if primary_member is None:
                raise ValueError('Nonempty export lacks primary binding: '+member)
            if Path(primary_member).is_absolute() or '..' in Path(primary_member).parts:
                raise ValueError('Unsafe primary source member')
            primary = primary_root/(primary_member+'_prim.csv')
            if staged is not None:
                primary = staged/'primary'/tag/(primary_member+'_prim.csv')
            if inventory_path and not primary.is_file():
                raise ValueError('Bound primary file missing: '+str(primary))
        # First three available files per potential are an engineering lineage
        # probe, not an unbiased primary-efficiency or physics measurement.
        if primary is not None and primary.is_file() and (inventory_path or result.get('primary_probe_files', 0) < 3):
            primary_hash = sha(primary)
            expected_hash = frozen.get('primary_sha256', {}).get(primary_member)
            if 'primary_sha256' in frozen and expected_hash != primary_hash:
                raise ValueError('Frozen primary content changed: '+primary_member)
            exact = frozen.get('binding_policy') == 'exact_initial_mctrack_all_files'
            probe = primary_probe(vf, primary, exact_kinematics=exact)
            if inventory_path and (probe['eligible_records'] != probe['matched_ids']
                                   or probe['matched_ids'] != probe['pdg_and_vertex_agree']
                                   or (exact and probe['matched_ids'] != probe.get('pdg_vertex_kinematics_agree'))):
                raise ValueError(f'Primary nucleon binding failed: {primary}; {json.dumps(probe, sort_keys=True)}')
            add(result['primary_binding_probe'], probe)
            result['primary_probe_files'] = result.get('primary_probe_files', 0)+1
            result['file_accounting']['bound_nonempty_files'] += 1
        saved = None
        if getattr(args, 'save_observations', False):
            folder = args.output/'observations'/hp.parent.name
            folder.mkdir(parents=True, exist_ok=True)
            saved_path = folder/(hp.name.removesuffix('_hits.csv')+'.pt')
            columns = [c for c in kept.columns if c != '_merge']
            torch.save({'columns': columns, 'values': torch.tensor(kept[columns].to_numpy(np.float64))}, saved_path)
            saved = {'path': str(saved_path.relative_to(args.output)), 'sha256': sha(saved_path)}
        result['files'].append({'run': hp.parent.name, 'stem': hp.name.removesuffix('_hits.csv'),
                                'sha256': hashes, 'observed_export_events': len(inventory),
                                'exported_hits': len(hf), 'exported_truth_records': len(vf),
                                'header_only_export': header_only,
                                'primary_binding_state': 'unresolved_header_only' if header_only else 'verified' if probe is not None else 'unverified',
                                'primary_file_available': primary is not None and primary.is_file(), 'primary_probe': probe,
                                'primary_probe_sha256': sha(primary) if probe is not None else None,
                                'primary_source_member': primary_member,
                                'geometry_bounds': {c: [float(obs[c].min()), float(obs[c].max())]
                                                    for c in ['LayerId', 'RowId', 'ColumnId']} if len(obs) else None,
                                'observations': saved,
                                'added_soft_hits': counts['added_soft_hits'], 'added_soft_clusters': counts['added_soft_clusters']})
        result['files_processed'] = i+1
        if i % 25 == 0 or i+1 == len(files):
            result['elapsed_seconds'] = time.monotonic()-started
            write(args.output/'progress.json', result)
            print(tag, i+1, '/', len(files), 'seconds', round(result['elapsed_seconds'], 1), flush=True)
    result['state'] = 'complete'
    result['source_script_sha256'] = sha(Path(__file__))
    write(args.output/'summary.json', result)


def record_failure(output, exc):
    """Keep the original failure even when its output filesystem also fails."""
    try:
        if output.is_dir() and not (output/'summary.json').exists():
            write(output/'failure.json', {'state': 'failed', 'error': repr(exc),
                                         'source_script_sha256': sha(Path(__file__))})
    except Exception as reporting_exc:
        print(f'Failure report unavailable: {reporting_exc!r}; original: {exc!r}', file=sys.stderr, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', choices=TAGS, required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--primary-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inventory', type=Path, default=None,
                   help='Frozen name inventory: restrict to common primary files; all are development')
    p.add_argument('--save-observations', action='store_true',
                   help='Save unscaled observed hits for a subsequent frozen-DCM transfer view')
    p.add_argument('--staged-root', type=Path, help='Hash-verified node-local input bundle')
    p.add_argument('--limit-files', type=int, default=None,
                   help='Bounded engineering smoke only; output is marked partial')
    args = p.parse_args()
    if args.limit_files is not None and args.limit_files < 1:
        p.error('--limit-files must be positive')
    torch.set_num_threads(4)
    try:
        run(args)
    except Exception as exc:
        record_failure(args.output, exc)
        raise


if __name__ == '__main__':
    main()
