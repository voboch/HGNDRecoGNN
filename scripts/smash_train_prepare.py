"""Per-file SMASH training packs: inherited observations, fold scalers, A and B arms.

CPU only. No model is built, trained or scored. For every content of the split
the raw detector export is copied node-local under its pinned SHA256, the audit
observation recipe (scripts/audit_smash_sources.observe) is applied, and two
packs are written on the DCM transfer graph view (data/smash_transfer.py):

  A  frozen DCM train-only scalers and affine remap (smash_transfer_infer.frozen_dcm)
  B  one StandardScaler per detector side, partial_fit on the fold's training
     files only, pooled over potentials, in sorted content order

Candidates, edges, labels and targets of A and B must be identical; only the
feature standardisation differs. Validation and test files never reach a scaler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import pickle
import sys
import tarfile
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.data.graph_dataset import FEATURES  # noqa: E402
from HGNDRecoGNN.data.smash_transfer import RADIUS_CELLS, TIME_NS, observed_features, transfer_pack  # noqa: E402
from HGNDRecoGNN.scripts import audit_smash_sources as audit  # noqa: E402
from HGNDRecoGNN.scripts.scratch_access import ACCESS_RETRIES, with_access_retry  # noqa: E402

SPLIT_SCHEMA = 'smash-train-split-v1'
TAGS = tuple(audit.TAGS)
ROLES = ('train', 'validation', 'test')
SIDES = ('bot', 'top')  # side 0 is fY <= 0, side 1 is fY > 0, as in the DCM scalers
# Identity remap for B: the fold scaler alone standardises. Multiplying a
# float32 by 1 and adding 0 is exact, so B features are the scaler output.
IDENTITY_A = np.ones((2, len(FEATURES) - 1))
IDENTITY_B = np.zeros((2, len(FEATURES) - 1))
CANDIDATE_KEYS = ('pe', 'edges', 'cluster', 'label', 'energy', 'row', 'side', 'index',
                  'node_ptr', 'edge_ptr', 'cluster_ptr')
PACK_PREFIX = 'prepare'  # member prefix of the prepare output inside its published bundle


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or str(path) != name:
        raise ValueError('Unsafe member name: ' + name)
    return name


# ── split ────────────────────────────────────────────────────────────────

def _from_group_manifest(raw):
    """scripts/smash_split.py manifest (smash-split-v1): groups, six parts, three folds."""
    contents, ids = {}, {}
    for g in raw['groups']:
        if g['hits_path'] != g['kept'] + '_hits.csv' or g['vacs_path'] != g['kept'] + '_vacs.csv':
            raise ValueError('Group paths differ from the kept member: ' + g['id'])
        cid = f"{g['tag']}/{g['kept']}"
        contents[cid] = {'tag': g['tag'], 'member': g['kept'], 'group': g['id'], 'truth_bound': bool(g['truth_bound']),
                         'primary_member': g['primary'] if g['truth_bound'] else None,
                         'sha256': {'hits': g['hits_sha256'], 'vacs': g['vacs_sha256']}}
        ids[g['id']] = cid
    folds = []
    for k in range(int(raw['folds_count'])):
        parts = raw['folds'][str(k)]['partitions']
        folds.append({'fold': k, 'train': [ids[i] for i in parts['train']],
                      'validation': [ids[i] for i in parts['val']], 'test': [ids[i] for i in parts['test']],
                      'unused': [ids[i] for i in parts['unused']]})
    return {'schema': SPLIT_SCHEMA, 'source_schema': raw['schema'],
            'tags': {t: {'raw_root': r} for t, r in raw['raw_roots'].items()}, 'contents': contents, 'folds': folds}


def _from_smoke_manifest(raw):
    """scripts/smash_split.py smoke split (smash-smoke-split-v1): one fold, no test."""
    contents, tags, fold = {}, {}, {'fold': 0, 'train': [], 'validation': [], 'test': []}
    for part, role in (('train', 'train'), ('val', 'validation')):
        for e in raw[part]:
            if e['hits_path'] != e['member'] + '_hits.csv' or e['vacs_path'] != e['member'] + '_vacs.csv':
                raise ValueError('Smoke paths differ from the member: ' + e['member'])
            cid = f"{e['tag']}/{e['member']}"
            contents[cid] = {'tag': e['tag'], 'member': e['member'], 'group': cid, 'truth_bound': True,
                             'primary_member': e['primary'],
                             'sha256': {'hits': e['hits_sha256'], 'vacs': e['vacs_sha256']}}
            if tags.setdefault(e['tag'], e['raw_root']) != e['raw_root']:
                raise ValueError('Smoke split names two raw roots for one potential')
            fold[role].append(cid)
    return {'schema': SPLIT_SCHEMA, 'source_schema': raw['schema'], 'smoke': True,
            'tags': {t: {'raw_root': r} for t, r in tags.items()}, 'contents': contents, 'folds': [fold]}


def load_split(path):
    """Validate the split manifest consumed by prepare, train, evaluate and the analyses.

    Accepts the manifests of scripts/smash_split.py (smash-split-v1 and its
    smoke split) and normalises them to smash-train-split-v1: `tags[tag].raw_root`;
    `contents[id]` with id == tag/member, `sha256.{hits,vacs}`, `truth_bound`,
    `primary_member` and `group`; `folds[k]` with `fold == k` and lists
    `train`, `validation`, `test` of content ids.
    """
    raw = json.loads(Path(path).read_text())
    schema = raw.get('schema')
    split = {'smash-split-v1': _from_group_manifest, 'smash-smoke-split-v1': _from_smoke_manifest}.get(
        schema, lambda x: x)(raw)
    if split.get('schema') != SPLIT_SCHEMA:
        raise ValueError('Unknown split schema')
    contents = split['contents']
    for cid, c in contents.items():
        if c['tag'] not in TAGS or cid != f"{c['tag']}/{safe_member(c['member'])}":
            raise ValueError('Content id differs from tag/member: ' + cid)
        if c['tag'] not in split['tags']:
            raise ValueError('Content tag lacks a raw root: ' + cid)
        if any(len(c['sha256'][k]) != 64 for k in ('hits', 'vacs')):
            raise ValueError('Content lacks pinned hashes: ' + cid)
        if c['truth_bound'] and not c.get('primary_member'):
            raise ValueError('Truth-bound content lacks its primary member: ' + cid)
    pairs = [(c['sha256']['hits'], c['sha256']['vacs']) for c in contents.values()]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Byte-identical contents listed twice')
    for k, fold in enumerate(split['folds']):
        if fold['fold'] != k:
            raise ValueError('Folds must be numbered 0..n-1 in order')
        flat = [cid for role in ROLES for cid in fold[role]]
        if len(set(flat)) != len(flat):
            raise ValueError(f'Content in several partitions of fold {k}')
        if set(flat) - set(contents):
            raise ValueError(f'Unknown content in fold {k}')
        if not fold['train'] or not fold['validation']:
            raise ValueError(f'Empty training or validation partition in fold {k}')
        if any(not contents[cid]['truth_bound'] for cid in fold['test']):
            raise ValueError(f'Unbound content in the test partition of fold {k}')
        role_of_group = {}
        for role in ROLES:
            for cid in fold[role]:
                group = contents[cid].get('group')
                if group is not None and role_of_group.setdefault(group, role) != role:
                    raise ValueError(f'Content group {group} crosses partitions of fold {k}')
    return split


def used_contents(split):
    return sorted({cid for fold in split['folds'] for role in ROLES for cid in fold[role]})


def roles_of(split, cid):
    return {fold['fold']: role for fold in split['folds'] for role in ROLES if cid in fold[role]}


# ── bundles ──────────────────────────────────────────────────────────────

def extract_members(bundle_dir, manifest_sha256, members, output, check_archive=True):
    """Hash the whole archive before extraction, then extract only `members`.

    Every extracted member is hash-checked against the pinned manifest. An
    unknown or missing member fails; nothing else in the archive is written.
    `check_archive=False` skips only the whole-archive re-hash for a caller
    that already verified this archive; per-member hashes are always checked.
    The bundle sits on /scratch, so an EACCES retries the whole extraction
    after removing the members this call wrote (RA-09).
    """
    output = Path(output)
    fresh = [output / safe_member(m) for m in members if not (output / safe_member(m)).exists()]

    def remove_partial():
        for path in fresh:
            path.unlink(missing_ok=True)

    return with_access_retry(_extract_members, bundle_dir, manifest_sha256, members, output, check_archive,
                             cleanup=remove_partial)


def _extract_members(bundle_dir, manifest_sha256, members, output, check_archive):
    bundle_dir, output = Path(bundle_dir), Path(output)
    manifest_path = bundle_dir / 'manifest.json'
    if sha(manifest_path) != manifest_sha256:
        raise ValueError('Bundle manifest differs from its pin: ' + str(bundle_dir))
    manifest = json.loads(manifest_path.read_text())
    archive = bundle_dir / 'inputs.tar'
    if manifest['state'] != 'complete' or (check_archive and sha(archive) != manifest['archive_sha256']):
        raise ValueError('Bundle archive hash or state failed: ' + str(bundle_dir))
    expected = {safe_member(x['member']): x for x in manifest['files']}
    wanted = {safe_member(m) for m in members}
    if wanted - set(expected):
        raise ValueError(f'Members absent from bundle: {sorted(wanted - set(expected))[:5]}')
    seen = set()
    with tarfile.open(archive, 'r') as tar:
        for item in tar:
            name = safe_member(item.name)
            if name not in wanted:
                continue
            if name in seen or not item.isfile():
                raise ValueError('Duplicate or non-file member: ' + name)
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(item)
            digest, size = hashlib.sha256(), 0
            with target.open('xb') as sink:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    sink.write(block)
                    digest.update(block)
                    size += len(block)
            if digest.hexdigest() != expected[name]['sha256'] or size != expected[name]['bytes']:
                raise ValueError('Extracted member hash failed: ' + name)
            seen.add(name)
    if seen != wanted:
        raise ValueError('Bundle members missing after extraction')
    return {name: output / name for name in sorted(wanted)}


def audit_references(index_path, ids, work):
    """Saved audit observation tensors for every requested content they cover."""
    index = json.loads(Path(index_path).read_text())
    refs = {}
    for tag in TAGS:
        item = index[tag]
        bundle = Path(item['bundle_dir'])
        if with_access_retry(sha, bundle / 'manifest.json') != item['manifest_sha256']:
            raise ValueError('Audit bundle manifest differs from its pin: ' + tag)
        manifest = json.loads(with_access_retry((bundle / 'manifest.json').read_text))
        listed = {x['member'] for x in manifest['files']}
        summary_member = f'audit/{tag}/summary.json'
        candidates = {cid: f'audit/{tag}/observations/' + cid.split('/', 1)[1] + '.pt'
                      for cid in ids if cid.startswith(tag + '/')}
        candidates = {cid: m for cid, m in candidates.items() if m in listed}
        paths = extract_members(bundle, item['manifest_sha256'], [summary_member, *candidates.values()],
                                work / 'audit' / tag)
        summary = json.loads(paths[summary_member].read_text())
        if summary['state'] != 'complete':
            raise ValueError('Audit summary incomplete: ' + tag)
        entries = {f"{tag}/{f['run']}/{f['stem']}": f for f in summary['files']}
        for cid, member in candidates.items():
            f = entries[cid]
            if (f.get('observations') is None or f'audit/{tag}/' + f['observations']['path'] != member
                    or sha(paths[member]) != f['observations']['sha256']):
                raise ValueError('Audit observation differs from its summary: ' + cid)
            refs[cid] = {'path': str(paths[member]), 'sha256': f['observations']['sha256'],
                         'raw_sha256': f['sha256']}
    return refs


# ── observation phase ────────────────────────────────────────────────────

def stage(source, target, expected):
    """Copy to node-local storage while hashing; parse only the verified copy.

    An EACCES on the /scratch source removes the partial copy and copies again
    (RA-09); the hash check below is unchanged.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = with_access_retry(_copy_hashing, source, target,
                               cleanup=lambda: target.unlink(missing_ok=True))
    if digest != expected:
        raise ValueError('Raw export differs from the split pin: ' + str(source))


def _copy_hashing(source, target):
    digest = hashlib.sha256()
    with Path(source).open('rb') as inp, target.open('xb') as out:
        for block in iter(lambda: inp.read(1 << 20), b''):
            digest.update(block)
            out.write(block)
    return digest.hexdigest()


def load_observation(path):
    payload = torch.load(path, weights_only=True, map_location='cpu')
    return pd.DataFrame(payload['values'].numpy(), columns=payload['columns'])


def observe_file(task):
    cid, content, raw_root, work, reference = task
    retries_before = len(ACCESS_RETRIES)
    tag, member = content['tag'], content['member']
    local = Path(work) / 'raw' / tag
    hp, vp = local / (member + '_hits.csv'), local / (member + '_vacs.csv')
    stage(Path(raw_root) / (member + '_hits.csv'), hp, content['sha256']['hits'])
    stage(Path(raw_root) / (member + '_vacs.csv'), vp, content['sha256']['vacs'])
    hf, vf = audit.read_csv(hp, audit.HITS), audit.read_csv(vp, audit.VACS)
    # The jitter key is the hits path relative to the tag root, as in the audit.
    _, kept, dropped = audit.observe(hf, vf, member + '_hits.csv', hp, vp)
    columns = [c for c in kept.columns if c != '_merge']
    values = kept[columns].to_numpy(np.float64)
    path = Path(work) / 'observations' / tag / (member + '.pt')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'columns': columns, 'values': torch.tensor(values)}, path)
    hp.unlink()
    vp.unlink()
    item = {'path': str(path), 'sha256': sha(path), 'observed_hits': len(kept), 'exported_hits': len(hf),
            'exported_truth_records': len(vf), 'header_only_export': len(hf) == len(vf) == 0,
            'rejected_pdg0_hits': dropped, 'audit_reference': None,
            'access_retries': ACCESS_RETRIES[retries_before:]}
    if reference is not None:
        if reference['raw_sha256'] != content['sha256']:
            raise ValueError('Audit reference was built from different raw bytes: ' + cid)
        saved = torch.load(reference['path'], weights_only=True, map_location='cpu')
        identical = (saved['columns'] == columns and saved['values'].dtype == torch.float64
                     and saved['values'].numpy().tobytes() == values.tobytes())
        item['audit_reference'] = {'observation_bit_identical': bool(identical),
                                   'file_sha256_identical': item['sha256'] == reference['sha256'],
                                   'audit_sha256': reference['sha256']}
        if not identical:
            raise ValueError('Observation differs from the saved audit tensor: ' + cid)
    return cid, item


# ── scalers ──────────────────────────────────────────────────────────────

def fit_fold_scalers(train_ids, observations, geometry):
    """One StandardScaler per detector side on training files only, sorted order."""
    scalers = [StandardScaler(), StandardScaler()]
    for cid in sorted(train_ids):
        features = observed_features(load_observation(observations[cid]), geometry)
        for side in (0, 1):
            if len(features[side]):
                scalers[side].partial_fit(features[side])
    if any(not hasattr(s, 'n_samples_seen_') for s in scalers):
        raise ValueError('A detector side received no training hits')
    if any(not np.all(np.isfinite(s.scale_)) or np.any(s.scale_ <= 0) for s in scalers):
        raise ValueError('Degenerate fold scaler')
    return scalers


def scaler_record(scalers, train_ids):
    return {'files': sorted(train_ids), 'fit': 'StandardScaler.partial_fit, sorted content id order',
            'features': list(FEATURES[:-1]),
            'sides': {name: {'mean': s.mean_, 'var': s.var_, 'scale': s.scale_,
                             'n_samples_seen': int(s.n_samples_seen_)}
                      for name, s in zip(SIDES, scalers)}}


# ── pack phase ───────────────────────────────────────────────────────────

_STATE = {}


def _init_worker(state):
    torch.set_num_threads(1)
    _STATE.update(state)


def same_candidates(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in CANDIDATE_KEYS)


def pack_population(pack):
    return {'graphs': len(pack['row']), 'physical_events': len(np.unique(pack['row'].numpy())),
            'hits': len(pack['x']), 'edges': int(pack['edges'].shape[1]), 'clusters': len(pack['label']),
            'signal_clusters': int(pack['label'].sum())}


def pack_file(task):
    cid, folds = task
    s = _STATE
    frame = load_observation(s['observations'][cid])
    out = Path(s['output'])
    result = {'packs': {}}
    a_pack = transfer_pack(frame, s['dcm_scalers'], s['dcm_a'], s['dcm_b'], s['geometry'])
    if a_pack is None:
        return cid, {'empty_after_cuts': True, 'packs': {}}
    rel = Path('packs/A') / (cid + '.pt')
    atomic_torch(out / rel, a_pack)
    result['packs']['A'] = {'path': str(rel), 'sha256': sha(out / rel), 'bytes': (out / rel).stat().st_size}
    for k in folds:
        b_pack = transfer_pack(frame, s['fold_scalers'][k], IDENTITY_A, IDENTITY_B, s['geometry'])
        if not same_candidates(a_pack, b_pack):
            raise ValueError('A and B packs differ in candidates, labels or targets: ' + cid)
        if torch.equal(a_pack['x'], b_pack['x']):
            raise ValueError('B features equal A features; the fold scaler was not applied: ' + cid)
        rel = Path(f'packs/B/fold_{k}') / (cid + '.pt')
        atomic_torch(out / rel, b_pack)
        result['packs'][f'B/fold_{k}'] = {'path': str(rel), 'sha256': sha(out / rel),
                                          'bytes': (out / rel).stat().st_size}
    result.update(empty_after_cuts=False, population=pack_population(a_pack))
    return cid, result


def run_pool(function, tasks, workers, state=None):
    if workers <= 1:
        _init_worker(state or {})
        torch.set_num_threads(int(os.getenv('SLURM_CPUS_PER_TASK', '1')))
        return [function(t) for t in tasks]
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(workers, initializer=_init_worker, initargs=(state or {},)) as pool:
        return pool.map(function, tasks, chunksize=1)


def prepare(args):
    started = time.monotonic()
    split = load_split(args.split)
    from HGNDRecoGNN.scripts.smash_transfer_infer import frozen_dcm
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    dcm_scalers, a, b, geometry, dcm_receipt = with_access_retry(frozen_dcm, argparse.Namespace(
        prepared=args.dcm_prepared, native_campaign=args.native_campaign))
    ids = used_contents(split)
    contents = split['contents']
    refs = audit_references(args.audit_index, set(ids), work) if args.audit_index else {}
    print('observing', len(ids), 'contents;', len(refs), 'with saved audit tensors', flush=True)
    tasks = [(cid, contents[cid], split['tags'][contents[cid]['tag']]['raw_root'], str(work), refs.get(cid))
             for cid in ids]
    observed = dict(run_pool(observe_file, tasks, args.workers))
    observations = {cid: observed[cid]['path'] for cid in ids}
    phases = {'audit_and_observations': time.monotonic() - started}
    print('observations done', round(time.monotonic() - started, 1), flush=True)
    fold_scalers, scaler_receipt = {}, {}
    (output / 'scalers').mkdir()
    for fold in split['folds']:
        k = fold['fold']
        leaked = set(fold['train']) & (set(fold['validation']) | set(fold['test']))
        if leaked:
            raise ValueError(f'Scaler inputs overlap evaluation files in fold {k}')
        scalers = fit_fold_scalers(fold['train'], observations, geometry)
        fold_scalers[k] = scalers
        path = output / 'scalers' / f'fold_{k}.pkl'
        with path.open('xb') as stream:
            pickle.dump({name: s for name, s in zip(SIDES, scalers)}, stream)
        scaler_receipt[f'fold_{k}'] = {**scaler_record(scalers, fold['train']),
                                       'pickle': {'path': str(path.relative_to(output)), 'sha256': sha(path)}}
    phases['scalers'] = time.monotonic() - started - sum(phases.values())
    print('scalers done', round(time.monotonic() - started, 1), flush=True)
    state = {'observations': observations, 'output': str(output), 'dcm_scalers': dcm_scalers,
             'dcm_a': a, 'dcm_b': b, 'geometry': geometry, 'fold_scalers': fold_scalers}
    tasks = [(cid, sorted(roles_of(split, cid))) for cid in ids]
    packed = dict(run_pool(pack_file, tasks, args.workers, state))
    phases['packs'] = time.monotonic() - started - sum(phases.values())
    files = {}
    for cid in ids:
        item = observed[cid]
        if args.keep_observations:
            rel = Path('observations') / (cid + '.pt')
            (output / rel).parent.mkdir(parents=True, exist_ok=True)
            (output / rel).write_bytes(Path(item['path']).read_bytes())
            item = {**item, 'path': str(rel)}
            if sha(output / rel) != item['sha256']:
                raise ValueError('Kept observation changed while copying')
        else:
            item = {k: v for k, v in item.items() if k != 'path'}
        files[cid] = {'tag': contents[cid]['tag'], 'member': contents[cid]['member'],
                      'raw_sha256': contents[cid]['sha256'], 'truth_bound': contents[cid]['truth_bound'],
                      'primary_member': contents[cid].get('primary_member'),
                      'roles': roles_of(split, cid), 'observation': item, **packed[cid]}
    bound_refs = [f['observation']['audit_reference'] for f in files.values() if f['observation']['audit_reference']]
    receipt = {
        'state': 'complete', 'schema': 'smash-train-prepare-v1',
        'scope': 'Per-file SMASH packs on the DCM transfer view; no model built, trained or scored',
        'split_sha256': sha(args.split), 'split_schema': SPLIT_SCHEMA,
        'source_sums_sha256': sha(args.source_sums) if args.source_sums else None,
        'scripts_sha256': {name: sha(Path(__file__).resolve().parents[1] / name) for name in
                           ('scripts/smash_train_prepare.py', 'scripts/audit_smash_sources.py',
                            'data/smash_transfer.py', 'scripts/smash_transfer_infer.py')},
        'observation_recipe': {'source': 'scripts/audit_smash_sources.observe', 'policy': audit.POLICY,
                               'jitter_key': '<member>_hits.csv relative to the tag raw root'},
        'graph_view': {'radius_cells': RADIUS_CELLS, 'time_ns': TIME_NS, 'max_neighbors': 32,
                       'features': list(FEATURES)},
        'arms': {'A': 'frozen DCM train-only scalers and affine remap',
                 'B': 'fold StandardScaler per detector side, training files only, identity remap'},
        'dcm_inputs': dcm_receipt, 'scalers': scaler_receipt,
        'audit_comparison': {'files_compared': len(bound_refs),
                             'all_bit_identical': all(r['observation_bit_identical'] for r in bound_refs),
                             'file_sha256_identical': sum(r['file_sha256_identical'] for r in bound_refs)},
        'folds': split['folds'], 'files': files, 'pack_prefix_in_bundle': PACK_PREFIX,
        'workers': args.workers, 'phase_seconds': phases, 'seconds': time.monotonic() - started,
        # Main-process and per-file retries; with workers <= 1 the two lists overlap.
        'access_retries': [json.loads(e) for e in sorted(
            {json.dumps(e, sort_keys=True) for e in ACCESS_RETRIES}
            | {json.dumps(e, sort_keys=True) for item in observed.values() for e in item['access_retries']})],
    }
    write_json(output / 'summary.json', receipt)
    print('prepare complete', len(files), 'files', round(receipt['seconds'], 1), 's', flush=True)
    return json.loads((output / 'summary.json').read_text())  # the receipt exactly as consumers read it


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True, help='Node-local scratch for staged raw exports')
    p.add_argument('--dcm-prepared', type=Path, required=True)
    p.add_argument('--native-campaign', type=Path, required=True)
    p.add_argument('--audit-index', type=Path, help='Audit bundle index; bound files must match its tensors')
    p.add_argument('--source-sums', type=Path, help='SHA256SUMS of the immutable source package')
    p.add_argument('--keep-observations', action='store_true', help='Smoke: export observation tensors')
    p.add_argument('--workers', type=int, default=int(os.getenv('SLURM_CPUS_PER_TASK', '1')))
    args = p.parse_args()
    try:
        prepare(args)
    except BaseException as exc:
        if args.output.is_dir():
            write_json(args.output / 'failure.json', {'state': 'failed', 'error': repr(exc),
                                                      'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
