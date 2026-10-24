"""Validate exact source-map records and freeze explicit full-audit inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# These exact header bytes were retained in the Oct22 corrected-default bundle.
# Both schemas and actual zero row counts are checked again by the CPU audit.
EMPTY_HASH = {'hits': '8cdfe6636b8691b82cb22c94eba1d6aaecdf66bd66c5f5c234060dd7984b0617',
              'vacs': 'ec748ac207cff811d7a0b59e6860cd7f9b3e09201cea10ae67ec1f29194b5084'}
EMPTY_BYTES = {'hits': 69, 'vacs': 88}


def verify_mapping(data, frozen, raw_manifest, inventory_sha, canonicalize_exact_duplicates=False):
    members = frozen['common_primary_pair_names']
    if (data['state'] != 'complete' or data['inventory_sha256'] != inventory_sha
            or [x['member'] for x in data['detector_files']] != members
            or [x['member'] for x in data['primary_files']] != members
            or [x['detector_member'] for x in data['binding_candidates']] != members):
        raise ValueError('Incomplete or changed source mapping')
    tag = data['tag']
    raw = {x['member']: x for x in raw_manifest['files']}
    inputs = {m: {s: raw[f'raw/{tag}/{m}_{s}.csv']['sha256'] for s in ['hits', 'vacs']}
              for m in members}
    if any(x['sha256'] != inputs[x['member']]['vacs'] for x in data['detector_files']):
        raise ValueError('Mapped detector input differs from byte-verified bundle')
    matches = pd.DataFrame(data['matches'])
    dkeys = ['detector_member', 'Row_det', 'Id_det']
    skeys = ['primary_member', 'Row_prim', 'Id_prim']
    identity = matches.drop_duplicates(dkeys+skeys)
    counts = identity.groupby(dkeys).size()
    if int((counts > 1).sum()):
        raise ValueError('Ambiguous source track')
    if not ((identity.Row_det == identity.Row_prim).all() and (identity.Id_det == identity.Id_prim).all()):
        raise ValueError('Changed Row/Id in exact source association')
    tracks = identity.groupby(skeys).detector_member.nunique()
    events = identity[['detector_member','Row_det','primary_member','Row_prim']].drop_duplicates()
    event_use = events.groupby(['primary_member','Row_prim']).detector_member.nunique()
    source_map, empty = {}, {}
    for candidate in data['binding_candidates']:
        member = candidate['detector_member']
        one = identity.loc[identity.detector_member == member]
        found = one.groupby(['Row_det', 'Id_det']).size()
        if (len(found) != candidate['matched_records'] or int((found > 1).sum()) != candidate['ambiguous_records']
                or sorted(one.primary_member.unique()) != candidate['primary_members']):
            raise ValueError('Candidate evidence disagrees with source-map records')
        if candidate['eligible_records'] == 0:
            if (len(found) or candidate['state'] != 'unresolved' or inputs[member] != EMPTY_HASH
                    or any(raw[f'raw/{tag}/{member}_{s}.csv']['bytes'] != EMPTY_BYTES[s] for s in EMPTY_BYTES)):
                raise ValueError('Unresolved nonempty export has no source binding: '+member)
            source_map[member] = None
            empty[member] = {'sha256': inputs[member], 'exported_hits': 0, 'exported_truth_records': 0}
        else:
            if (candidate['state'] != 'exact_unique_file_candidate' or len(found) != candidate['eligible_records']
                    or not (found == 1).all() or len(candidate['primary_members']) != 1):
                raise ValueError('Incomplete exact source binding: '+member)
            source_map[member] = candidate['primary_members'][0]
    if sum(x['eligible_records'] for x in data['binding_candidates']) != data['target_records'] or len(counts) != data['target_records']:
        raise ValueError('Target accounting differs')
    aliases = {}
    if canonicalize_exact_duplicates:
        # This is a byte-identity rule fixed before model scores. Different
        # detector realizations of one MC source require a separate policy.
        by_hash = {}
        for member in members:
            if member not in empty:
                by_hash.setdefault(tuple(inputs[member].values()), []).append(member)
        for group in by_hash.values():
            first = min(group)
            if len({source_map[m] for m in group}) != 1:
                raise ValueError('Identical raw exports have different primary bindings')
            aliases.update({m: first for m in group if m != first})
    canonical = identity.loc[~identity.detector_member.isin(aliases)]
    canonical_tracks = canonical.groupby(skeys).detector_member.nunique()
    canonical_events = canonical[['detector_member','primary_member','Row_prim']].drop_duplicates()
    canonical_event_use = canonical_events.groupby(['primary_member','Row_prim']).detector_member.nunique()
    if (canonical_tracks > 1).any() or (canonical_event_use > 1).any():
        raise ValueError('Source track/event reused across detector files')
    bound = [x for m,x in source_map.items() if x is not None and m not in aliases]
    if len(bound) != len(set(bound)):
        raise ValueError('Primary file reused across detector exports')
    pairs = [tuple(inputs[m].values()) for m in members if m not in empty and m not in aliases]
    if len(pairs) != len(set(pairs)):
        raise ValueError('Duplicate nonempty detector export')
    report = {'state': 'verified', 'tag': tag, 'detector_files': len(members),
              'exact_bound_nonempty_files': len(bound), 'header_only_exports': len(empty),
              'duplicate_byte_aliases': len(aliases),
              'eligible_records_before_canonicalization': len(counts),
              'eligible_records_after_canonicalization': len(canonical),
              'source_tracks_reused_before_canonicalization': int((tracks > 1).sum()),
              'source_events_reused_before_canonicalization': int((event_use > 1).sum()),
              'target_records': len(counts), 'ambiguous_tracks': 0,
              'source_tracks_reused': 0, 'source_events_reused_among_eligible_records': 0,
              'processed_root_events': None, 'fresh_holdout_established': False,
              'primary_minus_detector_filename_index_counts': dict(sorted(Counter(
                  int(Path(p).name)-int(Path(m).name) for m,p in source_map.items() if p is not None).items()))}
    return {**frozen, 'primary_member_map': source_map, 'empty_exports': empty,
            'duplicate_exports': aliases,
            'analysis_pair_names': [m for m in members if m not in aliases],
            'input_sha256': inputs, 'primary_sha256': {x['member']: x['sha256'] for x in data['primary_files']},
            'binding_policy': 'exact_initial_mctrack_all_files'}, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--mapping', action='append', required=True, help='tag=summary.json')
    parser.add_argument('--bundle', action='append', required=True, help='tag=manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--canonicalize-exact-duplicates', action='store_true',
                        help='Explicit pre-score policy: identical hits/vacs and exact source binding, first member only')
    args = parser.parse_args()
    maps = {t: Path(p) for t,p in (x.split('=',1) for x in args.mapping)}
    bundles = {t: Path(p) for t,p in (x.split('=',1) for x in args.bundle)}
    original = json.loads(args.inventory.read_text())
    if set(maps) != set(original['tags']) or set(bundles) != set(maps):
        raise ValueError('All three frozen source-map populations are required')
    args.output.mkdir(parents=True, exist_ok=True)
    full = {'population': 'retrospective corrected SMASH; conditional cluster metrics only; no fresh holdout',
            'original_inventory_sha256': sha(args.inventory), 'tags': {}, 'mapping_provenance': {}}
    reports = {}
    for tag, frozen in original['tags'].items():
        bound, report = verify_mapping(json.loads(maps[tag].read_text()), frozen,
                                       json.loads(bundles[tag].read_text()), sha(args.inventory),
                                       args.canonicalize_exact_duplicates)
        full['tags'][tag], reports[tag] = bound, report
        full['mapping_provenance'][tag] = {'summary_sha256': sha(maps[tag]), 'raw_manifest_sha256': sha(bundles[tag])}
        raw_entries = json.loads(bundles[tag].read_text())['files']
        files = [x for x in raw_entries if x['member'].startswith('raw/'+tag+'/')]
        sources = sorted({p for p in bound['primary_member_map'].values() if p is not None})
        files += [{'source': frozen['primary_root']+'/'+p+'_prim.csv', 'member': f'primary/{tag}/{p}_prim.csv',
                   'sha256': bound['primary_sha256'][p]} for p in sources]
        plan = {'scope': 'Complete frozen corrected SMASH audit inputs; no fitting',
                'original_inventory_sha256': sha(args.inventory), 'files': files}
        (args.output/(tag+'_audit_plan.json')).write_text(json.dumps(plan, indent=2)+'\n')
    (args.output/'bound_inventory.json').write_text(json.dumps(full, indent=2)+'\n')
    (args.output/'binding_verified.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
