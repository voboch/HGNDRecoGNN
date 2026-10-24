"""Check source-mapping receipts, content hashes and repeated MC events locally."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, original_root):
    summary_path = root/'mapping_inputs/summary.json'
    data = json.loads(summary_path.read_text())
    receipt = json.loads((root/'map_corrected_terminal_receipt.json').read_text())
    if '4435670|COMPLETED|0:0' not in receipt['slurm'] or data['state'] != 'complete':
        raise ValueError('Mapping job did not finish')
    if sha(summary_path) != receipt['export']['uncompressed_sha256']:
        raise ValueError('Mapping summary changed')
    original = original_root/'input_inventory.json'
    if data['inventory_sha256'] != sha(original):
        raise ValueError('Frozen inventory changed')
    members = json.loads(original.read_text())['tags']['defaultSpot']['common_primary_pair_names']
    if [x['member'] for x in data['detector_files']] != members or [x['member'] for x in data['primary_files']] != members:
        raise ValueError('Incomplete source scan')
    bundle = json.loads((root/'bundle_manifest.json').read_text())
    expected_raw = {x['member']: x['sha256'] for x in bundle['files']}
    for x in data['detector_files']:
        if x['sha256'] != expected_raw['raw/defaultSpot/'+x['member']+'_vacs.csv']:
            raise ValueError('Staged detector input differs from original bytes')
    earlier = json.loads((original_root/'locator_summary.json').read_text())
    expected_primary = {x['member']: x['sha256'] for x in earlier['files_checked']}
    if any(x['sha256'] != expected_primary[x['member']] for x in data['primary_files']):
        raise ValueError('Primary content changed since first locator')
    matches = pd.DataFrame(data['matches'])
    detector_keys = ['detector_member', 'Row_det', 'Id_det']
    source_keys = ['primary_member', 'Row_prim', 'Id_prim']
    identity = matches.drop_duplicates(detector_keys+source_keys)
    counts = identity.groupby(detector_keys).size()
    expected_targets = sum(x['eligible_records'] for x in data['binding_candidates'])
    if expected_targets != data['target_records']:
        raise ValueError('Target accounting differs')
    events = identity[['detector_member', 'Row_det', 'primary_member', 'Row_prim']].drop_duplicates()
    reused_events = events.groupby(['primary_member', 'Row_prim']).detector_member.nunique()
    reused_tracks = identity.groupby(source_keys).detector_member.nunique()
    candidates = data['binding_candidates']
    for candidate in candidates:
        one = identity.loc[identity.detector_member == candidate['detector_member']]
        found = one.groupby(['Row_det', 'Id_det']).size()
        if len(found) != candidate['matched_records'] or int((found > 1).sum()) != candidate['ambiguous_records']:
            raise ValueError('Per-file matching accounting differs')
        if sorted(one.primary_member.unique().tolist()) != candidate['primary_members']:
            raise ValueError('Per-file primary source differs')
        if candidate['state'] == 'exact_unique_file_candidate':
            if (len(found) != candidate['eligible_records'] or not (found == 1).all()
                    or one.primary_member.nunique() != 1
                    or not (one.Row_det == one.Row_prim).all() or not (one.Id_det == one.Id_prim).all()):
                raise ValueError('Invalid exact-file candidate')
    verified = [x for x in candidates if x['state'] == 'exact_unique_file_candidate']
    unresolved = [x for x in candidates if x['state'] != 'exact_unique_file_candidate']
    result = {'state': 'verified', 'scope': 'frozen defaultSpot source association only; no model evaluation or generation independence claim',
              'input_summary_sha256': sha(summary_path), 'inventory_sha256': sha(original),
              'source_script_sha256': data['source_script_sha256'],
              'detector_files_checked': len(data['detector_files']),
              'primary_files_checked': len(data['primary_files']),
              'target_records': data['target_records'], 'matched_detector_records': len(counts),
              'ambiguous_detector_records': int((counts > 1).sum()),
              'unique_source_tracks': len(identity[source_keys].drop_duplicates()),
              'source_tracks_reused_across_detector_files': int((reused_tracks > 1).sum()),
              'source_events_reused_across_detector_files': int((reused_events > 1).sum()),
              'all_matched_Row_Id_equal': bool((identity.Row_det == identity.Row_prim).all() and (identity.Id_det == identity.Id_prim).all()),
              'exact_unique_file_candidates': len(verified),
              'distinct_primary_source_files_for_candidates': len({x['primary_members'][0] for x in verified}),
              'primary_minus_detector_filename_index_counts': dict(sorted(Counter(
                  int(Path(x['primary_members'][0]).name)-int(Path(x['detector_member']).name)
                  for x in verified).items())),
              'unresolved_files': unresolved, 'candidates': candidates,
              'all_primary_hashes_unchanged_since_locator_4435586': True,
              'all_detector_hashes_match_staged_bundle': True,
              'generated_event_count_established': False,
              'generation_job_independence_established': False}
    (root/'source_mapping_verified.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original-root', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.original_root)
    print(json.dumps({k: v for k, v in result.items() if k not in ['candidates', 'unresolved_files']}, indent=2))
