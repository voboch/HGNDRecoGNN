"""Record source reuse and empty exports without relaxing the full-transfer gate."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.build_smash_bound_inventory import EMPTY_HASH, EMPTY_BYTES  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root):
    receipt = json.loads((root/'bigSpot_export_receipt.json').read_text())
    compressed = root/'bigSpot_mapping_complete.json.gz'
    if sha(compressed) != receipt['gzip_sha256'] or '4436975|COMPLETED|0:0' not in receipt['slurm']:
        raise ValueError('Incomplete or changed source-map result')
    content = gzip.decompress(compressed.read_bytes())
    if hashlib.sha256(content).hexdigest() != receipt['summary_sha256']:
        raise ValueError('Decompressed source-map hash differs')
    data = json.loads(content)
    inventory_path = root/'original_inventory.json'
    inventory = json.loads(inventory_path.read_text())['tags']['bigSpot']
    members = inventory['common_primary_pair_names']
    assert data['state'] == 'complete' and data['inventory_sha256'] == sha(inventory_path)
    assert [x['member'] for x in data['primary_files']] == members
    assert [x['member'] for x in data['detector_files']] == members
    manifest_path = root/'bigSpot_raw_manifest.json'
    submission = json.loads((root/'bigSpot_map_submission.json').read_text())
    assert sha(manifest_path) == submission['manifest_sha256']
    raw = {x['member']:x for x in json.loads(manifest_path.read_text())['files']}
    for item in data['detector_files']:
        assert item['sha256'] == raw['raw/bigSpot/'+item['member']+'_vacs.csv']['sha256']
    frame = pd.DataFrame(data['matches'])
    dkeys = ['detector_member','Row_det','Id_det']
    skeys = ['primary_member','Row_prim','Id_prim']
    identity = frame.drop_duplicates(dkeys+skeys)
    counts = identity.groupby(dkeys).size()
    assert len(counts) == data['target_records'] and (counts == 1).all()
    assert (identity.Row_det == identity.Row_prim).all() and (identity.Id_det == identity.Id_prim).all()
    tracks = identity.groupby(skeys).detector_member.nunique()
    events = identity[['detector_member','Row_det','primary_member','Row_prim']].drop_duplicates()
    event_use = events.groupby(['primary_member','Row_prim']).detector_member.nunique()
    bound = [x for x in data['binding_candidates'] if x['state']=='exact_unique_file_candidate']
    empty = [x for x in data['binding_candidates'] if x['state']=='unresolved']
    for item in empty:
        assert item['eligible_records'] == item['matched_records'] == 0
        for suffix in EMPTY_HASH:
            entry = raw[f"raw/bigSpot/{item['detector_member']}_{suffix}.csv"]
            assert entry['sha256'] == EMPTY_HASH[suffix] and entry['bytes'] == EMPTY_BYTES[suffix]
    groups = defaultdict(list)
    for item in bound:
        one = identity.loc[identity.detector_member == item['detector_member']]
        assert len(one)==item['eligible_records'] and one.primary_member.nunique()==1
        assert one.primary_member.unique().tolist()==item['primary_members']
        groups[item['primary_members'][0]].append(item['detector_member'])
    duplicates = {}
    for fields in [('hits',),('vacs',),('hits','vacs')]:
        by_hash = defaultdict(list)
        for item in bound:
            member=item['detector_member']
            key=tuple(raw[f'raw/bigSpot/{member}_{s}.csv']['sha256'] for s in fields)
            by_hash[key].append(member)
        duplicates['_and_'.join(fields)] = [v for v in by_hash.values() if len(v)>1]
    reused = tracks[tracks>1]
    first = identity.set_index(skeys).loc[reused.head(1).index].reset_index()
    report = {'state':'full_transfer_gate_failed_source_reuse', 'job_id':'4436975',
              'scope':'Exact initial MCTrack identity among exported eligible n/p only; not generated-event closure or production-job independence',
              'input_summary_sha256':receipt['summary_sha256'],'compressed_sha256':sha(compressed),
              'inventory_sha256':sha(inventory_path),'raw_bundle_manifest_sha256':sha(manifest_path),
              'source_script_sha256':sha(Path(__file__)), 'files_checked':len(members),
              'exact_bound_nonempty_files':len(bound),'header_only_exports':len(empty),
              'unresolved_nonempty_files':0,'target_records':data['target_records'],
              'matched_detector_records':len(counts),'ambiguous_detector_records':0,
              'all_matched_Row_Id_equal':True,'unique_primary_source_files':len(groups),
              'unique_source_tracks':len(tracks),'source_tracks_reused_across_detector_files':int((tracks>1).sum()),
              'unique_source_events_among_matched_records':len(event_use),
              'source_events_reused_across_detector_files':int((event_use>1).sum()),
              'source_track_detector_file_multiplicity':dict(sorted(Counter(tracks).items())),
              'source_file_detector_file_multiplicity':dict(sorted(Counter(map(len,groups.values())).items())),
              'source_file_groups':dict(sorted(groups.items())), 'duplicate_nonempty_content_groups':duplicates,
              'exact_repeated_track_example':first[dkeys+skeys].to_dict('records'),
              'header_only_members':[x['detector_member'] for x in empty],
              'generated_event_count':None,'processed_root_event_count':None,
              'deduplication_or_group_split_selected':False,'models_evaluated':False}
    (root/'bigSpot_binding_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_file_groups','duplicate_nonempty_content_groups','header_only_members']},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
