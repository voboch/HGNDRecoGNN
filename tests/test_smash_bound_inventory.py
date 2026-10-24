"""Guard the boundary between exact source evidence and empty-export accounting."""
from pathlib import Path
import sys
import copy

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.build_smash_bound_inventory import verify_mapping, EMPTY_HASH, EMPTY_BYTES  # noqa: E402


def evidence():
    members = ['run/0001', 'run/0002']
    frozen = dict(common_primary_pair_names=members)
    raw = dict(files=[dict(member=f'raw/bigSpot/{m}_{s}.csv', sha256=EMPTY_HASH[s] if i else 'nonempty_'+s,
                           bytes=EMPTY_BYTES[s] if i else 1000) for i,m in enumerate(members) for s in ['hits','vacs']])
    data = dict(state='complete', tag='bigSpot', inventory_sha256='fixed', target_records=1,
                detector_files=[dict(member=m,sha256=EMPTY_HASH['vacs'] if i else 'nonempty_vacs') for i,m in enumerate(members)],
                primary_files=[dict(member=m,sha256='primary_'+str(i)) for i,m in enumerate(members)],
                matches=[dict(detector_member=members[0],Row_det=0,Id_det=7,
                              primary_member=members[1],Row_prim=0,Id_prim=7)],
                binding_candidates=[dict(detector_member=members[0],eligible_records=1,matched_records=1,
                                         ambiguous_records=0,primary_members=[members[1]],state='exact_unique_file_candidate'),
                                    dict(detector_member=members[1],eligible_records=0,matched_records=0,
                                         ambiguous_records=0,primary_members=[],state='unresolved')])
    return data,frozen,raw


def test_exact_map_preserves_header_only_member_without_claiming_a_primary_source():
    data,frozen,raw = evidence()
    bound,report = verify_mapping(data,frozen,raw,'fixed')
    assert bound['primary_member_map'] == {'run/0001':'run/0002','run/0002':None}
    assert report['exact_bound_nonempty_files'] == report['header_only_exports'] == 1
    assert report['processed_root_events'] is None
    # Identical hashes alone cannot override contradictory nonzero byte counts.
    bad = copy.deepcopy(raw)
    bad['files'][-1]['bytes'] = 1000
    with pytest.raises(ValueError,match='Unresolved nonempty'):
        verify_mapping(data,frozen,bad,'fixed')


def test_full_binding_does_not_hide_changed_row_or_missing_track():
    data,frozen,raw = evidence()
    bad = copy.deepcopy(data)
    bad['matches'][0]['Row_prim'] = 2
    with pytest.raises(ValueError,match='Changed Row/Id'):
        verify_mapping(bad,frozen,raw,'fixed')
    bad = copy.deepcopy(data)
    bad['binding_candidates'][0]['eligible_records'] = 2
    with pytest.raises(ValueError,match='Incomplete exact'):
        verify_mapping(bad,frozen,raw,'fixed')


def test_source_overlap_cannot_be_counted_as_distinct_detector_population():
    data,frozen,raw = evidence()
    member = 'run/0003'
    frozen['common_primary_pair_names'].append(member)
    data['detector_files'].append(dict(member=member,sha256='another_vacs'))
    data['primary_files'].append(dict(member=member,sha256='another_primary'))
    for suffix in ['hits','vacs']:
        raw['files'].append(dict(member=f'raw/bigSpot/{member}_{suffix}.csv',sha256='another_'+suffix,bytes=1000))
    data['matches'].append({**data['matches'][0],'detector_member':member})
    data['binding_candidates'].append({**data['binding_candidates'][0],'detector_member':member})
    data['target_records'] += 1
    with pytest.raises(ValueError,match='Source track/event reused'):
        verify_mapping(data,frozen,raw,'fixed')
    # Canonicalization requires identical detector bytes as well as MC source.
    with pytest.raises(ValueError,match='Source track/event reused'):
        verify_mapping(data,frozen,raw,'fixed',canonicalize_exact_duplicates=True)
    for item in raw['files'][-2:]:
        item['sha256'] = item['sha256'].replace('another_', 'nonempty_')
    data['detector_files'][-1]['sha256'] = 'nonempty_vacs'
    bound,report = verify_mapping(data,frozen,raw,'fixed',canonicalize_exact_duplicates=True)
    assert bound['duplicate_exports'] == {'run/0003': 'run/0001'}
    assert report['source_tracks_reused_before_canonicalization'] == 1
    assert report['source_tracks_reused'] == 0
    assert report['eligible_records_after_canonicalization'] == 1
