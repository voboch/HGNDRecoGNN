from pathlib import Path
import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.map_smash_primary import binding_candidates, targets_from_vacs, run  # noqa: E402


def test_file_renumbering_is_candidate_only_if_every_track_has_one_source():
    targets = pd.DataFrame({'detector_member': ['0001', '0001', '0002'],
                            'Row': [0, 1, 0], 'Id': [5, 8, 1]})
    matches = pd.DataFrame({'detector_member': ['0001', '0001', '0002', '0002'],
                            'Row_det': [0, 1, 0, 0], 'Id_det': [5, 8, 1, 1],
                            'Row_prim': [0, 1, 0, 0], 'Id_prim': [5, 8, 1, 1],
                            'primary_member': ['0002', '0002', '0003', '0004']})
    first, ambiguous, missing = binding_candidates(targets, matches, ['0001', '0002', '0005'])
    assert first['state'] == 'exact_unique_file_candidate'
    assert first['primary_members'] == ['0002']
    assert ambiguous['state'] == 'unresolved' and ambiguous['ambiguous_records'] == 1
    assert missing['state'] == 'unresolved' and missing['eligible_records'] == 0
    matches.loc[0, 'Row_prim'] = 7
    assert binding_candidates(targets, matches, ['0001'])[0]['state'] == 'unresolved'


def test_conflicting_track_attributes_are_rejected_before_source_search():
    frame = pd.DataFrame({'Row': [0, 0], 'Id': [4, 4], 'PDG': [2112, 2112],
                          'Ekin': [1., 2.], 'Px': [0., 0.], 'Py': [0., 0.],
                          'Pz': [1., 1.], 'vX': [0., 0.], 'vY': [0., 0.],
                          'vZ': [0., 0.], 'fMotherId': [-1, -1], 'Side': [5, 5]})
    with pytest.raises(ValueError, match='Conflicting'):
        targets_from_vacs(frame, '0001')


def test_header_only_detector_export_has_no_binding_evidence():
    import io
    empty = pd.read_csv(io.StringIO('Row,Id,PDG,Ekin,Px,Py,Pz,vX,vY,vZ,fMotherId,Side\n'))
    targets = targets_from_vacs(empty, 'empty')
    assert len(targets) == 0
    result = binding_candidates(targets, pd.DataFrame(), ['empty'])[0]
    assert result['state'] == 'unresolved' and result['eligible_records'] == 0


def test_lookup_rejects_same_kinematics_with_different_vertex(tmp_path):
    raw, primary = tmp_path / 'raw', tmp_path / 'primary'
    raw.mkdir()
    primary.mkdir()
    base = {'Row': [0], 'Id': [4], 'PDG': [2112], 'Ekin': [1.],
            'Px': [.1], 'Py': [.2], 'Pz': [1.], 'vX': [0.], 'vY': [0.], 'vZ': [0.]}
    first = pd.DataFrame(base)
    for member, energy in [('0001', 1.), ('0002', 2.)]:
        detector = first.assign(Ekin=energy, fMotherId=-1, Side=5)
        detector.to_csv(raw / (member + '_vacs.csv'), index=False)
    first.assign(vZ=1.).to_csv(primary / '0001_prim.csv', index=False)
    first.to_csv(primary / '0002_prim.csv', index=False)
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'tags': {'defaultSpot': {
        'raw_root': str(raw), 'primary_root': str(primary),
        'common_primary_pair_names': ['0001', '0002']}}}))
    output = tmp_path / 'out'
    run(inventory, output, 'defaultSpot')
    result = json.loads((output / 'summary.json').read_text())
    assert result['state'] == 'complete'
    assert len(result['primary_files']) == 2 and len(result['matches']) == 1
    assert result['exact_unique_file_candidates'] == 1
    assert result['binding_candidates'][0]['primary_members'] == ['0002']
    assert result['binding_candidates'][1]['matched_records'] == 0
    assert result['downstream_audit_authorized_by_this_result'] is False
    # Changing only the primary read location must preserve every exact match.
    staged = tmp_path/'staged'
    dest = staged/'primary/defaultSpot'
    dest.mkdir(parents=True)
    for path in primary.glob('*.csv'):
        (dest/path.name).write_bytes(path.read_bytes())
    staged_out = tmp_path/'staged_out'
    run(inventory, staged_out, 'defaultSpot', staged_primary_root=staged)
    copied = json.loads((staged_out/'summary.json').read_text())
    for field in ['matches', 'primary_files', 'binding_candidates', 'target_records']:
        assert copied[field] == result[field]
