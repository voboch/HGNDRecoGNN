from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.publish_smash_directory import publish  # noqa: E402
from HGNDRecoGNN.scripts.smash_input_bundle import unpack  # noqa: E402


def test_directory_publisher_refuses_partial_and_roundtrips_complete_output(tmp_path):
    source=tmp_path/'source/audit/zeroSpot'
    source.mkdir(parents=True)
    summary=source/'summary.json'
    summary.write_text('{"state":"running"}')
    with pytest.raises(ValueError,match='Incomplete'):
        publish(tmp_path/'source',tmp_path/'bundle',tmp_path/'destination',tmp_path/'receipt.json')
    summary.write_text('{"state":"complete"}')
    observation=source/'obs.pt'
    observation.write_bytes(b'fixture tensor bytes')
    publish(tmp_path/'source',tmp_path/'bundle',tmp_path/'destination',tmp_path/'receipt.json')
    receipt=json.loads((tmp_path/'receipt.json').read_text())
    assert receipt['files']==2 and receipt['state']=='complete'
    unpack(tmp_path/'destination/inputs.tar',tmp_path/'destination/manifest.json',tmp_path/'readback')
    assert (tmp_path/'readback/audit/zeroSpot/obs.pt').read_bytes()==observation.read_bytes()
    assert (tmp_path/'readback/audit/zeroSpot/summary.json').read_bytes()==summary.read_bytes()
