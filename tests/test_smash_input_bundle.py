from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.smash_input_bundle import create, unpack, sha, safe_name  # noqa: E402


def test_bundle_preserves_pinned_bytes_and_rejects_modified_archive(tmp_path):
    source = tmp_path/'source.csv'
    source.write_bytes(b'Row,Id\n0,4\n')
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'scope': 'fixture', 'files': [
        {'source': str(source), 'member': 'raw/defaultSpot/run/0001_vacs.csv', 'sha256': sha(source)}]}))
    output = tmp_path/'bundle'
    create(plan, output)
    unpack(output/'inputs.tar', output/'manifest.json', tmp_path/'local')
    assert (tmp_path/'local/raw/defaultSpot/run/0001_vacs.csv').read_bytes() == source.read_bytes()
    with (output/'inputs.tar').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        unpack(output/'inputs.tar', output/'manifest.json', tmp_path/'tampered')


@pytest.mark.parametrize('name', ['/tmp/escape', '../escape', 'raw/../../escape'])
def test_bundle_rejects_member_outside_output(name):
    with pytest.raises(ValueError, match='Unsafe'):
        safe_name(name)
