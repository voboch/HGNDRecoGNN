from pathlib import Path
import sys
import json
import gzip

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.publish_smash_mapping import publish  # noqa: E402


def test_publish_rejects_orphan_partial_and_preserves_complete_bytes(tmp_path):
    source=tmp_path/'summary.json'
    data=dict(state='running',inventory_sha256='frozen',primary_files=[{}],files_expected=2,source_script_sha256='script')
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='incomplete'):
        publish(source,tmp_path/'export',tmp_path/'receipt.json','frozen')
    assert not (tmp_path/'export').exists()
    data.update(state='complete',primary_files=[{},{}])
    source.write_text(json.dumps(data))
    publish(source,tmp_path/'export',tmp_path/'receipt.json','frozen')
    assert gzip.decompress((tmp_path/'export/summary.json.gz').read_bytes())==source.read_bytes()
    with pytest.raises(FileExistsError):
        publish(source,tmp_path/'export',tmp_path/'second_receipt.json','frozen')
