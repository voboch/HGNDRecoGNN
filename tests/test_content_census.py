"""Identity must survive bookkeeping changes while detecting physical changes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import pytest
import json
from HGNDRecoGNN.scripts import content_census
from HGNDRecoGNN.scripts.content_census import event_signatures, signature, summarise, sha256, validate_smoke


def test_exact_multiset_identity():
    x = np.array([[1., -0.], [2., 3.]])
    h = signature(x, ('a', 'b'))
    assert signature(x[::-1], ('a', 'b')) == h
    assert signature([[1., 0.], [2., 3.]], ('a', 'b')) == h
    assert signature(np.r_[x, x[:1]], ('a', 'b')) != h
    changed = x.copy(); changed[1, 1] = np.nextafter(3., 4.)
    assert signature(changed, ('a', 'b')) != h
    assert signature(x, ('b', 'a')) != h
    with pytest.raises(ValueError):
        signature([[float('nan'), 0]], ('a', 'b'))


def test_event_bookkeeping_and_overlap():
    frame = pd.DataFrame({'Row': [1, 2, 1], 'value': [3., 5., 4.]})
    first = list(event_signatures(frame, ('value',)))
    second = list(event_signatures(frame.assign(Row=frame.Row + 100).iloc[::-1], ('value',)))
    assert [v[-1] for v in first] == [v[-1] for v in second]
    records = [('a', 'f1', 1, 2, first[0][-1]), ('a', 'f1', 2, 2, first[0][-1]),
               ('a', 'f2', 1, 2, first[0][-1]), ('b', 'f1', 1, 2, first[0][-1]),
               ('b', 'f1', 2, 1, first[1][-1])]
    result = summarise(records)
    assert result['unique_contents'] == 2
    assert result['events_in_repeated_groups'] == 4
    assert result['cross_directory_groups'] == result['cross_file_groups'] == 1
    assert result['within_file_repeated_groups'] == 1


@pytest.mark.parametrize('corrupt', [None, 'code', 'inventory', 'dcm', 'slurm', 'export', 'index'])
def test_full_gate_rejects_foreign_or_failed_smoke(tmp_path, corrupt):
    files, membership, tags = [], [], {}
    dcm = tmp_path / 'dcm'
    for run in ('988974', '988975', '988976'):
        p = dcm / run / '1_hits.csv'; p.parent.mkdir(parents=True); p.write_text('input')
        files.append({'path': str(p), 'sha256': sha256(p), 'events': 1, 'records': 1})
        membership.append({'relative': str(p.relative_to(dcm))})
        q = tmp_path / run / '1_prim.csv'; q.parent.mkdir(); q.write_text('primary')
        tags[run] = {'primary_root': str(q.parent), 'common_primary_pair_names': ['1']}
        files.append({'path': str(q), 'sha256': sha256(q), 'events': 1, 'records': 1})
    inventory = tmp_path / 'inventory.json'; inventory.write_text(json.dumps({'tags': tags}))
    dm = tmp_path / 'dcm.json'; dm.write_text(json.dumps({'files': membership}))
    identities = {}
    for name in ('dcm_hits', 'dcm_retained_hits', 'smash_full', 'smash_without_vertex'):
        p = tmp_path / f'{name}.csv.gz'; p.write_bytes(b'index')
        identities[name] = {'index_sha256': sha256(p)}
    summary = tmp_path / 'summary.json'
    x = {'status': 'complete', 'smoke': True, 'code_sha256': sha256(content_census.__file__),
         'smash_inventory_sha256': sha256(inventory), 'dcm_inventory_sha256': sha256(dm),
         'files': files, 'identities': identities}
    if corrupt in ('code', 'inventory', 'dcm'):
        x[{'code': 'code_sha256', 'inventory': 'smash_inventory_sha256', 'dcm': 'dcm_inventory_sha256'}[corrupt]] = 'wrong'
    summary.write_text(json.dumps(x))
    archive = tmp_path / 'archive.tar.gz'; archive.write_bytes(b'archive')
    v = {'slurm_state': 'COMPLETED', 'exit_code': '0:0', 'summary_path': str(summary),
         'summary_sha256': sha256(summary), 'archive_path': str(archive), 'archive_sha256': sha256(archive)}
    if corrupt == 'slurm': v['slurm_state'] = 'FAILED'
    if corrupt == 'export': archive.write_bytes(b'corrupted')
    if corrupt == 'index': (tmp_path / 'dcm_hits.csv.gz').write_bytes(b'corrupted')
    receipt = tmp_path / 'verified.json'; receipt.write_text(json.dumps(v))
    if corrupt:
        with pytest.raises(ValueError): validate_smoke(receipt, inventory, dcm, dm)
    else:
        validate_smoke(receipt, inventory, dcm, dm)
