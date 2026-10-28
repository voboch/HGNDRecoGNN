"""RA-09 retry in the SMASH training path: transient EACCES is absorbed and recorded, nothing else is."""
import errno
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytest

from HGNDRecoGNN.scripts import publish_smash_directory as pub
from HGNDRecoGNN.scripts import scratch_access
from HGNDRecoGNN.scripts import smash_train_prepare as prep
from HGNDRecoGNN.scripts.smash_input_bundle import create, sha


def denied(path='x'):
    return PermissionError(errno.EACCES, 'Permission denied', str(path))


@pytest.fixture
def retries(monkeypatch):
    sleeps = []
    monkeypatch.setattr(scratch_access.time, 'sleep', sleeps.append)
    scratch_access.ACCESS_RETRIES.clear()
    yield sleeps
    scratch_access.ACCESS_RETRIES.clear()


def fail_once(monkeypatch, module, name, before=None):
    """Make module.name raise EACCES on its first call, after `before` ran."""
    original, calls = getattr(module, name), []

    def wrapped(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            if before is not None:
                before(*args)
            raise denied(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapped)
    return calls


def test_retry_schedule_cleanup_and_other_errors(retries):
    calls, cleaned = [], []

    def always(path):
        calls.append(path)
        raise denied(path)

    with pytest.raises(PermissionError):
        scratch_access.with_access_retry(always, 'p', cleanup=lambda: cleaned.append(1))
    assert len(calls) == len(scratch_access.ACCESS_RETRY_WAITS) + 1
    assert retries == list(scratch_access.ACCESS_RETRY_WAITS)
    assert len(cleaned) == len(scratch_access.ACCESS_RETRY_WAITS)  # before every retry, never before the first
    assert [e['attempt'] for e in scratch_access.ACCESS_RETRIES] == [1, 2, 3, 4, 5, 6]
    assert scratch_access.ACCESS_RETRIES[0]['path'] == 'p'
    for error in (PermissionError(errno.EPERM, 'Operation not permitted'), FileExistsError(errno.EEXIST, 'exists')):
        calls.clear()

        def other(path, error=error):
            calls.append(path)
            raise error

        with pytest.raises(type(error)):
            scratch_access.with_access_retry(other, 'q')
        assert calls == ['q']


def test_stage_removes_partial_copy_and_keeps_hash_check(tmp_path, monkeypatch, retries):
    source = tmp_path / 'scratch' / '0001_hits.csv'
    source.parent.mkdir()
    source.write_bytes(b'Row,fX\n0,1.0\n' * 1000)
    target = tmp_path / 'local' / '0001_hits.csv'
    fail_once(monkeypatch, prep, '_copy_hashing', before=lambda s, t: t.write_bytes(b'partial'))
    prep.stage(source, target, sha(source))
    assert target.read_bytes() == source.read_bytes() and retries == [1]
    assert scratch_access.ACCESS_RETRIES[0]['path'] == str(source)
    with pytest.raises(ValueError, match='differs from the split pin'):
        prep.stage(source, tmp_path / 'local' / 'other.csv', '0' * 64)


def test_extract_members_retries_without_touching_existing_files(tmp_path, monkeypatch, retries):
    files = {'prepare/summary.json': b'{"state": "complete"}\n', 'prepare/packs/a.pt': b'a' * 4096,
             'prepare/packs/b.pt': b'b' * 4096}
    for name, data in files.items():
        (tmp_path / 'src' / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / 'src' / name).write_bytes(data)
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'scope': 'test', 'files': [{'source': str(tmp_path / 'src' / n), 'member': n}
                                                           for n in files]}))
    bundle = tmp_path / 'bundle'
    create(plan, bundle)
    out = tmp_path / 'out'
    (out / 'prepare/packs').mkdir(parents=True)
    (out / 'prepare/packs/a.pt').write_bytes(b'already here')  # extracted by an earlier call

    def partial(bundle_dir, manifest_sha256, members, output, check_archive):
        (Path(output) / 'prepare/packs/b.pt').write_bytes(b'cut')

    fail_once(monkeypatch, prep, '_extract_members', before=partial)
    paths = prep.extract_members(bundle, sha(bundle / 'manifest.json'), ['prepare/packs/b.pt'], out)
    assert paths['prepare/packs/b.pt'].read_bytes() == files['prepare/packs/b.pt'] and retries == [1]
    assert (out / 'prepare/packs/a.pt').read_bytes() == b'already here'


def test_publish_retries_partial_scratch_copy(tmp_path, monkeypatch, retries):
    source = tmp_path / 'output'
    (source / 'train').mkdir(parents=True)
    (source / 'train/summary.json').write_text(json.dumps({'state': 'complete'}))
    (source / 'train/history.json').write_text('[1, 2, 3]\n')
    destination = tmp_path / 'scratch' / 'train_bundles' / 'train_1_0'
    calls = fail_once(monkeypatch, pub, 'copy_new', before=lambda s, t: t.write_bytes(b'partial'))
    result = pub.publish(source, tmp_path / 'bundle', destination, tmp_path / 'receipt.json')
    assert len(calls) == 3  # inputs.tar twice, manifest.json once
    for name in ('inputs.tar', 'manifest.json'):
        assert sha(destination / name) == sha(tmp_path / 'bundle' / name)
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt == result and len(receipt['access_retries']) == 1
    assert receipt['access_retries'][0]['path'].endswith('inputs.tar')
