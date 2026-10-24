"""Copy explicit SMASH inputs into a hash-verified archive for node-local reads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts or str(path) != name:
        raise ValueError('Unsafe archive member: ' + name)
    return name


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.hash.update(block)
        return block


def create(plan_path, output):
    plan = json.loads(plan_path.read_text())
    entries = plan['files']
    names = [safe_name(x['member']) for x in entries]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate bundle member')
    output.mkdir(parents=True, exist_ok=False)
    partial = output / 'inputs.tar.partial'
    recorded = []
    with tarfile.open(partial, 'w') as archive:
        for i, item in enumerate(entries):
            source = Path(item['source'])
            with source.open('rb') as stream:
                # Validate the exact open file being copied. A second pathname
                # stat can fail on Lustre even after the read succeeded; using
                # the descriptor preserves the same content-stability gate.
                before = os.fstat(stream.fileno())
                info = tarfile.TarInfo(item['member'])
                info.size, info.mode, info.mtime = before.st_size, 0o444, 0
                reader = HashReader(stream)
                archive.addfile(info, reader)
                after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source changed while copying: ' + str(source))
            digest = reader.hash.hexdigest()
            if item.get('sha256') and digest != item['sha256']:
                raise ValueError('Pinned source hash changed: ' + str(source))
            recorded.append({**item, 'sha256': digest, 'bytes': before.st_size})
            if i % 25 == 0 or i + 1 == len(entries):
                print('copied', i + 1, '/', len(entries), flush=True)
    final = output / 'inputs.tar'
    partial.replace(final)
    result = {'state': 'complete', 'scope': plan['scope'], 'plan_sha256': sha(plan_path),
              'source_stat_method': 'fstat on the same read descriptor before and after copying',
              'source_script_sha256': sha(Path(__file__)),
              'archive_sha256': sha(final), 'archive_bytes': final.stat().st_size,
              'unpacked_bytes': sum(x['bytes'] for x in recorded), 'files': recorded}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print('bundle complete', len(recorded), result['archive_bytes'], flush=True)


def unpack(archive_path, manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    if manifest['state'] != 'complete' or sha(archive_path) != manifest['archive_sha256']:
        raise ValueError('Bundle archive hash or state failed')
    expected = {safe_name(x['member']): x for x in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise ValueError('Duplicate manifest members')
    output.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(archive_path, 'r') as archive:
        for item in archive:
            name = safe_name(item.name)
            if name in seen or name not in expected or not item.isfile():
                raise ValueError('Unexpected or duplicate bundle member: ' + name)
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(item)
            digest, size = hashlib.sha256(), 0
            with target.open('xb') as sink:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    sink.write(block)
                    digest.update(block)
                    size += len(block)
            if digest.hexdigest() != expected[name]['sha256'] or size != expected[name]['bytes']:
                raise ValueError('Extracted content hash failed: ' + name)
            seen.add(name)
    if seen != set(expected):
        raise ValueError('Bundle members missing')
    receipt = {'state': 'complete', 'manifest_sha256': sha(manifest_path),
               'archive_sha256': manifest['archive_sha256'], 'files_verified': len(seen)}
    (output / 'bundle_verified.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['create', 'unpack'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'create':
        if not args.plan:
            parser.error('--plan required for create')
        create(args.plan, args.output)
    else:
        if not args.archive or not args.manifest:
            parser.error('--archive and --manifest required for unpack')
        unpack(args.archive, args.manifest, args.output)


if __name__ == '__main__':
    main()
