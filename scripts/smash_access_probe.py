"""Bounded filesystem checks inside an allocated node; never change input ACLs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    result = {'node': socket.gethostname(), 'state': 'running', 'checks': [],
              'inventory_sha256': hashlib.sha256(args.inventory.read_bytes()).hexdigest()}
    try:
        for tag, data in inventory['tags'].items():
            member = data['common_primary_pair_names'][0]
            for root, suffix in [('raw_root', '_hits.csv'), ('raw_root', '_vacs.csv'), ('primary_root', '_prim.csv')]:
                path = Path(data[root])/(member+suffix)
                with path.open('rb') as stream:
                    assert stream.read(1024)
                result['checks'].append({'path': str(path), 'read': True})
            if tag == 'defaultSpot':
                # Explicitly include the file that caused the previous EACCES.
                path = Path(data['raw_root'])/'10943245/1054_vacs.csv'
                with path.open('rb') as stream:
                    assert stream.read(1024)
                result['checks'].append({'path': str(path), 'read': True})
        args.output.mkdir(parents=True, exist_ok=False)
        temporary, final = args.output/'atomic.tmp', args.output/'atomic.json'
        payload = b'{"write_read_replace": true}\n'
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(final)
        assert final.read_bytes() == payload
        result['state'] = 'complete'
        (args.output/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    except Exception as exc:
        result.update(state='failed', error=repr(exc))
        raise
    finally:
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
