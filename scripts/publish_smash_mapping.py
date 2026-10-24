"""Export a complete node-local mapping once, without replacing scratch files."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def publish(source, destination, receipt, inventory_sha):
    data=json.loads(source.read_text())
    if (data['state']!='complete' or data['inventory_sha256']!=inventory_sha
            or len(data['primary_files'])!=data['files_expected']):
        raise ValueError('Mapping is incomplete or belongs to another inventory')
    destination.mkdir(parents=True,exist_ok=False)
    target=destination/'summary.json.gz'
    with source.open('rb') as inp,target.open('xb') as out:
        with gzip.GzipFile(fileobj=out,mode='wb',mtime=0,filename='') as compressed:
            for block in iter(lambda:inp.read(1024*1024),b''):
                compressed.write(block)
    uncompressed, compressed_hash = hashlib.sha256(), hashlib.sha256()
    # Verify through one open descriptor; avoid a second pathname stat after
    # a successful read, which failed in the preserved Lustre copy attempt.
    with target.open('rb') as raw:
        for block in iter(lambda:raw.read(1024*1024),b''):
            compressed_hash.update(block)
        compressed_bytes = os.fstat(raw.fileno()).st_size
        raw.seek(0)
        with gzip.GzipFile(fileobj=raw,mode='rb') as copied:
            for block in iter(lambda:copied.read(1024*1024),b''):
                uncompressed.update(block)
    if uncompressed.hexdigest()!=sha(source):
        raise ValueError('Export content changed')
    result={'state':'complete','source_summary_sha256':sha(source),'gzip_sha256':compressed_hash.hexdigest(),
            'gzip_path':str(target),'gzip_bytes':compressed_bytes,
            'mapping_source_sha256':data['source_script_sha256'],'inventory_sha256':inventory_sha,
            'source_primary_files':len(data['primary_files']),
            'publisher_sha256':sha(Path(__file__)),'atomic_replace_on_scratch':False}
    with receipt.open('x') as out:
        out.write(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--inventory-sha',required=True)
    a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination,a.receipt,a.inventory_sha)))
