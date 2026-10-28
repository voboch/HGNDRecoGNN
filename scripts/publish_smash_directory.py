"""Publish node-local audit/prediction files as one verified immutable bundle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.scratch_access import ACCESS_RETRIES, with_access_retry  # noqa: E402
from HGNDRecoGNN.scripts.smash_input_bundle import create, sha  # noqa: E402


def copy_new(source, target):
    with source.open('rb') as inp, target.open('xb') as out:
        shutil.copyfileobj(inp, out, 1024*1024)


def publish(source, bundle, destination, receipt):
    files=sorted(p for p in source.rglob('*') if p.is_file())
    summaries=[p for p in files if p.name=='summary.json']
    if not summaries or any(json.loads(p.read_text()).get('state')!='complete' for p in summaries):
        raise ValueError('Incomplete audit/transfer outputs')
    plan=bundle.with_suffix('.plan.json')
    plan.write_text(json.dumps({'scope':'Complete node-local HGND outputs; immutable publication without scratch atomic replace',
                               'files':[{'source':str(p),'member':str(p.relative_to(source))} for p in files]},indent=2)+'\n')
    create(plan,bundle)
    manifest=json.loads((bundle/'manifest.json').read_text())
    # The destination is on /scratch: EACCES retries each step, removing a
    # partial copy first (RA-09); the hash comparison is unchanged.
    with_access_retry(destination.mkdir,parents=True,exist_ok=False)
    for name in ['inputs.tar','manifest.json']:
        target=destination/name
        with_access_retry(copy_new,bundle/name,target,cleanup=lambda target=target: target.unlink(missing_ok=True))
        if sha(bundle/name)!=with_access_retry(sha,target):
            raise ValueError('Published bundle content changed')
    result={'state':'complete','bundle_dir':str(destination),
            'manifest_sha256':with_access_retry(sha,destination/'manifest.json'),
            'archive_sha256':manifest['archive_sha256'],'archive_bytes':manifest['archive_bytes'],
            'files':len(files),'summary_files':{str(p.relative_to(source)):sha(p) for p in summaries},
            'publisher_sha256':sha(Path(__file__)),'scratch_atomic_replace':False,
            'access_retries':list(ACCESS_RETRIES)}
    with receipt.open('x') as out:
        out.write(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','bundle','destination','receipt']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(publish(a.source,a.bundle,a.destination,a.receipt)))
