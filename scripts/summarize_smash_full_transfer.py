"""Recompute full conditional SMASH metrics, preserving files and training seeds."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.diagnose_student_pipeline import measure, clean, SIZE_EDGES  # noqa: E402
from HGNDRecoGNN.scripts.summarize_smash_transfer import SEEDS, TAGS, METRICS, plot  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(metrics):
    result={}
    for key in METRICS:
        values=[m[key] for m in metrics]
        complete=all(v is not None for v in values)
        result[key]=dict(seed_values=values,mean=float(np.mean(values)) if complete else None,
                        sample_sd=float(np.std(values,ddof=1)) if complete else None)
    return result


def run(root):
    index_path=root/'full_transfer_summary_bundle.json'
    index=json.loads(index_path.read_text())
    if index['state']!='complete':
        raise ValueError('Incomplete full transfer')
    for job in index['job_ids'].values():
        if str(job)+'|COMPLETED|0:0' not in index['slurm_accounting']:
            raise ValueError('GPU transfer job did not complete')
    inventory_path=root/'bound_inventory.json'
    inventory=json.loads(inventory_path.read_text())
    hashes={str(index_path):sha(index_path),str(inventory_path):sha(inventory_path)}
    frames,rows={},{}
    for tag in TAGS:
        folder=root/'full_transfer_inputs/transfer'/tag
        path=folder/'summary.json'
        if sha(path)!=index['summary_sha256'][tag]:
            raise ValueError('Summary changed')
        data=json.loads(path.read_text())
        if (data['state']!='complete' or data['smoke'] or data['model_device']!='cuda:0'
                or data['inventory_sha256']!=sha(inventory_path) or not data['no_training_or_calibration']):
            raise ValueError('Unexpected transfer recipe/status')
        frozen=inventory['tags'][tag]
        if [f['member'] for f in data['files']]!=frozen['common_primary_pair_names']:
            raise ValueError('Incomplete file coverage')
        for file in data['files']:
            if file.get('excluded_exact_byte_alias'):
                if frozen['duplicate_exports'].get(file['member'])!=file['duplicate_of']:
                    raise ValueError('Undeclared byte alias')
        frames[tag],rows[tag]={},{}
        reference=None
        for seed in SEEDS:
            collected=[]
            for item in data['files']:
                if item.get('empty_after_cuts') or item.get('excluded_exact_byte_alias'):
                    continue
                output=item['outputs'][str(seed)]
                p=folder/output['path']
                if sha(p)!=output['sha256']:
                    raise ValueError('Prediction changed')
                hashes[str(p)]=sha(p)
                f=pd.read_csv(p)
                if len(f)!=item['clusters'] or int(f.cl_label.sum())!=item['signal'] or (f.cluster_size<1).any():
                    raise ValueError('Candidate accounting mismatch')
                if f.duplicated(['Row','cl_istop','ClusterID']).any():
                    raise ValueError('Duplicated candidate identity within file')
                f['detector_member']=item['member']
                collected.append(f)
            f=pd.concat(collected,ignore_index=True)
            identity=f[['detector_member','Row','cl_istop','ClusterID','cl_label','e_true','cluster_size']]
            if reference is not None and not identity.equals(reference):
                raise ValueError('Seed populations differ')
            reference=identity
            t=data['per_seed'][str(seed)]['threshold']
            metrics=measure(f,t)
            for k,v in metrics.items():
                expected=data['per_seed'][str(seed)]['metrics'][k]
                if v is None or expected is None:
                    assert v is expected
                elif not np.isclose(v,expected,rtol=1e-6,atol=1e-8):
                    raise ValueError(f'Metric mismatch: {tag}/{seed}/{k}')
            strata=[]
            for lo,hi in zip(SIZE_EDGES[:-1],SIZE_EDGES[1:]):
                subset=f.loc[(f.cluster_size>lo)&(f.cluster_size<=hi)]
                strata.append(dict(lo=lo,hi=hi,**measure(subset,t)))
            rows[tag][str(seed)]=dict(threshold=t,checkpoint_sha256=data['per_seed'][str(seed)]['checkpoint_sha256'],metrics=metrics,size_strata=strata)
            frames[tag][seed]=f
    result=dict(state='verified',scope='Retrospective canonical-file conditional SMASH cluster metrics; no fresh holdout',
                uncertainty='sample SD of three DCM training seeds, not MC/source uncertainty',
                no_smash_training_or_calibration=True,potential_sensitivity_measured=False,
                generated_neutron_efficiency_measured=False,tags={},input_sha256=hashes)
    for tag in TAGS:
        f=frames[tag][42]
        result['tags'][tag]=dict(clusters=len(f),MC_positive_clusters=int(f.cl_label.sum()),
                                events_with_candidates=int(len(f[['detector_member','Row']].drop_duplicates())),
                                event_key='canonical detector member + Row; arms kept together',
                                per_seed=rows[tag],summary=describe([rows[tag][str(s)]['metrics'] for s in SEEDS]),
                                size_strata=[dict(lo=lo,hi=hi,population=dict(n=rows[tag]['42']['size_strata'][i]['n'],signal=rows[tag]['42']['size_strata'][i]['signal']),
                                                  summary=describe([rows[tag][str(s)]['size_strata'][i] for s in SEEDS]))
                                             for i,(lo,hi) in enumerate(zip(SIZE_EDGES[:-1],SIZE_EDGES[1:]))])
    (root/'full_metrics_verified.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    plot(root,frames,scope='canonical-file development transfer')
    print(json.dumps(clean({t:v['summary'] for t,v in result['tags'].items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
