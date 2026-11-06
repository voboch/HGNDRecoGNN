"""Same-observation A/B tensor-feature ledger. No model, fit or inference."""
from pathlib import Path, PurePosixPath
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import os
import tarfile
import time
import traceback

import numpy as np
import torch

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def digest(a,dtype):
 return hashlib.sha256(np.ascontiguousarray(a,dtype=dtype).tobytes()).hexdigest()

def work(task):
 root,record=task;torch.set_num_threads(1)
 packs=[]
 for entry in record['members']:
  path=Path(root)/entry['member'];assert sha(path)==entry['sha256']
  packs.append(torch.load(path,map_location='cpu',weights_only=True))
 a,b=packs;assert set(a)==set(b)
 for k in a:
  assert isinstance(a[k],torch.Tensor) and isinstance(b[k],torch.Tensor),k
  if k!='x':
   assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape,(record['content'],k,'dtype/shape')
   assert a[k].numpy().tobytes()==b[k].numpy().tobytes(),(record['content'],k,'bytes')
 x,y=a['x'].numpy(),b['x'].numpy();assert x.shape==y.shape and x.shape[1]==10
 assert np.isfinite(x).all() and np.isfinite(y).all()
 assert x[:,-1].tobytes()==y[:,-1].tobytes() and np.isin(x[:,-1],[0,1]).all()
 nodeptr=a['node_ptr'].numpy();clptr=a['cluster_ptr'].numpy()
 assert nodeptr[0]==clptr[0]==0 and nodeptr[-1]==len(x)
 cluster=a['cluster'].numpy()+np.repeat(clptr[:-1],np.diff(nodeptr))
 ncl=len(a['label']);assert clptr[-1]==ncl and (np.diff(nodeptr)>=0).all() and (np.diff(clptr)>=0).all()
 assert ((cluster>=0)&(cluster<ncl)).all()
 sizes=np.bincount(cluster,minlength=ncl);assert (sizes>0).all()
 lengths=np.diff(clptr)
 keys=np.column_stack([np.repeat(a['row'].numpy(),lengths),np.arange(ncl)-np.repeat(clptr[:-1],lengths),np.repeat(a['side'].numpy(),lengths)])
 assert digest(keys,'<i8')==record['keys_sha256']
 assert digest(np.column_stack([a['label'].numpy(),a['energy'].numpy()]),'<f4')==record['targets_sha256']
 assert digest(sizes,'<i8')==record['sizes_sha256'] and ncl==record['clusters']
 side=np.repeat(a['side'].numpy(),np.diff(nodeptr)).astype(int)
 label=a['label'].numpy()[cluster].astype(int);size=np.digitize(sizes[cluster],[1,3,7],right=True)
 assert np.isin(side,[0,1]).all() and np.isin(label,[0,1]).all()
 group=side*8+label*4+size
 n=np.bincount(group,minlength=16)
 sums=np.zeros((2,16,10));squares=np.zeros_like(sums);diff2=np.zeros((16,10));diffmax=np.zeros((16,10))
 for arm,v in enumerate([x,y]):
  for j in range(10):
   vv=v[:,j].astype(np.float64)
   sums[arm,:,j]=np.bincount(group,weights=vv,minlength=16)
   squares[arm,:,j]=np.bincount(group,weights=vv*vv,minlength=16)
 for j in range(10):
  delta=x[:,j].astype(np.float64)-y[:,j]
  diff2[:,j]=np.bincount(group,weights=delta*delta,minlength=16)
  np.maximum.at(diffmax[:,j],group,np.abs(delta))
 assert diffmax[:,-1].max()==0
 return {'content':record['content'],'fold':record['fold'],'hits':len(x),'clusters':ncl,'graphs':len(a['row']),'n':n,'sum':sums,'sumsq':squares,'squared_encoding_difference':diff2,'max_encoding_difference':diffmax}

def main(args):
 start=time.monotonic();output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
 plan=json.loads(Path(args.plan).read_text());workdir=Path(args.work);workdir.mkdir(parents=True,exist_ok=False)
 # Copy and hash the immutable archive once; scientific data processing stays node-local.
 local=workdir/'inputs.tar';h=hashlib.sha256()
 with Path(plan['archive']).open('rb') as inp,local.open('xb') as dst:
  before=os.fstat(inp.fileno())
  for chunk in iter(lambda:inp.read(8*1024*1024),b''):
   h.update(chunk);dst.write(chunk)
  after=os.fstat(inp.fileno())
 assert before.st_size==after.st_size==plan['archive_bytes'] and before.st_mtime_ns==after.st_mtime_ns
 assert h.hexdigest()==plan['archive_sha256']
 expected={e['member']:e for r in plan['files'] for e in r['members']};assert len(expected)==572
 extracted=set();root=workdir/'selected';root.mkdir()
 with tarfile.open(local,'r:') as archive:
  for member in archive:
   if member.name not in expected:continue
   name=PurePosixPath(member.name)
   assert member.isfile() and not name.is_absolute() and '..' not in name.parts and str(name)==member.name
   assert member.name not in extracted
   p=root/member.name;p.parent.mkdir(parents=True,exist_ok=True)
   stream=archive.extractfile(member);hh=hashlib.sha256()
   with p.open('xb') as dst:
    for chunk in iter(lambda:stream.read(1024*1024),b''):hh.update(chunk);dst.write(chunk)
   assert p.stat().st_size==expected[member.name]['bytes'] and hh.hexdigest()==expected[member.name]['sha256']
   extracted.add(member.name)
 assert extracted==set(expected)
 rows=[]
 with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
  for r in pool.map(work,[(str(root),r) for r in plan['files']]):
   rows.append(r)
   if len(rows)%40==0 or len(rows)==len(plan['files']):print('Native pairs accepted',len(rows),'/',len(plan['files']),flush=True)
 assert len(rows)==286 and sum(r['clusters'] for r in rows)==plan['expected_clusters']
 assert [r['content'] for r in rows]==[r['content'] for r in plan['files']]
 arrays={k:np.stack([r[k] for r in rows]) for k in ['n','sum','sumsq','squared_encoding_difference','max_encoding_difference']}
 arrays.update(content=np.array([r['content'] for r in rows]),fold=np.array([r['fold'] for r in rows]),hits=np.array([r['hits'] for r in rows]),clusters=np.array([r['clusters'] for r in rows]))
 np.savez_compressed(output/'features.npz',**arrays)
 summary={'state':'complete','scope':'S286 original TEST-fold A/B input encodings on identical hits; node-weighted moments; descriptive preprocessing diagnostic, no causal weights ablation or OOD threshold','files':len(rows),'clusters':sum(r['clusters'] for r in rows),'hits':sum(r['hits'] for r in rows),'exact_non_x_replays':len(rows),'exact_prediction_key_target_size_replays':len(rows),'features':plan['features'],'arms':['A: frozen DCM scaler plus affine remap','B: original fold train-only SMASH scaler'],'groups':'side*8 + native_cluster_label*4 + size_bin(1,2-3,4-7,8+)','stat_axes':{'n':'file,group','sum_and_sumsq':'file,arm,group,feature','encoding_difference':'file,group,feature'},'manifest_sha256':plan['manifest_sha256'],'archive_sha256':h.hexdigest(),'plan_sha256':sha(args.plan),'script_sha256':sha(__file__),'features_sha256':sha(output/'features.npz'),'new_models':0,'new_inference':0,'seconds':time.monotonic()-start}
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({'state':'complete','seconds':summary['seconds'],'hits':summary['hits']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ['plan','work','output']:p.add_argument('--'+name,required=True)
 args=p.parse_args()
 if Path(args.output).exists():raise SystemExit('Existing output: refusing to modify execution evidence')
 try:main(args)
 except Exception:
  out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
  (out/'failure.txt').write_text(traceback.format_exc());raise
