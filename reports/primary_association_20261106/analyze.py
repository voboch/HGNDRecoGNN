"""Uncensored exported-primary counters and exact direct-neutron associations."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
import traceback
import numpy as np
import pandas as pd
import torch


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def write(p,d):
 Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def process(task):
 r,plan,work,output=task
 # Readers are inherited unchanged from accepted immutable stage-A source.
 sys.path.insert(0,plan['old_source'])
 from HGNDRecoGNN.scripts import smash_train_prepare as prep, audit_smash_sources as audit
 from HGNDRecoGNN.scripts.map_smash_primary import targets_from_vacs, KEYS, COLS
 from HGNDRecoGNN.scripts.hgnd_acceptance import in_acceptance, Z_FRONT, X_LO, X_HI, Y_LO, Y_HI
 from HGNDRecoGNN.data.smash_transfer import _ordered
 from HGNDRecoGNN.data.graph_dataset import _dfToF
 from scipy.sparse import csr_matrix
 from scipy.sparse.csgraph import maximum_bipartite_matching
 cid=r['content'];w=Path(work)/cid;w.mkdir(parents=True,exist_ok=False)
 raw={}
 for kind in ['hits','vacs']:
  p=w/(kind+'.csv');prep.stage(Path(r['raw_root'])/(r['member']+'_'+kind+'.csv'),p,r['sha256'][kind]);raw[kind]=p
 hf=audit.read_csv(raw['hits'],audit.HITS);vf=audit.read_csv(raw['vacs'],audit.VACS)
 if hf.duplicated(['Row','Instance']).any() or vf.duplicated(['Row','Instance']).any():raise ValueError('Duplicate hit/vacs key '+cid)
 targets=targets_from_vacs(vf,cid)
 if len(targets)!=r['binding']['eligible_records']:raise ValueError('Direct primary census count changed '+cid)
 _,frame,dropped=audit.observe(hf,vf,r['member']+'_hits.csv')
 # Exact accepted serialization uses torch.save to a named path, as prepare4465983.
 cols=[c for c in frame.columns if c!='_merge'];op=w/(Path(r['member']).name+'.pt')
 torch.save({'columns':cols,'values':torch.tensor(frame[cols].to_numpy(np.float64))},op)
 if sha(op)!=r['observation']['sha256']:raise ValueError('Observation SHA replay failed '+cid)
 frame,rows,sides=_ordered(frame)
 pack=torch.load(Path(work)/'packs'/r['pack']['member'],map_location='cpu',weights_only=True)
 a={k:v.numpy() for k,v in pack.items()};lengths=np.diff(a['node_ptr']);ncl=len(a['label'])
 if lengths.sum()!=len(frame) or not np.array_equal(np.repeat(a['row'],lengths),rows) or not np.array_equal(np.repeat(a['side'],lengths),sides):raise ValueError('Observation order mismatch')
 cluster=a['cluster']+np.repeat(a['cluster_ptr'][:-1],lengths)
 if (cluster<0).any() or len(np.unique(cluster))!=ncl:raise ValueError('Native cluster coverage')
 energy=frame.Ekin.to_numpy();distance=np.linalg.norm(frame[['fX','fY','fZ']].to_numpy()/100,axis=1);t=frame.fTime.to_numpy()
 prompt=(np.abs(frame.PDG.to_numpy())==2112)&frame.Side.isin([5,6]).to_numpy()&(energy>=_dfToF(distance,t+.3))&(energy<=_dfToF(distance,t-.3))
 native=prompt&(energy>.3);y=np.zeros(ncl,dtype=a['label'].dtype);e=np.zeros(ncl)
 np.maximum.at(y,cluster,native.astype(y.dtype));np.maximum.at(e,cluster,np.where(native,energy,0))
 if y.tobytes()!=a['label'].tobytes() or e.astype(a['energy'].dtype).tobytes()!=a['energy'].tobytes():raise ValueError('Native label/target replay failed')
 span=int(r['processed_span'])
 if int(frame.Row.max())+1!=span:raise ValueError('Historical processed span mismatch')
 # Hits that may fail observation still have their direct association ledger.
 joined=hf[['Row','Instance']].merge(vf[['Row','Instance','Id','PDG','fMotherId','Side']],on=['Row','Instance'],how='left',validate='one_to_one',indicator=True)
 if (joined._merge!='both').any():raise ValueError('Raw hit lacks vac association')
 def direct(f):return (f.fMotherId==-1)&f.Side.isin([5,6])&(f.PDG==2112)
 stage_keys=[];stages=[joined.loc[direct(joined)],frame.loc[direct(frame)],frame.loc[direct(frame)&prompt],frame.loc[direct(frame)&native]]
 for f in stages:stage_keys.append(set(map(tuple,f[['Row','Id']].drop_duplicates().to_numpy(np.int64))))
 edges=[]
 for mask in [direct(frame).to_numpy(),direct(frame).to_numpy()&prompt,direct(frame).to_numpy()&native]:
  edges.append(np.unique(np.column_stack([rows[mask].astype(np.int64),frame.Id.to_numpy(np.int64)[mask],cluster[mask]]),axis=0))
 # A single local primary copy bounds transient disk/memory to two files.
 pp=w/'primary.csv';prep.stage(Path(r['primary_path']),pp,r['primary_sha256'])
 eventpieces=[];selected=[];matched=[];lastrow=-1;lastids=set();records=0;other=0;all_hist=np.zeros((2,10),np.int64)
 bins=np.array(plan['energy_edges_GeV']+[np.inf]);target_keys=pd.MultiIndex.from_frame(targets[['Row','Id']]);rawrows=set(hf.Row.to_numpy(np.int64))
 for ch in pd.read_csv(pp,usecols=COLS,chunksize=100000):
  z=ch.to_numpy(float)
  if not np.isfinite(z).all() or (ch.Ekin<0).any():raise ValueError('Invalid primary kinematics')
  rr=ch.Row.to_numpy();ii=ch.Id.to_numpy()
  if (rr!=np.floor(rr)).any() or (ii!=np.floor(ii)).any() or (rr<0).any() or (ii<0).any() or (np.diff(rr)<0).any() or rr[0]<lastrow or ch.duplicated(['Row','Id']).any():raise ValueError('Invalid/duplicate primary identity')
  boundary=set(ch.loc[ch.Row==lastrow,'Id'])
  if boundary&lastids:raise ValueError('Duplicate primary identity across chunks')
  if rr[-1]==lastrow:lastids|=set(ii)
  else:lastids=set(ch.loc[ch.Row==rr[-1],'Id'])
  lastrow=rr[-1];records+=len(ch);other+=int((~ch.PDG.isin([2112,2212])).sum())
  acc=in_acceptance(*(ch[k].to_numpy() for k in ['Px','Py','Pz','vX','vY','vZ']))
  # Independent cross-multiplied rectangle, including identical boundaries.
  dz=Z_FRONT-ch.vZ.to_numpy();pz=ch.Pz.to_numpy();xx=ch.vX.to_numpy()*pz+dz*ch.Px.to_numpy();yy=ch.vY.to_numpy()*pz+dz*ch.Py.to_numpy()
  acc2=(pz>0)&(dz>0)&(xx>=X_LO*pz)&(xx<=X_HI*pz)&(yy>=Y_LO*pz)&(yy<=Y_HI*pz)
  if not np.array_equal(acc,acc2):raise ValueError('Independent rectangle mismatch')
  ev=pd.DataFrame({'Row':rr,'exported_particles':1})
  for sp,code in enumerate([2112,2212]):
   mask=acc&(ch.PDG.to_numpy()==code);en=ch.Ekin.to_numpy();all_hist[sp]+=np.histogram(en[mask],bins)[0]
   for suffix,m in [('all',mask),('hi',mask&(en>=2)),('low_uncensored',mask&(en<1)),('lo_reach',mask&(en>=.3)&(en<1))]:ev[('n' if sp==0 else 'p')+'_'+suffix]=m.astype(np.int64)
  eventpieces.append(ev.groupby('Row',sort=True).sum())
  hit=targets.merge(ch,on=['Row','Id'],how='inner',suffixes=('_det','_prim'),validate='one_to_one')
  if len(hit):
   if any(not np.array_equal(hit[k+'_det'],hit[k+'_prim']) for k in KEYS):raise ValueError('Exact initial MCTrack binding failed')
   matched.extend(map(tuple,hit[['Row','Id']].to_numpy(np.int64)))
  isdirect=pd.MultiIndex.from_frame(ch[['Row','Id']]).isin(target_keys)&(ch.PDG.to_numpy()==2112)
  keep=(acc&(ch.PDG.to_numpy()==2112))|isdirect
  part=ch.loc[keep,['Row','Id','Ekin']].copy();part['in_rectangle']=acc[keep];selected.append(part)
 events=pd.concat(eventpieces).groupby(level=0).sum().astype(np.int64);events.index=events.index.astype(np.int64)
 if len(matched)!=len(targets) or len(set(matched))!=len(targets):raise ValueError('Incomplete or ambiguous direct primary binding')
 if not rawrows.issubset(set(events.index)) or not set(a['row']).issubset(set(events.index)):raise ValueError('Detector Row absent from primary inventory')
 counter_columns=['exported_particles','n_all','p_all','n_hi','p_hi','n_low_uncensored','p_low_uncensored','n_lo_reach','p_lo_reach']
 certified=np.column_stack([events.index.to_numpy(),events[counter_columns].to_numpy()]).astype('<i8')
 if len(events)!=r['certified_full_events'] or hashlib.sha256(certified.tobytes()).hexdigest()!=r['certified_full_event_counter_sha256']:raise ValueError('Full certified event/multiplicity/counter replay failed')
 if not np.array_equal(all_hist.sum(axis=1),[events.n_all.sum(),events.p_all.sum()]):raise ValueError('Full energy histogram conservation')
 old=events.loc[events.index<span]
 for sp in ['n','p']:
  expected=r['reference_'+sp+'_hi_lo'];got=[int(old[sp+'_hi'].sum()),int(old[sp+'_lo_reach'].sum())]
  if got!=expected:raise ValueError(f'Historical T1 {sp} counter mismatch {got} != {expected}')
 prim=pd.concat(selected,ignore_index=True);key=list(map(tuple,prim[['Row','Id']].to_numpy(np.int64)))
 for name,keys in zip(['raw_hit','observed_hit','qualifying_disabled','qualifying_native'],stage_keys):prim[name]=[k in keys for k in key]
 if not (prim.raw_hit>=prim.observed_hit).all() or not (prim.observed_hit>=prim.qualifying_disabled).all() or not (prim.qualifying_disabled>=prim.qualifying_native).all():raise ValueError('Object stages not nested')
 dest=Path(output)/cid;dest.mkdir(parents=True,exist_ok=False)
 events.to_csv(dest/'events.csv.gz',index_label='Row')
 np.savez_compressed(dest/'neutrons.npz',**{k:prim[k].to_numpy() for k in prim},observed_edges=edges[0],disabled_edges=edges[1],native_edges=edges[2],cluster_row=np.repeat(a['row'],np.diff(a['cluster_ptr'])),cluster_side=np.repeat(a['side'],np.diff(a['cluster_ptr'])),cluster_local_id=np.arange(ncl)-np.repeat(a['cluster_ptr'][:-1],np.diff(a['cluster_ptr'])))
 hist=np.zeros((2,2,5,10),np.int64);cardinality={}
 for domain,domainmask in [('all_export',np.ones(len(prim),bool)),('processed_span',prim.Row.to_numpy()<span)]:
  di=int(domain=='processed_span');cardinality[domain]={}
  for geo in [False,True]:
   select=domainmask&(prim.in_rectangle.to_numpy()==geo)
   # Out-of-window data contains linked direct tracks only, never a denominator.
   for st,name in enumerate(['denominator','raw_hit','observed_hit','qualifying_disabled','qualifying_native']):
    sm=select if st==0 else select&prim[name].to_numpy();hist[di,int(geo),st]=np.histogram(prim.Ekin.to_numpy()[sm],bins)[0]
   objects=prim.loc[select,['Row','Id']];objkeys=pd.MultiIndex.from_frame(objects)
   for stage,edge in zip(['observed','disabled','native'],edges):
    idx=objkeys.get_indexer(pd.MultiIndex.from_arrays([edge[:,0],edge[:,1]]));use=idx>=0
    adj=csr_matrix((np.ones(int(use.sum()),np.int64),(edge[use,2],idx[use])),shape=(ncl,len(objects)))
    nc=np.diff(adj.indptr);no=np.diff(adj.tocsc().indptr)
    cardinality[domain][str(geo)+'_'+stage]={'linked_objects':int((no>0).sum()),'linked_clusters':int((nc>0).sum()),'split_objects':int((no>1).sum()),'merged_clusters':int((nc>1).sum()),'maximum_bipartite_cardinality':int((maximum_bipartite_matching(adj,perm_type='column')>=0).sum()) if len(objects) else 0}
 gaps=int(events.index.max()+1-len(events))
 result={'content':cid,'clusters':ncl,'hits':len(frame),'raw_hits':len(hf),'primary_export_records':records,'export_events':len(events),'missing_Row_in_0_max':gaps,'processed_span':span,'export_events_in_span':len(old),'events_zero_rectangle_neutrons':int((events.n_all==0).sum()),'other_export_particle_records':other,'direct_binding_records_exact':len(matched),'observation_sha_exact':True,'native_replay_exact':True,'certified_full_event_counter_replay_exact':True,'hist':hist.tolist(),'all_species_rectangle_hist':all_hist.tolist(),'cardinality':cardinality,'out_of_rectangle_population':'linked direct neutrons only; denominator incomplete','T1_n_hi_lo_reach':r['reference_n_hi_lo'],'T1_p_hi_lo_reach':r['reference_p_hi_lo'],'access_retries':list(prep.ACCESS_RETRIES),'files':{p.name:sha(p) for p in dest.iterdir()}}
 write(dest/'record.json',result)
 # Only node-local scratch copies made by this task are removed.
 for p in [pp,*raw.values(),op]:p.unlink()
 return result


def run(args):
 start=time.monotonic();out=Path(args.output)
 # Existing outputs, including failed evidence, are never reused.
 if out.exists() or Path(args.work).exists():raise FileExistsError('Existing output/work')
 out.mkdir(parents=True);w=Path(args.work);w.mkdir(parents=True)
 try:
  plan=json.loads(Path(args.plan).read_text())
  if sha(Path(plan['old_source'])/'SHA256SUMS')!=plan['old_source_sums_sha256']:raise ValueError('Inherited source identity mismatch')
  local=w/'inputs.tar';h=hashlib.sha256()
  with Path(plan['archive']).open('rb') as src,local.open('xb') as dst:
   before=os.fstat(src.fileno())
   for b in iter(lambda:src.read(8<<20),b''):h.update(b);dst.write(b)
   after=os.fstat(src.fileno())
  if h.hexdigest()!=plan['archive_sha256'] or before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns or before.st_size!=plan['archive_bytes']:raise ValueError('Native archive changed')
  expected={r['pack']['member']:r['pack'] for r in plan['files']};seen=set()
  with tarfile.open(local,'r:') as tf:
   for m in tf:
    if m.name not in expected:continue
    name=PurePosixPath(m.name)
    if not m.isfile() or name.is_absolute() or '..' in name.parts or m.name in seen:raise ValueError('Unsafe/duplicate member')
    p=w/'packs'/m.name;p.parent.mkdir(parents=True,exist_ok=True)
    with tf.extractfile(m) as src,p.open('xb') as dst:
     for b in iter(lambda:src.read(1<<20),b''):dst.write(b)
    if p.stat().st_size!=expected[m.name]['bytes'] or sha(p)!=expected[m.name]['sha256']:raise ValueError('Native member SHA')
    seen.add(m.name)
  if seen!=set(expected) or len(seen)!=286:raise ValueError('Native membership')
  rows=[]
  tasks=iter([(r,plan,str(w),str(out)) for r in plan['files']])
  with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
   active={pool.submit(process,next(tasks)) for _ in range(plan['workers'])}
   while active:
    done,active=wait(active,return_when=FIRST_COMPLETED)
    # Resolve every completed future before scheduling any further case.
    batch=[future.result() for future in done]
    for r in batch:
     rows.append(r);print('Accepted primary association',len(rows),'/286',r['content'],flush=True)
     write(out/'progress.json',{'state':'running','completed_files':len(rows),'last_content':r['content']})
    for _ in batch:
     task=next(tasks,None)
     if task is not None:active.add(pool.submit(process,task))
  rows.sort(key=lambda r:r['content'])
  if len(rows)!=286 or sum(r['clusters'] for r in rows)!=plan['expected_clusters'] or sum(r['hits'] for r in rows)!=plan['expected_hits']:raise ValueError('Population totals')
  write(out/'summary.json',{'state':'complete','schema':'primary-association-v1','files':rows,'plan_sha256':sha(args.plan),'script_sha256':sha(__file__),'seconds':time.monotonic()-start,'hist_axes':'file,domain(full/span),rectangle(false/true),stage(denominator/raw/observed/disabled/native),energy','energy_edges_GeV':plan['energy_edges_GeV'],'last_bin':'overflow >=8 GeV','scope':'uncensored energy exported nucleons; out-of-rectangle table is direct-linked only; missing genealogy prevents full neutron detection efficiency','new_models':0,'new_inference':0,'extra_tests':False})
 except BaseException:
  write(out/'failure.json',{'state':'failed','traceback':traceback.format_exc(),'seconds':time.monotonic()-start});raise


if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['plan','work','output']:p.add_argument('--'+k,required=True)
 run(p.parse_args())
