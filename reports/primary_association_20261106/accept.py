"""Read-only numerical acceptance of one published primary-association export."""
import hashlib,json,tarfile
from pathlib import Path,PurePosixPath
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def main():
 plan=json.load(open(H/'plan.json'));launch=json.load(open(H/'submission.json'));job=launch['job_id'];e=H/('export_'+job);receipt=json.load(open(e/'receipt.json'));manifest=json.load(open(e/'manifest.json'))
 assert receipt['state']==manifest['state']=='complete'
 assert sha(e/'manifest.json')==receipt['manifest_sha256']
 assert sha(e/'inputs.tar')==receipt['archive_sha256']==manifest['archive_sha256'] and (e/'inputs.tar').stat().st_size==manifest['archive_bytes']==receipt['archive_bytes']
 expected={r['member']:r for r in manifest['files']};assert len(expected)==receipt['files']==len(manifest['files'])
 root=H/'result';root.mkdir(exist_ok=False);seen=set()
 with tarfile.open(e/'inputs.tar','r:') as tf:
  for m in tf:
   p=PurePosixPath(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and str(p)==m.name and m.name in expected and m.name not in seen
   data=tf.extractfile(m).read();assert len(data)==expected[m.name]['bytes'] and hashlib.sha256(data).hexdigest()==expected[m.name]['sha256'];dest=root/m.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);seen.add(m.name)
 assert seen==set(expected)
 s=json.load(open(root/'summary.json'));assert s['state']=='complete' and s['plan_sha256']==sha(H/'plan.json') and s['script_sha256']==sha(H/'analyze.py') and len(s['files'])==286
 assert sha(root/'summary.json')==receipt['summary_files']['summary.json']
 configs={r['content']:r for r in plan['files']};records={r['content']:r for r in s['files']};assert len(records)==len(configs)==286 and set(records)==set(configs)
 bins=np.array(plan['energy_edges_GeV']+[np.inf]);hists=[];mixed=[];zero=0
 from scipy.sparse import csr_matrix
 from scipy.sparse.csgraph import maximum_bipartite_matching
 for cid in sorted(records):
  r=records[cid];cfg=configs[cid];d=root/cid;record=json.load(open(d/'record.json'));assert record==r
  assert all(sha(d/name)==value for name,value in r['files'].items())
  assert r['observation_sha_exact'] and r['native_replay_exact'] and r['certified_full_event_counter_replay_exact']
  ev=pd.read_csv(d/'events.csv.gz').set_index('Row').sort_index();cols=['exported_particles','n_all','p_all','n_hi','p_hi','n_low_uncensored','p_low_uncensored','n_lo_reach','p_lo_reach'];matrix=np.column_stack([ev.index.to_numpy(),ev[cols].to_numpy()]).astype('<i8');assert hashlib.sha256(matrix.tobytes()).hexdigest()==cfg['certified_full_event_counter_sha256'];assert not ev.index.has_duplicates and len(ev)==cfg['certified_full_events']
  z=np.load(d/'neutrons.npz');row=z['Row'].astype(np.int64);pid=z['Id'].astype(np.int64);energy=z['Ekin'];rectangle=z['in_rectangle'];keys=pd.MultiIndex.from_arrays([row,pid]);assert not keys.has_duplicates and np.isfinite(energy).all() and (energy>=0).all();assert (energy[z['qualifying_native']]>.3).all();zero+=int(((energy==0)&rectangle).sum())
  for a,b in zip(['raw_hit','observed_hit','qualifying_disabled'],['observed_hit','qualifying_disabled','qualifying_native']):assert (z[a]>=z[b]).all()
  hist=np.zeros((2,2,5,10),np.int64)
  for di in [0,1]:
   domain=np.ones(len(row),bool) if di==0 else row<cfg['processed_span']
   for geo in [False,True]:
    base=domain&(rectangle==geo)
    for si,name in enumerate(['denominator','raw_hit','observed_hit','qualifying_disabled','qualifying_native']):
     mask=base if si==0 else base&z[name];hist[di,int(geo),si]=np.histogram(energy[mask],bins)[0]
  assert np.array_equal(hist,np.asarray(r['hist']))
  assert hist[0,1,0].sum()==ev.n_all.sum() and hist[1,1,0].sum()==ev.loc[ev.index<cfg['processed_span'],'n_all'].sum()
  # Cluster keys include file (cid), Row, detector side and local ID.
  clusterkeys=np.column_stack([z['cluster_row'],z['cluster_side'],z['cluster_local_id']]);assert len(np.unique(clusterkeys,axis=0))==r['clusters']
  m={}
  for name in ['observed','disabled','native']:
   edge=z[name+'_edges'];assert edge.shape[1]==3 and len(np.unique(edge,axis=0))==len(edge)
   idx=keys.get_indexer(pd.MultiIndex.from_arrays([edge[:,0],edge[:,1]]));assert (idx>=0).all() and (edge[:,2]>=0).all() and (edge[:,2]<r['clusters']).all();assert np.array_equal(row[idx],z['cluster_row'][edge[:,2]])
   g=csr_matrix((np.ones(len(edge),np.int64),(edge[:,2],idx)),shape=(r['clusters'],len(row)));total=np.diff(g.indptr);inside=np.asarray(g[:,rectangle].sum(axis=1)).ravel();outside=np.asarray(g[:,~rectangle].sum(axis=1)).ravel()
   m[name]={'all_direct_merged_clusters':int((total>1).sum()),'mixed_rectangle_clusters':int(((inside>0)&(outside>0)).sum())}
   for di,domain in enumerate(['all_export','processed_span']):
    dm=np.ones(len(row),bool) if di==0 else row<cfg['processed_span']
    for geo in [False,True]:
     select=dm&(rectangle==geo);a=g[:,select];nc=np.diff(a.indptr);no=np.diff(a.tocsc().indptr);v={'linked_objects':int((no>0).sum()),'linked_clusters':int((nc>0).sum()),'split_objects':int((no>1).sum()),'merged_clusters':int((nc>1).sum()),'maximum_bipartite_cardinality':int((maximum_bipartite_matching(a,perm_type='column')>=0).sum()) if select.any() else 0};assert v==r['cardinality'][domain][str(geo)+'_'+name]
  hists.append(hist);mixed.append(m)
 totals={'clusters':sum(r['clusters'] for r in records.values()),'hits':sum(r['hits'] for r in records.values()),'export_events':sum(r['export_events'] for r in records.values()),'primary_export_records':sum(r['primary_export_records'] for r in records.values()),'direct_binding_records_exact':sum(r['direct_binding_records_exact'] for r in records.values()),'events_zero_rectangle_neutrons':sum(r['events_zero_rectangle_neutrons'] for r in records.values()),'export_Row_gaps':sum(r['missing_Row_in_0_max'] for r in records.values()),'zero_energy_accepted_neutrons':zero}
 assert totals['clusters']==plan['expected_clusters'] and totals['hits']==plan['expected_hits']
 np.savez_compressed(H/'counts.npz',content=np.array(sorted(records)),hist=np.stack(hists))
 out={'state':'accepted_numeric_export','job_id':job,'files':286,**totals,'source_sha256sums':launch['source_sha256sums'],'plan_sha256':sha(H/'plan.json'),'receipt_sha256':sha(e/'receipt.json'),'manifest_sha256':sha(e/'manifest.json'),'archive_sha256':sha(e/'inputs.tar'),'summary_sha256':sha(root/'summary.json'),'counts_sha256':sha(H/'counts.npz'),'checks':['archive/member/summary SHA','full per-event certified matrix SHA','old T1 n/p counters runtime','observation SHA/native dtype-byte targets runtime','unbinned spectra/histograms and nested direct objects','cluster key uniqueness and per-event edges','independent structural split/merge/cardinality replay'],'mixed_cohort_merges':{name:{k:sum(x[name][k] for x in mixed) for k in mixed[0][name]} for name in mixed[0]},'scope':'direct-only associations; not complete descendant genealogy or full neutron detection efficiency','scientific_review':'pending','extra_tests':False}
 (H/'acceptance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
