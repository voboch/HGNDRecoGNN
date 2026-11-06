"""Even-part, saved-output response diagnostics and frozen correction transfer."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import sys
import time
import traceback
import signal

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT.parent))
from HGNDRecoGNN.scripts.smash_train import apply_calibration

TAGS = ['zeroSpot', 'defaultSpot', 'bigSpot']
SEEDS = [42, 123, 456]
ARMS = ['A1', 'B1']
STAGES = ['T1', 'C1', 'C2', 'C3', 'C4', 'raw_C3', 'raw_C4', 'M_hard']
PAIRS = {'18/0': (1, 0), '90/0': (2, 0), '90/18': (2, 1)}
COLUMNS = ['Row', 'ClusterID', 'cl_istop', 'cl_label', 'e_true', 'cluster_size', 'cl_score', 'e_pred']
KEYS = ['Row', 'ClusterID', 'cl_istop']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, obj):
    (HERE/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def bands(e):
    return [e >= 2, (e >= .3) & (e < 1)]


def count(e, mask):
    return np.array([int((m & mask).sum()) for m in bands(e)])


def state(e):
    # Negative predictions are retained, rather than absorbed into the soft band.
    return np.digitize(e, [0., .3, 1., 2.])


def main():
    start = time.monotonic()
    if (HERE/'results.json').exists():
        raise FileExistsError('Completed output exists; no automatic repeated calculation')
    write('execution.json', {'state': 'running', 'date': '2026-11-06', 'new_jobs': [], 'new_model_calls': 0})
    pins = json.loads((HERE/'input_pins.json').read_text())
    for path, digest in pins['files'].items():
        assert sha(ROOT/path) == digest, path
    split = json.loads((ROOT/pins['split']).read_text())
    groups = {g['tag']+'/'+g['kept']: g for g in split['groups'] if g['truth_bound'] and g['part'] in [0,2,4]}
    ids = sorted(groups)
    assert len(ids) == 286
    index = {cid: i for i, cid in enumerate(ids)}
    hybrid = json.loads((ROOT/'reports/hybrid_np_20261105/results.json').read_text())
    primary = {r['content']: r for r in hybrid['records'] if r['content'] in groups}
    assert set(primary) == set(ids)
    meta = json.loads((ROOT/'reports/smash_performance_20261103/data.json').read_text())
    E = np.r_[-np.inf, meta['energy_edges_GeV'], np.inf]
    source = {}
    registry = {}
    for job in ['4466339','4467086','4467091']:
        base = ROOT/f'results/smash_train_20261028/evaluate_{job}'
        s = json.loads((base/'evaluation_summary.json').read_text())
        assert s['state'] == 'complete' and not s['test_threshold_tuning']
        for cid in s['test_files']:
            if cid in groups:
                assert cid not in registry
                assert groups[cid]['part'] in split['folds'][str(s['fold'])]['test_parts']
                registry[cid] = (base, s)
    assert set(registry) == set(ids)
    n = len(ids)
    ledger = np.zeros((n,3,2,6,2), np.int64)
    joint = np.zeros((n,3,2,4,2,2), np.int64)
    energy = np.zeros((n,3,2,len(E)-1,2), np.int64)
    migration = np.zeros((n,3,2,2,5,5), np.int64)
    errors = np.zeros((n,3,2,5,3), float)
    cand_n = np.zeros(n, np.int64)
    for i, cid in enumerate(ids):
        base, summary = registry[cid]
        ref = None
        for si, seed in enumerate(SEEDS):
            frames = {}
            for arm in ['A0','A1','B0','B1']:
                entry = summary['predictions'][arm][str(seed)][cid]
                p = base/'extracted/evaluation'/entry['path']
                raw = p.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == entry['sha256'], p
                source[str(p.relative_to(ROOT))] = entry['sha256']
                f = pd.read_csv(BytesIO(raw), usecols=COLUMNS)
                assert not f.duplicated(KEYS).any()
                if ref is None:
                    ref = f[KEYS+['cl_label','e_true','cluster_size']].to_numpy()
                    cand_n[i] = len(f)
                else:
                    assert np.array_equal(ref, f[KEYS+['cl_label','e_true','cluster_size']].to_numpy()), (cid,arm,seed)
                assert np.isfinite(f[['e_true','e_pred','cl_score']].to_numpy()).all()
                assert f.cl_score.between(0,1).all() and f.cl_label.isin([0,1]).all()
                frames[arm] = f
            positive = frames['A1'].cl_label.to_numpy() == 1
            target = frames['A1'].e_true.to_numpy()
            size = np.digitize(frames['A1'].cluster_size.to_numpy(), [1,3,7], right=True)
            assert ((size>=0)&(size<4)).all()
            selected = {a: frames[a].cl_score.to_numpy() > summary['training'][str(seed)]['cuts'][a] for a in ARMS}
            sa,sb = selected['A1'], selected['B1']
            np.add.at(joint[i,si], (positive.astype(int), size, sa.astype(int), sb.astype(int)), 1)
            assert joint[i,si].sum() == len(target)
            for ai, arm in enumerate(ARMS):
                raw_arm = 'A0' if arm == 'A1' else 'B0'
                assert np.array_equal(frames[raw_arm].cl_score.to_numpy(),frames[arm].cl_score.to_numpy())
                ep = frames[arm].e_pred.to_numpy(); eraw = frames[raw_arm].e_pred.to_numpy(); sel = selected[arm]
                ledger[i,si,ai] = [count(target,positive),count(target,positive&sel),count(ep,positive&sel),count(ep,sel),count(eraw,positive&sel),count(eraw,sel)]
                old = np.asarray(primary[cid]['counts'])[si,2:6] if arm=='A1' else np.asarray(primary[cid]['counts'])[si,6:10]
                assert np.array_equal(ledger[i,si,ai,:4],old), (cid,seed,arm,'old rung replay')
                energy[i,si,ai,:,0] = np.histogram(target[positive], E)[0]
                energy[i,si,ai,:,1] = np.histogram(target[positive&sel], E)[0]
                for chosen in [0,1]:
                    m=positive & (sel==bool(chosen))
                    np.add.at(migration[i,si,ai,chosen],(state(target[m]),state(ep[m])),1)
                assert migration[i,si,ai].sum() == positive.sum()
                populations=[positive,positive&sa&sb,positive&sa&~sb,positive&~sa&sb,positive&~sa&~sb]
                for pi,m in enumerate(populations):
                    residual=ep[m]-target[m]
                    errors[i,si,ai,pi]=[int(m.sum()),float(residual.sum()),float(np.abs(residual).sum())]
        if i%40==0 or i==n-1:
            print('Accepted even-part CSV content',i+1,'/',n,flush=True)
    assert cand_n.sum() == 1798225
    # Validation coefficients of each original fold/seed. No candidate joins or new fits.
    vcounts = np.zeros((n,3,2,4),np.int64) # NH,NL,SH,SL per physical content
    vfold = np.full(n,-1,int)
    old_bg = json.loads((ROOT/'results/rn_background_correction_20261029/summary.json').read_text())
    coef = np.zeros((3,3,2),float)
    for fold in range(3):
        members=[cid for cid in ids if groups[cid]['part']==split['folds'][str(fold)]['val_part']]
        assert not set(members)&{cid for cid in ids if registry[cid][1]['fold']==fold}
        ix=np.array([index[cid] for cid in members]);vfold[ix]=fold
        for si,seed in enumerate(SEEDS):
            task=3*fold+si
            base=ROOT/f'results/smash_train_20261028/train_4466016/extracted/task_{task}/train'
            complete=json.loads((base/'complete.json').read_text())
            expected_complete=next(s['training'][str(seed)]['complete_sha256'] for _,s in registry.values() if s['fold']==fold)
            assert sha(base/'complete.json')==expected_complete
            assert complete['fold']==fold and int(complete['seed'])==seed
            for ai,arm in enumerate(ARMS):
                rel='validation/predictions_'+('A' if arm=='A1' else 'B')+'.npz'
                path=base/rel;digest=sha(path);assert digest==complete['files'][rel]
                source[str(path.relative_to(ROOT))]=digest
                with np.load(path,allow_pickle=False) as d:
                    keep=np.isin(d['content'],members)
                    content=d['content'][keep];lab=d['cl_label'][keep]==1
                    score=d['cl_score'][keep];ep=apply_calibration(complete['calibrations'][arm],d['e_pred'][keep])
                sel=score>complete['cuts'][arm];hi,lo=bands(ep)
                names,inv=np.unique(content,return_inverse=True)
                assert set(names)==set(members)
                totals=np.stack([np.bincount(inv,weights=m,minlength=len(names)) for m in [sel&hi,sel&lo,sel&hi&lab,sel&lo&lab]],axis=1).astype(np.int64)
                for name,row in zip(names,totals):vcounts[index[name],si,ai]=row
                c=totals.sum(axis=0);coef[fold,si,ai]=(c[1]-c[3])/c[0]
                old=old_bg['per_run'][f'{arm}/fold_{fold}/seed_{seed}']['pooled']['c']
                assert np.isclose(coef[fold,si,ai],old,rtol=0,atol=1e-14)
    assert (vfold>=0).all()
    rng=np.random.default_rng(20261106)
    weights=np.zeros((400,n),np.int64)
    tags=np.array([groups[c]['tag'] for c in ids]);parts=np.array([groups[c]['part'] for c in ids]);testfold=np.array([registry[c][1]['fold'] for c in ids])
    for tag in TAGS:
        for part in [0,2,4]:
            ix=np.flatnonzero((tags==tag)&(parts==part))
            weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),400)
    assert (weights.sum(axis=1)==n).all()
    truth=np.array([np.asarray(primary[c]['counts'])[0,0] for c in ids])
    proton=np.array([np.asarray(primary[c]['counts'])[0,1] for c in ids])
    # Bootstrap coefficient and application share the weights of each physical content.
    bc=np.zeros((400,3,3,2))
    for fold in range(3):
        ix=vfold==fold
        vc=np.einsum('df,fsac->dsac',weights[:,ix],vcounts[ix])
        assert (vc[...,0]>0).all()
        bc[:,fold]=(vc[...,1]-vc[...,3])/vc[...,0]
    central=[];replicas=[];pcentral=[];preplicas=[]
    for tag in TAGS:
        ix=tags==tag
        c=ledger[ix].sum(axis=0);b=np.einsum('df,fsarb->dsarb',weights[:,ix],ledger[ix])
        th=truth[ix].sum(axis=0);bt=weights[:,ix]@truth[ix]
        pc=proton[ix].sum(axis=0);pb=weights[:,ix]@proton[ix]
        corr=c[:,:,3].astype(float).copy();corrb=b[:,:,:,3].astype(float).copy()
        for fold in range(3):
            use=ix&(testfold==fold)
            cc=ledger[use,:,:,3].sum(axis=0)
            bb=np.einsum('df,fsab->dsab',weights[:,use],ledger[use,:,:,3])
            corr[...,1]-=coef[fold]*cc[...,0]
            corrb[...,1]-=bc[:,fold]*bb[...,0]
        # No clipping or redraw: invalid aggregate denominators are a retained failure.
        if not ((corr[...,1]>0).all() and (corrb[...,1]>0).all()):
            bad=np.argwhere(corrb[...,1]<=0)
            write('invalid_denominators.json',{'potential':tag,'invalid_bootstrap_cells':len(bad),'bootstrap_draw_seed_arm_indices':bad.tolist(),'central_seed_arm_indices':np.argwhere(corr[...,1]<=0).tolist(),'corrected_central_low':corr[...,1].tolist(),'corrected_bootstrap_low':corrb[...,1].tolist()})
            raise ValueError('Invalid corrected aggregate denominator; see invalid_denominators.json')
        vals=np.concatenate([np.broadcast_to(th,(3,2,1,2)),c,corr[:,:,None,:]],axis=2)
        boots=np.concatenate([np.broadcast_to(bt[:,None,None,None,:],(400,3,2,1,2)),b,corrb[:,:,:,None,:]],axis=3)
        assert (vals[...,1]>0).all() and (boots[...,1]>0).all()
        central.append(vals[...,0]/vals[...,1]);replicas.append(boots[...,0]/boots[...,1])
        pcentral.append(pc[0]/pc[1]);preplicas.append(pb[:,0]/pb[:,1])
    central=np.stack(central,-1);replicas=np.stack(replicas,-1)
    pcentral=np.array(pcentral);preplicas=np.stack(preplicas,-1)
    pairs={}
    for pair,(hi,lo) in PAIRS.items():
        c=100*(central[...,hi]/central[...,lo]-1);b=100*(replicas[...,hi]/replicas[...,lo]-1)
        adjacent=np.diff(c[:,:,:5],axis=2);ab=np.diff(b[:,:,:,:5],axis=3)
        assert np.allclose(adjacent.sum(axis=2),c[:,:,4]-c[:,:,0],rtol=0,atol=1e-12)
        rawcal=c[:,:,4]-c[:,:,6];br=b[:,:,:,4]-b[:,:,:,6]
        cor=c[:,:,7]-c[:,:,4];corb=b[:,:,:,7]-b[:,:,:,4]
        residual=c[:,:,7]-c[:,:,0];rb=b[:,:,:,7]-b[:,:,:,0]
        dr=central[:,:,4]/pcentral;bd=replicas[:,:,:,4]/preplicas[:,None,None,:]
        k=100*(dr[...,hi]/dr[...,lo]-1);bk=100*(bd[...,hi]/bd[...,lo]-1)
        pairs[pair]={'central_percent':c.tolist(),'file_SD_percent':b.std(axis=0,ddof=1).tolist(),
                     'adjacent_rung_delta_pp':adjacent.tolist(),'adjacent_rung_file_SD_pp':ab.std(axis=0,ddof=1).tolist(),
                     'calibrated_minus_raw_C4_pp':rawcal.tolist(),'calibrated_minus_raw_C4_SD_pp':br.std(axis=0,ddof=1).tolist(),
                     'correction_minus_C4_pp':cor.tolist(),'correction_minus_C4_SD_pp':corb.std(axis=0,ddof=1).tolist(),
                     'corrected_minus_T1_pp':residual.tolist(),'corrected_minus_T1_SD_pp':rb.std(axis=0,ddof=1).tolist(),
                     'hybrid_D_percent':k.tolist(),'hybrid_D_file_SD_percent':bk.std(axis=0,ddof=1).tolist(),
                     'D_shift_pp_per_1pct_log_proton_response':(-(100+k)/100).tolist()}
    np.savez_compressed(HERE/'counts.npz',ledger=ledger,joint=joint,energy=energy,migration=migration,error_sums=errors,validation_counts=vcounts,content=np.array(ids),testfold=testfold,valfold=vfold,tag=tags,candidates=cand_n,weights=weights,central_Rn=central,replicas_Rn=replicas,proton_R=pcentral,proton_replicas_R=preplicas,coefficients=coef,bootstrap_coefficients=bc)
    result={'date':'2026-11-06','scope':'S286 even-part retrospective TEST predictions; cross-fit validation coefficients; no fresh holdout','files':n,'candidates':int(cand_n.sum()),'seeds':SEEDS,'arms':ARMS,'stages':STAGES,'pairs':pairs,'members':ids,'sources':source,'input_pins':pins,'protocol_sha256':sha(HERE/'protocol.md'),'code_sha256':sha(Path(__file__)),'counts_sha256':sha(HERE/'counts.npz'),'draws':{'n':400,'seed':20261106,'strata':'potential x even part','shared_validation_application_content':True},'invalid_aggregate_denominators':0,'unavailable_in_saved_CSV':['raw input features and scaler transforms','primary-to-hit inefficiency','full ancestry','proton tracking/PID'],'elapsed_seconds':time.monotonic()-start}
    write('results.json',result)
    write('execution.json',{'state':'complete','elapsed_seconds':time.monotonic()-start,'new_jobs':[],'new_model_calls':0,'results_sha256':sha(HERE/'results.json')})
    print('COMPLETE',result['elapsed_seconds'],flush=True)


if __name__=='__main__':
    if (HERE/'execution.json').exists() or (HERE/'results.json').exists():
        raise SystemExit('Existing execution evidence: inspect it before any explicit new attempt')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('20-minute execution cap')))
    signal.alarm(1200)
    try:
        main()
    except Exception:
        (HERE/'failure.txt').write_text(traceback.format_exc())
        write('execution.json',{'state':'failed','failure':'failure.txt','new_jobs':[]})
        raise
