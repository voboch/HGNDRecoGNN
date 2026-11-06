"""Read accepted native feature moments; paired physical-file uncertainty."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
summary=json.loads((HERE/'result/summary.json').read_text());assert summary['state']=='complete'
path=HERE/'result/features.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==summary['features_sha256']
z=np.load(path,allow_pickle=False);prior_path=ROOT/'reports/response_diagnostics_20261106/counts.npz'
prior_sha=hashlib.sha256(prior_path.read_bytes()).hexdigest()
prior_acceptance=json.loads((ROOT/'reports/response_diagnostics_20261106/validation.json').read_text());assert prior_sha==prior_acceptance['sha256']['counts.npz']
prior=np.load(prior_path,allow_pickle=False)
assert np.array_equal(z['content'],prior['content']);W=prior['weights']
features=summary['features'];counts=z['n'].sum(0);sums=z['sum'].sum(0);sq=z['sumsq'].sum(0)
assert counts.sum()==summary['hits'] and summary['clusters']==1798225
mu=np.divide(sums,counts[None,:,None],out=np.full_like(sums,np.nan),where=counts[None,:,None]>0)
var=np.divide(sq,counts[None,:,None],out=np.full_like(sq,np.nan),where=counts[None,:,None]>0)-mu*mu
assert (var[np.isfinite(var)]>=-1e-9).all();std=np.sqrt(np.maximum(var,0))
ns=z['n'].sum(1);total=ns.sum();total_mu=z['sum'].sum((0,2))/total
rms=np.sqrt(z['squared_encoding_difference'].sum((0,1))/total)
bootN=W@ns
bootS=np.einsum('df,faj->daj',W,z['sum'].sum(2));bootmu=bootS/bootN[:,None,None]
delta=total_mu[1]-total_mu[0];bdelta=bootmu[:,1]-bootmu[:,0]
boot_rms=np.sqrt((W@z['squared_encoding_difference'].sum(1))/bootN[:,None])
def nullable(a):
 if isinstance(a,list):return [nullable(v) for v in a]
 return a if a is None or np.isfinite(a) else None
assert rms[-1]==0 and delta[-1]==0
result={'scope':summary['scope'],'files':summary['files'],'clusters':summary['clusters'],'hits':summary['hits'],'features':features,'arms':summary['arms'],'node_weighted_mean':total_mu.tolist(),'mean_B_minus_A':delta.tolist(),'paired_file_SD_mean_difference':bdelta.std(0,ddof=1).tolist(),'encoding_RMS_A_minus_B':rms.tolist(),'encoding_RMS_file_SD':boot_rms.std(0,ddof=1).tolist(),'per_group_count':counts.tolist(),'per_group_mean':nullable(mu.tolist()),'per_group_SD':nullable(std.tolist()),'draws':{'n':400,'source':'response_diagnostics_20261106/counts.npz','source_sha256':prior_sha,'strata':'potential x even part; same physical files'},'claim_limit':'Input encodings differ; no model swap or causal weights/scaler attribution. Standardised coordinates have pipeline-specific train units.'}
(HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
plt.rcParams.update({'font.size':13,'axes.labelsize':14,'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
fig,axes=plt.subplots(1,2,figsize=(11.3,4.2),layout='constrained')
for a,label,color,marker in [(0,'A: DCM encoding','#a55022','s'),(1,'B: SMASH fold encoding','#174d78','^')]:
 sd=bootmu[:,a].std(0,ddof=1)
 axes[0].errorbar(np.arange(9)+(a-.5)*.12,total_mu[a,:9],yerr=sd[:9],fmt=marker,color=color,label=label,capsize=3)
axes[0].axhline(0,color='.5',ls='--');axes[0].set(ylabel='Mean encoded input (train units)');axes[0].set_xticks(range(9),features[:9],rotation=55,ha='right')
axes[1].errorbar(range(9),rms[:9],yerr=boot_rms[:,:9].std(0,ddof=1),fmt='o',color='#174d78',capsize=3);axes[1].set(ylabel='Same-hit A/B encoding RMS');axes[1].set_xticks(range(9),features[:9],rotation=55,ha='right')
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False)
fig.text(.99,-.02,'SMASH Xe+Cs · Ekin/A = 2.87 GeV · S286 · no B selection · bars: 400 whole-file SD',ha='right',fontsize=10)
for ext in ['png','pdf']:fig.savefig(HERE/('encoding_comparison.'+ext),dpi=180,bbox_inches='tight',pad_inches=.15)
plt.close(fig)
s='''# Native input feature/scaler diagnostic\n\n6 November 2026. Existing S286 TEST-fold native packs only, no new models,\ninference, training or fits. A uses frozen DCM scalers plus affine remap, B\nuses the original fold train-only SMASH scaler. Different coordinates have\npipeline-specific train units. This is a descriptive input comparison, not\na causal explanation of the A1/B1 performance gap or an OOD test.\n\n'''
s+=f"Accepted {summary['files']} paired files, {summary['clusters']:,} clusters, {summary['hits']:,} hits. Every non-x tensor replays byte-exactly; candidate keys, native labels/targets and cluster sizes match pinned prediction digests. Unscaled binary SurfaceHit is identical.\n\n"
s+='Node-weighted means across identical hits. Differences use paired SD of400whole-file draws stratified potential×evenpart. Hit counts do not act as independent observations. RMS measures encoding distance; its bars are also whole-file SD.\n\n'
s+=f'Largest observed numeric encoding distances are eToF (B−A={delta[8]:.4f}, RMS={rms[8]:.4f}) and fTime (B−A={delta[6]:.4f}, RMS={rms[6]:.4f}). The binary SurfaceHit difference is exactly zero. These describe the preprocessing map on common observations. They do not measure feature importance or the fraction of the A1/B1 performance gap explained by preprocessing. All coordinates are in pipeline-specific train units, with no direction of better. Small error bars can be hidden by markers.\n\n'
s+='|Feature|Mean A|Mean B|B−A ± paired file SD|Same-hit encoding RMS ± file SD|\n|---|---:|---:|---:|---:|\n'
for j,f in enumerate(features):s+=f'|{f}|{total_mu[0,j]:.4g}|{total_mu[1,j]:.4g}|{delta[j]:.4g} ± {bdelta[:,j].std(ddof=1):.2g}|{rms[j]:.4g} ± {boot_rms[:,j].std(ddof=1):.2g}|\n'
s+='''\n![Same-observation encodings](encoding_comparison.png)\n\nThe left panel identifies mean displacement, the right distinguishes encoding\nRMS from a simple mean shift. Bars on both panels are whole-file SD. Zero is only\nthe encoded origin, not proof of in-domain behaviour. SurfaceHit is omitted\nfrom the graph because it is an exact binary control, and retained in the table.\nNative cluster labels describe the qualifying-hit rule, not full genealogy.\n\nNo scaler was substituted into a ready model. Weights, score cuts and energy\ncalibration were not evaluated anew. Raw primary efficiency, ancestry and\nactual proton tracking/PID remain separate measurements.\n\nReproduction/provenance: source_package.json, plan.json, preflight.md, the\naccepted publication receipt and result/summary.json. Per-file/group moments\nand encoding differences remain in result/features.npz; results.json is the\nsole generated descriptive table. Monitoring uses Luna/low under shared §11.\n'''
(HERE/'README.md').write_text(s)
