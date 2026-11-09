"""Describe accepted direct-only associations with paired physical-file errors."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 acceptance=json.load(open(H/'acceptance.json'));assert acceptance['state']=='accepted_numeric_export'
 assert sha(H/'counts.npz')==acceptance['counts_sha256']
 data=np.load(H/'counts.npz');source=R/'reports/response_diagnostics_20261106/counts.npz';assert sha(source)=='f14d46c9c435a522f8e1519f594dd05c6d9f58e3f3c5f5e7b6f982264df12e9f';draw=np.load(source)
 ids=data['content'].tolist();idx=[draw['content'].tolist().index(k) for k in ids];weights=draw['weights'][:,idx];h=data['hist'];bins=['0–0.1','0.1–0.2','0.2–0.3','0.3–0.4','0.4–0.5','0.5–1','1–2','2–4','4–8','≥8'];tags=['zeroSpot','defaultSpot','bigSpot'];pot=[0,18,90];stage=['Geometric primary denominator','Direct raw hit','Retained observed hit','Qualifying: ToF, no energy floor','Qualifying: ToF, E>0.3 GeV'];results={};ratios=[];errors=[];spectrum=[]
 for tag in tags:
  select=np.array([k.startswith(tag+'/') for k in ids]);x=h[select,:,1];c=x.sum(axis=0);b=np.einsum('df,fsqe->dsqe',weights[:,select],x)
  den=c[1,0];rr=np.divide(c[1,1:],den,out=np.full((4,10),np.nan),where=den>0);br=np.divide(b[:,1,1:],b[:,1,0,None,:],out=np.full((len(weights),4,10),np.nan),where=b[:,1,0,None,:]>0)
  valid=(b[:,1,0,:]>0).sum(axis=0);sd=np.full((4,10),np.nan)
  for j in range(10):
   if valid[j]==len(weights):sd[:,j]=br[:,:,j].std(axis=0,ddof=1)
  ratios.append(rr);errors.append(sd);spectrum.append(c[:,0])
  tot=c[1].sum(axis=-1);bt=b[:,1].sum(axis=-1);assert tot[0]>0 and (bt[:,0]>0).all();tr=tot/tot[0];bt=bt/bt[:,0,None]
  reach=c[1,:,3:].sum(axis=-1);breach=b[:,1,:,3:].sum(axis=-1);assert reach[0]>0 and (breach[:,0]>0).all();breach=breach/breach[:,0,None]
  band_fraction={};band_boot={}
  for label,sl in [('low_0p3_1',slice(3,6)),('high_ge2',slice(7,10))]:
   v=c[1,:,sl].sum(axis=-1);vb=b[:,1,:,sl].sum(axis=-1);assert v[0]>0 and (vb[:,0]>0).all();fraction=v/v[0];fb=vb/vb[:,0,None];band_fraction[label]={'counts':v.tolist(),'fraction':fraction.tolist(),'file_SD':fb.std(axis=0,ddof=1).tolist()};band_boot[label]=fb
  response=band_fraction['high_ge2']['fraction'][-1]/band_fraction['low_0p3_1']['fraction'][-1];responseb=band_boot['high_ge2'][:,-1]/band_boot['low_0p3_1'][:,-1]
  results[tag]={'frozen_band_fractions':band_fraction,'direct_primary_native_hardness_response':response,'direct_primary_native_hardness_response_file_SD':float(responseb.std(ddof=1)),'full_primary_rectangle_neutrons':int(c[0,0].sum()),'span_primary_rectangle_neutrons':int(tot[0]),'stage_counts_span_all_energies':tot.tolist(),'stage_fraction_span_all_energies':tr.tolist(),'stage_fraction_span_file_SD':bt.std(axis=0,ddof=1).tolist(),'stage_counts_span_ge_0p3':reach.tolist(),'stage_fraction_span_ge_0p3':(reach/reach[0]).tolist(),'stage_fraction_span_ge_0p3_file_SD':breach.std(axis=0,ddof=1).tolist(),'span_over_full_counts_by_energy':np.divide(c[1,0],c[0,0],out=np.full(10,np.nan),where=c[0,0]>0).tolist(),'direct_stage_fractions_by_energy':rr.tolist(),'direct_stage_fraction_file_SD':errors[-1].tolist(),'valid_bootstrap_denominators_by_energy':valid.tolist(),'SD_policy':'suppress per-bin SD unless all400draws have positive denominators','bootstrap_zero_denominator_cells':int((b[:,1,0,:]==0).sum())}
 plt.rcParams.update({'font.size':11,'axes.labelsize':11,'axes.titlesize':12,'legend.fontsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42})
 colors=['#0072B2','#009E73','#D55E00','#CC79A7'];markers=['o','s','^','D']
 production='SMASH Xe+Cs at 2.87 GeV/nucleon • S286 even parts 0/2/4 • no impact-parameter selection'
 fig,axes=plt.subplots(1,3,figsize=(12.4,4.5),sharey=True);xx=np.arange(10)
 for i,ax in enumerate(axes):
  for st in range(4):ax.errorbar(xx+(st-1.5)*.08,ratios[i][st],yerr=errors[i][st],marker=markers[st],ms=4,linestyle='none',elinewidth=1,color=colors[st],capsize=2,label=stage[st+1])
  ax.set_title(r'$U_{\mathrm{sym}}='+str(pot[i])+r'\,\mathrm{MeV}$');ax.set_xticks(xx,bins,rotation=60,ha='right');ax.set_ylim(0,.4);ax.grid(axis='y',alpha=.2);ax.set_xlabel('Initial primary neutron $E_{kin}$ [GeV]')
 axes[0].set_ylabel('Directly linked objects / geometric primaries')
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.995),frameon=False);fig.text(.5,.869,production,ha='center',fontsize=10);fig.text(.5,.819,'MC truth + observed-hit association • exact rectangle • frozen processed span • bars: paired file-bootstrap SD',ha='center',fontsize=10);fig.subplots_adjust(top=.739,bottom=.28,left=.07,right=.99,wspace=.1)
 fig.text(.5,.025,'Direct means Mother=−1; descendants are not counted. ≥8 GeV has no denominator: points and SD omitted.',ha='center',fontsize=10)
 fig.savefig(H/'direct_observability.png',dpi=180);fig.savefig(H/'direct_observability.pdf');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12.4,4.3),sharey=True)
 for i,ax in enumerate(axes):
  counts=spectrum[i]
  ax.stairs(counts[0],np.arange(11)-.5,label='Full exported event inventory T0',color='#0072B2',lw=1.7);ax.stairs(counts[1],np.arange(11)-.5,label='Frozen processed span T1',color='#D55E00',lw=1.7)
  ax.set_yscale('log');ax.set_xticks(xx,bins,rotation=60,ha='right');ax.set_title(r'$U_{\mathrm{sym}}='+str(pot[i])+r'\,\mathrm{MeV}$');ax.set_xlabel('Initial primary neutron $E_{kin}$ [GeV]');ax.grid(axis='y',alpha=.2)
 axes[0].set_ylabel('MC-truth neutrons per displayed energy bin');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.995),frameon=False);fig.text(.5,.9,production,ha='center',fontsize=10);fig.text(.5,.85,'MC truth • exact per-vertex rectangle • no energy floor • variable-width bins; final bin includes overflow',ha='center',fontsize=10);fig.subplots_adjust(top=.77,bottom=.28,left=.07,right=.99,wspace=.1);fig.text(.5,.025,f'T1 retains {100*sum(v[1].sum() for v in spectrum)/sum(v[0].sum() for v in spectrum):.4f}% of T0 geometric neutrons; spectra overlap at this scale. The ≥8 GeV bin is empty.',ha='center',fontsize=10);fig.savefig(H/'uncensored_inventory.png',dpi=180);fig.savefig(H/'uncensored_inventory.pdf');plt.close(fig)
 # JSON contains missing bins explicitly, never a silent zero fraction.
 def clean(v):
  if isinstance(v,dict):return {k:clean(a) for k,a in v.items()}
  if isinstance(v,list):return [clean(a) for a in v]
  if isinstance(v,float) and not np.isfinite(v):return None
  return v
 out={'state':'measured','scope':'nominal straight-ray geometric primary window; direct-only observability, not measured boundary crossings or full neutron detection efficiency','stages':stage,'results':results,'replicas':400,'uncertainty':'paired whole-file bootstrap SD; same frozen potential x even-part draws as response diagnostics','weights_source_sha256':sha(source),'acceptance_sha256':sha(H/'acceptance.json'),'counts_sha256':sha(H/'counts.npz'),'script_sha256':sha(__file__),'graphs':{'uncensored_inventory':'Tests the energy coverage and processed-span truncation before reconstruction; count per variable-width bin, no density claim.','direct_observability':'Localises direct-object losses at raw, observation and qualifying-hit stages on one denominator; three potentials test composition dependence, no training-seed duplication.'}}
 (H/'results.json').write_text(json.dumps(clean(out),indent=2,allow_nan=False)+'\n');print(json.dumps(clean(results),indent=2))

if __name__=='__main__':main()
