"""Plot accepted common-reference validation reductions, without model calls."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = json.loads((HERE/'results.json').read_text())
Z = np.load(HERE/'plot_data.npz')
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.grid': False,
                     'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True, 'savefig.dpi': 180})
COLORS = {'baseline': '#222222', 'no_floor': '#0072B2'}
LABELS = {'baseline': 'Native baseline B0', 'no_floor': 'No energy clause'}
FILES = []


def finish(fig, name, title, foot):
    fig.suptitle(title, fontsize=13)
    fig.text(.5, .014, foot, ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            b = item.get_window_extent(renderer)
            assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.x1+1 and b.y1 <= fig.bbox.y1+1, (name,item.get_text())
    for ext in ('png','pdf'):
        path = HERE / (name+'.'+ext)
        fig.savefig(path)
        FILES.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)


def curves():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    grid = Z['grid']
    for col, ref in enumerate(('native','disabled')):
        for row, curve in enumerate(('roc','pr')):
            ax = axes[row,col]
            for arm in COLORS:
                values = np.array([np.mean([Z[f'{f*3+s}__{arm}__{ref}__{curve}'] for f in range(3)],axis=0) for s in range(3)])
                mean, sd = values.mean(axis=0), values.std(axis=0,ddof=1)
                ax.plot(grid, mean, color=COLORS[arm], ls='-' if arm=='baseline' else '--', label=LABELS[arm])
                ax.fill_between(grid,np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=COLORS[arm],alpha=.16)
            if curve=='roc':
                ax.plot([0,1],[0,1],'-.' ,color='.55',lw=.8,label='Random ROC reference')
                ax.set(xlabel='False-positive fraction',ylabel='True-positive fraction')
            else:
                prevalence=np.mean([r['metrics']['baseline'][ref]['positive']/r['n'] for r in R['records'] if r['seed']==42])
                ax.axhline(prevalence,color='.55',ls=':',lw=.8,label='MC-positive prevalence')
                ax.set(xlabel='Recall',ylabel='Precision')
            ax.set(xlim=(0,1),ylim=(0,1),title=('Native 0.3-GeV reference' if ref=='native' else 'Disabled-clause reference'))
            ax.text(.96,.04,'Reconstructed scores / MC labels',transform=ax.transAxes,ha='right',fontsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    pr_handles,pr_labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles+[pr_handles[-1]],labels+[pr_labels[-1]],ncol=4,loc='upper center',bbox_to_anchor=(.5,.955),fontsize=8,frameon=False)
    finish(fig,'01_common_reference_curves','SMASH validation: both arms on the same MC-label tasks',
           '290 validation groups; 2.87 GeV/A; no B selection. Curves: equal fold macro, then mean +/- SD of 3 seeds.\nEpochs selected on this validation. Interpolated curves are display summaries; AP is computed directly per fold.')


def metric_pairs():
    items=[('native/auc','Native-reference ROC-AUC [1], higher better',1),
           ('disabled/auc','Disabled-reference ROC-AUC [1], higher better',1),
           ('native/ap','Native-reference AP [1], higher better',1),
           ('disabled/ap','Disabled-reference AP [1], higher better',1),
           ('common_positive/raw_mae_GeV','Common-positive raw MAE [MeV], lower better',1000),
           ('added_soft/raw_mae_GeV','Added-soft raw MAE [MeV], lower better',1000)]
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    for ax,(key,title,scale) in zip(axes.flat,items):
        a=R['macro_by_seed'][key]
        b=np.array(a['baseline']['by_seed_42_123_456'])*scale
        n=np.array(a['no_floor']['by_seed_42_123_456'])*scale
        for i,seed in enumerate((42,123,456)):
            ax.plot([0,1],[b[i],n[i]],color='.65',lw=.8)
            ax.plot(0,b[i],marker=('o','s','^')[i],color=COLORS['baseline'],ms=5)
            ax.plot(1,n[i],marker=('o','s','^')[i],color=COLORS['no_floor'],ms=5,label=f'Seed {seed}')
        vals=np.r_[b,n];pad=max(np.ptp(vals)*.18,abs(vals.mean())*.0002,1e-6)
        ax.set(xlim=(-.25,1.25),ylim=(vals.min()-pad,vals.max()+pad),title=title)
        ax.set_xticks([0,1],['Native baseline','No energy clause'])
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.text(.04,.96,'MC-labelled validation',transform=ax.transAxes,va='top',fontsize=8)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker=m,color='.4',ls='None',label=f'Seed {seed}') for m,seed in zip(('o','s','^'),(42,123,456))]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,.955),frameon=False,fontsize=9)
    finish(fig,'02_paired_metrics','One paired fold-macro result per training seed',
           'Each point averages the same 3 validation folds. The 9 fold x seed cases are not 9 independent samples.\nSoft objects were outside the baseline regression training target; this comparison does not measure neutron efficiency.')


def response():
    fig,axes=plt.subplots(1,2,figsize=(11,4.9))
    for ax,pop,title in zip(axes,('common_positive','added_soft'),('Common positives','Added soft positives')):
        for arm in COLORS:
            counts=[];xs=[];ys=[]
            for s in range(3):
                n=sum(Z[f'{f*3+s}__{arm}__{pop}__n'] for f in range(3))
                x=sum(Z[f'{f*3+s}__{arm}__{pop}__target_sum'] for f in range(3))
                y=sum(Z[f'{f*3+s}__{arm}__{pop}__pred_sum'] for f in range(3))
                counts.append(n);xs.append(np.divide(x,n,out=np.zeros_like(x),where=n>0));ys.append(np.divide(y,n,out=np.zeros_like(y),where=n>0))
            assert all(np.array_equal(counts[0],n) for n in counts[1:])
            mask=counts[0]>0;x=np.array(xs).mean(axis=0)[mask];y=np.array(ys).mean(axis=0)[mask];sd=np.array(ys).std(axis=0,ddof=1)[mask]
            assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(sd).all()
            ax.errorbar(x,y,yerr=sd,color=COLORS[arm],marker='o' if arm=='baseline' else 's',ms=4,ls='-' if arm=='baseline' else '--',label=LABELS[arm],capsize=2)
        xmax=max(x.max()*1.08,.3 if pop=='added_soft' else 0)
        ax.plot([0,xmax],[0,xmax],':',color='.5',lw=1,label='Identity')
        ax.set(xlim=(0,xmax),xlabel='MC max initial Ekin of qualifying tracks [GeV]',ylabel='Mean raw reconstructed energy [GeV]',title=title)
        ax.text(.04,.96,'Reconstructed / MC target\n'+f'N = {int(counts[0].sum()):,}',transform=ax.transAxes,va='top',fontsize=8)
        ax.margins(y=.1)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.5,.94),frameon=False,fontsize=9)
    finish(fig,'03_raw_energy_response','Raw energy response on two fixed candidate populations',
           'Points use the actual mean target in each display bin; bars are SD over 3 seeds, not event/file uncertainty.\nAll validation groups and all target bins retained. Baseline energy head was not trained on added-soft positives.')


def histories():
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    for task,ax in enumerate(axes.flat):
        right=ax.twinx()
        for arm in COLORS:
            cls=Z[f'{task}__{arm}__cluster_cls__history'];en=Z[f'{task}__{arm}__cluster_energy__history']
            assert len(cls)==len(en)==30
            x=np.arange(1,31)
            ax.plot(x,cls,color=COLORS[arm],ls='-',lw=1.1)
            right.plot(x,en,color=COLORS[arm],ls='--',lw=1.1,alpha=.85)
            c=int(np.argmin(cls));e=int(np.argmin(en))
            ax.plot(c+1,cls[c],'o',color=COLORS[arm],ms=4)
            right.plot(e+1,en[e],'s',color=COLORS[arm],ms=4)
        ax.axvline(15,color='.65',ls=':',lw=.8)
        ax.set(xlim=(1,30),title=f'Fold {task//3}, seed {(42,123,456)[task%3]}',xlabel='Committed epoch (one-based)',ylabel='Classification component (BCE)')
        right.set_ylabel('Energy component (loss units)',fontsize=8)
        ax.text(.96,.96,'MC-labelled validation',transform=ax.transAxes,ha='right',va='top',fontsize=7)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=COLORS[a],lw=1.5,label=LABELS[a]) for a in COLORS]+[Line2D([],[],color='.3',ls='-',label='Classification'),Line2D([],[],color='.3',ls='--',label='Energy')]
    fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.5,.96),frameon=False,fontsize=9)
    finish(fig,'04_validation_histories','30-epoch validation histories; each arm uses its own MC targets',
           'Circles/squares mark per-head minima; the dotted line is the registered 15-epoch continuation boundary.\nLoss curves for different target definitions are not a common-task quality comparison. No additional epochs or cut fits.')


if __name__=='__main__':
    curves();metric_pairs();response();histories()
    (HERE/'figure_manifest.json').write_text(json.dumps({'files':FILES,'bounds':'all visible text inside canvas; data axes use finite data/full probability ranges'},indent=2)+'\n')
