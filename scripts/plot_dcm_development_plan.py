"""Record the fixed rate perturbation and the preceding validation trajectories."""
from pathlib import Path
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from HGNDRecoGNN.scripts.plot_b_full import apply_style
from HGNDRecoGNN.scripts.dcm_development import recipe
from HGNDRecoGNN.training.train import TrainConfig, _build_scheduler


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'results/dcm_development_20261017/plan';out.mkdir(parents=True, exist_ok=True)
    apply_style();record = {'recipes': {n: recipe(n, 42) for n in ('reference','layernorm','pool','low_lr')}}
    fig, ax = plt.subplots(figsize=(3.30, 2.55))
    record['learning_rates_used'] = {}
    for name, colour, style in [('reference','#2b2b2b','-'),('low_lr','#1f77b4','--')]:
        cfg = TrainConfig(lr=recipe(name,42)['lr'], epochs=30)
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(()))], lr=cfg.lr)
        scheduler,_ = _build_scheduler(cfg,opt);rates=[]
        for _ in range(cfg.epochs):
            rates.append(opt.param_groups[0]['lr']);opt.step();scheduler.step()
        record['learning_rates_used'][name]=rates
        ax.step(np.arange(len(rates)), rates, where='post', color=colour, ls=style, label=name)
    ax.set(xlabel='Epoch (zero-based)', ylabel='Learning rate (dimensionless)', yscale='log')
    ax.tick_params(which='both', top=True, right=True);ax.legend()
    fig.suptitle('DCM development | fixed schedule control', fontsize=8)
    fig.tight_layout();fig.savefig(out/'learning_rates.pdf');fig.savefig(out/'learning_rates.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.84,2.8),sharey=True)
    sources={};record['preceding_validation_best_epochs']={}
    for task,(arm,seed) in enumerate([(a,s) for s in (42,123,456) for a in ('multistep','cosine')]):
        p=root/f'results/dcm_grouped_20261017/current/logs/dcm_group_train_4412700_{task}.out'
        data=p.read_bytes();sources[str(p.relative_to(root))]=hashlib.sha256(data).hexdigest()
        values=[tuple(map(float,r)) for r in re.findall(r'^epoch\s+(\d+)\s+train ([\d.]+)\s+val ([\d.]+)',data.decode(),re.M)]
        assert len(values)==30
        x,tr,va=np.array(values).T;ax=axes[int(arm=='cosine')];colour=['#2b2b2b','#1f77b4','#B22222'][(42,123,456).index(seed)]
        ax.plot(x,tr,color=colour);ax.plot(x,va,color=colour,ls='--')
        k=int(np.argmin(va));marker=('o','s','^')[(42,123,456).index(seed)]
        ax.plot(x[k],va[k],marker=marker,color=colour,mfc='white',ms=4)
        record['preceding_validation_best_epochs'][f'{arm}_{seed}']=k
    for ax,arm in zip(axes,('MultiStepLR','Cosine')):
        ax.set(xlabel='Epoch (zero-based)',title=arm);ax.tick_params(which='both',top=True,right=True)
    axes[0].set_ylabel('Historical composite loss (batch mean)')
    axes[0].legend(handles=[Line2D([],[],color=c,marker=m,mfc='white',label=str(s))
        for s,c,m in zip((42,123,456),('#2b2b2b','#1f77b4','#B22222'),('o','s','^'))],title='Training seed',title_fontsize=7)
    axes[1].legend(handles=[Line2D([],[],color='.2',label='Train'),Line2D([],[],color='.2',ls='--',label='Validation')])
    fig.suptitle('DCM simulation | preceding source-run training/validation | individual trajectories, no error bands',fontsize=8)
    fig.tight_layout();fig.savefig(out/'preceding_losses.pdf');fig.savefig(out/'preceding_losses.png',dpi=200);plt.close(fig)
    record['sources_sha256']=sources
    (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
