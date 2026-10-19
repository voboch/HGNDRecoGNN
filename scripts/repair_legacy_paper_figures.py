"""Re-export the archived SMASH notebook diagnostics as labelled vector PDFs.

No checkpoint is loaded and no new model evaluation is performed. Inputs are
the original frozen prediction pickles. The evaluation is graph-random and
superseded; do not call it an independent test or DCM result.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from HGNDRecoGNN.analysis.metrics import classifier_curves
from HGNDRecoGNN.scripts.plot_b_full import apply_style

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO.parent/'HGNDPaper'
INPUT = REPO/'notebooks/results/hpc_defaultSpot_v2_seed42'


def save(fig, name):
    fig.savefig(PAPER/'figs'/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(PAPER/'figs'/f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    apply_style()
    record = {'scope':'superseded SMASH graph-random evaluation; single training seed',
              'sources_sha256':{}, 'classifiers':{}}
    frames = {}
    # Keep full print width and the 7-8 pt type while avoiding a whole extra
    # manuscript page for the two archived diagnostics and bibliography.
    fig, axes = plt.subplots(1, 2, figsize=(6.84, 2.5))
    for name, score, label, color, ls in [
        ('hits','score','label','#2b2b2b','-'),
        ('clusters','cl_score','cl_label','#B22222','--'),
        ('edges','link_score','link_label','#1f77b4',':')]:
        source = INPUT/f'pred_{name}_hpc.pkl'
        record['sources_sha256'][str(source.relative_to(REPO))] = hashlib.sha256(source.read_bytes()).hexdigest()
        frame = pd.read_pickle(source)
        if name == 'clusters':
            frames[name] = frame
        scores, labels = frame[score].to_numpy(), frame[label].to_numpy()
        curves = classifier_curves(scores, labels)
        # Drop only PR points whose removal preserves the empirical curve;
        # scores and reported metrics use every original prediction row.
        precision, recall, _ = precision_recall_curve(labels, scores, drop_intermediate=True)
        axes[0].plot(curves.roc_fpr, curves.roc_tpr, color=color, ls=ls,
                     label=f'{name}, AUC {curves.roc_auc:.4f}')
        axes[1].plot(recall, precision, color=color, ls=ls,
                     label=f'{name}, AP {curves.pr_ap:.4f}')
        record['classifiers'][name] = {'n':len(frame), 'roc_auc':curves.roc_auc, 'average_precision':curves.pr_ap}
    axes[0].plot([0,1],[0,1], color='.5', ls='--', lw=.6)
    for ax in axes:
        ax.set(xlim=(0,1), ylim=(0,1.02))
        ax.legend(loc='lower left' if ax is axes[1] else 'lower right', fontsize=7)
    axes[0].set(xlabel='False-positive fraction', ylabel='True-positive fraction')
    axes[1].set(xlabel='Recall', ylabel='Precision')
    fig.suptitle('Archived SMASH Xe+Cs, beam 2.87 AGeV; '+r'$U_{\rm sym}=18$ MeV'+'\n'
                 'Reconstructed classifiers against MC truth; graph-random evaluation', fontsize=8)
    fig.tight_layout();save(fig, 'performance_roc')
    assert abs(record['classifiers']['clusters']['roc_auc']-.9444)<.0001

    clusters = frames['clusters']
    # The 0.5 score cut is the archived notebook convention, not an optimised
    # operating point. The accompanying ROC/PR curves expose its trade-off.
    signal = clusters[(clusters.cl_label == 1) & (clusters.cl_score > .5)]
    true, pred = signal.e_true.to_numpy(), signal.e_pred.to_numpy()
    assert len(signal) and np.all(true > 0) and np.all(np.isfinite(pred))
    # 0.25 GeV is inherited display binning, not a fitted physical scale.
    # Include the full observed energy range rather than the old 0-6 GeV crop.
    edges = np.arange(0, np.ceil(max(true.max(),pred.max())/.25)*.25+.125, .25)
    hist, _, _ = np.histogram2d(true, pred, bins=(edges,edges))
    positive = hist[hist>0]
    fig, axes = plt.subplots(1, 2, figsize=(6.84, 2.5))
    mesh = axes[0].pcolormesh(edges, edges, np.ma.masked_equal(hist.T,0),
        norm=LogNorm(vmin=positive.min(),vmax=positive.max()), cmap='viridis', shading='flat', rasterized=False)
    axes[0].plot([edges[0],edges[-1]],[edges[0],edges[-1]],color='.5',ls='--',lw=.7)
    axes[0].set(xlabel='MC-truth cluster energy (GeV)',ylabel='Reconstructed cluster energy (GeV)')
    fig.colorbar(mesh,ax=axes[0],label='Clusters / 0.25 GeV square bin',pad=.02)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (true>=low)&(true<high)
        if mask.any():
            residual = (pred[mask]-true[mask])/true[mask]
            rows.append({'energy_GeV':(low+high)/2,'n':int(mask.sum()),
                         'mean':float(residual.mean()),'sd':float(residual.std())})
    bins = pd.DataFrame(rows)
    axes[1].errorbar(bins.energy_GeV, bins['mean'], yerr=bins.sd, fmt='o',
                     ms=3,capsize=2,color='#1f77b4',label='Mean residual +/- empirical SD')
    axes[1].axhline(0,color='.5',ls='--',lw=.7)
    axes[1].set(xlabel='MC-truth cluster energy (GeV)',ylabel=r'$(E_{\rm pred}-E_{\rm true})/E_{\rm true}$')
    axes[1].legend(fontsize=6.8)
    fig.suptitle('Archived SMASH Xe+Cs, 2.87 AGeV; MC truth + reconstructed energy\n'
                 'Selected signal clusters, score > 0.5; SD bars show spread, not uncertainty of the mean', fontsize=8)
    fig.tight_layout();save(fig, 'ereco_vs_etrue')
    record['energy'] = {'n_selected_signal_clusters':len(signal), 'threshold':.5,
        'threshold_provenance':'inherited notebook display convention, not optimised',
        'energy_edges_GeV':edges.tolist(),'residual_bins':rows,
        'error_bars':'empirical within-bin SD, not uncertainty of the mean'}
    (PAPER/'data/legacy_diagnostics.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    main()
