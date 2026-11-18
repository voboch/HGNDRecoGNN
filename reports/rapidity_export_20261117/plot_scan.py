"""Render the S286 rapidity diagnostics from the accepted scan ledger."""
from argparse import ArgumentParser
from json import loads
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ('Rules + ToF', 'DynamicEdgeConv')
METHOD_COLORS = ('#986128', '#145d8d')
MARKERS = ('o', 's')
RAPIDITY = r'$y_{\mathrm{NN\,CM}}$'
CONTEXT = r'SMASH Xe+Cs, 2.87 GeV/nucleon · 286 legacy even files · $T$ from 0.3 to 4 GeV'


def series(ax, edges, metric, index, color, label, marker):
    center = (edges[1:] + edges[:-1]) / 2
    y = np.asarray(metric['value'], dtype=float)[index]
    s = np.asarray(metric['file_sd'], dtype=float)[index]
    valid = np.isfinite(y)
    with_error = valid & np.isfinite(s)
    ax.errorbar(center[with_error], y[with_error], yerr=s[with_error], fmt=marker, ms=3.7,
                color=color, lw=1, capsize=1.5, linestyle='none', label=label)
    missing = valid & ~np.isfinite(s)
    ax.plot(center[missing], y[missing], marker, mfc='none', mec=color, ms=4.2, ls='none')
    ax.set_xlim(edges[0], edges[-1])


def save(fig, directory, name, foot):
    fig.text(.5, .02, foot, ha='center', color='#29323a', fontsize=10)
    for ext in ('pdf', 'png'):
        fig.savefig(directory / (name + '.' + ext), dpi=180)
    plt.close(fig)


def yields(directory, edges, data):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.55), layout='constrained')
    fig.suptitle(CONTEXT, fontsize=12)
    axis_n, axis_ratio, axis_c = axes
    for i, (name, color) in enumerate(zip(('Neutrons', 'Protons'), ('#262626', '#35775f'), strict=True)):
        series(axis_n, edges, data['mc_yield_per_event_per_rapidity'], (i, 0), color, name, MARKERS[i])
    axis_n.set(title='MC geometrical primaries', ylabel=r'$\mathrm{d}N/(N_{\mathrm{evt}}\,\mathrm{d}y)$')
    series(axis_ratio, edges, data['mc_n_over_p'], (0,), '#262626', 'MC', 'o')
    axis_ratio.axhline(1, color='.62', ls='--', lw=.8)
    axis_ratio.set(title='MC yield ratio', ylabel=r'$N_n/N_p$')
    for i, (label, color) in enumerate(zip(METHODS, METHOD_COLORS, strict=True)):
        series(axis_c, edges, data['candidate_yield_per_event_per_rapidity'], (i, 0), color, label, MARKERS[i])
    axis_c.set(title='Selected reconstructed candidates', ylabel=r'$\mathrm{d}N_{\mathrm{cand}}/(N_{\mathrm{evt}}\,\mathrm{d}y)$')
    axis_n.legend(loc='lower center', bbox_to_anchor=(.5, 1.105), ncol=2, frameon=False)
    axis_c.legend(loc='lower center', bbox_to_anchor=(.5, 1.105), ncol=2, frameon=False)
    for ax, level in zip(axes, ('MC', 'MC', 'reco'), strict=True):
        ax.set_xlabel(RAPIDITY + ' (' + level + ')')
        ax.set_ylim(bottom=0)
    fig.get_layout_engine().set(rect=(0, .074, 1, .926))
    save(fig, directory, 'rapidity_yields', 'Per processed event · all transverse momenta · vertical bars show whole-file bootstrap SD · open markers have undefined SD')


def links(directory, edges, data):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), layout='constrained')
    fig.suptitle(CONTEXT, fontsize=12)
    cover, gap, rules, graph = axes.flat
    for i, (label, color) in enumerate(zip(METHODS, METHOD_COLORS, strict=True)):
        series(cover, edges, data['geometric_rates_percent'], (i, 0, 0), color, label, MARKERS[i])
    cover.set(title='Coverage of geometrical MC neutrons', ylabel='With selected direct link (%)')
    cover.legend(loc='lower center', bbox_to_anchor=(.5, 1.105), ncol=2, frameon=False)
    pt_labels = ('All', r'$p_T<0.5$ GeV/$c$', r'$p_T\geq0.5$ GeV/$c$')
    for i, (label, color, marker) in enumerate(zip(pt_labels, ('#262626', '#35775f', '#825077'), ('o', '<', '>'), strict=True)):
        series(gap, edges, data['paired_dynamic_minus_rules_geometric_pp'], (0, i), color, label, marker)
    gap.set(title='Paired difference in direct coverage', ylabel='DynamicEdgeConv − rules (pp)')
    gap.axhline(0, color='.65', ls='--', lw=.8)
    gap.legend(loc='lower center', bbox_to_anchor=(.5, 1.105), ncol=3, frameon=False, fontsize=9)
    for ax, i in ((rules, 0), (graph, 1)):
        for j, (label, color, marker) in enumerate(zip(('Within geometry', 'Outside only', 'No direct link'),
                  ('#35775f', '#986128', '#515d70'), ('o', 's', '^'), strict=True), start=1):
            series(ax, edges, data['candidate_composition_percent'], (i, j, 0), color, label, marker)
        ax.set(title=METHODS[i] + ' candidate composition', ylabel='Selected candidates (%)', ylim=(-5, 105))
        ax.set_xlabel(RAPIDITY + ' (reco)')
    rules.legend(loc='lower left', bbox_to_anchor=(0, 1.105), ncol=3, frameon=False, fontsize=9)
    for ax in (cover, gap):
        ax.set_xlabel(RAPIDITY + ' (MC)')
    fig.get_layout_engine().set(rect=(0, .05, 1, .95), hspace=.11)
    save(fig, directory, 'rapidity_direct_links', 'Seed-averaged estimates · vertical bars show paired whole-file bootstrap SD · open markers have undefined SD')


def migration(directory, edges, data):
    fig = plt.figure(figsize=(11.2, 3.85), layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.15))
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    fig.suptitle(CONTEXT, fontsize=12)
    matrices = np.asarray(data['common_isolated_y_migration'])[:, :, 0]
    denom = matrices.sum(axis=-1, keepdims=True)
    fractions = np.divide(matrices, denom, out=np.full(matrices.shape, np.nan), where=denom > 0).mean(axis=0)
    for i, ax in enumerate(axes[:2]):
        cells = ax.pcolormesh(edges, edges, 100 * fractions[i].T, shading='flat', cmap='Blues', vmin=0, vmax=100)
        ax.plot(edges[[0, -1]], edges[[0, -1]], '--', color='#a95e32', lw=.85)
        ax.set(aspect='equal', title=METHODS[i], xlabel=RAPIDITY + ' (MC)', xlim=(edges[0], edges[-1]), ylim=(edges[0], edges[-1]))
    axes[0].set_ylabel(RAPIDITY + ' (reco)')
    axes[1].tick_params(labelleft=False)
    fig.colorbar(cells, ax=axes[:2], shrink=.82, label='Within MC bin (%)')
    ax = axes[2]
    for i, (label, color) in enumerate((('MAE difference', '#262626'), ('Bias difference', '#825077'))):
        series(ax, edges, data['paired_dynamic_minus_rules_y_error'], (0, i), color, label, MARKERS[i])
    ax.axhline(0, color='.62', ls='--', lw=.8)
    ax.set(title='Paired rapidity error', xlabel=RAPIDITY + ' (MC)', ylabel='DynamicEdgeConv − rules')
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.105), ncol=2, frameon=False, fontsize=9)
    fig.get_layout_engine().set(rect=(0, .074, 1, .926))
    save(fig, directory, 'rapidity_migration', 'Shared selected isolated links · matrices normalize each MC bin · vertical bars show whole-file bootstrap SD')


def plot(scan, output):
    plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 11, 'axes.labelsize': 11,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True,
                         'ytick.right': True, 'pdf.fonttype': 42, 'axes.spines.top': True})
    result = loads(scan.read_bytes())
    if result['state'] != 'complete' or result['files'] != 286:
        raise ValueError('Expected the complete S286 scan')
    output.mkdir(parents=True, exist_ok=True)
    edges = np.asarray(result['y_nn_cm_edges'])
    data = result['scopes']['pooled']
    yields(output, edges, data)
    links(output, edges, data)
    migration(output, edges, data)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('scan', type=Path)
    parser.add_argument('output', type=Path)
    a = parser.parse_args()
    plot(a.scan, a.output)
