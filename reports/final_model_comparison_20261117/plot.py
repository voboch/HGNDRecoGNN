"""Figures from the frozen final-model CPU summaries."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.linewidth': .7,
                     'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.top': True, 'ytick.right': True})
STYLES = {'rules': ('#252525', 'o', 'Rules + ToF'),
          'original_tof': ('#252525', 'o', 'First-hit ToF'),
          'jittered_tof': ('#a33b35', '^', 'First-hit ToF + jitter'),
          'gnn': ('#235e9d', 's', 'DynamicEdgeConv')}


def energy_figure(items, methods, name):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.35), layout='constrained')
    for ax, field, ylabel in zip(axes, ('median_relative_error_percent', 'central_half_width_percent'),
                                 ('Median relative error (%)', 'Central 68% half-width (%)'), strict=True):
        for method in methods:
            color, marker, label = STYLES[method]
            rows = [row for key, row in items.items() if key != 'primary' and row[field][method]['value'] is not None]
            ax.errorbar([np.mean(row['median_truth_GeV_by_seed']) for row in rows],
                        [row[field][method]['value'] for row in rows],
                        yerr=[row[field][method]['file_sd'] for row in rows],
                        color=color, fmt=marker, markersize=4, linewidth=.8, capsize=2, label=label)
        if len(methods) == 3:
            ax.axvspan(4, 6.1, color='#e6e7e8', zorder=0)
            ax.set_xlim(.25, 6.1)
        else:
            ax.set_xlim(.25, 4)
        ax.set_xlabel('MC neutron kinetic energy (GeV)')
        ax.set_ylabel(ylabel)
    axes[0].axhline(0, color='.55', linestyle='--', linewidth=.8)
    axes[1].set_ylim(bottom=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncols=len(methods), frameon=False)
    fig.savefig(HERE / f'{name}.pdf')
    fig.savefig(HERE / f'{name}.png', dpi=210)
    plt.close(fig)


def contrast_figure(cases):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3.25), layout='constrained')
    data = {target: result['open_upper']['contrasts'] for target, result in cases.items()}
    pairs = ('18/0', '90/0', '90/18')
    y = np.arange(3)[::-1]
    entries = [('rules', 'raw70', '#252525', 'o', 'Rules + ToF'),
               ('gnn', 'raw70', '#235e9d', 's', 'DynamicEdgeConv at 70%'),
               ('gnn', 'raw75_frozen', '#8b3c37', '^', 'DynamicEdgeConv at 75%')]
    for shift, (method, target, color, marker, label) in zip((.22, 0, -.22), entries, strict=True):
        rows = [data[target][pair][method] for pair in pairs]
        left.errorbar([r['value'] for r in rows], y + shift, xerr=[r['file_sd'] for r in rows],
                      fmt=marker, color=color, markersize=4, capsize=2, label=label)
        if method == 'gnn':
            rows = [data[target][pair]['absolute_residual_gnn_minus_rules'] for pair in pairs]
            x = np.asarray([r['value'] for r in rows])
            lo, hi = np.asarray([[r[k] for r in rows] for k in ('q16', 'q84')])
            right.hlines(y + shift, lo, hi, color=color, linewidth=.9)
            right.plot(x, y + shift, marker, color=color, markersize=4)
    for pair, index in zip(pairs, y, strict=True):
        mc = data['raw70'][pair]['mc']['value']
        left.vlines(mc, index - .37, index + .37, color='.50', linestyle='--', linewidth=.8)
    for ax in (left, right):
        ax.set_yticks(y, pairs)
        ax.set_ylim(-.6, 2.6)
        ax.axvline(0, color='.75', linewidth=.7)
    left.set_xlabel('Reconstructed contrast (%)')
    left.set_ylabel('Nominal setting pair')
    right.set_xlabel('Change in absolute MC residual (pp)')
    fig.legend(*left.get_legend_handles_labels(), loc='outside upper center', ncols=3, frameon=False, fontsize=9)
    fig.savefig(HERE / 'odd_contrast.pdf')
    fig.savefig(HERE / 'odd_contrast.png', dpi=210)
    plt.close(fig)


if __name__ == '__main__':
    even = json.loads((HERE / 'even.json').read_text())
    odd = json.loads((HERE / 'odd.json').read_text())
    energy_figure(even['first_hit_energy'], ('original_tof', 'jittered_tof', 'gnn'), 'first_hit_response')
    energy_figure(even['rules_energy'], ('rules', 'gnn'), 'rules_response')
    contrast_figure(odd['cases'])
