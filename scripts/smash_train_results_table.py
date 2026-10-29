"""One results table for the SMASH-training campaign, read from the verified analysis outputs.

Protocol: reports/smash_train_protocol_20261028.md. Every number in the results note
comes from the JSON this writes (one number, one source). Reads
results/smash_train_20261028/analysis/summary.json (classification, energy, Delta bias),
results/smash_train_20261028/heldout_rn/summary.json (R_n ladder) and the nine verified
complete.json files (validation AUC); fits and chooses nothing.

Spreads: "seed SD" is the sample SD over the three training seeds (42/123/456); "boot SD"
is the SD of 400 whole-file bootstrap resamples. Arm comparisons have different weights,
so they are paired over the three shared seed labels: mean of the per-seed differences,
SE of that mean, t on 2 df against t_crit = 4.303 (result-representation section 11a, case 1).

usage: python scripts/smash_train_results_table.py
"""
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/smash_train_20261028'
SEEDS = ('42', '123', '456')
FOLDS = ('fold_0', 'fold_1', 'fold_2')
ARMS = ('A0', 'A1', 'B0', 'B1', 'Bcomp')
TASK = {(f, s): 3 * f + i for f in range(3) for i, s in enumerate(SEEDS)}
T_CRIT_2DF = 4.303


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(a, b):
    """Case 1: per-seed differences b - a, their mean, SE of the mean, t on 2 df."""
    d = [y - x for x, y in zip(a, b)]
    mean, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
    t = mean / se if se > 0 else None
    return {'per_seed': d, 'mean': mean, 'se_of_mean': se, 't_2df': t,
            'resolved': t is not None and abs(t) >= T_CRIT_2DF}


def seeds(values):
    return {'per_seed': values, 'mean': st.mean(values), 'seed_sd': st.stdev(values)}


def main():
    ana = json.loads((BASE / 'analysis/summary.json').read_text())
    rn = json.loads((BASE / 'heldout_rn/summary.json').read_text())
    out = {'sources': {'analysis': sha(BASE / 'analysis/summary.json'), 'heldout_rn': sha(BASE / 'heldout_rn/summary.json')},
           'scope': ('cross-fitted test files of corrected, deduplicated SMASH Xe+Cs 2.87 GeV/nucleon '
                     '(U_sym 0/18/90 MeV), split 848ebf49; each file scored only by models that never saw it; '
                     'the files were scored by the frozen transfer before, so not a blind holdout')}

    def cls(arm, key, fold, scope='pooled'):
        return [ana['arms'][arm][s]['classification'][scope][fold]['all'][key] for s in SEEDS]

    # Classification and energy: per-seed values averaged over the three folds (AUC is per fold,
    # since pooling across folds would mix three models' score scales).
    n_clusters = sum(ana['arms']['B1']['42']['classification']['pooled'][f]['all']['n'] for f in FOLDS)
    n_signal = sum(ana['arms']['B1']['42']['classification']['pooled'][f]['all']['signal'] for f in FOLDS)
    table = {}
    for arm in ARMS:
        row = {}
        for key in ('auc', 'ap', 'purity', 'efficiency', 'bias_GeV', 'mae_GeV'):
            per_seed = [st.mean(cls(arm, key, f)[i] for f in FOLDS) for i in range(3)]
            row[key] = seeds(per_seed)
        row['auc_per_fold_seed_mean'] = [st.mean(cls(arm, 'auc', f)) for f in FOLDS]
        table[arm] = row
    out['classification_energy'] = {
        'n': {'test_files': 568, 'clusters': n_clusters, 'mc_positive_clusters': n_signal},
        'definitions': {'auc/ap': 'ROC-AUC and average precision of cl_score against the native label, per fold, '
                                  'then averaged over folds per seed (reconstructed score, MC-truth label)',
                        'purity/efficiency': 'at the cut frozen on fold validation (count purity 0.70)',
                        'bias/mae': 'mean and mean absolute e_pred - e_true over all MC-positive clusters, GeV '
                                    '(MC-truth label and energy, reconstructed energy)'},
        'arms': table,
        'B1_minus_A1': {k: paired(table['A1'][k]['per_seed'], table['B1'][k]['per_seed'])
                        for k in ('auc', 'efficiency', 'mae_GeV')},
        'B0_minus_A0': {k: paired(table['A0'][k]['per_seed'], table['B0'][k]['per_seed'])
                        for k in ('auc', 'bias_GeV', 'mae_GeV')}}

    # Leakage check: validation AUC (complete.json, before test) minus test AUC, per fold and seed.
    gaps = {}
    for arm in ('A0', 'B1'):
        gaps[arm] = []
        for f in range(3):
            for i, s in enumerate(SEEDS):
                c = json.loads((BASE / f'train_4466016/task_{TASK[(f, s)]}/complete.json').read_text())
                gaps[arm].append(c['validation_metrics'][arm]['pooled']['auc'] - cls(arm, 'auc', f'fold_{f}')[i])
    out['auc_validation_minus_test'] = {arm: {'values': v, 'max_abs': max(abs(x) for x in v)} for arm, v in gaps.items()}

    # Calibrated median response on test, 0.3-3 GeV bins, MC-positive clusters.
    resp = {}
    for arm in ('A1', 'B0', 'B1'):
        resp[arm] = [max(abs(r['median_response'] - 1) for r in ana['arms'][arm][s]['energy']['mc_positive']['pooled']
                         if r['e_lo'] >= 0.3 and r['e_hi'] <= 3.0) for s in SEEDS]
    out['median_response_max_abs_dev_0p3_3GeV'] = resp

    # Delta bias in the reviewer's [1.75, 3) GeV window, out of fold, MC-positive clusters.
    delta = {}
    for arm in ARMS:
        delta[arm] = {}
        for key in ('90-0', '18-0'):
            w = [next(r for r in ana['arms'][arm][s]['delta_bias']['mc_positive'][key] if r['window']) for s in SEEDS]
            delta[arm][key] = {'delta_bias_MeV': seeds([r['delta_bias_MeV'] for r in w]),
                               'boot_sd_MeV': [r['delta_bias_sd_MeV'] for r in w],
                               'delta_mean_e_true_MeV': w[0]['delta_mean_e_true_MeV'],
                               'n_high_n_zero': [w[0]['n_high'], w[0]['n_zero']]}
    # The B1 - A1 difference of Delta bias has no paired bootstrap in the analysis; its seed SE
    # misses the shared file-sampling error, so it is reported with the per-arm boot SDs beside it.
    out['delta_bias_window'] = {'window_GeV': [1.75, 3.0], 'arms': delta,
                                'B1_minus_A1_90-0_seed_paired': paired(
                                    delta['A1']['90-0']['delta_bias_MeV']['per_seed'],
                                    delta['B1']['90-0']['delta_bias_MeV']['per_seed']),
                                'note': 'seed SE excludes the file-sampling error shared by all seeds; '
                                        'compare against the per-arm boot SDs'}

    # R_n ladder, primary band [0.3, 1) GeV, pooled out of fold and per fold.
    ladder = {}
    for scope in ('pooled',) + FOLDS:
        v = rn['variants']['primary'][scope]
        ladder[scope] = {'n_files': v['n_files'], 'arms': {}}
        for arm in ARMS:
            e, steps, g = v['arms'][arm]['endpoint'], v['arms'][arm]['steps'], v['gates'][arm]
            ladder[scope]['arms'][arm] = {
                'T1_percent': e['T1/seed_42']['central_percent'], 'T1_boot_sd': e['T1/seed_42']['bootstrap_sd_percent'],
                'C4_percent': [e[f'C4/seed_{s}']['central_percent'] for s in SEEDS],
                'C4_boot_sd': [e[f'C4/seed_{s}']['bootstrap_sd_percent'] for s in SEEDS],
                'retention_C4_over_T1': [e[f'C4/seed_{s}']['central_percent'] / e[f'T1/seed_{s}']['central_percent']
                                         for s in SEEDS],
                'steps_percent': {k: [steps[f'{k}/seed_{s}']['central_percent'] for s in SEEDS]
                                  for k in ('C1-T2', 'C2-C1', 'C3-C2', 'C4-C3', 'C4-T1')},
                'steps_boot_sd': {k: [steps[f'{k}/seed_{s}']['bootstrap_sd_percent'] for s in SEEDS]
                                  for k in ('C1-T2', 'C2-C1', 'C3-C2', 'C4-C3', 'C4-T1')},
                'gates': {'G1_same_sign': g['all_seeds']['G1_same_sign'], 'G2_closure': g['all_seeds']['G2_closure'],
                          'G3_energy_migration': g['all_seeds']['G3_energy_migration'], 'G5_null': g['G5_null'],
                          'G6_seed_stability': g['G6_seed_stability']}}
        p = v['pairs']['B1-A1']['endpoint']
        # The seed SE misses the file-sampling error all seeds share; the paired whole-file
        # bootstrap SD of each seed's difference is the larger, governing uncertainty here.
        ladder[scope]['B1_minus_A1_C4'] = {
            'per_seed': [p[f'C4/seed_{s}']['central_percent'] for s in SEEDS],
            'boot_sd': [p[f'C4/seed_{s}']['bootstrap_sd_percent'] for s in SEEDS],
            'seed_paired': paired(ladder[scope]['arms']['A1']['C4_percent'], ladder[scope]['arms']['B1']['C4_percent']),
            'resolved_against_boot_sd': [abs(x) >= 2 * sd for x, sd in zip(
                [p[f'C4/seed_{s}']['central_percent'] for s in SEEDS],
                [p[f'C4/seed_{s}']['bootstrap_sd_percent'] for s in SEEDS])]}
    out['rn_ladder_primary'] = ladder
    below = rn['variants']['below1']['pooled']['arms']
    out['rn_ladder_below1_pooled_C4'] = {arm: [below[arm]['endpoint'][f'C4/seed_{s}']['central_percent'] for s in SEEDS]
                                         for arm in ('A1', 'B1')}
    out['rn_prediction_rows'] = rn['predictions']

    # Retention C4/T1 with percentile intervals from the ladder's 400 shared whole-file
    # replicas (pooled, primary band): a ratio of two noisy contrasts is skewed, so the
    # interval is quoted rather than a symmetric SD.
    import numpy as np
    reps = np.load(BASE / 'heldout_rn/replicas.npz')
    j1, j4 = 1, 6  # RUNGS = T0, T1, T2, C1, C2, C3, C4
    retention = {}
    for arm in ('A1', 'B1'):
        r = reps[f'primary__pooled__{arm}__endpoint']
        ratio = r[:, :, j4] / r[:, :, j1]
        retention[arm] = {'per_seed_interval_68': np.percentile(ratio, [16, 84], axis=0).T.tolist(),
                          'per_seed_interval_95': np.percentile(ratio, [2.5, 97.5], axis=0).T.tolist(),
                          'fraction_of_replicas_with_C4_not_positive': (r[:, :, j4] <= 0).mean(axis=0).tolist()}
    out['retention_intervals'] = {'source': 'heldout_rn/replicas.npz, primary__pooled__<arm>__endpoint',
                                  'arms': retention}

    # Selection and stop rules.
    out['selection'] = ana['selection']
    out['stop_rules'] = ana['stop_rules']
    out['a0_parity_with_transfer'] = ana['a0_parity_with_transfer']
    (BASE / 'results_table.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'written': str((BASE / 'results_table.json').relative_to(ROOT))}))


if __name__ == '__main__':
    main()
