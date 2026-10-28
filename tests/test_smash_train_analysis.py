"""Energy-response summaries and the differential bias with its E_true control."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd

from HGNDRecoGNN.scripts import smash_train_analysis as an


def frame(rng, tag, shift, files=10, n=20000, fold=0):
    e_true = rng.uniform(.3, 6.5, n)
    e_pred = e_true + shift + rng.normal(0, .02, n)
    return pd.DataFrame({'e_true': e_true, 'e_pred': e_pred, 'cl_label': 1, 'cl_score': .9, 'fold': fold,
                         'tag': tag, 'content': [f'{tag}/{i % files}' for i in range(n)], 'cluster_size': 2})


def test_energy_bins_report_bias_mae_response_and_width():
    rng = np.random.default_rng(1)
    e_true = rng.uniform(.3, 6.5, 20000)
    rows = an.energy_bins(e_true, 1.1 * e_true)
    assert all(abs(r['median_response'] - 1.1) < 1e-9 and r['half_width_16_84'] < 1e-9 for r in rows)
    assert all(np.isclose(r['bias_GeV'], .1 * r['mean_e_true']) for r in rows)
    assert rows[0]['e_lo'] == .3  # no MC-positive cluster below the native 0.3 GeV floor


def test_delta_bias_recovers_an_injected_potential_dependent_offset():
    rng = np.random.default_rng(2)
    data = pd.concat([frame(rng, 'zeroSpot', 0.), frame(rng, 'defaultSpot', .004), frame(rng, 'bigSpot', -.010)],
                     ignore_index=True)
    result = an.delta_bias(data, {0: .5})
    window = next(r for r in result['mc_positive']['90-0'] if r['window'])
    assert (window['e_lo'], window['e_hi']) == an.WINDOW
    assert 0 < window['delta_bias_sd_MeV'] < 1.5
    assert abs(window['delta_bias_MeV'] + 10) < 4 * window['delta_bias_sd_MeV']
    assert abs(window['delta_mean_e_true_MeV']) < 3 * window['delta_mean_e_true_sd_MeV'] + 1
    mid = next(r for r in result['mc_positive']['18-0'] if r['window'])
    assert abs(mid['delta_bias_MeV'] - 4) < 4 * mid['delta_bias_sd_MeV']
    # A cut above every score empties the selected population; nothing is invented.
    empty = an.delta_bias(data.assign(cl_score=.1), {0: .5})
    assert all(r['delta_bias_MeV'] is None for r in empty['selected_positive']['90-0'])


def test_selection_uses_each_rows_own_fold_cut():
    f = pd.DataFrame({'cl_score': [.3, .3, .6, .6], 'fold': [0, 1, 0, 1]})
    assert an.selected(f, {0: .2, 1: .5}).tolist() == [True, False, True, True]
