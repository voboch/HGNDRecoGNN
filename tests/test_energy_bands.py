"""Engineering fixtures pin soft/tail exclusions and zero-event accounting."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
from HGNDRecoGNN.analysis.energy_bands import EnergyBands
from HGNDRecoGNN.scripts.hardness_threshold_opt import _counts_reco, _counts_truth


def test_half_open_boundaries_exclude_soft_denominator_and_open_tail():
    # Toy boundaries test interval semantics; no physics window is selected.
    b = EnergyBands(1., 2., 3., 4.)
    low, high = b.masks([.3, 1., 1.9, 2., 3., 3.9, 4., 10.])
    assert low.tolist() == [False, True, True, False, False, False, False, False]
    assert high.tolist() == [False, False, False, False, True, True, False, False]


def test_event_inventory_order_and_empty_events_are_preserved():
    low, high = EnergyBands(1., 2., 3., 4.).counts([1.5, 3.5], [20, 10], [20, 10, 30])
    assert low.tolist() == [1., 0., 0.]
    assert high.tolist() == [0., 1., 0.]
    with pytest.raises(ValueError):
        EnergyBands(1., 2., 3., 4.).counts([1.5], [99], [1, 2])


def test_histograms_require_exact_band_edges():
    b = EnergyBands(1., 2., 3., 4.)
    lo, hi = b.histogram_masks([0, 1, 2, 3, 4, 5])
    assert np.where(lo)[0].tolist() == [1]
    assert np.where(hi)[0].tolist() == [3]
    with pytest.raises(ValueError):
        b.histogram_masks([0, 1, 2.5, 3, 4])


@pytest.mark.parametrize('edges', [(2, 1, 3, 4), (0, 2, 1, 4), (0, 1, 2, np.inf)])
def test_invalid_order_and_infinite_tail_are_refused(edges):
    with pytest.raises(ValueError):
        EnergyBands(*edges)


def test_hardness_counter_finite_mode_and_legacy_compatibility():
    e = np.array([.3, 1.5, 3.5, 8.])
    u = np.array([1, 1, 2, 2])
    lo, hi = _counts_reco(e, u, (2., 3.))
    assert lo.tolist() == [2., 0.] and hi.tolist() == [0., 2.]
    lo, hi = _counts_reco(e, u, (2., 3.), energy_bounds=(1., 4.), all_units=[1, 2, 3])
    assert lo.tolist() == [1., 0., 0.] and hi.tolist() == [0., 1., 0.]
    hist = np.ones((2, 1, 5))
    lo, hi = _counts_truth(hist, np.arange(6), (2., 3.), energy_bounds=(1., 4.))
    assert lo.tolist() == [1., 1.] and hi.tolist() == [1., 1.]
