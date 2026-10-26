"""Composition matching must remove a purely compositional difference."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.reach_controls import reweighted, classes
from HGNDRecoGNN.analysis.truth_statistics import values


def test_composition_difference_removed():
    def cell(n, ratio):
        return [n, 100 * n, 100 * n, ratio * n, 10 * n, 10 * n, 10 * n]

    a = np.array(
        [
            [cell(100, 2), cell(300, 8)],
            [cell(200, 2), cell(200, 8)],
            [cell(300, 2), cell(100, 8)],
        ],
        float,
    )
    assert np.ptp(values(a[:, :, 1:].sum(axis=1))[:, 0]) > 0.1
    assert np.ptp(values(reweighted(a))[:, 0]) == pytest.approx(0, abs=1e-12)


def test_five_classes_conserve_events_and_keep_ties():
    h = np.ones((3, 20, 7), int)
    c, cuts = classes(h)
    assert c.shape == (3, 5, 7) and len(cuts) == 4
    assert np.array_equal(c.sum(axis=1), h.sum(axis=1))
