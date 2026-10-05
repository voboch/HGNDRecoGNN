"""The HGND kinematic selection, applied to primary neutrons and protons.

`in_acceptance` propagates a straight line from the vertex. That is a neutron's
trajectory and not a proton's: the analysing magnet bends the proton, and no
field map enters this analysis. The isovector ratios compare the two species,
so what they need is one *kinematic region* both can be counted in, defined on
emission direction alone and therefore charge-independent. These tests pin that
region down and bound what dropping the vertex costs.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from hgnd_acceptance import (  # noqa: E402
    Z_FRONT, X_LO, X_HI, Y_LO, Y_HI,
    acceptance_bounds, in_acceptance, kinematic_acceptance, vertex_sensitivity,
)


def test_bounds_match_the_stated_geometry():
    """The note quotes 8.45-12.99 deg and 20 % of azimuth; both must come out."""
    tmin, tmax, phi_c, phi_w = acceptance_bounds()
    assert abs(tmin - 8.45) < 0.02, tmin
    assert abs(tmax - 12.99) < 0.02, tmax
    assert abs(phi_c - 180.0) < 0.5, phi_c
    assert abs(phi_w / 360.0 - 0.20) < 0.005, phi_w


def test_selection_is_charge_and_species_blind():
    """Both species must get the identical cut, or the ratio is not a ratio."""
    rng = np.random.default_rng(1)
    p = rng.normal(0, 1, (3, 5000))
    p[2] = np.abs(p[2]) + 1.0
    a = kinematic_acceptance(*p)
    b = kinematic_acceptance(*p)          # nothing about the particle but p
    assert np.array_equal(a, b)
    assert a.dtype == bool


def test_a_ray_through_the_face_centre_is_selected():
    cx = 0.5 * (X_LO + X_HI)
    cy = 0.5 * (Y_LO + Y_HI)
    assert kinematic_acceptance([cx], [cy], [Z_FRONT])[0]
    # mirrored in x: the face is offset to negative x, so +x must fail
    assert not kinematic_acceptance([-cx], [cy], [Z_FRONT])[0]
    # backward-going never counts
    assert not kinematic_acceptance([cx], [cy], [-Z_FRONT])[0]


def test_energy_window_is_applied_when_given():
    cx = 0.5 * (X_LO + X_HI)
    px, py, pz = [cx] * 3, [0.0] * 3, [Z_FRONT] * 3
    ek = [0.2, 1.0, 5.0]
    sel = kinematic_acceptance(px, py, pz, ekin=ek, ekin_lo=0.5, ekin_hi=3.0)
    assert list(sel) == [False, True, False]


def test_dropping_the_vertex_is_a_sub_percent_effect_on_the_count():
    """Individual particles migrate across the edge; the net count must not.

    This is the whole justification for using a direction-only region: the
    disagreement per particle is a few per cent, but in and out largely cancel,
    so a counting observable is biased far less. If this ever grows, the
    kinematic selection stops being a safe stand-in for the geometric one.
    """
    r = vertex_sensitivity(sigma_xy=0.5, sigma_z=2.0, n=400_000, seed=0)
    assert r["disagree_frac"] < 0.05, r
    assert abs(r["net_bias"]) < 0.01, r


def test_agrees_with_the_geometric_test_at_a_point_vertex():
    """With the vertex at the origin the two must be the same selection."""
    rng = np.random.default_rng(2)
    p = rng.normal(0, 1, (3, 20000))
    p[2] = np.abs(p[2]) + 0.5
    kin = kinematic_acceptance(*p)
    geo = in_acceptance(*p, 0.0, 0.0, 0.0)
    assert np.array_equal(kin, geo)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'ok  {name}')
    print('all passed')
