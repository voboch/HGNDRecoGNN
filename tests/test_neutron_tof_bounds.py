"""Analytic identities and numerical derivatives, without physics event samples."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest

from HGNDRecoGNN.analysis import neutron_tof_bounds as tof


def test_exact_kinematics_roundtrip_and_dimensionless_timing_derivative():
    energy = np.array([.01, .1, .3, 1, 2, 3, 4])
    time = tof.flight_time_ns(energy, 7.3)
    assert np.allclose(tof.kinetic_from_time_GeV(time, 7.3), energy, rtol=1e-12)
    # Finite difference of the exact inverse verifies the propagated Jacobian.
    dt = 1e-5
    deriv = (tof.kinetic_from_time_GeV(time + dt, 7.3)
             - tof.kinetic_from_time_GeV(time - dt, 7.3)) / (2 * dt)
    assert np.allclose(abs(deriv) * .15 / energy, tof.timing_fraction(energy, 7.3, .15), rtol=1e-7)
    assert np.all(np.diff(time) < 0)
    assert np.all(np.diff(tof.timing_fraction(energy, 7.3, .15)) > 0)


def test_known_time_gate_and_no_intrinsic_0p3_kinematic_cut():
    assert np.isclose(tof.kinetic_from_time_GeV(35, 7), .3216928254256843)
    assert tof.flight_time_ns(.1, 7.3) > tof.flight_time_ns(.3, 7.3) > 35
    for energy in [.001, .003, .1, .3, 3.]:
        assert np.isclose(tof.elastic_recoil_max_GeV(energy), energy)


def test_prompt_window_has_gaussian_coverage_and_velocity_dependent_path_penalty():
    coverage = tof.prompt_acceptance(np.array([.1, .3, 1.]), 0.)
    assert np.allclose(coverage, .9544997361036416)
    delayed = tof.prompt_acceptance(np.array([.1, .3, 1.]), .03)
    assert np.all(delayed < coverage) and np.all(np.diff(delayed) > 0)
    lengths = np.linspace(0, .15, 20)
    assert np.all(np.diff(tof.prompt_acceptance(.3, lengths)) < 0)


@pytest.mark.parametrize("call", [lambda: tof.beta(0), lambda: tof.beta(-1),
                                  lambda: tof.flight_time_ns(.3, 0),
                                  lambda: tof.kinetic_from_time_GeV(1, 7.3),
                                  lambda: tof.timing_fraction(.3, 7.3, -.1),
                                  lambda: tof.prompt_acceptance(.3, -.1)])
def test_unphysical_domains_are_refused(call):
    with pytest.raises(ValueError):
        call()
