"""Relativistic bounds for a theoretical assessment, without event samples.

The mass is the PDG/CODATA2018 value in GeV; c is exact SI in m/ns.
Timing/path error propagation is local and is not full detector resolution.
"""
from __future__ import annotations

from math import erf, sqrt

import numpy as np

NEUTRON_MASS_GEV = .93956542052
C_M_PER_NS = .299792458


def beta(kinetic_GeV):
    kinetic = np.asarray(kinetic_GeV, dtype=float)
    if not np.isfinite(kinetic).all() or np.any(kinetic <= 0):
        raise ValueError("Flight kinetic energy must be finite and positive")
    # This form avoids subtracting two nearly equal numbers at small T/m.
    return np.sqrt(kinetic * (kinetic + 2 * NEUTRON_MASS_GEV)) / (kinetic + NEUTRON_MASS_GEV)


def flight_time_ns(kinetic_GeV, length_m):
    if not np.isfinite(length_m) or length_m <= 0:
        raise ValueError("Flight length must be finite and positive")
    return length_m / (C_M_PER_NS * beta(kinetic_GeV))


def kinetic_from_time_GeV(time_ns, length_m):
    time = np.asarray(time_ns, dtype=float)
    if (not np.isfinite(length_m) or length_m <= 0 or not np.isfinite(time).all()
            or np.any(time <= length_m / C_M_PER_NS)):
        raise ValueError("Physical ToF must exceed the light flight time")
    speed = length_m / (C_M_PER_NS * time)
    root = np.sqrt(1 - speed ** 2)
    return NEUTRON_MASS_GEV * speed ** 2 / (root * (1 + root))


def timing_fraction(kinetic_GeV, length_m, sigma_time_ns):
    if not np.isfinite(sigma_time_ns) or sigma_time_ns < 0:
        raise ValueError("Timing SD must be finite and nonnegative")
    gamma = 1 + np.asarray(kinetic_GeV) / NEUTRON_MASS_GEV
    return gamma * (gamma + 1) * sigma_time_ns / flight_time_ns(kinetic_GeV, length_m)


def prompt_acceptance(kinetic_GeV, extra_path_m, sigma_time_ns=.15, halfwidth_ns=.3):
    """Single-hit constant-energy path-delay toy, not a shower or efficiency model.

    150 ps and +/-300 ps are the unchanged jitter/window of the current
    observation/label recipe. Extra path is a hypothetical control parameter.
    """
    if sigma_time_ns <= 0 or halfwidth_ns < 0 or not np.isfinite([sigma_time_ns, halfwidth_ns]).all():
        raise ValueError("Finite positive jitter and nonnegative window required")
    path = np.asarray(extra_path_m, dtype=float)
    if not np.isfinite(path).all() or np.any(path < 0):
        raise ValueError("Extra path must be finite and nonnegative")
    delay = path / (C_M_PER_NS * beta(kinetic_GeV))
    normal_cdf = np.vectorize(lambda x: .5 * (1 + erf(x / sqrt(2))), otypes=[float])
    return (normal_cdf((halfwidth_ns - delay) / sigma_time_ns)
            - normal_cdf((-halfwidth_ns - delay) / sigma_time_ns))


def elastic_recoil_max_GeV(kinetic_GeV, target_mass_GeV=NEUTRON_MASS_GEV):
    """Exact two-body elastic recoil maximum for a free target initially at rest.

    Equal neutron/proton masses are the stated illustrative default: T_R,max=T.
    This kinematic allowance says nothing about a cross section or light yield.
    """
    kinetic = np.asarray(kinetic_GeV, dtype=float)
    beta(kinetic)  # Validate the incident kinetic-energy domain.
    if not np.isfinite(target_mass_GeV) or target_mass_GeV <= 0:
        raise ValueError("Target rest energy must be finite and positive")
    invariant_s = (NEUTRON_MASS_GEV + target_mass_GeV) ** 2 + 2 * target_mass_GeV * kinetic
    return 2 * target_mass_GeV * kinetic * (kinetic + 2 * NEUTRON_MASS_GEV) / invariant_s
