"""Background models of the R_n correction: each is exact when its own assumption holds, and only then."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts import rn_background_correction as bc  # noqa: E402

# Signal (high, low) per tag with a +5 % hardness contrast at 90 MeV, and events per tag.
SIGNAL = {"zeroSpot": (6866.0, 3460.0), "defaultSpot": (6900.0, 3450.0), "bigSpot": (7210.0, 3460.0)}
EVENTS = {"zeroSpot": 1e5, "defaultSpot": 1e5, "bigSpot": 1e5}
TRUE = bc.contrast({t: np.array(SIGNAL[t][0]) for t in SIGNAL}, {t: np.array(SIGNAL[t][1]) for t in SIGNAL})


def pooled(background):
    """Counts [N_H, N_L, S_H, S_L] per tag for a background rule, and pooled-over-potential coefficients."""
    c = {t: np.array([SIGNAL[t][0] + background[t][0], SIGNAL[t][1] + background[t][1], *SIGNAL[t]]) for t in SIGNAL}
    coef = bc.coefficients(sum(c.values()), sum(EVENTS.values()))
    return c, coef


def corrected_contrast(c, coef, model):
    h, low = {}, {}
    for t in c:
        h[t], low[t] = bc.corrected(c[t], np.array(EVENTS[t]), coef, model)
    return float(bc.contrast(h, low))


def test_potential_independent_purity_cancels():
    c, coef = pooled({t: (1600.0, 2700.0) for t in SIGNAL})
    raw = bc.contrast({t: c[t][bc.NH] for t in c}, {t: c[t][bc.NL] for t in c})
    scaled = bc.contrast({t: coef["pi_H"] * c[t][bc.NH] for t in c}, {t: coef["pi_L"] * c[t][bc.NL] for t in c})
    assert abs(raw - scaled) < 1e-12


def test_m_event_exact_when_background_is_constant_per_event():
    c, coef = pooled({t: (1600.0, 2700.0) for t in SIGNAL})
    assert corrected_contrast(c, coef, "M_event") == pytest.approx(TRUE, abs=1e-9)
    assert abs(corrected_contrast(c, coef, "C4") - TRUE) > 0.5


def test_m_hard_exact_when_background_follows_hard_signal_and_not_otherwise():
    k_h, k_l = 0.235, 0.395
    c, coef = pooled({t: (k_h * SIGNAL[t][0], k_l * SIGNAL[t][0]) for t in SIGNAL})
    assert corrected_contrast(c, coef, "M_hard") == pytest.approx(TRUE, abs=1e-9)
    assert abs(corrected_contrast(c, coef, "M_event") - TRUE) > 0.5  # wrong model, wrong answer
    c, coef = pooled({t: (1600.0, 2700.0) for t in SIGNAL})
    assert abs(corrected_contrast(c, coef, "M_hard") - TRUE) > 0.5


def explicit(c, coef):
    return corrected_contrast(c, coef, "M_hard")


def test_removed_fraction_and_k_bias_match_explicit_computation():
    k_h, k_l = 0.235, 0.395
    c, coef = pooled({t: (k_h * SIGNAL[t][0], k_l * SIGNAL[t][0]) for t in SIGNAL})
    # Dilute the raw contrast so the correction is non-trivial: extra low-band background at 90 MeV.
    c["bigSpot"] = c["bigSpot"] + np.array([0.0, 150.0, 0.0, 0.0])
    coef = bc.coefficients(sum(c.values()), sum(EVENTS.values()))
    c4 = bc.contrast({t: c[t][bc.NH] for t in c}, {t: c[t][bc.NL] for t in c})
    k_c = explicit(c, coef)
    for r in (0.5, 0.8):
        swapped = explicit(c, {**coef, "c": r * coef["c"]})
        assert (k_c - swapped) / (k_c - c4) == pytest.approx(bc.removed_fraction(r, coef["pi_L"]), rel=0.05)
    # A c that is 2 % larger at 90 MeV than the pooled value biases the pooled-c contrast by about k_bias.
    base, _ = pooled({t: (k_h * SIGNAL[t][0], k_l * SIGNAL[t][0]) for t in SIGNAL})
    eps = 0.02
    base["bigSpot"] = base["bigSpot"] + np.array([0.0, eps * k_l * SIGNAL["bigSpot"][0], 0.0, 0.0])
    coef = bc.coefficients(sum(base.values()), sum(EVENTS.values()))
    by = {t: bc.coefficients(base[t]) for t in ("zeroSpot", "bigSpot")}
    ratio = by["bigSpot"]["c"] / by["zeroSpot"]["c"]
    assert explicit(base, coef) - TRUE == pytest.approx(bc.k_bias(ratio, coef["pi_L"]), rel=0.25)
