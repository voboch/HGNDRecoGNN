"""Parfenov slide-22/23 PID, with one frozen calibration for all SMASH arms.

Calibration may use matched MC on a declared development sample. Application
uses only reconstructed track quantities; truth labels are audit columns.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MASS_GEV = {2212: 0.9382720813, 211: 0.13957039}
# Particle masses in GeV/c^2; these constants set the PID hypotheses, not cuts.
KEY = ["source_id", "entry", "track_index"]


def bethe_bloch(p, mass, pars):
    """Blum parametrisation printed on Parfenov slide 22 (dE/dx in a.u.)."""
    p = np.asarray(p, dtype=float)
    a, b, c, d, e = np.asarray(pars, dtype=float)
    bg = p / mass
    beta = bg / np.sqrt(1 + bg * bg)
    return a / beta**d * (b - beta**d - np.log(c + bg**(-e)))


def calibration_hash(cal):
    return hashlib.sha256(json.dumps(cal, sort_keys=True).encode()).hexdigest()


def validate_tracks(d):
    required = KEY + ["p", "dedx", "charge_sign", "from_vertex",
                      "m2_400", "m2_700", "tof400_valid", "tof700_valid"]
    missing = set(required) - set(d.columns)
    if missing:
        raise ValueError("Missing reconstructed columns: " + ", ".join(sorted(missing)))
    if d.duplicated(KEY).any():
        raise ValueError("Duplicate (source_id, entry, track_index); TOF arms must not double-count tracks")


def validate_calibration(cal):
    if cal.get("schema_version") != 1 or cal.get("pid_method") != "parfenov_combined":
        raise ValueError("Expected schema-1 combined dE/dx + m2 calibration")
    if str(cal.get("tof")) not in ("400", "700"):
        raise ValueError("Calibration must identify one TOF arm")
    for species in ("proton", "pion"):
        v = cal["species"][species]
        for k in ("p", "sigma_dedx_relative", "sigma_m2"):
            a = np.asarray(v[k], float)
            if len(a) < 3 or not np.isfinite(a).all():
                raise ValueError("Calibration needs at least three finite momentum slices")
        if np.any(np.diff(v["p"]) <= 0):
            raise ValueError("Momentum slices must be strictly ordered")
        if np.any(np.asarray(v["sigma_dedx_relative"]) <= 0) or np.any(np.asarray(v["sigma_m2"]) <= 0):
            raise ValueError("Calibration widths must be positive")
        if len(v["bethe_parameters"]) != 5:
            raise ValueError("Bethe parametrisation needs five fitted coefficients")


def pid_scores(d, cal):
    """Compute distances within the measured calibration domain; no extrapolation."""
    validate_tracks(d)
    validate_calibration(cal)
    tof = str(cal["tof"])
    p = d.p.to_numpy(float)
    dedx = d.dedx.to_numpy(float)
    m2 = d["m2_" + tof].to_numpy(float)
    out = d.copy()
    valid = (np.isfinite(p) & (p > 0) & np.isfinite(dedx) & (dedx > 0)
             & np.isfinite(m2) & d["tof" + tof + "_valid"].eq(1).to_numpy())
    for species in ("proton", "pion"):
        v = cal["species"][species]
        xp = np.asarray(v["p"], float)
        domain = (p >= xp.min()) & (p <= xp.max())
        valid &= domain
        mu = bethe_bloch(p, v["mass_GeV"], v["bethe_parameters"])
        sigx = np.interp(p, xp, v["sigma_dedx_relative"])
        sigy = np.interp(p, xp, v["sigma_m2"])
        # Slide 22 divides the dE/dx residual by mu * relative width;
        # the mass-squared residual is centred at the physical hypothesis.
        with np.errstate(invalid="ignore", divide="ignore"):
            x = (dedx - mu) / (mu * sigx)
            y = (m2 - v["mass_GeV"]**2) / sigy
        out["x_" + species] = x
        out["y_" + species] = y
        out["r_" + species] = np.hypot(x, y)
        valid &= np.isfinite(x) & np.isfinite(y) & (mu > 0)
    out["pid_valid"] = valid
    return out


def select_protons(d, cal, proton_radius=2.0, pion_veto=3.0,
                   require_vertex=True, max_chi2_ndf=None, max_dca_cm=None):
    out = pid_scores(d, cal)
    quality = out.charge_sign.eq(1)
    if require_vertex:
        # DST guide §5.2: reconstructed primary association via the inner
        # track used in MpdVertex fit. This is not an MC mother-ID selection.
        quality &= out.from_vertex.eq(1)
    if max_chi2_ndf is not None:
        quality &= out.chi2_ndf.ge(0) & out.chi2_ndf.le(max_chi2_ndf)
    if max_dca_cm is not None:
        quality &= out.dca_cm.ge(0) & out.dca_cm.le(max_dca_cm)
    # Strict inequalities reproduce slide 23. Radius 2 trades efficiency
    # against proton purity; pion distance 3 suppresses band overlap.
    out["selected_proton"] = (quality & out.pid_valid
                              & out.r_proton.lt(proton_radius)
                              & out.r_pion.gt(pion_veto))
    out["ekin_proton_hypothesis"] = np.hypot(out.p, MASS_GEV[2212]) - MASS_GEV[2212]
    return out


def fit_calibration(d, tof, p_edges, min_slice_tracks, sample_description):
    """MC-development calibration; Gaussian MLE widths in explicit p slices.

    Slice centres and widths are saved with fit diagnostics. Empty/poor slices
    are omitted; interpolation never extends beyond the measured centres.
    """
    from scipy.optimize import least_squares
    from scipy.stats import norm

    validate_tracks(d)
    if "mc_pdg" not in d:
        raise ValueError("MC development calibration needs mc_pdg labels")
    edges = np.asarray(p_edges, float)
    if len(edges) < 4 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("Supply at least three ordered momentum bins")
    if min_slice_tracks < 10:
        raise ValueError("Gaussian slice calibration requires at least ten entries")
    base = (d.charge_sign.eq(1) & d.from_vertex.eq(1)
            & d["tof" + str(tof) + "_valid"].eq(1)
            & np.isfinite(d.dedx) & d.dedx.gt(0) & np.isfinite(d["m2_" + str(tof)])
            & d.p.ge(edges[0]) & d.p.lt(edges[-1]))
    cal = {"schema_version": 1, "pid_method": "parfenov_combined", "tof": str(tof),
           "calibration_provenance": "MC-labelled development sample; not a truth-free data calibration",
           "sample_description": sample_description, "p_edges": edges.tolist(),
           "min_slice_tracks": min_slice_tracks,
           "width_estimator": "Gaussian MLE per momentum slice; diagnostic means and correlations retained",
           "species": {}}
    for pdg, name in ((2212, "proton"), (211, "pion")):
        part = d[base & d.mc_pdg.eq(pdg)].copy()
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = part[part.p.ge(lo) & part.p.lt(hi)]
            if len(s) >= min_slice_tracks:
                bins.append((lo, hi, s))
        if len(bins) < 5:
            raise ValueError(f"{name}: fewer than five populated slices for the five-parameter Bethe fit")
        positions = [int(np.searchsorted(edges, lo)) for lo, _, _ in bins]
        if any(b-a != 1 for a,b in zip(positions[:-1], positions[1:])):
            raise ValueError(f"{name}: empty internal momentum slice; merge bins instead of interpolating across a gap")
        pc = np.array([s.p.mean() for _, _, s in bins])
        mu = np.array([s.dedx.mean() for _, _, s in bins])
        se = np.array([max(s.dedx.std(ddof=1) / np.sqrt(len(s)), np.finfo(float).eps)
                       for _, _, s in bins])
        start = np.array([np.median(mu) / 2, 5., 1., 2., 2.])
        # Positive log parameters prevent undefined logarithms; scales are
        # fitted to this production, never copied from the slide's ADC axes.
        # Average the model over actual reconstructed momenta in each slice;
        # evaluating f(mean(p)) would bias a curved low-momentum Bethe band.
        momenta = [s.p.to_numpy() for _, _, s in bins]
        fit = least_squares(lambda z: (np.array([
                                bethe_bloch(ps, MASS_GEV[pdg], np.exp(z)).mean()
                                for ps in momenta])-mu)/se,
                            np.log(start), bounds=(-14, 14), max_nfev=10000)
        pars = np.exp(fit.x)
        if not fit.success or np.any(bethe_bloch(pc, MASS_GEV[pdg], pars) <= 0):
            raise ValueError(f"{name}: Bethe fit failed")
        rows = []
        for lo, hi, s in bins:
            expected = bethe_bloch(s.p.to_numpy(), MASS_GEV[pdg], pars)
            rel = (s.dedx.to_numpy() - expected) / expected
            mass2 = s["m2_" + str(tof)].to_numpy()
            dx_mu, dx_sigma = norm.fit(rel)
            m2_mu, m2_sigma = norm.fit(mass2)
            if dx_sigma <= 0 or m2_sigma <= 0:
                raise ValueError(f"{name}: zero width in [{lo}, {hi})")
            rows.append({"lo": float(lo), "hi": float(hi), "n": len(s),
                         "p": float(s.p.mean()), "dedx_mean": float(s.dedx.mean()),
                         "dedx_mean_se": float(s.dedx.std(ddof=1)/np.sqrt(len(s))),
                         "dedx_residual_mean": float(dx_mu),
                         "sigma_dedx_relative": float(dx_sigma),
                         "m2_mean": float(m2_mu), "sigma_m2": float(m2_sigma),
                         "dedx_m2_correlation": float(np.corrcoef(rel, mass2)[0, 1])})
        cal["species"][name] = {"pdg": pdg, "mass_GeV": MASS_GEV[pdg],
            "bethe_parameters": pars.tolist(), "fit_cost": float(fit.cost),
            "fit_dof": len(bins)-5, "p": [r["p"] for r in rows],
            "sigma_dedx_relative": [r["sigma_dedx_relative"] for r in rows],
            "sigma_m2": [r["sigma_m2"] for r in rows], "slices": rows}
    validate_calibration(cal)
    return cal


def read_tables(paths):
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise ValueError("No input tables")
    d = pd.concat(frames, ignore_index=True)
    validate_tracks(d)
    return d


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
