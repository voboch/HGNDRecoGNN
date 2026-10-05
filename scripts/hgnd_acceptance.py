"""Geometric acceptance of the HGND front face, seen from the interaction point.

The earlier analyses selected nucleons with a polar-angle band, theta in
[8.9, 13.1) deg.  The detector is not a theta band.  Its front face is a
rectangle offset to negative x:

    z = 718.42 cm,  x in [-146.61, -106.67] cm,  y in [-77.34, +77.34] cm

which subtends theta from 8.45 deg at the near edge to 12.99 deg at the far
corner, and only 71.9 deg of azimuth -- 20 % of 2*pi -- centred on phi = 180 deg.
A theta band therefore counts particles at azimuths the detector does not cover,
and its theta range does not match the rectangle's.

Two ways to apply the real window are provided:

  in_acceptance()  exact, per particle: propagate a straight line from the
                   particle's vertex and test whether it crosses the rectangle.
                   Needs momentum and vertex, i.e. the primary CSV.

  theta_acceptance()  the azimuthal fraction of each theta shell that lies
                   inside the rectangle, A(theta).  Applied as a weight to a
                   theta-binned histogram it gives the same answer as the exact
                   test whenever the laboratory azimuthal distribution is flat,
                   which for minimum-bias events with randomly oriented reaction
                   planes it is.  This lets the full production be re-analysed
                   from histograms already on disk instead of re-reading 25 GB.

`scripts/check_acceptance.py` tests the second against the first.
"""
from __future__ import annotations
import numpy as np

Z_FRONT = 718.42
X_LO, X_HI = -146.61, -106.67
Y_LO, Y_HI = -77.34, 77.34


def in_acceptance(px, py, pz, vx=0.0, vy=0.0, vz=0.0, z_front=Z_FRONT):
    """Does a straight line from the vertex cross the HGND front face?"""
    px, py, pz = (np.asarray(v, float) for v in (px, py, pz))
    vx, vy, vz = (np.asarray(v, float) for v in np.broadcast_arrays(vx, vy, vz))
    dz = z_front - vz
    ok = (pz > 0) & (dz > 0)
    t = np.divide(dz, pz, out=np.zeros_like(px), where=ok)
    x = vx + px * t
    y = vy + py * t
    return ok & (x >= X_LO) & (x <= X_HI) & (y >= Y_LO) & (y <= Y_HI)


def theta_acceptance(theta_deg, n_phi=4096, z_front=Z_FRONT):
    """A(theta): azimuthal fraction of a theta shell inside the rectangle."""
    th = np.atleast_1d(np.asarray(theta_deg, float))
    r = z_front * np.tan(np.radians(th))
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    x = r[:, None] * np.cos(phi)[None, :]
    y = r[:, None] * np.sin(phi)[None, :]
    inside = (x >= X_LO) & (x <= X_HI) & (y >= Y_LO) & (y <= Y_HI)
    return inside.mean(axis=1)


def bin_acceptance(theta_edges, n_sub=64, **kw):
    """Average A(theta) within each histogram bin, weighted by solid angle.

    The solid-angle weight sin(theta) matters little across a 0.5 deg bin but
    costs nothing and keeps the weight unbiased for wider binnings.
    """
    edges = np.asarray(theta_edges, float)
    out = np.empty(len(edges) - 1)
    for k in range(len(edges) - 1):
        t = np.linspace(edges[k], edges[k + 1], n_sub)
        a = theta_acceptance(t, **kw)
        w = np.sin(np.radians(t))
        out[k] = np.trapezoid(a * w, t) / np.trapezoid(w, t)
    return out


def summary():
    th = np.linspace(0.0, 20.0, 20001)
    a = theta_acceptance(th)
    nz = th[a > 0]
    return {"z_front": Z_FRONT, "x": [X_LO, X_HI], "y": [Y_LO, Y_HI],
            "theta_min": float(nz.min()), "theta_max": float(nz.max()),
            "max_azimuthal_fraction": float(a.max()),
            "solid_angle_sr": float(np.trapezoid(
                a * 2 * np.pi * np.sin(np.radians(th)), np.radians(th)))}


if __name__ == "__main__":
    import json
    s = summary()
    print(json.dumps(s, indent=1))
    print(f"\n  full 2pi band 8.9-13.1 deg would be "
          f"{2*np.pi*(np.cos(np.radians(8.9))-np.cos(np.radians(13.1))):.5f} sr")
    print(f"  the real acceptance is {s['solid_angle_sr']:.5f} sr "
          f"({s['solid_angle_sr']/(2*np.pi*(np.cos(np.radians(8.9))-np.cos(np.radians(13.1))))*100:.1f} % of it)")

# ── Kinematic selection ──────────────────────────────────────────────────────
#
# `in_acceptance` propagates a straight line from the particle's vertex. That is
# the right test for a neutron, which is neutral and travels straight, and it is
# the wrong test for a proton, which the analysing magnet bends: a proton aimed
# at the front face need not arrive, and one aimed elsewhere may. Nothing in
# this file models that field.
#
# So the proton selection cannot be read as a proton acceptance. What it can be
# is a *kinematic region*, defined in the emission direction alone, applied
# identically to both species, so that a neutron numerator and a proton
# denominator are formed over the same phase space. That is what the isovector
# ratios need — they compare two species in one region — and it is what the
# function below provides. Any statement about protons actually reaching a
# detector needs the field map and the charged arm's own acceptance.


def kinematic_acceptance(px, py, pz, ekin=None, ekin_lo=None, ekin_hi=None):
    """The HGND front-face region as a cut on emission direction alone.

    Charge- and vertex-independent: a particle is selected when a ray from the
    origin along its momentum crosses the front-face rectangle. For neutrons
    from a target-sized vertex this agrees with `in_acceptance` to the level
    quantified by `vertex_sensitivity`; for protons it is a phase-space
    definition rather than a trajectory, for the reason given above.

    `ekin_lo` / `ekin_hi` optionally restrict the kinetic energy, so a single
    call defines the full kinematic selection a ratio is formed over.
    """
    sel = in_acceptance(px, py, pz, 0.0, 0.0, 0.0)
    if ekin is not None and (ekin_lo is not None or ekin_hi is not None):
        ek = np.asarray(ekin, float)
        if ekin_lo is not None:
            sel = sel & (ek >= ekin_lo)
        if ekin_hi is not None:
            sel = sel & (ek < ekin_hi)
    return sel


def acceptance_bounds(n=20001):
    """(theta_min, theta_max, phi_centre, phi_width) of the front face, degrees."""
    xs = np.linspace(X_LO, X_HI, 401)
    ys = np.linspace(Y_LO, Y_HI, 401)
    X, Y = np.meshgrid(xs, ys)
    r = np.hypot(X, Y)
    th = np.degrees(np.arctan2(r, Z_FRONT))
    ph = np.degrees(np.arctan2(Y, X)) % 360.0
    return (float(th.min()), float(th.max()),
            float(0.5 * (ph.min() + ph.max())), float(ph.max() - ph.min()))


def vertex_sensitivity(sigma_xy=0.5, sigma_z=2.0, n=400000, seed=0):
    """How often the direction-only cut and the vertex-aware test disagree.

    The kinematic selection ignores the vertex. This measures what that costs
    for a realistic target-sized interaction region, by throwing isotropic
    directions, giving them vertices, and comparing the two selections.
    """
    rng = np.random.default_rng(seed)
    cos_t = rng.uniform(0.90, 1.0, n)          # forward cone covering the face
    sin_t = np.sqrt(1.0 - cos_t ** 2)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    px, py, pz = sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t
    vx = rng.normal(0.0, sigma_xy, n)
    vy = rng.normal(0.0, sigma_xy, n)
    vz = rng.normal(0.0, sigma_z, n)

    kin = kinematic_acceptance(px, py, pz)
    geo = in_acceptance(px, py, pz, vx, vy, vz)
    both = int((kin & geo).sum())
    return {"n_kinematic": int(kin.sum()), "n_geometric": int(geo.sum()),
            "n_both": both,
            "disagree_frac": float((kin ^ geo).sum() / max(kin.sum(), 1)),
            "net_bias": float(kin.sum() / max(geo.sum(), 1) - 1.0)}
