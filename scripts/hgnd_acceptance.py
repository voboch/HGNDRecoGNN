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
