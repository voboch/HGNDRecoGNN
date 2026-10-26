"""Exact reach-window counters and independent straight-ray smoke check."""

from __future__ import annotations

import numpy as np
import pandas as pd
from HGNDRecoGNN.scripts import hgnd_acceptance as geometry

# User-adopted lower ToF reach; upper/low boundary and high band are inherited.
FLOOR = 0.3
# Proton rest mass (GeV); y_beam/2 is the fixed-target CM rapidity. The old
# 0.9863 remains only in the replayed legacy mid_* counters, never this proxy.
BEAM_KINETIC, NUCLEON_MASS = 2.87, 0.9382720813
Y_CM = np.arccosh(1 + BEAM_KINETIC / NUCLEON_MASS) / 2
COLUMNS = (
    "acc_n_reach",
    "acc_p_reach",
    "acc_n_hi",
    "acc_n_lo_reach",
    "acc_p_hi",
    "acc_p_lo_reach",
)


def reach_counters(primary, floor, *, independent=False):
    if floor != FLOOR:
        raise ValueError("Only the adopted exact 0.3 GeV floor is supported")
    acc = geometry.in_acceptance(
        *(primary[c].to_numpy() for c in ("Px", "Py", "Pz", "vX", "vY", "vZ"))
    )
    if independent:
        # No projection helper/division: intersect each forward ray using the
        # equivalent cross-multiplied plane inequalities, including boundaries.
        z, px, py = (primary[c].to_numpy() for c in ("Pz", "Px", "Py"))
        distance = geometry.Z_FRONT - primary.vZ.to_numpy()
        xz = primary.vX.to_numpy() * z + distance * px
        yz = primary.vY.to_numpy() * z + distance * py
        other = (
            (z > 0)
            & (distance > 0)
            & (xz >= geometry.X_LO * z)
            & (xz <= geometry.X_HI * z)
            & (yz >= geometry.Y_LO * z)
            & (yz <= geometry.Y_HI * z)
        )
        if not np.array_equal(acc, other):
            raise ValueError("Independent ray/window smoke check failed")
    e, pdg = primary.Ekin.to_numpy(), primary.PDG.to_numpy()
    masks = {}
    for name, code in [("n", 2112), ("p", 2212)]:
        masks["acc_" + name + "_reach"] = acc & (pdg == code) & (e >= floor)
        masks["acc_" + name + "_lo_reach"] = (
            acc & (pdg == code) & (e >= floor) & (e < 1.0)
        )
    # MC-truth composition proxy, disjoint from every particle in the window.
    masks["ref_mult"] = (
        (~acc)
        & np.isin(pdg, [2112, 2212])
        & (np.abs(primary.Rapid.to_numpy() - Y_CM) < 0.5)
    )
    return (
        pd.DataFrame(masks)
        .astype(np.int64)
        .groupby(primary.Row.to_numpy(), sort=True)
        .sum()
        .rename_axis("Row")
    )
