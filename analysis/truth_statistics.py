"""Portable replay of six per-file integer counters; NumPy is the only dependency."""

from __future__ import annotations
import numpy as np

TAGS = ("zeroSpot", "defaultSpot", "bigSpot")
OBS = ("R_n", "R_p", "D", "n_over_p")


def values(c):
    c = np.asarray(c)
    if not np.isfinite(c).all() or (c[..., [1, 2, 3, 4, 5]] <= 0).any():
        raise ValueError("Nonfinite or empty observable denominator/numerator")
    rn = c[..., 2] / c[..., 3]
    rp = c[..., 4] / c[..., 5]
    return np.stack((rn, rp, rn / rp, c[..., 0] / c[..., 1]), axis=-1)


def poisson_sd(c, m):
    n, p, h, low, hp, lp = c
    rn, rp = h / low, hp / lp
    gn = np.array([0, 0, 1 / low, -h / low**2, 0, 0])
    gp = np.array([0, 0, 0, 0, 1 / lp, -hp / lp**2])
    gradients = np.array(
        [gn, gp, gn / rp - rn * gp / rp**2, [1 / p, -n / p**2, 0, 0, 0, 0]]
    )
    return np.sqrt(np.maximum(0, np.einsum("oi,ij,oj->o", gradients, m, gradients)))


def replay(data, replicates=400, seed=20261025):
    if replicates != 400 or seed != 20261025:
        raise ValueError("Bootstrap recipe differs from the fixed campaign")
    if set(r["tag"] for r in data["rows"]) != set(TAGS):
        raise ValueError("Incomplete potential cohort")
    rng = np.random.default_rng(seed)
    samples = {}
    draws = {}
    centers = {}
    event_sd = {}
    for tag in TAGS:
        rows = [r for r in data["rows"] if r["tag"] == tag]
        if len({r["member"] for r in rows}) != len(rows):
            raise ValueError("Duplicate production file")
        c = np.array([r["counts"] for r in rows])
        if (
            c.shape != (len(rows), 6)
            or not np.issubdtype(c.dtype, np.integer)
            or (c < 0).any()
        ):
            raise ValueError("Six nonnegative integer counters required")
        w = rng.multinomial(len(rows), np.full(len(rows), 1 / len(rows)), replicates)
        draws[tag] = values(w @ c)
        centers[tag] = values(c.sum(axis=0))
        result = {
            "files": len(rows),
            "events": sum(r["events"] for r in rows),
            "counts": dict(zip(data["columns"], c.sum(axis=0).tolist())),
            "observables": dict(zip(OBS, centers[tag].tolist())),
            "bootstrap_sd": dict(zip(OBS, draws[tag].std(axis=0, ddof=1).tolist())),
        }
        if all("second_moments" in r for r in rows):
            m = np.array([r["second_moments"] for r in rows]).sum(axis=0)
            if m.shape != (6, 6) or not np.array_equal(m, m.T):
                raise ValueError("Invalid second moments")
            event_sd[tag] = poisson_sd(c.sum(axis=0), m)
            result["event_poisson_sd"] = dict(zip(OBS, event_sd[tag].tolist()))
        samples[tag] = result
    result = {"samples": samples, "observable_order": list(OBS)}
    for label, tag in [("endpoint", "bigSpot"), ("midpoint", "defaultSpot")]:
        point = centers[tag] / centers["zeroSpot"] - 1
        sd = (draws[tag] / draws["zeroSpot"] - 1).std(axis=0, ddof=1)
        result[
            label + "_fraction_big_over_zero_minus_one"
            if label == "endpoint"
            else "midpoint_fraction_middle_over_zero_minus_one"
        ] = point.tolist()
        result[label + "_bootstrap_sd"] = sd.tolist()
        if event_sd:
            psd = (point + 1) * np.sqrt(
                (event_sd[tag] / centers[tag]) ** 2
                + (event_sd["zeroSpot"] / centers["zeroSpot"]) ** 2
            )
            result[label + "_event_poisson_sd"] = psd.tolist()
            result[label + "_quoted_sd"] = np.maximum(sd, psd).tolist()
            result[label + "_file_over_event_sd"] = (sd / psd).tolist()
    return result
