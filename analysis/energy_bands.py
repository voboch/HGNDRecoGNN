"""Explicit finite energy bands; no implicit soft denominator or open tail."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EnergyBands:
    minimum: float
    low_stop: float
    high_start: float
    maximum: float

    def __post_init__(self):
        edges = np.asarray([self.minimum, self.low_stop, self.high_start, self.maximum])
        if not np.isfinite(edges).all() or not (0 <= edges[0] < edges[1] <= edges[2] < edges[3]):
            raise ValueError("Require finite 0 <= minimum < low_stop <= high_start < maximum")

    def masks(self, energy):
        energy = np.asarray(energy, dtype=float)
        if not np.isfinite(energy).all():
            raise ValueError("Nonfinite energy cannot be silently removed")
        return ((energy >= self.minimum) & (energy < self.low_stop),
                (energy >= self.high_start) & (energy < self.maximum))

    def counts(self, energy, unit, all_units=None):
        """Count both half-open bands, retaining events with zero candidates."""
        energy, unit = np.asarray(energy), np.asarray(unit)
        if energy.shape != unit.shape or energy.ndim != 1:
            raise ValueError("One event key is required per candidate")
        keys = np.unique(unit) if all_units is None else np.asarray(all_units)
        if len(np.unique(keys)) != len(keys):
            raise ValueError("Duplicate event inventory keys")
        order = np.argsort(keys)
        positions = np.searchsorted(keys[order], unit)
        if len(unit) and (not len(keys) or np.any(positions >= len(keys))):
            raise ValueError("Candidate event outside inventory")
        if len(unit) and not np.array_equal(keys[order][positions], unit):
            raise ValueError("Candidate event outside inventory")
        inv = order[positions]
        lo, hi = self.masks(energy)
        return (np.bincount(inv, weights=lo, minlength=len(keys)),
                np.bincount(inv, weights=hi, minlength=len(keys)))

    def histogram_masks(self, edges):
        """Reject cuts through a histogram bin instead of assuming its shape."""
        edges = np.asarray(edges, dtype=float)
        if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
            raise ValueError("Invalid histogram edges")
        for cut in (self.minimum, self.low_stop, self.high_start, self.maximum):
            # Numerical tolerance concerns serialized edge arithmetic, not an
            # energy acceptance margin. Whole bins are required.
            if not np.any(np.isclose(edges, cut, rtol=0, atol=1e-12)):
                raise ValueError("Energy boundary cuts a histogram bin")
        return ((edges[:-1] >= self.minimum) & (edges[1:] <= self.low_stop),
                (edges[:-1] >= self.high_start) & (edges[1:] <= self.maximum))
