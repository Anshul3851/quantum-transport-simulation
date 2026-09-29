"""Summary statistics for disorder-ensemble transmission samples."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def transmission_statistics(
    transmissions: ArrayLike,
    log_floor: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Summarize a (realization, condition) transmission array.

    The standard deviation uses the sample convention (``ddof=1``). Values
    below ``log_floor`` are floored only when taking logarithms; their counts
    are returned so this numerical protection is distinguishable from a
    physical zero transmission.
    """
    values = np.asarray(transmissions)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("transmissions must have shape (n_realizations >= 2, n_conditions >= 1)")
    if values.dtype.kind not in "iuf":
        raise TypeError("transmissions must contain real numeric values")
    values = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("transmissions must be finite")
    if np.any(values < 0.0):
        raise ValueError("transmissions must be non-negative")
    if np.any(values > 1.0 + 1e-10):
        raise ValueError("transmissions exceed the single-channel upper bound")

    floor_array = np.asarray(log_floor)
    if floor_array.ndim != 0 or floor_array.dtype.kind not in "iuf":
        raise TypeError("log_floor must be a real numeric scalar")
    floor = float(floor_array)
    if not np.isfinite(floor) or not 0.0 < floor < 1.0:
        raise ValueError("log_floor must be finite and between zero and one")

    standard_deviation = np.std(values, axis=0, ddof=1)
    mean_log = np.mean(np.log(np.maximum(values, floor)), axis=0)
    return {
        "mean": np.mean(values, axis=0),
        "median": np.median(values, axis=0),
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / np.sqrt(values.shape[0]),
        "mean_log": mean_log,
        "typical": np.exp(mean_log),
        "log_floor_count": np.count_nonzero(values < floor, axis=0),
    }
