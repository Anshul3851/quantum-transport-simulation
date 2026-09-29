"""Reproducible independent onsite disorder for finite 1D devices."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def disorder_potential(
    n_sites: int,
    disorder_strength: float,
    onsite_energy: float = 0.0,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Return ``epsilon_i = onsite_energy + w_i`` with iid uniform disorder.

    Each ``w_i`` is drawn from ``Uniform(-W/2, W/2)``. Supply exactly one
    explicit non-negative integer ``seed`` or NumPy ``Generator``; no global
    random state is used.
    """
    if isinstance(n_sites, (bool, np.bool_)) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be a positive integer")
    if n_sites <= 0:
        raise ValueError("n_sites must be greater than zero")

    strength_array = np.asarray(disorder_strength)
    if strength_array.ndim != 0 or strength_array.dtype.kind not in "iuf":
        raise TypeError("disorder_strength must be a real numeric scalar")
    strength = float(strength_array)
    if not np.isfinite(strength):
        raise ValueError("disorder_strength must be finite")
    if strength < 0.0:
        raise ValueError("disorder_strength must be non-negative")

    onsite_array = np.asarray(onsite_energy)
    if onsite_array.ndim != 0 or onsite_array.dtype.kind not in "iuf":
        raise TypeError("onsite_energy must be a real numeric scalar")
    background = float(onsite_array)
    if not np.isfinite(background):
        raise ValueError("onsite_energy must be finite")

    if (seed is None) == (rng is None):
        raise ValueError("supply exactly one of seed or rng")
    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
            raise TypeError("seed must be a non-negative integer")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        generator = np.random.default_rng(int(seed))
    else:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        generator = rng

    disorder = generator.uniform(-strength / 2.0, strength / 2.0, size=int(n_sites))
    return background + disorder
