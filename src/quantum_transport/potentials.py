"""Simple onsite-potential profiles for finite one-dimensional devices."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def _device_size(n_sites: int) -> int:
    if isinstance(n_sites, (bool, np.bool_)) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be a positive integer")
    if n_sites <= 0:
        raise ValueError("n_sites must be greater than zero")
    return int(n_sites)


def _finite_real_scalar(value: float, name: str, *, strictly_positive: bool = False) -> float:
    scalar = np.asarray(value)
    if scalar.ndim != 0 or scalar.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real numeric scalar")
    result = float(scalar)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def clean_potential(n_sites: int, onsite_energy: float = 0.0) -> NDArray[np.float64]:
    """Return a uniform onsite profile for a clean N-site device."""
    size = _device_size(n_sites)
    epsilon_0 = _finite_real_scalar(onsite_energy, "onsite_energy")
    return np.full(size, epsilon_0, dtype=np.float64)


def single_barrier_potential(
    n_sites: int,
    onsite_energy: float,
    barrier_height: float,
    site_index: int,
) -> NDArray[np.float64]:
    """Add a positive onsite barrier at one zero-based device site."""
    profile = clean_potential(n_sites, onsite_energy)
    height = _finite_real_scalar(barrier_height, "barrier_height", strictly_positive=True)
    if isinstance(site_index, (bool, np.bool_)) or not isinstance(site_index, Integral):
        raise TypeError("site_index must be an integer")
    if not 0 <= site_index < profile.size:
        raise ValueError("site_index must be within the device")
    profile[int(site_index)] += height
    return profile


def finite_well_potential(
    n_sites: int,
    onsite_energy: float,
    well_depth: float,
    start_index: int,
    stop_index: int,
) -> NDArray[np.float64]:
    """Lower onsite energies by ``well_depth`` on the half-open site interval."""
    profile = clean_potential(n_sites, onsite_energy)
    depth = _finite_real_scalar(well_depth, "well_depth", strictly_positive=True)
    if isinstance(start_index, (bool, np.bool_)) or not isinstance(start_index, Integral):
        raise TypeError("start_index must be an integer")
    if isinstance(stop_index, (bool, np.bool_)) or not isinstance(stop_index, Integral):
        raise TypeError("stop_index must be an integer")
    start = int(start_index)
    stop = int(stop_index)
    if start < 0 or stop > profile.size or start >= stop:
        raise ValueError("well interval must satisfy 0 <= start_index < stop_index <= n_sites")
    profile[start:stop] -= depth
    return profile


def symmetric_double_barrier_potential(
    n_sites: int,
    onsite_energy: float,
    barrier_height: float,
    well_depth: float,
    barrier_width: int,
    well_width: int,
) -> NDArray[np.float64]:
    """Return a centered, mirror-symmetric barrier/well/barrier profile.

    The two barriers have equal widths and heights, and the central well has
    the requested width and depth. Any remaining sites retain ``onsite_energy``.
    """
    size = _device_size(n_sites)
    epsilon_0 = _finite_real_scalar(onsite_energy, "onsite_energy")
    height = _finite_real_scalar(barrier_height, "barrier_height", strictly_positive=True)
    depth = _finite_real_scalar(well_depth, "well_depth", strictly_positive=True)
    for value, name in ((barrier_width, "barrier_width"), (well_width, "well_width")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    barrier_width = int(barrier_width)
    well_width = int(well_width)
    structure_width = 2 * barrier_width + well_width
    margin = size - structure_width
    if margin < 0:
        raise ValueError("barriers and well do not fit inside the device")
    if margin % 2:
        raise ValueError("profile cannot be centered symmetrically on this site lattice")

    start = margin // 2
    left_barrier = slice(start, start + barrier_width)
    well = slice(start + barrier_width, start + barrier_width + well_width)
    right_barrier = slice(start + barrier_width + well_width, start + structure_width)
    profile = np.full(size, epsilon_0, dtype=np.float64)
    profile[left_barrier] += height
    profile[well] -= depth
    profile[right_barrier] += height
    return profile
