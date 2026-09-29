"""Transverse eigenmodes and propagating-channel counts for 2D strips."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .two_dimensional import transverse_slice_hamiltonian


def transverse_modes(
    width: int,
    onsite_energy: float = 0.0,
    hopping: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return transverse eigenenergies and normalized site-basis modes.

    Eigenvectors are columns, ordered by ascending energy. For open hard-wall
    boundaries the reference spectrum is ``epsilon_0 - 2t cos(n pi/(W+1))``.
    """
    transverse = transverse_slice_hamiltonian(width, onsite_energy, hopping)
    return np.linalg.eigh(transverse)


def open_mode_count(
    energy: ArrayLike,
    mode_energies: ArrayLike,
    hopping: float = 1.0,
) -> int | NDArray[np.int64]:
    """Count modes with ``|E - epsilon_n| <= 2*hopping``.

    Scalar energy input returns an integer; array input preserves its shape.
    This analytic channel count is independent of the finite device length.
    """
    energies = np.asarray(energy)
    modes = np.asarray(mode_energies)
    t_array = np.asarray(hopping)
    if energies.dtype.kind not in "iuf" or not np.all(np.isfinite(energies)):
        raise ValueError("energy must contain finite real values")
    if modes.ndim != 1 or modes.size == 0 or modes.dtype.kind not in "iuf":
        raise ValueError("mode_energies must be a non-empty one-dimensional real array")
    if not np.all(np.isfinite(modes)):
        raise ValueError("mode_energies must be finite")
    if t_array.ndim != 0 or t_array.dtype.kind not in "iuf":
        raise TypeError("hopping must be a real numeric scalar")
    t = float(t_array)
    if not np.isfinite(t):
        raise ValueError("hopping must be finite")
    if t <= 0.0:
        raise ValueError("hopping must be positive")

    counts = np.count_nonzero(np.abs(energies[..., None] - modes) <= 2.0 * t, axis=-1)
    if energies.ndim == 0:
        return int(counts.item())
    return counts.astype(np.int64, copy=False)
