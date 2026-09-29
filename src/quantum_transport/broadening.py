"""Lead broadening values and endpoint broadening matrices."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def _finite_self_energy(self_energy: complex | float) -> complex:
    """Validate and convert one finite numeric self-energy scalar."""
    value = np.asarray(self_energy)
    if value.ndim != 0 or value.dtype.kind not in "iufc":
        raise TypeError("self_energy must be a complex-compatible numeric scalar")
    if not np.isfinite(value):
        raise ValueError("self_energy must be finite")
    return complex(value)


def _positive_device_size(n_sites: int) -> int:
    """Validate and convert a positive integer device size."""
    if isinstance(n_sites, (bool, np.bool_)) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be a positive integer")
    if n_sites <= 0:
        raise ValueError("n_sites must be greater than zero")
    return int(n_sites)


def scalar_broadening(self_energy: complex | float) -> float:
    """Return scalar broadening ``gamma = i (Sigma^r - Sigma^{r*})``.

    For a scalar retarded self-energy this expression is real and equals
    ``-2 Im(Sigma^r)``; its sign follows directly from the supplied self-energy.
    """
    sigma = _finite_self_energy(self_energy)
    gamma = 1j * (sigma - sigma.conjugate())
    return float(gamma.real)


def left_broadening_matrix(
    n_sites: int,
    self_energy: complex | float,
) -> NDArray[np.float64]:
    """Place the scalar lead broadening only on the first device site."""
    size = _positive_device_size(n_sites)
    gamma = scalar_broadening(self_energy)
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[0, 0] = gamma
    return matrix


def right_broadening_matrix(
    n_sites: int,
    self_energy: complex | float,
) -> NDArray[np.float64]:
    """Place the scalar lead broadening only on the last device site."""
    size = _positive_device_size(n_sites)
    gamma = scalar_broadening(self_energy)
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[-1, -1] = gamma
    return matrix
