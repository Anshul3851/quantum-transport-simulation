"""Local and total density of states derived from a retarded Green matrix."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _green_matrix(value: ArrayLike) -> NDArray[np.complex128]:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("retarded_green_function must be a non-empty square matrix")
    if matrix.dtype.kind not in "iufc":
        raise TypeError("retarded_green_function must contain numeric values")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("retarded_green_function values must be finite")
    return matrix.astype(np.complex128, copy=False)


def local_density_of_states(retarded_green_function: ArrayLike) -> NDArray[np.float64]:
    """Return site-resolved ``rho_i(E) = -Im(G^r_ii(E))/pi``."""
    green = _green_matrix(retarded_green_function)
    return -np.diag(green).imag / np.pi


def total_density_of_states(retarded_green_function: ArrayLike) -> float:
    """Return total device DOS ``rho(E) = -Im(Tr(G^r(E)))/pi``."""
    green = _green_matrix(retarded_green_function)
    return float(-np.trace(green).imag / np.pi)
