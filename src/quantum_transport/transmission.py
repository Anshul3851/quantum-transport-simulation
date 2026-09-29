"""Landauer transmission for a finite device coupled to two leads."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _square_complex_matrix(value: ArrayLike, name: str) -> NDArray[np.complex128]:
    """Validate a finite, non-empty square matrix and convert to complex128."""
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if matrix.dtype.kind not in "iufc":
        raise TypeError(f"{name} must contain numeric values")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} values must be finite")
    return matrix.astype(np.complex128, copy=False)


def landauer_transmission(
    retarded_green_function: ArrayLike,
    left_broadening: ArrayLike,
    right_broadening: ArrayLike,
) -> float:
    """Calculate ``T = Tr[Gamma_L G^r Gamma_R (G^r)^dagger]``.

    Small imaginary or negative values within ``1e-10 * max(1, |Re(T)|)``
    are treated as floating-point roundoff. Larger violations raise an error;
    transmission is not clipped to an upper bound or normalized.
    """
    green = _square_complex_matrix(retarded_green_function, "retarded_green_function")
    gamma_left = _square_complex_matrix(left_broadening, "left_broadening")
    gamma_right = _square_complex_matrix(right_broadening, "right_broadening")
    if gamma_left.shape != green.shape or gamma_right.shape != green.shape:
        raise ValueError("Green function and broadening matrices must have matching shapes")

    for name, gamma in (("left_broadening", gamma_left), ("right_broadening", gamma_right)):
        scale = max(1.0, float(np.linalg.norm(gamma, ord="fro")))
        if not np.allclose(gamma, gamma.conj().T, rtol=0.0, atol=1e-12 * scale):
            raise ValueError(f"{name} must be Hermitian")

    value = np.trace(gamma_left @ green @ gamma_right @ green.conj().T)
    if not np.isfinite(value):
        raise ValueError("transmission calculation produced a non-finite value")
    tolerance = 1e-10 * max(1.0, abs(float(value.real)))
    if abs(float(value.imag)) > tolerance:
        raise ValueError("transmission has a non-negligible imaginary component")
    if value.real < -tolerance:
        raise ValueError("transmission is significantly negative; check broadening matrices")
    if value.real < 0.0:
        return 0.0
    return float(value.real)
