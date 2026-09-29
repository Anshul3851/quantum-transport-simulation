"""Retarded and advanced Green functions for finite devices coupled to leads."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_numeric_scalar(value: complex | float, name: str) -> complex:
    """Validate and convert a finite numeric scalar."""
    scalar = np.asarray(value)
    if scalar.ndim != 0 or scalar.dtype.kind not in "iufc":
        raise TypeError(f"{name} must be a numeric scalar")
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return complex(scalar)


def _device_matrix(value: ArrayLike) -> NDArray[np.complex128]:
    """Validate and convert a non-empty square device Hamiltonian."""
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("device_hamiltonian must be a non-empty square matrix")
    if matrix.dtype.kind not in "iufc":
        raise TypeError("device_hamiltonian must contain numeric values")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("device_hamiltonian values must be finite")
    return matrix.astype(np.complex128, copy=False)


def retarded_device_green_function(
    energy: complex | float,
    device_hamiltonian: ArrayLike,
    left_self_energy: complex,
    right_self_energy: complex,
) -> NDArray[np.complex128]:
    """Return the retarded Green matrix for a finite device attached at its ends.

    The convention is ``G^r = [E I - H_device - Sigma_L - Sigma_R]^-1``.
    The scalar left self-energy acts on the first site and the right one on
    the last. No additional imaginary energy broadening is introduced.
    """
    energy_value = _finite_numeric_scalar(energy, "energy")
    hamiltonian = _device_matrix(device_hamiltonian)
    sigma_left = _finite_numeric_scalar(left_self_energy, "left_self_energy")
    sigma_right = _finite_numeric_scalar(right_self_energy, "right_self_energy")

    n_sites = hamiltonian.shape[0]
    effective_matrix = energy_value * np.eye(n_sites, dtype=np.complex128) - hamiltonian
    effective_matrix[0, 0] -= sigma_left
    effective_matrix[-1, -1] -= sigma_right
    if not np.all(np.isfinite(effective_matrix)):
        raise ValueError("effective device matrix contains non-finite values")

    identity = np.eye(n_sites, dtype=np.complex128)
    try:
        return np.linalg.solve(effective_matrix, identity)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "effective device matrix is singular at this energy"
        ) from error


def advanced_device_green_function(
    energy: complex | float,
    device_hamiltonian: ArrayLike,
    left_self_energy: complex,
    right_self_energy: complex,
) -> NDArray[np.complex128]:
    """Return the advanced Green matrix as the conjugate transpose of ``G^r``."""
    retarded = retarded_device_green_function(
        energy,
        device_hamiltonian,
        left_self_energy,
        right_self_energy,
    )
    return retarded.conj().T
