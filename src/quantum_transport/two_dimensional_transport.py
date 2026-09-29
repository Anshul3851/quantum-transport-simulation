"""Dense Green-function and Landauer transport for finite-width 2D devices."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
try:
    from scipy.constants import elementary_charge, Planck

    PHYSICAL_CONSTANTS_SOURCE = "scipy.constants"
except ImportError:  # Exact SI defining constants, used only without SciPy.
    elementary_charge = 1.602176634e-19  # C
    Planck = 6.62607015e-34  # J s
    PHYSICAL_CONSTANTS_SOURCE = "exact SI defining constants (fallback)"

from .transmission import landauer_transmission


def _finite_real_energy(energy: float) -> float:
    value = np.asarray(energy)
    if value.ndim != 0 or value.dtype.kind not in "iuf":
        raise TypeError("energy must be a real numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("energy must be finite")
    return result


def _finite_square_matrix(value: ArrayLike, name: str) -> NDArray[np.complex128]:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if matrix.dtype.kind not in "iufc":
        raise TypeError(f"{name} must contain numeric values")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix.astype(np.complex128, copy=False)


def retarded_2d_device_green_function(
    energy: float,
    device_hamiltonian: ArrayLike,
    left_self_energy: ArrayLike,
    right_self_energy: ArrayLike,
) -> NDArray[np.complex128]:
    """Solve ``(E I - H - Sigma_L - Sigma_R) G^r = I`` for the device."""
    energy_value = _finite_real_energy(energy)
    hamiltonian = _finite_square_matrix(device_hamiltonian, "device_hamiltonian")
    sigma_left = _finite_square_matrix(left_self_energy, "left_self_energy")
    sigma_right = _finite_square_matrix(right_self_energy, "right_self_energy")
    if sigma_left.shape != hamiltonian.shape or sigma_right.shape != hamiltonian.shape:
        raise ValueError("device Hamiltonian and self-energy matrices must have matching shapes")
    effective = energy_value * np.eye(hamiltonian.shape[0]) - hamiltonian - sigma_left - sigma_right
    if not np.all(np.isfinite(effective)):
        raise ValueError("effective device matrix must be finite")
    try:
        return np.linalg.solve(effective, np.eye(hamiltonian.shape[0], dtype=np.complex128))
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError("effective 2D device matrix is singular at this energy") from error


def advanced_2d_device_green_function(
    energy: float,
    device_hamiltonian: ArrayLike,
    left_self_energy: ArrayLike,
    right_self_energy: ArrayLike,
) -> NDArray[np.complex128]:
    """Return the advanced device Green function ``(G^r)^dagger``."""
    return retarded_2d_device_green_function(
        energy, device_hamiltonian, left_self_energy, right_self_energy
    ).conj().T


def broadening_matrix(self_energy: ArrayLike) -> NDArray[np.complex128]:
    """Calculate ``Gamma = i (Sigma^r - Sigma^{r dagger})`` without clipping."""
    sigma = _finite_square_matrix(self_energy, "self_energy")
    return 1j * (sigma - sigma.conj().T)


def multichannel_transmission(
    retarded_green_function: ArrayLike,
    left_broadening: ArrayLike,
    right_broadening: ArrayLike,
) -> float:
    """Evaluate the full multi-channel Landauer trace for 2D device matrices."""
    return landauer_transmission(retarded_green_function, left_broadening, right_broadening)


def conductance_quantum() -> float:
    """Return ``2 e^2 / h`` in siemens.

SciPy constants are used when installed; otherwise the exact SI defining
values of the elementary charge and Planck constant are used.
"""
    return float(2.0 * elementary_charge**2 / Planck)


def physical_conductance(transmission: ArrayLike) -> float | NDArray[np.float64]:
    """Convert dimensionless Landauer transmission to conductance in siemens."""
    values = np.asarray(transmission)
    if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError("transmission must contain finite real values")
    if np.any(values < 0.0):
        raise ValueError("transmission must be non-negative")
    result = values.astype(np.float64, copy=False) * conductance_quantum()
    if values.ndim == 0:
        return float(result.item())
    return result
