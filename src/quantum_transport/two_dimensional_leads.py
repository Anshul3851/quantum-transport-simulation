"""Semi-infinite finite-width lead surface Green functions and self-energies."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .leads import surface_green_function
from .modes import transverse_modes


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return int(value)


def _real_scalar(value: float, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real numeric scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def two_dimensional_surface_green_function(
    energy: float,
    width: int,
    onsite_energy: float = 0.0,
    hopping: float = 1.0,
) -> NDArray[np.complex128]:
    """Return the exact semi-infinite strip surface Green matrix.

    The lead is infinite along x and has open width ``width`` along y. The
    transverse Hamiltonian is diagonalized as ``U diag(epsilon_n) U^dagger``;
    each mode uses the existing semi-infinite 1D retarded surface function
    with onsite ``epsilon_n`` and longitudinal hopping ``hopping``. The
    resulting ``U diag(g_n^r) U^dagger`` is in the transverse site basis.
    """
    energy_value = _real_scalar(energy, "energy")
    width = _positive_integer(width, "width")
    epsilon = _real_scalar(onsite_energy, "onsite_energy")
    t = _real_scalar(hopping, "hopping", positive=True)
    mode_energies, mode_vectors = transverse_modes(width, epsilon, t)
    mode_surface = np.diag(
        [surface_green_function(energy_value, mode_energy, t) for mode_energy in mode_energies]
    )
    return np.asarray(mode_vectors @ mode_surface @ mode_vectors.conj().T, dtype=np.complex128)


def embed_contact_slice(
    slice_operator: ArrayLike,
    length: int,
    contact: str,
) -> NDArray[np.complex128]:
    """Embed a transverse-slice operator on only the left or right endpoint."""
    length = _positive_integer(length, "length")
    operator = np.asarray(slice_operator)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1] or operator.shape[0] == 0:
        raise ValueError("slice_operator must be a non-empty square matrix")
    if operator.dtype.kind not in "iufc" or not np.all(np.isfinite(operator)):
        raise ValueError("slice_operator must contain finite numeric values")
    if contact not in ("left", "right"):
        raise ValueError("contact must be 'left' or 'right'")
    width = operator.shape[0]
    embedded = np.zeros((length * width, length * width), dtype=np.complex128)
    start = 0 if contact == "left" else (length - 1) * width
    embedded[start:start + width, start:start + width] = operator
    return embedded


def two_dimensional_self_energy_matrices(
    energy: float,
    length: int,
    width: int,
    onsite_energy: float = 0.0,
    hopping: float = 1.0,
    coupling: float = 1.0,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return retarded left/right self-energy matrices for matched strip leads.

    The lead-device hopping is ``-coupling`` times the transverse identity,
    so the boundary-slice self-energy is ``coupling**2 * g_surface`` and is
    embedded only on the first or last longitudinal slice.
    """
    length = _positive_integer(length, "length")
    width = _positive_integer(width, "width")
    contact = _real_scalar(coupling, "coupling", nonnegative=True)
    surface = two_dimensional_surface_green_function(
        energy, width, onsite_energy=onsite_energy, hopping=hopping
    )
    sigma_slice = contact**2 * surface
    return (
        embed_contact_slice(sigma_slice, length, "left"),
        embed_contact_slice(sigma_slice, length, "right"),
    )
