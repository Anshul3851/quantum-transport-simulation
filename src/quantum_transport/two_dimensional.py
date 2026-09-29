"""Finite-width rectangular 2D tight-binding device Hamiltonians."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .hamiltonian import tight_binding_hamiltonian


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return int(value)


def _hopping_value(hopping: float) -> float:
    value = np.asarray(hopping)
    if value.ndim != 0 or value.dtype.kind not in "iuf":
        raise TypeError("hopping must be a real numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("hopping must be finite")
    if result <= 0.0:
        raise ValueError("hopping must be positive")
    return result


def _onsite_grid(onsite_energy: ArrayLike, shape: tuple[int, int]) -> NDArray[np.float64]:
    values = np.asarray(onsite_energy)
    if values.ndim == 0:
        values = np.full(shape, values.item())
    elif values.shape == (shape[0] * shape[1],):
        values = values.reshape(shape)
    elif values.shape != shape:
        raise ValueError(
            f"onsite_energy must be a scalar or have shape {shape} "
            f"(or flattened length {shape[0] * shape[1]})"
        )
    if values.dtype.kind not in "iufc":
        raise TypeError("onsite_energy must contain real numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError("onsite_energy values must be finite")
    if np.iscomplexobj(values) and np.any(values.imag != 0.0):
        raise ValueError("onsite_energy values must be real")
    return values.real.astype(np.float64, copy=False)


def transverse_slice_hamiltonian(
    width: int,
    onsite_energy: float = 0.0,
    hopping: float = 1.0,
) -> NDArray[np.float64]:
    """Build one open-boundary transverse slice of width ``width``.

    The onsite entries are ``onsite_energy`` and adjacent transverse sites
    have hopping ``-hopping``. There is no wraparound coupling.
    """
    width = _positive_integer(width, "width")
    return np.asarray(tight_binding_hamiltonian(width, onsite_energy, hopping), dtype=np.float64)


def rectangular_device_hamiltonian(
    length: int,
    width: int,
    onsite_energy: ArrayLike = 0.0,
    hopping: float = 1.0,
) -> NDArray[np.float64]:
    """Construct an open-boundary rectangular 2D tight-binding Hamiltonian.

    Sites are ordered by longitudinal slice then transverse coordinate:
    ``index(x, y) = x * width + y``. Nearest neighbours along either axis
    have matrix element ``-hopping``; neither axis is periodic. The onsite
    input may be scalar, a ``(length, width)`` array, or a flattened array.
    """
    length = _positive_integer(length, "length")
    width = _positive_integer(width, "width")
    t = _hopping_value(hopping)
    onsite = _onsite_grid(onsite_energy, (length, width))
    n_sites = length * width
    hamiltonian = np.zeros((n_sites, n_sites), dtype=np.float64)
    diagonal = np.arange(n_sites)
    hamiltonian[diagonal, diagonal] = onsite.ravel(order="C")

    for x in range(length):
        for y in range(width):
            site = x * width + y
            if x + 1 < length:
                neighbour = (x + 1) * width + y
                hamiltonian[site, neighbour] = -t
                hamiltonian[neighbour, site] = -t
            if y + 1 < width:
                neighbour = x * width + (y + 1)
                hamiltonian[site, neighbour] = -t
                hamiltonian[neighbour, site] = -t
    return hamiltonian
