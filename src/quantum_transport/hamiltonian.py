"""Finite one-dimensional tight-binding device Hamiltonian."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


Hamiltonian: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]


def tight_binding_hamiltonian(
    n_sites: int,
    onsite_energy: ArrayLike = 0.0,
    hopping: float = 1.0,
) -> Hamiltonian:
    """Construct the finite 1D tight-binding Hamiltonian.

    The convention is H[i, i] = epsilon_i and H[i, i+1] = H[i+1, i] = -t
    for t > 0. Open boundary conditions are used, with no coupling from the
    last site back to the first.
    """
    if isinstance(n_sites, (bool, np.bool_)) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be a positive integer")
    if n_sites <= 0:
        raise ValueError("n_sites must be greater than zero")
    n_sites = int(n_sites)

    hopping_array = np.asarray(hopping)
    if hopping_array.ndim != 0 or hopping_array.dtype.kind not in "iuf":
        raise TypeError("hopping must be a real numeric scalar")
    hopping_value = float(hopping_array)
    if not np.isfinite(hopping_value):
        raise ValueError("hopping must be finite")
    if hopping_value <= 0:
        raise ValueError("hopping must be positive")

    onsite = np.asarray(onsite_energy)
    if onsite.ndim == 0:
        onsite = np.full(n_sites, onsite.item())
    elif onsite.ndim != 1:
        raise ValueError("onsite_energy must be a scalar or a one-dimensional array")
    elif onsite.size != n_sites:
        raise ValueError(
            f"onsite_energy must contain exactly {n_sites} values; got {onsite.size}"
        )

    if onsite.dtype.kind not in "iufc":
        raise TypeError("onsite_energy must contain real numeric values")
    if not np.all(np.isfinite(onsite)):
        raise ValueError("onsite_energy values must be finite")
    if np.iscomplexobj(onsite) and np.any(onsite.imag != 0):
        raise ValueError("onsite_energy values must be real for a Hermitian Hamiltonian")

    dtype = np.result_type(onsite.dtype, np.asarray(hopping_value).dtype)
    hamiltonian = np.zeros((n_sites, n_sites), dtype=dtype)
    indices = np.arange(n_sites)
    hamiltonian[indices, indices] = onsite
    if n_sites > 1:
        off_diagonal = np.arange(n_sites - 1)
        hamiltonian[off_diagonal, off_diagonal + 1] = -hopping_value
        hamiltonian[off_diagonal + 1, off_diagonal] = -hopping_value
    return hamiltonian
