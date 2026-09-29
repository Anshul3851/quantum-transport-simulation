"""Smooth gate potential profile for a finite 2D quantum point contact."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return int(value)


def _finite_scalar(value: float, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
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


def quantum_point_contact_potential(
    length: int,
    width: int,
    gate_strength: float,
    *,
    longitudinal_sigma: float = 6.0,
    transverse_strength: float = 1.5,
    onsite_background: float = 0.0,
) -> NDArray[np.float64]:
    r"""Return a smooth, x- and y-symmetric quantum point contact profile.

    For site coordinates ``x=0..Lx-1`` and ``y=0..Ly-1`` the profile is

    ``epsilon(x,y) = epsilon_background + Vg exp[-(x-xc)^2/(2 sigma_x^2)]
    [1 + alpha ((y-yc)/y_scale)^2]``

    where ``xc=(Lx-1)/2``, ``yc=(Ly-1)/2`` and
    ``y_scale=max((Ly-1)/2, 1/2)``. Thus the gate tends smoothly to zero
    away from the constriction, is largest near its longitudinal centre, and
    raises sites farther from the central transverse channel more strongly.
    ``gate_strength`` and ``onsite_background`` use the tight-binding energy
    units; ``transverse_strength`` is dimensionless.
    """
    length = _positive_integer(length, "length")
    width = _positive_integer(width, "width")
    gate = _finite_scalar(gate_strength, "gate_strength", nonnegative=True)
    sigma_x = _finite_scalar(longitudinal_sigma, "longitudinal_sigma", positive=True)
    alpha = _finite_scalar(transverse_strength, "transverse_strength", nonnegative=True)
    background = _finite_scalar(onsite_background, "onsite_background")

    x_center = (length - 1) / 2.0
    y_center = (width - 1) / 2.0
    y_scale = max(y_center, 0.5)
    x = np.arange(length, dtype=np.float64)
    y = np.arange(width, dtype=np.float64)
    longitudinal_envelope = np.exp(-0.5 * ((x - x_center) / sigma_x) ** 2)
    transverse_profile = 1.0 + alpha * ((y - y_center) / y_scale) ** 2
    return background + gate * longitudinal_envelope[:, None] * transverse_profile[None, :]
