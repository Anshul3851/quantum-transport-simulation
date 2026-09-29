"""Retarded surface Green functions and self-energies for 1D leads."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


ComplexOutput = complex | NDArray[np.complex128]


def _real_finite_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert numeric real input to float64 after validating its values."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} values must be finite")
    return array.astype(np.float64, copy=False)


def _real_finite_scalar(
    value: float,
    name: str,
    *,
    strictly_positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Validate one real finite scalar parameter."""
    scalar = np.asarray(value)
    if scalar.ndim != 0 or scalar.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real numeric scalar")
    result = float(scalar)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def surface_green_function(
    energy: ArrayLike,
    onsite_energy: float = 0.0,
    hopping: float = 1.0,
) -> ComplexOutput:
    """Return the retarded surface Green function of a semi-infinite 1D lead.

    The lead has onsite energy ``onsite_energy`` and hopping ``-hopping`` with
    ``hopping > 0``. Its band is [onsite_energy - 2*hopping,
    onsite_energy + 2*hopping]. Inside the band the retarded branch has a
    negative imaginary part; outside it is real and chosen to decay as 1/E.
    At an exact band edge the square-root term is zero. Energies very near an
    edge need care because the square root changes rapidly there.
    """
    energies = _real_finite_array(energy, "energy")
    epsilon_0 = _real_finite_scalar(onsite_energy, "onsite_energy")
    t = _real_finite_scalar(hopping, "hopping", strictly_positive=True)

    offset = energies - epsilon_0
    scaled_offset = (offset / t) * 0.5
    magnitude = np.abs(scaled_offset)
    inside_band = magnitude < 1.0
    result = np.empty(energies.shape, dtype=np.complex128)

    # The dimensionless radicands reduce cancellation near |E-epsilon_0|=2t.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inside_root = np.sqrt(np.maximum((1.0 - magnitude) * (1.0 + magnitude), 0.0))
        result[inside_band] = (
            scaled_offset[inside_band] - 1j * inside_root[inside_band]
        ) / t

        outside_magnitude = magnitude[~inside_band]
        # sqrt(q^2 - 1), evaluated in two forms to retain precision near q=1
        # and avoid squaring large q far outside the band.
        outside_root = np.empty_like(outside_magnitude)
        near_edge = outside_magnitude < 1.5
        outside_root[near_edge] = np.sqrt(
            np.maximum(
                (outside_magnitude[near_edge] - 1.0)
                * (outside_magnitude[near_edge] + 1.0),
                0.0,
            )
        )
        q_far = outside_magnitude[~near_edge]
        outside_root[~near_edge] = q_far * np.sqrt(
            np.maximum(1.0 - (1.0 / q_far) ** 2, 0.0)
        )
        outside_sign = np.sign(scaled_offset[~inside_band])
        result[~inside_band] = outside_sign / (
            t * (outside_magnitude + outside_root)
        )

    if energies.ndim == 0:
        return complex(result.item())
    return result


def lead_self_energy(
    energy: ArrayLike,
    onsite_energy: float = 0.0,
    hopping: float = 1.0,
    coupling: float = 1.0,
) -> ComplexOutput:
    """Return the retarded lead self-energy ``Sigma^r = coupling^2 * g_s^r``.

    ``g_s^r`` is the semi-infinite 1D lead surface Green function using the
    hopping ``-hopping`` convention. The contact hopping ``coupling`` is
    finite and non-negative; this sign convention follows from multiplying
    the surface Green function by the square of the device-lead coupling.
    """
    contact_hopping = _real_finite_scalar(coupling, "coupling", nonnegative=True)
    surface = surface_green_function(energy, onsite_energy, hopping)
    return contact_hopping**2 * surface


