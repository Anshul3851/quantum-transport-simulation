import numpy as np
import pytest

from quantum_transport.broadening import (
    left_broadening_matrix,
    right_broadening_matrix,
    scalar_broadening,
)
from quantum_transport.leads import lead_self_energy


def test_scalar_broadening_uses_retarded_self_energy_formula():
    sigma = 0.3 - 0.4j

    assert scalar_broadening(sigma) == pytest.approx((1j * (sigma - sigma.conjugate())).real)
    assert scalar_broadening(sigma) == pytest.approx(0.8)


def test_negative_imaginary_self_energy_gives_positive_broadening():
    assert scalar_broadening(0.2 - 0.75j) == pytest.approx(1.5)


def test_real_self_energy_gives_zero_broadening():
    assert scalar_broadening(1.25 + 0j) == 0.0


def test_left_broadening_matrix_acts_only_on_first_site():
    gamma = scalar_broadening(0.2 - 0.3j)

    matrix = left_broadening_matrix(4, 0.2 - 0.3j)

    np.testing.assert_allclose(matrix, np.diag([gamma, 0.0, 0.0, 0.0]))


def test_right_broadening_matrix_acts_only_on_last_site():
    gamma = scalar_broadening(0.2 - 0.3j)

    matrix = right_broadening_matrix(4, 0.2 - 0.3j)

    np.testing.assert_allclose(matrix, np.diag([0.0, 0.0, 0.0, gamma]))


@pytest.mark.parametrize("n_sites", [1, 2, 4, 7])
def test_broadening_matrices_have_device_shape(n_sites):
    assert left_broadening_matrix(n_sites, -0.2j).shape == (n_sites, n_sites)
    assert right_broadening_matrix(n_sites, -0.2j).shape == (n_sites, n_sites)


def test_broadening_matrices_are_hermitian_and_real():
    left = left_broadening_matrix(4, 0.15 - 0.3j)
    right = right_broadening_matrix(4, 0.15 - 0.3j)

    np.testing.assert_allclose(left, left.conj().T, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(right, right.conj().T, atol=0.0, rtol=0.0)
    assert np.isrealobj(left)
    assert np.isrealobj(right)


def test_part3_in_band_self_energy_has_nonnegative_broadening():
    sigma = lead_self_energy(energy=0.5, onsite_energy=0.0, hopping=1.0, coupling=0.7)

    assert sigma.imag < 0
    assert scalar_broadening(sigma) >= 0.0


def test_zero_coupling_produces_zero_self_energy_and_broadening():
    sigma = lead_self_energy(energy=0.5, onsite_energy=0.0, hopping=1.0, coupling=0.0)

    assert sigma == 0j
    assert scalar_broadening(sigma) == 0.0
    np.testing.assert_array_equal(left_broadening_matrix(3, sigma), np.zeros((3, 3)))
    np.testing.assert_array_equal(right_broadening_matrix(3, sigma), np.zeros((3, 3)))


def test_repeated_broadening_calculations_are_deterministic():
    assert scalar_broadening(0.2 - 0.35j) == scalar_broadening(0.2 - 0.35j)
    np.testing.assert_array_equal(
        left_broadening_matrix(4, 0.2 - 0.35j),
        left_broadening_matrix(4, 0.2 - 0.35j),
    )


def test_scalar_numeric_edge_cases():
    assert scalar_broadening(0.0) == 0.0
    assert scalar_broadening(np.complex128(-0.1j)) == pytest.approx(0.2)
    assert scalar_broadening(0.1j) == pytest.approx(-0.2)


@pytest.mark.parametrize("self_energy", [np.nan, complex(np.inf, 0.0), [0.1j], "-0.2j"])
def test_invalid_self_energy_is_rejected(self_energy):
    with pytest.raises((TypeError, ValueError), match="self_energy"):
        scalar_broadening(self_energy)


@pytest.mark.parametrize("n_sites", [0, -2])
def test_non_positive_device_size_is_rejected(n_sites):
    with pytest.raises(ValueError, match="greater than zero"):
        left_broadening_matrix(n_sites, -0.1j)


@pytest.mark.parametrize("n_sites", [2.5, True])
def test_non_integer_device_size_is_rejected(n_sites):
    with pytest.raises(TypeError, match="positive integer"):
        right_broadening_matrix(n_sites, -0.1j)


def test_scalar_broadening_matches_part3_inside_and_outside_band_values():
    inside_sigma = lead_self_energy(0.5, onsite_energy=0.0, hopping=1.0, coupling=0.7)
    outside_sigma = lead_self_energy(2.5, onsite_energy=0.0, hopping=1.0, coupling=0.7)

    assert scalar_broadening(inside_sigma) == pytest.approx(0.948880919820817, abs=1e-12)
    assert outside_sigma.imag == pytest.approx(0.0, abs=1e-15)
    assert scalar_broadening(outside_sigma) == pytest.approx(0.0, abs=1e-15)

