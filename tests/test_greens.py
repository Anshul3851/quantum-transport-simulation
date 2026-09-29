import numpy as np
import pytest

from quantum_transport.greens import (
    advanced_device_green_function,
    retarded_device_green_function,
)
from quantum_transport.hamiltonian import tight_binding_hamiltonian
from quantum_transport.leads import lead_self_energy


def sample_device():
    return tight_binding_hamiltonian(4, onsite_energy=0.0, hopping=1.0)


def test_retarded_green_function_shape_and_complex_dtype():
    green = retarded_device_green_function(0.5, sample_device(), 0.1j, 0.1j)

    assert green.shape == (4, 4)
    assert np.iscomplexobj(green)


def test_retarded_green_function_solves_defining_linear_system():
    energy = 0.5
    hamiltonian = sample_device()
    sigma_left = 0.12 - 0.47j
    sigma_right = 0.12 - 0.47j
    green = retarded_device_green_function(energy, hamiltonian, sigma_left, sigma_right)
    effective = energy * np.eye(4, dtype=complex) - hamiltonian
    effective[0, 0] -= sigma_left
    effective[-1, -1] -= sigma_right

    np.testing.assert_allclose(effective @ green, np.eye(4), atol=1e-12, rtol=0.0)


def test_advanced_green_function_is_retarded_conjugate_transpose():
    hamiltonian = sample_device()
    retarded = retarded_device_green_function(0.5, hamiltonian, 0.1 - 0.2j, -0.05 - 0.3j)
    advanced = advanced_device_green_function(0.5, hamiltonian, 0.1 - 0.2j, -0.05 - 0.3j)

    np.testing.assert_allclose(advanced, retarded.conj().T, atol=1e-13, rtol=0.0)


def test_advanced_function_transposes_while_conjugating():
    hamiltonian = np.array([[0.2, 0.4 + 0.1j], [-0.3j, -0.1]], dtype=np.complex128)
    retarded = retarded_device_green_function(0.8, hamiltonian, 0.07j, -0.11j)
    advanced = advanced_device_green_function(0.8, hamiltonian, 0.07j, -0.11j)

    assert advanced[0, 1] == pytest.approx(np.conj(retarded[1, 0]))
    assert advanced[0, 1] != pytest.approx(np.conj(retarded[0, 1]))


def test_left_self_energy_acts_only_on_first_site():
    energy = 0.7
    hamiltonian = sample_device()
    sigma_left = -0.1j
    green = retarded_device_green_function(energy, hamiltonian, sigma_left, 0.0)
    effective = energy * np.eye(4, dtype=complex) - hamiltonian
    effective[0, 0] -= sigma_left

    np.testing.assert_allclose(effective @ green, np.eye(4), atol=1e-12, rtol=0.0)


def test_right_self_energy_acts_only_on_last_site():
    energy = 0.7
    hamiltonian = sample_device()
    sigma_right = -0.1j
    green = retarded_device_green_function(energy, hamiltonian, 0.0, sigma_right)
    effective = energy * np.eye(4, dtype=complex) - hamiltonian
    effective[-1, -1] -= sigma_right

    np.testing.assert_allclose(effective @ green, np.eye(4), atol=1e-12, rtol=0.0)


def test_zero_self_energies_give_isolated_device_green_function():
    energy = 2.5
    hamiltonian = sample_device()
    green = retarded_device_green_function(energy, hamiltonian, 0.0, 0.0)
    isolated_matrix = energy * np.eye(4, dtype=complex) - hamiltonian

    np.testing.assert_allclose(isolated_matrix @ green, np.eye(4), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        green,
        retarded_device_green_function(energy, hamiltonian, 0j, 0j),
        atol=0.0,
        rtol=0.0,
    )


def test_repeated_green_function_calculations_are_identical():
    hamiltonian = sample_device()
    first = retarded_device_green_function(0.5, hamiltonian, -0.2j, -0.3j)
    second = retarded_device_green_function(0.5, hamiltonian, -0.2j, -0.3j)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "hamiltonian",
    [np.zeros((2, 3)), np.zeros((2, 2, 2)), np.zeros((0, 0))],
)
def test_invalid_device_dimensions_are_rejected(hamiltonian):
    with pytest.raises(ValueError, match="non-empty square"):
        retarded_device_green_function(1.0, hamiltonian, 0j, 0j)


@pytest.mark.parametrize(
    "left_sigma, right_sigma",
    [([0.1j], 0j), (0j, np.array([0.2j])), (np.nan, 0j)],
)
def test_incompatible_or_nonfinite_self_energies_are_rejected(left_sigma, right_sigma):
    with pytest.raises((TypeError, ValueError), match="self_energy"):
        retarded_device_green_function(1.0, sample_device(), left_sigma, right_sigma)

