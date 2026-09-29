import numpy as np
import pytest

from quantum_transport.broadening import left_broadening_matrix, right_broadening_matrix
from quantum_transport.greens import retarded_device_green_function
from quantum_transport.hamiltonian import tight_binding_hamiltonian
from quantum_transport.leads import lead_self_energy
from quantum_transport.transmission import landauer_transmission


def transport_value(energy, onsite_energies, n_sites=10, t=1.0, epsilon_0=0.0, t_c=1.0):
    hamiltonian = tight_binding_hamiltonian(n_sites, onsite_energies, t)
    sigma_left = lead_self_energy(energy, epsilon_0, t, t_c)
    sigma_right = lead_self_energy(energy, epsilon_0, t, t_c)
    green = retarded_device_green_function(energy, hamiltonian, sigma_left, sigma_right)
    gamma_left = left_broadening_matrix(n_sites, sigma_left)
    gamma_right = right_broadening_matrix(n_sites, sigma_right)
    return landauer_transmission(green, gamma_left, gamma_right)


def test_transmission_is_a_real_scalar():
    green = np.array([[0.2 + 0.3j, 0.1j], [-0.4, 0.5 - 0.1j]])
    gamma_left = np.diag([1.0, 0.0])
    gamma_right = np.diag([0.0, 2.0])

    transmission = landauer_transmission(green, gamma_left, gamma_right)

    assert isinstance(transmission, float)
    assert np.isfinite(transmission)


def test_transmission_is_nonnegative_for_positive_endpoint_broadenings():
    green = np.array([[0.2 + 0.3j, 0.1j], [-0.4, 0.5 - 0.1j]])
    gamma_left = np.diag([1.0, 0.0])
    gamma_right = np.diag([0.0, 2.0])

    assert landauer_transmission(green, gamma_left, gamma_right) >= 0.0


def test_hermitian_conjugation_is_used_in_the_caroli_formula():
    green = np.array([[0.1 + 0.2j, 0.7 - 0.4j], [0.3 + 0.8j, -0.2j]])
    gamma_left = np.diag([2.0, 0.0])
    gamma_right = np.diag([0.0, 3.0])

    expected = 6.0 * abs(green[0, 1]) ** 2

    assert landauer_transmission(green, gamma_left, gamma_right) == pytest.approx(expected)


@pytest.mark.parametrize("energy", [-1.7, -1.0, -0.35, 0.4, 1.2, 1.75])
def test_clean_matched_finite_device_transmits_nearly_one_inside_band(energy):
    transmission = transport_value(energy, onsite_energies=0.0)

    # The energy points stay away from the band edges, where Gamma vanishes
    # with a square root and finite-precision checks become less robust.
    assert transmission == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("energy", [-1.0, -0.3, 0.4, 1.2, 1.75])
def test_single_onsite_barrier_reduces_transmission_at_representative_energies(energy):
    clean = transport_value(energy, onsite_energies=0.0)
    onsite = np.zeros(10)
    onsite[4] = 1.5
    with_barrier = transport_value(energy, onsite_energies=onsite)

    assert with_barrier < clean


def test_transmission_is_deterministic_for_identical_inputs():
    first = transport_value(0.4, onsite_energies=0.0)
    second = transport_value(0.4, onsite_energies=0.0)

    assert first == second


def test_zero_broadening_gives_zero_transmission():
    green = np.eye(3, dtype=complex)
    gamma_left = np.zeros((3, 3))
    gamma_right = np.zeros((3, 3))

    assert landauer_transmission(green, gamma_left, gamma_right) == 0.0


@pytest.mark.parametrize(
    "green, gamma_left, gamma_right",
    [
        (np.ones((2, 3)), np.eye(2), np.eye(2)),
        (np.eye(2), np.eye(3), np.eye(2)),
        (np.eye(2), np.eye(2), np.eye(3)),
        (np.eye(2), np.array([[0.0, 1.0], [0.0, 0.0]]), np.eye(2)),
        (np.eye(2), np.eye(2), np.array([[1.0, 2.0], [0.0, 1.0]])),
    ],
)
def test_invalid_shapes_or_nonhermitian_broadenings_are_rejected(green, gamma_left, gamma_right):
    with pytest.raises(ValueError):
        landauer_transmission(green, gamma_left, gamma_right)


def test_nonfinite_inputs_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        landauer_transmission(np.array([[np.inf]]), np.ones((1, 1)), np.ones((1, 1)))


def test_significantly_negative_result_is_not_hidden():
    green = np.eye(2, dtype=complex)
    gamma_left = np.diag([1.0, 0.0])
    gamma_right = np.diag([-1.0, 0.0])

    with pytest.raises(ValueError, match="negative"):
        landauer_transmission(green, gamma_left, gamma_right)


def test_energy_outside_lead_band_has_zero_transmission():
    assert transport_value(2.2, onsite_energies=0.0) == pytest.approx(0.0, abs=1e-14)
    assert transport_value(-2.2, onsite_energies=0.0) == pytest.approx(0.0, abs=1e-14)
