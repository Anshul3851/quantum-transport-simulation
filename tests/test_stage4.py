import numpy as np
import pytest

from quantum_transport.modes import open_mode_count, transverse_modes
from quantum_transport.leads import surface_green_function
from quantum_transport.two_dimensional import (
    rectangular_device_hamiltonian,
    transverse_slice_hamiltonian,
)
from quantum_transport.two_dimensional_leads import (
    embed_contact_slice,
    two_dimensional_self_energy_matrices,
    two_dimensional_surface_green_function,
)
from quantum_transport.two_dimensional_transport import (
    broadening_matrix,
    conductance_quantum,
    multichannel_transmission,
    physical_conductance,
    retarded_2d_device_green_function,
)


def calculate_clean_transmission(length=12, width=4, energy=0.0):
    hopping = 1.0
    hamiltonian = rectangular_device_hamiltonian(length, width, 0.0, hopping)
    sigma_left, sigma_right = two_dimensional_self_energy_matrices(
        energy, length, width, 0.0, hopping, hopping
    )
    green = retarded_2d_device_green_function(
        energy, hamiltonian, sigma_left, sigma_right
    )
    transmission = multichannel_transmission(
        green, broadening_matrix(sigma_left), broadening_matrix(sigma_right)
    )
    return transmission


def test_rectangular_hamiltonian_has_expected_shape_and_is_hermitian():
    hamiltonian = rectangular_device_hamiltonian(3, 4, onsite_energy=0.2)

    assert hamiltonian.shape == (12, 12)
    np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T)


def test_longitudinal_nearest_neighbour_links_follow_slice_ordering():
    hamiltonian = rectangular_device_hamiltonian(3, 2, hopping=1.5)

    # index(x, y) = x * width + y
    assert hamiltonian[0, 2] == pytest.approx(-1.5)
    assert hamiltonian[1, 3] == pytest.approx(-1.5)
    assert hamiltonian[2, 4] == pytest.approx(-1.5)


def test_transverse_nearest_neighbour_links_are_present():
    hamiltonian = rectangular_device_hamiltonian(2, 3, hopping=0.8)

    assert hamiltonian[0, 1] == pytest.approx(-0.8)
    assert hamiltonian[1, 2] == pytest.approx(-0.8)
    assert hamiltonian[3, 4] == pytest.approx(-0.8)


def test_no_diagonal_or_periodic_transverse_links_are_added():
    hamiltonian = rectangular_device_hamiltonian(3, 4, hopping=1.0)

    # Diagonal in x and y, and wraparound between the first and last y sites.
    assert hamiltonian[0, 5] == 0.0
    assert hamiltonian[3, 0] == 0.0
    assert hamiltonian[4, 7] == 0.0


def test_transverse_slice_uses_open_chain_hamiltonian():
    actual = transverse_slice_hamiltonian(4, onsite_energy=0.3, hopping=1.2)
    expected = np.array(
        [[0.3, -1.2, 0.0, 0.0], [-1.2, 0.3, -1.2, 0.0],
         [0.0, -1.2, 0.3, -1.2], [0.0, 0.0, -1.2, 0.3]]
    )
    np.testing.assert_allclose(actual, expected)


def test_transverse_modes_match_analytic_hard_wall_spectrum():
    width, onsite, hopping = 5, -0.2, 0.9
    energies, vectors = transverse_modes(width, onsite, hopping)
    n = np.arange(1, width + 1)
    expected = onsite - 2.0 * hopping * np.cos(n * np.pi / (width + 1))

    np.testing.assert_allclose(energies, expected, atol=1e-13)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(width), atol=1e-13)


def test_open_mode_count_for_representative_energies():
    modes, _ = transverse_modes(4)

    assert open_mode_count(0.0, modes) == 4
    assert open_mode_count(3.0, modes) == 1
    np.testing.assert_array_equal(open_mode_count([-3.0, 0.0, 3.0, 4.0], modes), [1, 4, 1, 0])


def test_surface_green_function_has_transverse_matrix_shape_and_is_symmetric():
    surface = two_dimensional_surface_green_function(0.4, width=4)

    assert surface.shape == (4, 4)
    assert np.iscomplexobj(surface)
    np.testing.assert_allclose(surface, surface.T, atol=1e-13)


def test_embedded_self_energies_modify_only_the_contact_slice():
    length, width = 5, 3
    sigma_left, sigma_right = two_dimensional_self_energy_matrices(
        0.2, length, width, coupling=0.7
    )
    n_sites = length * width
    nonzero_left = np.argwhere(np.abs(sigma_left) > 1e-14)
    nonzero_right = np.argwhere(np.abs(sigma_right) > 1e-14)

    assert sigma_left.shape == sigma_right.shape == (n_sites, n_sites)
    assert np.all((nonzero_left < width).all(axis=1))
    assert np.all((nonzero_right >= (length - 1) * width).all(axis=1))
    assert np.any(np.abs(sigma_left[:width, :width]) > 0.0)
    assert np.any(np.abs(sigma_right[-width:, -width:]) > 0.0)


def test_contact_slice_embedding_checks_boundary_and_shape():
    block = np.array([[1.0, 0.2j], [-0.2j, 2.0]])
    embedded = embed_contact_slice(block, length=3, contact="right")

    assert embedded.shape == (6, 6)
    np.testing.assert_array_equal(embedded[-2:, -2:], block)
    assert np.count_nonzero(embedded[:-2, :]) == 0


def test_broadening_matrices_are_hermitian_and_positive_semidefinite():
    sigma_left, sigma_right = two_dimensional_self_energy_matrices(
        0.25, length=8, width=4
    )
    gamma_left = broadening_matrix(sigma_left)
    gamma_right = broadening_matrix(sigma_right)

    np.testing.assert_allclose(gamma_left, gamma_left.conj().T, atol=1e-13)
    np.testing.assert_allclose(gamma_right, gamma_right.conj().T, atol=1e-13)
    assert np.linalg.eigvalsh(gamma_left).min() >= -1e-12
    assert np.linalg.eigvalsh(gamma_right).min() >= -1e-12


def test_retarded_green_function_has_full_complex_device_shape():
    length, width, energy = 4, 3, 0.35
    hamiltonian = rectangular_device_hamiltonian(length, width)
    sigma_left, sigma_right = two_dimensional_self_energy_matrices(
        energy, length, width
    )
    green = retarded_2d_device_green_function(
        energy, hamiltonian, sigma_left, sigma_right
    )

    assert green.shape == (length * width, length * width)
    assert np.iscomplexobj(green)
    assert np.all(np.isfinite(green))


def test_clean_multichannel_transmission_is_real_and_nonnegative():
    transmission = calculate_clean_transmission(energy=0.5)

    assert isinstance(transmission, float)
    assert np.isfinite(transmission)
    assert transmission >= 0.0


@pytest.mark.parametrize("energy", [-3.2, -2.8, -1.0, 0.0, 1.0, 2.8, 3.2])
def test_clean_matched_strip_transmission_matches_open_mode_count(energy):
    modes, _ = transverse_modes(4)
    count = open_mode_count(energy, modes)

    assert calculate_clean_transmission(length=12, width=4, energy=energy) == pytest.approx(
        count, abs=2e-10, rel=2e-10
    )


def test_repeated_clean_calculations_are_deterministic():
    first = calculate_clean_transmission(energy=0.75)
    second = calculate_clean_transmission(energy=0.75)

    assert first == second


def test_physical_conductance_uses_two_e_squared_over_h():
    assert conductance_quantum() == pytest.approx(7.748091729863649e-5, rel=1e-14)
    assert physical_conductance(2.0) == pytest.approx(2.0 * conductance_quantum())
    np.testing.assert_allclose(physical_conductance([0.0, 1.0]), [0.0, conductance_quantum()])


@pytest.mark.parametrize(
    "length,width,onsite,hopping",
    [(0, 2, 0.0, 1.0), (2, 0, 0.0, 1.0), (2, 2, 0.0, 0.0), (2, 2, np.nan, 1.0)],
)
def test_invalid_device_parameters_are_rejected(length, width, onsite, hopping):
    with pytest.raises((TypeError, ValueError)):
        rectangular_device_hamiltonian(length, width, onsite, hopping)


def test_invalid_mode_and_contact_inputs_are_rejected():
    with pytest.raises(ValueError, match="hopping must be positive"):
        open_mode_count(0.0, [-0.5, 0.5], hopping=0.0)
    with pytest.raises(ValueError, match="contact must"):
        embed_contact_slice(np.eye(2), 3, "top")
    with pytest.raises(ValueError, match="coupling must be non-negative"):
        two_dimensional_self_energy_matrices(0.0, 3, 2, coupling=-0.1)


def test_incompatible_green_function_dimensions_are_rejected():
    with pytest.raises(ValueError, match="matching shapes"):
        retarded_2d_device_green_function(0.2, np.eye(3), np.eye(2), np.eye(3))


def test_surface_green_matrix_matches_mode_by_mode_1d_functions():
    energy, width, onsite, hopping = 0.27, 4, -0.1, 0.9
    modes, vectors = transverse_modes(width, onsite, hopping)
    surface = two_dimensional_surface_green_function(energy, width, onsite, hopping)
    projected = vectors.T @ surface @ vectors
    expected = np.diag([surface_green_function(energy, mode, hopping) for mode in modes])

    np.testing.assert_allclose(projected, expected, atol=2e-13)


def test_two_dimensional_green_function_solves_the_retarded_matrix_equation():
    length, width, energy = 5, 3, 0.31
    hamiltonian = rectangular_device_hamiltonian(length, width)
    sigma_left, sigma_right = two_dimensional_self_energy_matrices(
        energy, length, width
    )
    green = retarded_2d_device_green_function(
        energy, hamiltonian, sigma_left, sigma_right
    )
    residual = (energy * np.eye(length * width) - hamiltonian - sigma_left - sigma_right) @ green

    np.testing.assert_allclose(residual, np.eye(length * width), atol=2e-13)
