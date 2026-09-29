import numpy as np
import pytest

from quantum_transport import tight_binding_hamiltonian


@pytest.mark.parametrize("n_sites", [1, 2, 4, 7])
def test_hamiltonian_has_correct_dimensions(n_sites):
    hamiltonian = tight_binding_hamiltonian(n_sites)

    assert hamiltonian.shape == (n_sites, n_sites)


def test_uniform_onsite_energy_fills_diagonal():
    hamiltonian = tight_binding_hamiltonian(4, onsite_energy=2.5)

    np.testing.assert_allclose(np.diag(hamiltonian), [2.5] * 4)


def test_site_dependent_onsite_energies_fill_diagonal():
    onsite = np.array([-1.0, 0.25, 2.0, 3.5])

    hamiltonian = tight_binding_hamiltonian(4, onsite_energy=onsite)

    np.testing.assert_allclose(np.diag(hamiltonian), onsite)


def test_nearest_neighbour_hopping_is_minus_t_on_both_sides():
    hopping = 0.75

    hamiltonian = tight_binding_hamiltonian(4, hopping=hopping)

    np.testing.assert_allclose(np.diag(hamiltonian, k=1), [-hopping] * 3)
    np.testing.assert_allclose(np.diag(hamiltonian, k=-1), [-hopping] * 3)


def test_no_next_nearest_neighbour_terms():
    hamiltonian = tight_binding_hamiltonian(5)

    np.testing.assert_allclose(np.diag(hamiltonian, k=2), np.zeros(3))
    np.testing.assert_allclose(np.diag(hamiltonian, k=-2), np.zeros(3))


def test_open_boundaries_have_no_first_to_last_coupling():
    hamiltonian = tight_binding_hamiltonian(4)

    assert hamiltonian[0, -1] == 0
    assert hamiltonian[-1, 0] == 0


def test_hamiltonian_is_hermitian():
    hamiltonian = tight_binding_hamiltonian(4, onsite_energy=[0.5, 1.0, -2.0, 0.0])

    np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T)


@pytest.mark.parametrize("n_sites", [0, -1])
def test_non_positive_site_count_is_rejected(n_sites):
    with pytest.raises(ValueError, match="greater than zero"):
        tight_binding_hamiltonian(n_sites)


@pytest.mark.parametrize("n_sites", [1.5, True])
def test_non_integer_site_count_is_rejected(n_sites):
    with pytest.raises(TypeError, match="positive integer"):
        tight_binding_hamiltonian(n_sites)


def test_onsite_array_with_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="exactly 4 values"):
        tight_binding_hamiltonian(4, onsite_energy=[0.0, 1.0, 2.0])


@pytest.mark.parametrize("hopping", [0.0, -1.0])
def test_non_positive_hopping_is_rejected(hopping):
    with pytest.raises(ValueError, match="must be positive"):
        tight_binding_hamiltonian(3, hopping=hopping)
