import numpy as np
import pytest

from quantum_transport.broadening import left_broadening_matrix, right_broadening_matrix
from quantum_transport.density import local_density_of_states, total_density_of_states
from quantum_transport.greens import retarded_device_green_function
from quantum_transport.hamiltonian import tight_binding_hamiltonian
from quantum_transport.leads import lead_self_energy
from quantum_transport.potentials import (
    clean_potential,
    finite_well_potential,
    single_barrier_potential,
    symmetric_double_barrier_potential,
)
from quantum_transport.transmission import landauer_transmission


def transmission_for_profile(onsite, energy=0.4, hopping=1.0, lead_onsite=0.0, contact=1.0):
    hamiltonian = tight_binding_hamiltonian(len(onsite), onsite, hopping)
    sigma_left = lead_self_energy(energy, lead_onsite, hopping, contact)
    sigma_right = lead_self_energy(energy, lead_onsite, hopping, contact)
    green = retarded_device_green_function(energy, hamiltonian, sigma_left, sigma_right)
    return landauer_transmission(
        green,
        left_broadening_matrix(len(onsite), sigma_left),
        right_broadening_matrix(len(onsite), sigma_right),
    )


def test_profile_dimensions_and_clean_uniform_value():
    profile = clean_potential(12, onsite_energy=-0.3)

    assert profile.shape == (12,)
    np.testing.assert_allclose(profile, np.full(12, -0.3))


def test_single_barrier_changes_only_requested_site():
    profile = single_barrier_potential(9, onsite_energy=0.2, barrier_height=1.4, site_index=3)
    expected = np.full(9, 0.2)
    expected[3] += 1.4

    np.testing.assert_allclose(profile, expected)


def test_finite_well_changes_only_half_open_region():
    profile = finite_well_potential(10, onsite_energy=0.2, well_depth=0.7, start_index=3, stop_index=7)
    expected = np.full(10, 0.2)
    expected[3:7] -= 0.7

    np.testing.assert_allclose(profile, expected)


def test_double_barrier_is_mirror_symmetric_by_default():
    profile = symmetric_double_barrier_potential(60, 0.0, 1.4, 0.8, 4, 14)

    np.testing.assert_array_equal(profile, profile[::-1])
    assert np.count_nonzero(profile == 1.4) == 8
    assert np.count_nonzero(profile == -0.8) == 14


def test_invalid_profile_parameters_are_rejected():
    with pytest.raises(ValueError, match="within the device"):
        single_barrier_potential(6, 0.0, 1.0, 6)
    with pytest.raises(ValueError, match="well interval"):
        finite_well_potential(6, 0.0, 0.5, 4, 2)
    with pytest.raises(ValueError, match="do not fit"):
        symmetric_double_barrier_potential(10, 0.0, 1.0, 0.5, 3, 6)
    with pytest.raises(ValueError, match="centered symmetrically"):
        symmetric_double_barrier_potential(10, 0.0, 1.0, 0.5, 2, 3)
    with pytest.raises(ValueError, match="positive"):
        single_barrier_potential(6, 0.0, 0.0, 1)


def test_reversing_onsite_profile_preserves_transmission_for_identical_leads():
    onsite = np.array([0.0, 0.0, 0.8, -0.2, 0.0, 0.0, 0.4, 0.0])

    forward = transmission_for_profile(onsite, energy=0.37)
    reversed_profile = transmission_for_profile(onsite[::-1], energy=0.37)

    assert forward == pytest.approx(reversed_profile, abs=1e-12)


def test_ldos_values_are_real_finite_and_match_diagonal_formula():
    green = np.array([[0.2 - 0.5j, 0.1j], [-0.4, 0.3 - 0.25j]])

    ldos = local_density_of_states(green)

    assert ldos.dtype == np.float64
    np.testing.assert_allclose(ldos, -np.diag(green).imag / np.pi)
    assert np.isfinite(ldos).all()


def test_total_dos_from_ldos_sum_matches_trace_formula():
    green = np.array([[0.2 - 0.5j, 0.1j], [-0.4, 0.3 - 0.25j]])

    summed_ldos = local_density_of_states(green).sum()
    trace_dos = total_density_of_states(green)

    assert summed_ldos == pytest.approx(-np.trace(green).imag / np.pi, abs=1e-14)
    assert trace_dos == pytest.approx(summed_ldos, abs=1e-14)


def test_clean_matched_device_remains_unit_transmission():
    profile = clean_potential(10, 0.0)

    assert transmission_for_profile(profile, energy=0.63) == pytest.approx(1.0, abs=1e-10)


def test_transmission_is_nonnegative_and_no_greater_than_one():
    profile = symmetric_double_barrier_potential(60, 0.0, 1.4, 0.8, 4, 14)
    energies = np.linspace(-1.8, 1.8, 21)
    values = np.array([transmission_for_profile(profile, energy=float(e)) for e in energies])

    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0 + 1e-10)


def test_barrier_suppresses_transmission_at_representative_energy():
    clean = clean_potential(12, 0.0)
    barrier = single_barrier_potential(12, 0.0, 1.2, 5)

    assert transmission_for_profile(barrier, 0.4) < transmission_for_profile(clean, 0.4)


def test_identical_profile_calculations_are_deterministic():
    profile = finite_well_potential(12, 0.0, 0.6, 3, 9)

    assert transmission_for_profile(profile, 0.4) == transmission_for_profile(profile, 0.4)


def test_double_barrier_resonance_scan_is_deterministic():
    profile = symmetric_double_barrier_potential(60, 0.0, 1.4, 0.8, 4, 14)
    energies = [-0.4, 0.0, 0.4]
    first = [transmission_for_profile(profile, energy) for energy in energies]
    second = [transmission_for_profile(profile, energy) for energy in energies]

    np.testing.assert_array_equal(first, second)




def test_double_barrier_resonance_has_enhanced_central_well_ldos():
    energy = -0.73095
    profile = symmetric_double_barrier_potential(60, 0.0, 1.5, 0.8, 3, 14)
    hamiltonian = tight_binding_hamiltonian(60, profile, 1.0)
    sigma = lead_self_energy(energy, 0.0, 1.0, 1.0)
    green = retarded_device_green_function(energy, hamiltonian, sigma, sigma)
    transmission = landauer_transmission(
        green,
        left_broadening_matrix(60, sigma),
        right_broadening_matrix(60, sigma),
    )
    ldos = local_density_of_states(green)
    central_fraction = ldos[23:37].sum() / ldos.sum()

    assert transmission > 0.95
    assert central_fraction > 0.75

def test_density_functions_reject_invalid_green_matrices():
    with pytest.raises(ValueError, match="square"):
        local_density_of_states(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        total_density_of_states(np.array([[np.inf + 0j]]))

