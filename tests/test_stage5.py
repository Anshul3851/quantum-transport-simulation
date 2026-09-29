import numpy as np
import pytest

from quantum_transport.density import local_density_of_states
from quantum_transport.modes import open_mode_count, transverse_modes
from quantum_transport.qpc import quantum_point_contact_potential
from quantum_transport.two_dimensional import rectangular_device_hamiltonian
from quantum_transport.two_dimensional_leads import two_dimensional_self_energy_matrices
from quantum_transport.two_dimensional_transport import (
    broadening_matrix,
    multichannel_transmission,
    retarded_2d_device_green_function,
)


def calculate_transmission(energy, gate, length=24, width=6, sigma=4.0):
    potential = quantum_point_contact_potential(
        length, width, gate, longitudinal_sigma=sigma
    )
    device = rectangular_device_hamiltonian(length, width, potential)
    sigma_left, sigma_right = two_dimensional_self_energy_matrices(
        energy, length, width
    )
    green = retarded_2d_device_green_function(
        energy, device, sigma_left, sigma_right
    )
    return multichannel_transmission(
        green, broadening_matrix(sigma_left), broadening_matrix(sigma_right)
    )


def test_qpc_potential_shape_and_finite_values():
    potential = quantum_point_contact_potential(40, 6, 0.5)

    assert potential.shape == (40, 6)
    assert np.all(np.isfinite(potential))


def test_qpc_potential_is_transversely_and_longitudinally_symmetric():
    potential = quantum_point_contact_potential(40, 6, 0.75)

    np.testing.assert_allclose(potential, potential[:, ::-1], atol=1e-15)
    np.testing.assert_allclose(potential, potential[::-1, :], atol=1e-15)


def test_zero_gate_reproduces_clean_potential():
    potential = quantum_point_contact_potential(18, 5, 0.0)

    np.testing.assert_array_equal(potential, np.zeros((18, 5)))


def test_increasing_gate_raises_central_potential():
    weak = quantum_point_contact_potential(40, 6, 0.25)
    strong = quantum_point_contact_potential(40, 6, 0.75)
    center = 20

    assert np.all(strong[center] > weak[center])
    assert strong[center, 0] > strong[center, 2]


def test_qpc_gate_tends_toward_zero_at_longitudinal_ends():
    potential = quantum_point_contact_potential(40, 6, 1.0)

    assert np.max(potential[[0, -1]]) < 0.01 * np.max(potential)


def test_clean_baseline_matches_open_channels_away_from_thresholds():
    length, width = 10, 6
    modes, _ = transverse_modes(width)
    for energy in (-3.5, -3.0, -2.2, -1.3):
        count = open_mode_count(energy, modes)
        assert calculate_transmission(energy, 0.0, length=length, width=width, sigma=3.0) == pytest.approx(
            count, abs=3e-10, rel=3e-10
        )


def test_qpc_transmission_is_real_nonnegative_and_bounded_by_open_modes():
    energy = -2.0
    modes, _ = transverse_modes(6)
    count = open_mode_count(energy, modes)
    transmission = calculate_transmission(energy, 0.75)

    assert isinstance(transmission, float)
    assert np.isfinite(transmission)
    assert transmission >= 0.0
    assert transmission <= count + 1e-10


def test_qpc_suppresses_transmission_for_a_representative_gate_scan():
    clean = calculate_transmission(-2.0, 0.0)
    weak = calculate_transmission(-2.0, 0.25)
    stronger = calculate_transmission(-2.0, 0.75)

    assert clean > weak > stronger


def test_repeated_qpc_transmission_is_deterministic():
    first = calculate_transmission(-2.5, 0.5)
    second = calculate_transmission(-2.5, 0.5)

    assert first == second


def test_ldos_has_spatial_shape_and_is_real():
    length, width, energy = 12, 6, -2.0
    potential = quantum_point_contact_potential(length, width, 0.5, longitudinal_sigma=2.0)
    device = rectangular_device_hamiltonian(length, width, potential)
    sigma_left, sigma_right = two_dimensional_self_energy_matrices(energy, length, width)
    green = retarded_2d_device_green_function(energy, device, sigma_left, sigma_right)
    ldos = local_density_of_states(green).reshape(length, width)

    assert ldos.shape == (length, width)
    assert np.isrealobj(ldos)
    assert np.all(np.isfinite(ldos))
    assert np.min(ldos) >= -1e-12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 0, "width": 6, "gate_strength": 0.5},
        {"length": 20, "width": 0, "gate_strength": 0.5},
        {"length": 20, "width": 6, "gate_strength": -0.1},
        {"length": 20, "width": 6, "gate_strength": 0.5, "longitudinal_sigma": 0.0},
        {"length": 20, "width": 6, "gate_strength": 0.5, "transverse_strength": np.nan},
    ],
)
def test_invalid_qpc_parameters_are_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        quantum_point_contact_potential(**kwargs)


def test_gate_scan_transmission_never_exceeds_incoming_channel_count():
    modes, _ = transverse_modes(6)
    for energy in (-3.5, -3.0, -2.5, -2.0, -1.5):
        count = open_mode_count(energy, modes)
        for gate in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = calculate_transmission(energy, gate)
            assert value <= count + 1e-9
