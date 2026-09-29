import numpy as np
import pytest

from quantum_transport.leads import lead_self_energy, surface_green_function


def test_retarded_surface_green_function_has_negative_imaginary_part_inside_band():
    green = surface_green_function(energy=0.6, onsite_energy=0.0, hopping=1.0)

    assert green.imag < 0


def test_retarded_self_energy_has_negative_imaginary_part_inside_band():
    sigma = lead_self_energy(energy=0.6, onsite_energy=0.0, hopping=1.0, coupling=0.8)

    assert sigma.imag < 0


def test_surface_green_function_satisfies_quadratic_relation():
    energies = np.array([-1.2, -0.4, 0.3, 1.1])
    epsilon_0 = 0.25
    hopping = 1.3
    green = surface_green_function(energies, epsilon_0, hopping)

    residual = hopping**2 * green**2 - (energies - epsilon_0) * green + 1.0

    np.testing.assert_allclose(residual, 0.0, atol=1e-12, rtol=0.0)


def test_outside_band_green_function_is_real():
    green = surface_green_function(np.array([-2.5, 2.5]), onsite_energy=0.0, hopping=1.0)

    np.testing.assert_allclose(green.imag, 0.0, atol=0.0, rtol=0.0)
    assert np.all(np.isfinite(green.real))


def test_green_function_is_complex_inside_and_real_outside_band():
    inside = surface_green_function(0.5, hopping=1.0)
    outside = surface_green_function(2.5, hopping=1.0)

    assert isinstance(inside, complex)
    assert inside.imag < 0
    assert isinstance(outside, complex)
    assert outside.imag == 0.0


def test_self_energy_scales_with_square_of_contact_hopping():
    energy = 0.5
    sigma_1 = lead_self_energy(energy, coupling=0.4)
    sigma_2 = lead_self_energy(energy, coupling=0.8)

    assert sigma_2 == pytest.approx(4.0 * sigma_1)


def test_onsite_energy_shifts_the_propagating_band():
    epsilon_0 = -1.5
    inside_shifted_band = surface_green_function(0.0, onsite_energy=epsilon_0, hopping=1.0)
    outside_shifted_band = surface_green_function(0.75, onsite_energy=epsilon_0, hopping=1.0)

    assert inside_shifted_band.imag < 0
    assert outside_shifted_band.imag == 0.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: surface_green_function(0.0, hopping=0.0),
        lambda: surface_green_function(0.0, hopping=-1.0),
        lambda: surface_green_function(0.0, hopping=np.inf),
        lambda: surface_green_function(np.nan),
        lambda: surface_green_function(1.0 + 0.1j),
        lambda: surface_green_function(0.0, onsite_energy=np.inf),
        lambda: lead_self_energy(0.0, coupling=-0.1),
        lambda: lead_self_energy(0.0, coupling=np.inf),
    ],
)
def test_invalid_parameters_are_rejected(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def test_repeated_calls_are_deterministic():
    energies = np.array([-2.5, -0.3, 0.4, 2.6])

    first = surface_green_function(energies, onsite_energy=0.1, hopping=1.2)
    second = surface_green_function(energies, onsite_energy=0.1, hopping=1.2)

    np.testing.assert_array_equal(first, second)


def test_scalar_and_vector_energy_inputs_preserve_shape():
    scalar = surface_green_function(0.5)
    energies = np.array([[0.5, 2.5], [-2.5, -0.5]])
    vector = surface_green_function(energies)
    sigma = lead_self_energy(energies, coupling=0.7)

    assert isinstance(scalar, complex)
    assert vector.shape == energies.shape
    assert sigma.shape == energies.shape
    assert np.iscomplexobj(vector)
    assert np.iscomplexobj(sigma)


