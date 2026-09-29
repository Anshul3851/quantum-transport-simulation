import numpy as np
import pytest

from quantum_transport.broadening import left_broadening_matrix, right_broadening_matrix
from quantum_transport.disorder import disorder_potential
from quantum_transport.ensemble import transmission_statistics
from quantum_transport.greens import retarded_device_green_function
from quantum_transport.hamiltonian import tight_binding_hamiltonian
from quantum_transport.leads import lead_self_energy
from quantum_transport.transmission import landauer_transmission


def calculate_transmission(onsite, energy=0.5):
    n_sites = len(onsite)
    hamiltonian = tight_binding_hamiltonian(n_sites, onsite, hopping=1.0)
    sigma = lead_self_energy(energy, onsite_energy=0.0, hopping=1.0, coupling=1.0)
    green = retarded_device_green_function(energy, hamiltonian, sigma, sigma)
    return landauer_transmission(
        green,
        left_broadening_matrix(n_sites, sigma),
        right_broadening_matrix(n_sites, sigma),
    )


def test_zero_disorder_gives_clean_onsite_profile():
    profile = disorder_potential(12, 0.0, onsite_energy=-0.25, seed=31)

    np.testing.assert_array_equal(profile, np.full(12, -0.25))


def test_disorder_profile_has_requested_length_and_background():
    profile = disorder_potential(17, 0.8, onsite_energy=0.3, seed=12)

    assert profile.shape == (17,)
    assert np.all((profile - 0.3 >= -0.4) & (profile - 0.3 <= 0.4))


def test_same_seed_produces_identical_disorder():
    first = disorder_potential(24, 1.2, seed=1234)
    second = disorder_potential(24, 1.2, seed=1234)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_normally_produce_different_disorder():
    first = disorder_potential(24, 1.2, seed=1234)
    second = disorder_potential(24, 1.2, seed=1235)

    assert not np.array_equal(first, second)


def test_numpy_generator_is_supported_and_reproducible():
    first = disorder_potential(10, 0.4, rng=np.random.default_rng(19))
    second = disorder_potential(10, 0.4, rng=np.random.default_rng(19))

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("n_sites", [0, -2])
def test_invalid_site_counts_are_rejected(n_sites):
    with pytest.raises(ValueError, match="greater than zero"):
        disorder_potential(n_sites, 0.5, seed=2)


def test_negative_disorder_strength_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        disorder_potential(8, -0.1, seed=2)


def test_disorder_requires_exactly_one_explicit_random_source():
    with pytest.raises(ValueError, match="exactly one"):
        disorder_potential(8, 0.5)
    with pytest.raises(ValueError, match="exactly one"):
        disorder_potential(8, 0.5, seed=2, rng=np.random.default_rng(2))
    with pytest.raises(TypeError, match="Generator"):
        disorder_potential(8, 0.5, rng=np.random.RandomState(2))


def test_ensemble_statistics_have_expected_dimensions_and_finite_values():
    samples = np.array([[0.5, 0.2], [0.7, 0.4], [0.6, 0.3]])

    summary = transmission_statistics(samples)

    assert all(summary[name].shape == (2,) for name in summary)
    assert all(np.isfinite(summary[name]).all() for name in summary)
    np.testing.assert_allclose(summary["mean"], [0.6, 0.3])
    np.testing.assert_allclose(summary["median"], [0.6, 0.3])


def test_log_floor_makes_mean_log_finite_and_is_counted():
    samples = np.array([[0.0, 0.5], [1e-15, 0.4], [0.2, 0.3]])

    summary = transmission_statistics(samples, log_floor=1e-12)

    assert np.isfinite(summary["mean_log"]).all()
    np.testing.assert_array_equal(summary["log_floor_count"], [2, 0])


def test_typical_transmission_equals_exponential_mean_log():
    samples = np.array([[0.2], [0.4], [0.8]])

    summary = transmission_statistics(samples)

    assert summary["typical"][0] == pytest.approx(np.exp(np.mean(np.log(samples[:, 0]))))


def test_clean_transport_matches_stage1_unit_transmission():
    profile = np.zeros(40)

    assert calculate_transmission(profile, energy=0.5) == pytest.approx(1.0, abs=1e-10)


def test_zero_disorder_reproduces_clean_transport():
    clean = disorder_potential(40, 0.0, seed=8)
    reference = np.zeros(40)

    assert calculate_transmission(clean, 0.5) == pytest.approx(calculate_transmission(reference, 0.5), abs=1e-13)


def test_multiple_disorder_seeds_do_not_reuse_realizations():
    profiles = np.array([disorder_potential(30, 1.0, seed=seed) for seed in range(20)])

    assert np.unique(profiles, axis=0).shape[0] == 20


def test_transmission_is_nonnegative_and_within_single_channel_bound():
    profile = disorder_potential(40, 1.0, seed=31)
    transmission = calculate_transmission(profile, energy=0.5)

    assert 0.0 <= transmission <= 1.0 + 1e-10


def test_repeated_ensemble_summary_is_deterministic():
    samples = np.array([[0.3, 0.5], [0.2, 0.7], [0.1, 0.4]])
    first = transmission_statistics(samples)
    second = transmission_statistics(samples)

    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_invalid_ensemble_arrays_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        transmission_statistics(np.ones(3))
    with pytest.raises(ValueError, match="non-negative"):
        transmission_statistics(np.array([[0.2], [-0.1]]))
    with pytest.raises(ValueError, match="upper bound"):
        transmission_statistics(np.array([[1.1], [0.9]]))


@pytest.mark.parametrize("strength", [np.nan, np.inf, -np.inf])
def test_nonfinite_disorder_strength_is_rejected(strength):
    with pytest.raises(ValueError, match="finite"):
        disorder_potential(8, strength, seed=2)


def test_invalid_background_and_boolean_site_count_are_rejected():
    with pytest.raises(TypeError, match="positive integer"):
        disorder_potential(True, 0.5, seed=2)
    with pytest.raises(ValueError, match="finite"):
        disorder_potential(8, 0.5, onsite_energy=np.nan, seed=2)


def test_nonfinite_or_complex_transmission_values_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        transmission_statistics(np.array([[0.2], [np.inf]]))
    with pytest.raises(TypeError, match="real numeric"):
        transmission_statistics(np.array([[0.2 + 0.1j], [0.4 + 0.2j]]))


def test_invalid_log_floor_is_rejected():
    with pytest.raises(ValueError, match="between zero and one"):
        transmission_statistics(np.array([[0.2], [0.4]]), log_floor=1.0)
