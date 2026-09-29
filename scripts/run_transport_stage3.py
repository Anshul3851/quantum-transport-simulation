"""Run seeded disorder ensembles and length scaling for 1D transport."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantum_transport.broadening import left_broadening_matrix, right_broadening_matrix
from quantum_transport.disorder import disorder_potential
from quantum_transport.ensemble import transmission_statistics
from quantum_transport.greens import retarded_device_green_function
from quantum_transport.hamiltonian import tight_binding_hamiltonian
from quantum_transport.leads import lead_self_energy
from quantum_transport.transmission import landauer_transmission


# Stage 3 parameters, in tight-binding units with t=1.
MASTER_SEED = 20260317
N_SINGLE = 40
N_ENSEMBLE = 40
LENGTHS = [10, 20, 40, 80]
HOPPING = 1.0
ONSITE_ENERGY = 0.0
CONTACT_HOPPING = 1.0
DISORDER_STRENGTHS = [0.0, 0.5, 1.0, 2.0]
LENGTH_DISORDER_STRENGTHS = [0.5, 1.0, 2.0]
REPRESENTATIVE_ENERGIES = [0.0, 0.5, 1.0]
LENGTH_ENERGIES = [0.0, 0.5]
N_REALIZATIONS = 100
N_PLOTTED_REALIZATIONS = 4
SINGLE_ENERGY_GRID = np.linspace(-1.8, 1.8, 181)
LOG_TRANSMISSION_FLOOR = 1e-12

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def _transmission_for_profile(
    onsite: np.ndarray,
    energy: float,
    *,
    hopping: float = HOPPING,
    epsilon_0: float = ONSITE_ENERGY,
    t_c: float = CONTACT_HOPPING,
) -> float:
    """Calculate transmission through one fixed onsite profile and energy."""
    n_sites = onsite.size
    hamiltonian = tight_binding_hamiltonian(n_sites, onsite, hopping)
    sigma_left = lead_self_energy(energy, epsilon_0, hopping, t_c)
    sigma_right = lead_self_energy(energy, epsilon_0, hopping, t_c)
    green = retarded_device_green_function(energy, hamiltonian, sigma_left, sigma_right)
    gamma_left = left_broadening_matrix(n_sites, sigma_left)
    gamma_right = right_broadening_matrix(n_sites, sigma_right)
    return landauer_transmission(green, gamma_left, gamma_right)


def _write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        writer.writerows(rows)


def _seed_list_text(seeds: list[int]) -> str:
    """Serialize the actual per-realization seeds for output tables."""
    return ";".join(str(seed) for seed in seeds)


def main() -> None:
    # Preallocate distinct child streams in a fixed group order so reruns are
    # reproducible and no nonzero-W realization is accidentally reused.
    n_plot_seeds = len(DISORDER_STRENGTHS) * N_PLOTTED_REALIZATIONS
    n_ensemble_seeds = len(DISORDER_STRENGTHS) * N_REALIZATIONS
    n_length_seeds = len(LENGTHS) * len(LENGTH_DISORDER_STRENGTHS) * N_REALIZATIONS
    n_seed_streams = n_plot_seeds + n_ensemble_seeds + n_length_seeds
    child_sequences = np.random.SeedSequence(MASTER_SEED).spawn(n_seed_streams)
    all_seeds = [
        int(child.generate_state(1, dtype=np.uint64)[0]) for child in child_sequences
    ]
    if len(set(all_seeds)) != n_seed_streams:
        raise RuntimeError("SeedSequence produced duplicate child seeds")

    cursor = 0
    plot_seed_bank: dict[float, list[int]] = {}
    for strength in DISORDER_STRENGTHS:
        plot_seed_bank[strength] = all_seeds[cursor:cursor + N_PLOTTED_REALIZATIONS]
        cursor += N_PLOTTED_REALIZATIONS
    plot_seed_range = [0, cursor]

    ensemble_seed_bank: dict[float, list[int]] = {}
    for strength in DISORDER_STRENGTHS:
        ensemble_seed_bank[strength] = all_seeds[cursor:cursor + N_REALIZATIONS]
        cursor += N_REALIZATIONS
    ensemble_seed_range = [plot_seed_range[1], cursor]

    length_seed_bank: dict[tuple[int, float], list[int]] = {}
    for n_sites in LENGTHS:
        for strength in LENGTH_DISORDER_STRENGTHS:
            length_seed_bank[(n_sites, strength)] = all_seeds[cursor:cursor + N_REALIZATIONS]
            cursor += N_REALIZATIONS
    length_seed_range = [ensemble_seed_range[1], cursor]
    if cursor != n_seed_streams:
        raise RuntimeError("Internal seed allocation did not consume the full seed bank")

    # Individual spectra and profiles: four independent realizations per W.
    single_transmission_rows = []
    realization_profiles: dict[float, list[np.ndarray]] = {}
    for strength in DISORDER_STRENGTHS:
        realization_profiles[strength] = []
        for realization_index, seed in enumerate(plot_seed_bank[strength]):
            onsite = disorder_potential(
                N_SINGLE, strength, ONSITE_ENERGY, seed=seed
            )
            realization_profiles[strength].append(onsite)
            for energy in SINGLE_ENERGY_GRID:
                value = _transmission_for_profile(onsite, float(energy))
                single_transmission_rows.append(
                    (N_SINGLE, strength, float(energy), realization_index, seed, value)
                )

    # Ensemble statistics at N=40, using the same independent realization at
    # all three selected energies within each (W, realization) group.
    ensemble_samples: dict[float, np.ndarray] = {}
    ensemble_stats: dict[float, dict[str, np.ndarray]] = {}
    ensemble_rows = []
    for strength in DISORDER_STRENGTHS:
        samples = np.empty((N_REALIZATIONS, len(REPRESENTATIVE_ENERGIES)))
        seeds = ensemble_seed_bank[strength]
        for realization_index, seed in enumerate(seeds):
            onsite = disorder_potential(
                N_ENSEMBLE, strength, ONSITE_ENERGY, seed=seed
            )
            for energy_index, energy in enumerate(REPRESENTATIVE_ENERGIES):
                samples[realization_index, energy_index] = _transmission_for_profile(
                    onsite, energy
                )
        summary = transmission_statistics(samples, LOG_TRANSMISSION_FLOOR)
        ensemble_samples[strength] = samples
        ensemble_stats[strength] = summary
        seed_text = _seed_list_text(seeds)
        for energy_index, energy in enumerate(REPRESENTATIVE_ENERGIES):
            ensemble_rows.append(
                (
                    N_ENSEMBLE, HOPPING, ONSITE_ENERGY, CONTACT_HOPPING,
                    strength, energy, N_REALIZATIONS, MASTER_SEED,
                    "SeedSequence(master_seed).spawn; child converted to uint64 seed",
                    seed_text,
                    summary["mean"][energy_index],
                    summary["median"][energy_index],
                    summary["standard_deviation"][energy_index],
                    summary["standard_error"][energy_index],
                    summary["mean_log"][energy_index],
                    summary["typical"][energy_index],
                    LOG_TRANSMISSION_FLOOR,
                    summary["log_floor_count"][energy_index],
                )
            )

    # Length dependence: independent seed sets for every (N, W), paired over E.
    length_samples: dict[tuple[int, float], np.ndarray] = {}
    length_stats: dict[tuple[int, float], dict[str, np.ndarray]] = {}
    length_rows = []
    for n_sites in LENGTHS:
        for strength in LENGTH_DISORDER_STRENGTHS:
            samples = np.empty((N_REALIZATIONS, len(LENGTH_ENERGIES)))
            seeds = length_seed_bank[(n_sites, strength)]
            for realization_index, seed in enumerate(seeds):
                onsite = disorder_potential(
                    n_sites, strength, ONSITE_ENERGY, seed=seed
                )
                for energy_index, energy in enumerate(LENGTH_ENERGIES):
                    samples[realization_index, energy_index] = _transmission_for_profile(
                        onsite, energy
                    )
            summary = transmission_statistics(samples, LOG_TRANSMISSION_FLOOR)
            length_samples[(n_sites, strength)] = samples
            length_stats[(n_sites, strength)] = summary
            seed_text = _seed_list_text(seeds)
            for energy_index, energy in enumerate(LENGTH_ENERGIES):
                length_rows.append(
                    (
                        n_sites, HOPPING, ONSITE_ENERGY, CONTACT_HOPPING,
                        strength, energy, N_REALIZATIONS, MASTER_SEED,
                        "SeedSequence(master_seed).spawn; child converted to uint64 seed",
                        seed_text,
                        summary["mean"][energy_index],
                        summary["median"][energy_index],
                        summary["standard_deviation"][energy_index],
                        summary["standard_error"][energy_index],
                        summary["mean_log"][energy_index],
                        summary["typical"][energy_index],
                        LOG_TRANSMISSION_FLOOR,
                        summary["log_floor_count"][energy_index],
                    )
                )

    # Scientific sanity checks. T is not normalized or clipped here.
    clean_summary = ensemble_stats[0.0]
    np.testing.assert_allclose(clean_summary["mean"], 1.0, atol=1e-10, rtol=0.0)
    for summary in list(ensemble_stats.values()) + list(length_stats.values()):
        if not np.isfinite(summary["mean"]).all() or not np.isfinite(summary["mean_log"]).all():
            raise RuntimeError("ensemble summary contains non-finite values")
        if np.any(summary["mean"] < 0.0) or np.any(summary["mean"] > 1.0 + 1e-10):
            raise RuntimeError("ensemble transmission violates single-channel bounds")
    if len(set(seed for seeds in ensemble_seed_bank.values() for seed in seeds)) != n_ensemble_seeds:
        raise RuntimeError("ensemble realization seeds are not unique")

    # Exploratory linear fits are reported only when log(T) versus N is
    # decreasing and has R^2 >= 0.95 across the four sampled lengths.
    exploratory_fits = []
    length_array = np.asarray(LENGTHS, dtype=np.float64)
    for strength in LENGTH_DISORDER_STRENGTHS:
        for energy_index, energy in enumerate(LENGTH_ENERGIES):
            means = np.array(
                [length_stats[(n_sites, strength)]["mean_log"][energy_index] for n_sites in LENGTHS]
            )
            slope, intercept = np.polyfit(length_array, means, deg=1)
            fitted = intercept + slope * length_array
            residual_sum = float(np.sum((means - fitted) ** 2))
            total_sum = float(np.sum((means - means.mean()) ** 2))
            r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
            if slope < 0.0 and r_squared >= 0.95:
                exploratory_fits.append(
                    {"W": strength, "energy": energy, "intercept": float(intercept), "slope_per_site": float(slope), "r_squared": float(r_squared), "localization_length_convention": "xi=-1/slope only as exploratory finite-size estimate"}
                )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ensemble_header = [
        "N", "hopping_t", "epsilon_0", "contact_t_c", "W", "energy",
        "n_realizations", "master_seed", "seed_generation_convention", "seeds",
        "mean_T", "median_T", "standard_deviation_T", "standard_error_mean_T",
        "mean_log_T", "typical_T", "log_floor", "log_floor_count",
    ]
    _write_csv(RESULTS_DIR / "stage3_ensemble.csv", ensemble_header, ensemble_rows)
    _write_csv(RESULTS_DIR / "stage3_length_scaling.csv", ensemble_header, length_rows)
    _write_csv(
        RESULTS_DIR / "stage3_single_realization.csv",
        ["N", "W", "energy", "realization_index", "seed", "transmission"],
        single_transmission_rows,
    )
    profile_rows = []
    for strength, profiles in realization_profiles.items():
        for realization_index, (seed, onsite) in enumerate(zip(plot_seed_bank[strength], profiles)):
            profile_rows.extend(
                (N_SINGLE, strength, realization_index, seed, site, value)
                for site, value in enumerate(onsite)
            )
    _write_csv(
        RESULTS_DIR / "stage3_disorder_realizations.csv",
        ["N", "W", "realization_index", "seed", "site_index", "onsite_energy"],
        profile_rows,
    )

    seed_convention = "SeedSequence(MASTER_SEED).spawn(total_streams), each child converted to a uint64 seed; fixed group order and distinct seed recorded for each realization"
    metadata = {
        "stage": "Stage 3 disorder and localization in coherent quantum transport",
        "model": "epsilon_i = epsilon_0 + w_i, independently sampled w_i ~ Uniform(-W/2, W/2)",
        "master_seed": MASTER_SEED,
        "seed_generation_convention": seed_convention,
        "seed_streams": {
            "single_realization_spectra": {"count": n_plot_seeds, "stream_index_range_half_open": plot_seed_range},
            "N40_ensemble": {"count": n_ensemble_seeds, "stream_index_range_half_open": ensemble_seed_range},
            "length_scaling": {"count": n_length_seeds, "stream_index_range_half_open": length_seed_range},
            "total_unique_seed_streams": n_seed_streams,
        },
        "single_realization_spectra": {"N": N_SINGLE, "W_values": DISORDER_STRENGTHS, "realizations_per_W": N_PLOTTED_REALIZATIONS, "energy_range": [float(SINGLE_ENERGY_GRID[0]), float(SINGLE_ENERGY_GRID[-1])], "energy_points": len(SINGLE_ENERGY_GRID)},
        "ensemble": {"N": N_ENSEMBLE, "W_values": DISORDER_STRENGTHS, "energies": REPRESENTATIVE_ENERGIES, "realizations_per_W": N_REALIZATIONS},
        "length_scaling": {"N_values": LENGTHS, "W_values": LENGTH_DISORDER_STRENGTHS, "energies": LENGTH_ENERGIES, "realizations_per_N_W": N_REALIZATIONS},
        "hopping_t": HOPPING,
        "onsite_energy_epsilon_0": ONSITE_ENERGY,
        "contact_hopping_t_c": CONTACT_HOPPING,
        "log_transmission_floor": LOG_TRANSMISSION_FLOOR,
        "log_floor_policy": "floor is applied only inside logarithms; counts of samples below the floor are stored per summary; transmission values themselves are unchanged",
        "standard_deviation": "sample standard deviation with ddof=1",
        "standard_error": "sample standard deviation divided by sqrt(number of realizations)",
        "localization_fit_policy": "fit mean(log(T)) against N only for negative slope and R^2 >= 0.95 across all four sampled lengths; any fit is exploratory",
        "numerical_conventions": {
            "device_hamiltonian": "H_ii=epsilon_i and nearest-neighbour terms -t with open device boundaries",
            "leads": "semi-infinite 1D nearest-neighbour leads with onsite epsilon_0 and hopping -t",
            "self_energy": "Sigma^r=t_c^2*g_s^r at the endpoints",
            "broadening": "Gamma=i(Sigma^r-Sigma^{r dagger})",
            "transmission": "Tr[Gamma_L G^r Gamma_R (G^r)^dagger]",
            "no_transmission_clipping_or_normalization": True,
        },
        "exploratory_linear_fits": exploratory_fits,
    }
    with (RESULTS_DIR / "stage3_metadata.json").open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)
        output_file.write("\n")

    # Disorder-potential realizations for each W.
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex=True, sharey=True, constrained_layout=True)
    for axis, strength in zip(axes.flat, DISORDER_STRENGTHS):
        for realization_index, onsite in enumerate(realization_profiles[strength]):
            axis.plot(np.arange(N_SINGLE), onsite, linewidth=0.9, alpha=0.78, label=f"r{realization_index + 1}")
        axis.axhline(ONSITE_ENERGY, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"W = {strength:g}")
        axis.grid(alpha=0.18)
    axes[1, 0].set_xlabel("Site index")
    axes[1, 1].set_xlabel("Site index")
    axes[0, 0].set_ylabel("Onsite energy")
    axes[1, 0].set_ylabel("Onsite energy")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    figure.savefig(FIGURES_DIR / "stage3_disorder_realizations.png", dpi=300)
    plt.close(figure)

    # Individual realization spectra across disorder strengths.
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex=True, sharey=True, constrained_layout=True)
    for axis, strength in zip(axes.flat, DISORDER_STRENGTHS):
        for realization_index in range(N_PLOTTED_REALIZATIONS):
            selected = [row for row in single_transmission_rows if row[1] == strength and row[3] == realization_index]
            axis.plot([row[2] for row in selected], [row[5] for row in selected], linewidth=0.8, alpha=0.75)
        axis.set_title(f"W = {strength:g}")
        axis.grid(alpha=0.18)
    axes[1, 0].set_xlabel("Energy E (units of t)")
    axes[1, 1].set_xlabel("Energy E (units of t)")
    axes[0, 0].set_ylabel("Transmission T(E)")
    axes[1, 0].set_ylabel("Transmission T(E)")
    figure.savefig(FIGURES_DIR / "stage3_transmission_disorder.png", dpi=300)
    plt.close(figure)

    # Ensemble mean and median at the representative energies.
    figure, axes = plt.subplots(1, len(REPRESENTATIVE_ENERGIES), figsize=(10.5, 3.5), sharey=True, constrained_layout=True)
    for energy_index, (axis, energy) in enumerate(zip(axes, REPRESENTATIVE_ENERGIES)):
        means = [ensemble_stats[w]["mean"][energy_index] for w in DISORDER_STRENGTHS]
        medians = [ensemble_stats[w]["median"][energy_index] for w in DISORDER_STRENGTHS]
        errors = [ensemble_stats[w]["standard_error"][energy_index] for w in DISORDER_STRENGTHS]
        axis.errorbar(DISORDER_STRENGTHS, means, yerr=errors, marker="o", capsize=2, label="Mean ± SEM")
        axis.plot(DISORDER_STRENGTHS, medians, marker="s", linestyle="--", label="Median")
        axis.set_title(f"E = {energy:g}")
        axis.set_xlabel("Disorder strength W")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Transmission")
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(FIGURES_DIR / "stage3_ensemble_statistics.png", dpi=300)
    plt.close(figure)

    # Mean log transmission and typical transmission versus device length.
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    colors = {0.5: "#2b6f9c", 1.0: "#4c8c4a", 2.0: "#a34a3a"}
    for strength in LENGTH_DISORDER_STRENGTHS:
        for energy_index, energy in enumerate(LENGTH_ENERGIES):
            means_log = [length_stats[(n_sites, strength)]["mean_log"][energy_index] for n_sites in LENGTHS]
            typical = [length_stats[(n_sites, strength)]["typical"][energy_index] for n_sites in LENGTHS]
            label = f"W={strength:g}, E={energy:g}"
            axes[0].plot(LENGTHS, means_log, marker="o", color=colors[strength], linestyle="-" if energy_index == 0 else "--", label=label)
            axes[1].semilogy(LENGTHS, typical, marker="o", color=colors[strength], linestyle="-" if energy_index == 0 else "--", label=label)
    axes[0].set_xlabel("Device length N")
    axes[0].set_ylabel("Mean log-transmission ⟨ln T⟩")
    axes[1].set_xlabel("Device length N")
    axes[1].set_ylabel("Typical transmission exp(⟨ln T⟩)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
    figure.savefig(FIGURES_DIR / "stage3_length_dependence.png", dpi=300)
    plt.close(figure)

    print("N=40 ensemble mean T by W and E:")
    for strength in DISORDER_STRENGTHS:
        print(f"W={strength:g}: " + ", ".join(f"E={energy:g}: {ensemble_stats[strength]['mean'][i]:.6g} (median {ensemble_stats[strength]['median'][i]:.6g}, mean logT {ensemble_stats[strength]['mean_log'][i]:.6g})" for i, energy in enumerate(REPRESENTATIVE_ENERGIES)))
    print("Length-scaling mean logT and typical T at E=0:")
    for strength in LENGTH_DISORDER_STRENGTHS:
        print(f"W={strength:g}: " + ", ".join(f"N={n}: <lnT>={length_stats[(n, strength)]['mean_log'][0]:.6g}, Ttyp={length_stats[(n, strength)]['typical'][0]:.6g}" for n in LENGTHS))
    print(f"Exploratory fits passing slope/R^2 criteria: {len(exploratory_fits)}")
    for name in (
        "stage3_ensemble.csv", "stage3_length_scaling.csv", "stage3_single_realization.csv",
        "stage3_disorder_realizations.csv", "stage3_metadata.json",
    ):
        print(f"Wrote {RESULTS_DIR / name}")
    for name in (
        "stage3_disorder_realizations.png", "stage3_transmission_disorder.png",
        "stage3_ensemble_statistics.png", "stage3_length_dependence.png",
    ):
        print(f"Wrote {FIGURES_DIR / name}")


if __name__ == "__main__":
    main()
