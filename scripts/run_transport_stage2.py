"""Run Stage 2 clean, barrier, well, and double-barrier transport studies."""

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


# Stage 2 parameters; energies are in units set by t=1.
N = 60
t = 1.0
epsilon_0 = 0.0
t_c = 1.0
barrier_height = 1.5
single_barrier_site = 29
well_depth = 0.8
well_start = 23
well_stop = 37
barrier_width = 3
double_well_width = 14
energy_min = -1.98
energy_max = 1.98
number_of_energy_points = 2401
peak_prominence = 0.02
peak_minimum_distance_points = 20
peak_window_points = 80

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def _find_prominent_peaks(
    values: np.ndarray,
    *,
    minimum_prominence: float,
    minimum_distance: int,
    window_points: int,
) -> tuple[np.ndarray, dict[int, float]]:
    """Find local maxima using a fixed-window, transparent prominence rule."""
    candidates = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    ) + 1
    prominences: dict[int, float] = {}
    for index in candidates:
        left = values[max(0, index - window_points):index]
        right = values[index + 1:min(values.size, index + window_points + 1)]
        if left.size == 0 or right.size == 0:
            continue
        prominence = float(values[index] - max(np.min(left), np.min(right)))
        if prominence >= minimum_prominence:
            prominences[int(index)] = prominence

    selected: list[int] = []
    for index in sorted(prominences, key=lambda i: values[i], reverse=True):
        if all(abs(index - accepted) >= minimum_distance for accepted in selected):
            selected.append(index)
    selected.sort()
    return np.asarray(selected, dtype=int), prominences

def _device_observables(energy: float, onsite: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Return transmission, site LDOS, and trace DOS at one energy."""
    device = tight_binding_hamiltonian(N, onsite, t)
    sigma_left = lead_self_energy(energy, epsilon_0, t, t_c)
    sigma_right = lead_self_energy(energy, epsilon_0, t, t_c)
    green = retarded_device_green_function(energy, device, sigma_left, sigma_right)
    gamma_left = left_broadening_matrix(N, sigma_left)
    gamma_right = right_broadening_matrix(N, sigma_right)
    transmission = landauer_transmission(green, gamma_left, gamma_right)
    ldos = local_density_of_states(green)
    dos = total_density_of_states(green)
    return transmission, ldos, dos


def _write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    energies = np.linspace(energy_min, energy_max, number_of_energy_points)
    clean = clean_potential(N, epsilon_0)
    barrier = single_barrier_potential(
        N, epsilon_0, barrier_height, single_barrier_site
    )
    well = finite_well_potential(N, epsilon_0, well_depth, well_start, well_stop)
    double_barrier = symmetric_double_barrier_potential(
        N,
        epsilon_0,
        barrier_height,
        well_depth,
        barrier_width,
        double_well_width,
    )
    profiles = {
        "clean": clean,
        "single_barrier": barrier,
        "finite_well": well,
        "double_barrier": double_barrier,
    }

    transmission: dict[str, np.ndarray] = {}
    double_ldos = np.empty((energies.size, N), dtype=np.float64)
    double_dos_trace = np.empty(energies.size, dtype=np.float64)
    for name, profile in profiles.items():
        t_values = np.empty(energies.size, dtype=np.float64)
        for index, energy in enumerate(energies):
            value, ldos, dos = _device_observables(float(energy), profile)
            t_values[index] = value
            if name == "double_barrier":
                double_ldos[index] = ldos
                double_dos_trace[index] = dos
        transmission[name] = t_values

    for name, values in transmission.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{name} transmission contains non-finite values")
        if np.any(values < 0.0) or np.any(values > 1.0 + 1e-9):
            raise RuntimeError(f"{name} transmission violates the single-channel bounds")
    if not np.all(np.isfinite(double_ldos)) or not np.all(np.isfinite(double_dos_trace)):
        raise RuntimeError("double-barrier DOS contains non-finite values")
    double_dos_sum = double_ldos.sum(axis=1)
    np.testing.assert_allclose(double_dos_trace, double_dos_sum, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(transmission["clean"], transmission["clean"][::-1], atol=1e-10, rtol=0.0)
    np.testing.assert_array_equal(double_barrier, double_barrier[::-1])

    peak_indices, peak_prominences = _find_prominent_peaks(
        transmission["double_barrier"],
        minimum_prominence=peak_prominence,
        minimum_distance=peak_minimum_distance_points,
        window_points=peak_window_points,
    )
    safe_peak_indices = peak_indices[np.abs(energies[peak_indices]) < 1.9]
    if safe_peak_indices.size == 0:
        raise RuntimeError(
            "No double-barrier transmission peaks met the predeclared prominence; "
            "inspect profile parameters and energy resolution."
        )
    strongest = safe_peak_indices[
        np.argsort(transmission["double_barrier"][safe_peak_indices])[::-1]
    ][:10]

    central_start = (N - (2 * barrier_width + double_well_width)) // 2 + barrier_width
    central_stop = central_start + double_well_width
    resonance_records = []
    for rank, index in enumerate(strongest, start=1):
        density = double_ldos[index]
        total = float(double_dos_trace[index])
        central_density = float(density[central_start:central_stop].sum())
        central_fraction = central_density / total if total > 0.0 else float("nan")
        resonance_records.append(
            (
                rank,
                float(energies[index]),
                float(transmission["double_barrier"][index]),
                total,
                central_density,
                central_fraction,
                peak_prominences[int(index)],
            )
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(
        RESULTS_DIR / "stage2_transmission.csv",
        ["energy", "clean", "single_barrier", "finite_well", "double_barrier"],
        zip(energies, transmission["clean"], transmission["single_barrier"], transmission["finite_well"], transmission["double_barrier"]),
    )
    _write_csv(
        RESULTS_DIR / "stage2_dos.csv",
        ["energy", "total_dos_from_trace", "total_dos_from_summed_ldos"],
        zip(energies, double_dos_trace, double_dos_sum),
    )
    _write_csv(
        RESULTS_DIR / "stage2_resonances.csv",
        ["rank_by_transmission", "energy", "transmission", "total_dos", "central_well_dos", "central_well_dos_fraction", "peak_prominence"],
        resonance_records,
    )

    # Use the strongest transmission resonance and nearby detuned energies to
    # compare its spatial LDOS profile with the off-resonant device response.
    barrier_band_lower_edge = epsilon_0 + barrier_height - 2.0 * t
    barrier_tunnelling_candidates = [
        record for record in resonance_records
        if record[1] < barrier_band_lower_edge
    ]
    if not barrier_tunnelling_candidates:
        raise RuntimeError("No prominent transmission peak lies below the barrier-region band edge")
    primary_record = max(barrier_tunnelling_candidates, key=lambda record: record[2])
    primary_index = int(np.argmin(np.abs(energies - primary_record[1])))
    detuning = 0.04
    ldos_indices = [
        int(np.argmin(np.abs(energies - (energies[primary_index] - detuning)))),
        primary_index,
        int(np.argmin(np.abs(energies - (energies[primary_index] + detuning)))),
    ]
    ldos_records = [
        (float(energies[index]), site_index, float(double_ldos[index, site_index]))
        for index in ldos_indices
        for site_index in range(N)
    ]
    _write_csv(
        RESULTS_DIR / "stage2_resonance_ldos.csv",
        ["energy", "site_index_zero_based", "local_density_of_states"],
        ldos_records,
    )

    site_positions = np.arange(N)
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    axis.step(site_positions, clean, where="mid", color="black", label="Clean")
    axis.step(site_positions, barrier, where="mid", color="#c04b36", label="Single barrier")
    axis.step(site_positions, double_barrier, where="mid", color="#2864a5", label="Double barrier")
    axis.set_xlabel("Device site index (zero-based)")
    axis.set_ylabel("Onsite energy (units of t)")
    axis.set_title("Stage 2 onsite-potential profiles")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.savefig(FIGURES_DIR / "stage2_potential_profiles.png", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.plot(energies, transmission["clean"], color="black", label="Clean")
    axis.plot(energies, transmission["single_barrier"], color="#c04b36", label="Single barrier")
    axis.plot(energies, transmission["finite_well"], color="#38845b", label="Finite well")
    axis.plot(energies, transmission["double_barrier"], color="#2864a5", label="Double barrier")
    axis.set_xlabel("Energy E (units of t)")
    axis.set_ylabel("Transmission T(E)")
    axis.set_title("Finite 1D device transmission")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.savefig(FIGURES_DIR / "stage2_transmission.png", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.plot(energies, double_dos_trace, color="#533a71", linewidth=1.4)
    axis.set_xlabel("Energy E (units of t)")
    axis.set_ylabel("Total device DOS (1 / energy unit)")
    axis.set_title("Double-barrier device density of states")
    axis.grid(alpha=0.2)
    figure.savefig(FIGURES_DIR / "stage2_dos.png", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    colors = ["#999999", "#b02e0c", "#2864a5"]
    labels = ["Below resonance", "At transmission resonance", "Above resonance"]
    for index, color, label in zip(ldos_indices, colors, labels):
        axis.plot(site_positions, double_ldos[index], color=color, linewidth=1.5, label=f"{label}, E={energies[index]:.4f}")
    axis.axvspan(central_start - 0.5, central_stop - 0.5, color="#d9cce8", alpha=0.25, label="Central well")
    axis.set_xlabel("Device site index (zero-based)")
    axis.set_ylabel("Local DOS (1 / energy unit)")
    axis.set_title("Double-barrier LDOS near a well-localized resonance")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.2)
    figure.savefig(FIGURES_DIR / "stage2_resonance_ldos.png", dpi=300)
    plt.close(figure)

    metadata = {
        "stage": "Stage 2 quantum barriers, wells, resonant tunnelling, and LDOS",
        "N": N,
        "hopping_t": t,
        "lead_and_background_onsite_epsilon_0": epsilon_0,
        "contact_hopping_t_c": t_c,
        "single_barrier": {"height": barrier_height, "site_index_zero_based": single_barrier_site, "width_sites": 1},
        "finite_well": {"depth": well_depth, "start_index_zero_based": well_start, "stop_index_exclusive": well_stop, "width_sites": well_stop - well_start},
        "double_barrier": {"height": barrier_height, "width_sites_each": barrier_width, "well_depth": well_depth, "well_width_sites": double_well_width, "central_well_start_zero_based": central_start, "central_well_stop_exclusive": central_stop, "symmetric": True},
        "energy_range": [energy_min, energy_max],
        "number_of_energy_points": number_of_energy_points,
        "peak_detection": {"method": "local maxima with a fixed-window prominence estimate", "minimum_prominence": peak_prominence, "minimum_distance_points": peak_minimum_distance_points, "prominence_window_points_each_side": peak_window_points, "safe_band_filter": "abs(E) < 1.9t"},
        "resonance_ldos_detuning": detuning,
        "primary_ldos_resonance_selection": "highest-transmission detected peak below the barrier-region band edge; central-well LDOS fraction is then measured independently",
        "units": "consistent tight-binding energy units with hopping magnitude t=1",
        "boundary_conditions": "open finite device; semi-infinite 1D leads",
        "conventions": {
            "device_hamiltonian": "H_ii=epsilon_i and nearest-neighbour terms -t",
            "lead_band": [epsilon_0 - 2.0 * t, epsilon_0 + 2.0 * t],
            "self_energy": "Sigma^r=t_c^2*g_s^r at first and last device sites",
            "device_green_function": "G^r=[E I-H_device-Sigma_L-Sigma_R]^-1; no added imaginary shift",
            "broadening": "Gamma=i(Sigma^r-Sigma^{r dagger})",
            "transmission": "Tr[Gamma_L G^r Gamma_R (G^r)^dagger]",
            "local_density_of_states": "rho_i=-Im(G^r_ii)/pi",
            "total_density_of_states": "rho=-Im(Tr(G^r))/pi",
            "transmission_processing": "no normalization; only roundoff-scale negative results within the transmission function tolerance are set to zero",
        },
    }
    with (RESULTS_DIR / "stage2_metadata.json").open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)
        output_file.write("\n")

    safe_interior = np.abs(energies) <= 1.5 * t
    barrier_reference = int(np.argmin(np.abs(energies - 0.5)))
    top = resonance_records[0]
    print(f"Clean min over |E| <= 1.5t: {np.min(transmission['clean'][safe_interior]):.12g}")
    print(f"Clean max |T-1| over |E| <= 1.5t: {np.max(np.abs(transmission['clean'][safe_interior] - 1.0)):.12g}")
    print(f"Single-barrier T({energies[barrier_reference]:.6f})={transmission['single_barrier'][barrier_reference]:.12g}")
    print(f"Finite-well T({energies[barrier_reference]:.6f})={transmission['finite_well'][barrier_reference]:.12g}")
    print(f"Strongest DB T peak E={top[1]:.8f}, T={top[2]:.8f}, central-well DOS fraction={top[5]:.6f}")
    print(f"Well-localized resonance E={primary_record[1]:.8f}, T={primary_record[2]:.8f}, DOS={primary_record[3]:.8f}, central-well DOS fraction={primary_record[5]:.6f}")
    print(f"Maximum |DOS(trace)-sum(LDOS)|: {np.max(np.abs(double_dos_trace - double_dos_sum)):.3e}")
    print(f"Maximum transmission over all structures: {max(np.max(v) for v in transmission.values()):.12g}")
    for name in ("stage2_transmission.csv", "stage2_dos.csv", "stage2_resonances.csv", "stage2_resonance_ldos.csv", "stage2_metadata.json"):
        print(f"Wrote {RESULTS_DIR / name}")
    for name in ("stage2_potential_profiles.png", "stage2_transmission.png", "stage2_dos.png", "stage2_resonance_ldos.png"):
        print(f"Wrote {FIGURES_DIR / name}")


if __name__ == "__main__":
    main()




