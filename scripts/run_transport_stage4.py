"""Run the clean finite-width strip benchmark for Stage 4."""

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

from quantum_transport.modes import open_mode_count, transverse_modes
from quantum_transport.two_dimensional import rectangular_device_hamiltonian
from quantum_transport.two_dimensional_leads import two_dimensional_self_energy_matrices
from quantum_transport.two_dimensional_transport import (
    PHYSICAL_CONSTANTS_SOURCE,
    broadening_matrix,
    multichannel_transmission,
    physical_conductance,
    retarded_2d_device_green_function,
)


# Tight-binding model values; all energy quantities use units where t=1.
LX = 30
LY = 4
HOPPING = 1.0
ONSITE_ENERGY = 0.0
CONTACT_HOPPING = HOPPING
ENERGY_MIN = -3.75
ENERGY_MAX = 3.75
ENERGY_POINTS = 751
THRESHOLD_EXCLUSION = 0.05 * HOPPING

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    device = rectangular_device_hamiltonian(
        LX, LY, onsite_energy=ONSITE_ENERGY, hopping=HOPPING
    )
    mode_energies, mode_vectors = transverse_modes(
        LY, onsite_energy=ONSITE_ENERGY, hopping=HOPPING
    )
    energy_grid = np.linspace(ENERGY_MIN, ENERGY_MAX, ENERGY_POINTS)
    transmissions = np.empty_like(energy_grid)
    channel_counts = open_mode_count(energy_grid, mode_energies, HOPPING)
    conductances = np.empty_like(energy_grid)

    for index, energy in enumerate(energy_grid):
        sigma_left, sigma_right = two_dimensional_self_energy_matrices(
            float(energy), LX, LY, ONSITE_ENERGY, HOPPING, CONTACT_HOPPING
        )
        green = retarded_2d_device_green_function(
            float(energy), device, sigma_left, sigma_right
        )
        gamma_left = broadening_matrix(sigma_left)
        gamma_right = broadening_matrix(sigma_right)
        transmissions[index] = multichannel_transmission(
            green, gamma_left, gamma_right
        )
        conductances[index] = physical_conductance(transmissions[index])

    if not np.all(np.isfinite(transmissions)):
        raise RuntimeError("Stage 4 transmission contains non-finite values")
    if np.any(transmissions < 0.0):
        raise RuntimeError("Stage 4 transmission contains negative values")

    lower_edges = mode_energies - 2.0 * HOPPING
    upper_edges = mode_energies + 2.0 * HOPPING
    thresholds = sorted(
        [
            {"mode_index": mode + 1, "transverse_energy": float(mode_energies[mode]),
             "threshold_type": "lower/opening", "energy": float(lower_edges[mode])}
            for mode in range(LY)
        ]
        + [
            {"mode_index": mode + 1, "transverse_energy": float(mode_energies[mode]),
             "threshold_type": "upper/closing", "energy": float(upper_edges[mode])}
            for mode in range(LY)
        ],
        key=lambda item: item["energy"],
    )
    threshold_energies = np.array([item["energy"] for item in thresholds])
    distance_to_threshold = np.min(
        np.abs(energy_grid[:, None] - threshold_energies[None, :]), axis=1
    )
    away = distance_to_threshold >= THRESHOLD_EXCLUSION
    absolute_error = np.abs(transmissions[away] - channel_counts[away])
    if absolute_error.size == 0:
        raise RuntimeError("No energy samples remain after threshold exclusion")
    max_error = float(np.max(absolute_error))
    median_error = float(np.median(absolute_error))
    mean_error = float(np.mean(absolute_error))
    if max_error > 1e-8:
        raise RuntimeError(
            f"Clean-strip transmission does not match open modes away from thresholds: "
            f"maximum absolute difference {max_error:.3e}"
        )

    transport_rows = [
        {
            "energy": float(energy),
            "transmission": float(transmission),
            "N_open": int(n_open),
            "dimensionless_conductance_g": float(transmission),
            "conductance_S": float(conductance),
            "nearest_mode_threshold_distance": float(distance),
        }
        for energy, transmission, n_open, conductance, distance in zip(
            energy_grid, transmissions, channel_counts, conductances, distance_to_threshold
        )
    ]
    _write_csv(
        RESULTS_DIR / "stage4_multichannel.csv",
        ["energy", "transmission", "N_open", "dimensionless_conductance_g",
         "conductance_S", "nearest_mode_threshold_distance"],
        transport_rows,
    )
    mode_rows = [
        {
            "mode_index": mode + 1,
            "transverse_energy": float(mode_energies[mode]),
            "lower_opening_threshold": float(lower_edges[mode]),
            "upper_closing_threshold": float(upper_edges[mode]),
            "mode_vector_components": ";".join(f"{value:.16g}" for value in mode_vectors[:, mode]),
        }
        for mode in range(LY)
    ]
    _write_csv(
        RESULTS_DIR / "stage4_modes.csv",
        ["mode_index", "transverse_energy", "lower_opening_threshold",
         "upper_closing_threshold", "mode_vector_components"],
        mode_rows,
    )

    metadata = {
        "stage": "Stage 4: 2D multi-channel transport and conductance quantization",
        "geometry": {"longitudinal_length_Lx": LX, "transverse_width_Ly": LY,
                     "site_ordering": "index(x,y) = x*Ly + y",
                     "boundary_conditions": "open along both device axes and hard-wall in lead transverse direction"},
        "model": {"hopping_t": HOPPING, "onsite_epsilon_0": ONSITE_ENERGY,
                  "lead_device_coupling_t_c": CONTACT_HOPPING,
                  "hamiltonian_convention": "H_ii=epsilon_i; all nearest-neighbour x/y matrix elements are -t"},
        "energy_grid": {"minimum": ENERGY_MIN, "maximum": ENERGY_MAX,
                        "points": ENERGY_POINTS, "units": "tight-binding energy units"},
        "transverse_modes": [
            {"mode_index": row["mode_index"], "transverse_energy": row["transverse_energy"],
             "lower_opening_threshold": row["lower_opening_threshold"],
             "upper_closing_threshold": row["upper_closing_threshold"]}
            for row in mode_rows
        ],
        "E_mode_thresholds": thresholds,
        "surface_green_function": "Diagonalize H_y; evaluate existing retarded semi-infinite 1D surface g_n^r(E) with onsite epsilon_n and longitudinal hopping t; transform U diag(g_n) U^dagger back to transverse-site basis.",
        "self_energy": "Sigma_boundary = t_c^2 g_surface; embedded only on the first/last longitudinal slice.",
        "transmission": "Tr[Gamma_L G^r Gamma_R G^a], Gamma=i(Sigma-Sigma^dagger); no artificial broadening, clipping, or normalization.",
        "conductance": {"dimensionless": "g=T", "physical": "G=(2e^2/h)T in siemens",
                        "constants_source": PHYSICAL_CONSTANTS_SOURCE},
        "clean_matched_check": {
            "energy_samples_excluded_near_any_mode_threshold": int(np.count_nonzero(~away)),
            "threshold_exclusion_distance": THRESHOLD_EXCLUSION,
            "energy_samples_checked_away_from_thresholds": int(np.count_nonzero(away)),
            "maximum_absolute_T_minus_N_open_away_from_thresholds": max_error,
            "median_absolute_T_minus_N_open_away_from_thresholds": median_error,
            "mean_absolute_T_minus_N_open_away_from_thresholds": mean_error,
            "validation_tolerance": 1e-8,
        },
        "dense_method_limit": "Dense solve used for Lx*Ly=120 device sites; intended for moderate strip sizes.",
    }
    (RESULTS_DIR / "stage4_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    plt.figure(figsize=(8, 4.5))
    for mode, epsilon in enumerate(mode_energies):
        low, high = epsilon - 2.0 * HOPPING, epsilon + 2.0 * HOPPING
        plt.hlines(mode + 1, low, high, color="tab:blue", linewidth=6, alpha=0.75)
        plt.scatter([low, high], [mode + 1, mode + 1], color=["tab:green", "tab:red"], zorder=3)
    plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    plt.yticks(np.arange(1, LY + 1), [f"n={n}" for n in range(1, LY + 1)])
    plt.xlabel("Energy (tight-binding units)")
    plt.ylabel("Transverse mode")
    plt.title("Hard-wall transverse mode propagation windows")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage4_mode_thresholds.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(energy_grid, transmissions, label=r"Numerical $T(E)$", color="tab:blue")
    plt.step(energy_grid, channel_counts, where="mid", label=r"Analytic $N_{open}(E)$",
             color="tab:orange", linestyle="--")
    for threshold in threshold_energies:
        plt.axvline(threshold, color="gray", linewidth=0.6, alpha=0.2)
    plt.xlabel("Energy (tight-binding units)")
    plt.ylabel("Transmission / open-channel count")
    plt.title(f"Clean matched 2D strip: Lx={LX}, Ly={LY}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage4_multichannel_transport.png", dpi=180)
    plt.close()

    figure, (axis_g, axis_G) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    axis_g.plot(energy_grid, transmissions, color="tab:blue")
    axis_g.set_ylabel(r"Dimensionless $g=T$")
    axis_g.grid(alpha=0.25)
    axis_G.plot(energy_grid, conductances * 1e6, color="tab:purple")
    axis_G.set_ylabel(r"Conductance ($\mu$S)")
    axis_G.set_xlabel("Energy (tight-binding units)")
    axis_G.grid(alpha=0.25)
    figure.suptitle(r"Conductance $G=(2e^2/h)T$")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "stage4_conductance.png", dpi=180)
    plt.close(figure)

    print(f"Device: Lx={LX}, Ly={LY}, sites={LX * LY}; modes={LY}")
    print("Transverse energies:", ", ".join(f"{value:.8f}" for value in mode_energies))
    print("Lower opening thresholds:", ", ".join(f"{value:.8f}" for value in lower_edges))
    print("Upper closing thresholds:", ", ".join(f"{value:.8f}" for value in upper_edges))
    print(f"Checked {np.count_nonzero(away)} energies >= {THRESHOLD_EXCLUSION:g} from any threshold")
    print(f"|T-N_open| away from thresholds: max={max_error:.3e}, median={median_error:.3e}, mean={mean_error:.3e}")
    print(f"Physical conductance constant source: {PHYSICAL_CONSTANTS_SOURCE}")
    print(f"Wrote {RESULTS_DIR / 'stage4_multichannel.csv'}")
    print(f"Wrote {RESULTS_DIR / 'stage4_modes.csv'}")
    print(f"Wrote {RESULTS_DIR / 'stage4_metadata.json'}")
    print("Wrote figures/stage4_mode_thresholds.png, stage4_multichannel_transport.png, stage4_conductance.png")


if __name__ == "__main__":
    main()
