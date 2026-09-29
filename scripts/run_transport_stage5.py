"""Run a clean-lead 2D quantum point contact channel-filtering study."""

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

from quantum_transport.density import local_density_of_states
from quantum_transport.modes import open_mode_count, transverse_modes
from quantum_transport.qpc import quantum_point_contact_potential
from quantum_transport.two_dimensional import rectangular_device_hamiltonian
from quantum_transport.two_dimensional_leads import two_dimensional_self_energy_matrices
from quantum_transport.two_dimensional_transport import (
    PHYSICAL_CONSTANTS_SOURCE,
    broadening_matrix,
    conductance_quantum,
    multichannel_transmission,
    physical_conductance,
    retarded_2d_device_green_function,
)
from quantum_transport.hamiltonian import tight_binding_hamiltonian


# Stage 5 parameters; energies are in units of t=1.
LX = 40
LY = 6
HOPPING = 1.0
ONSITE_ENERGY = 0.0
CONTACT_HOPPING = 1.0
GATE_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
LONGITUDINAL_SIGMA = 6.0
TRANSVERSE_STRENGTH = 1.5
ENERGY_GRID = np.linspace(-3.6, -1.2, 121)
GATE_SCAN_ENERGIES = [-3.5, -3.0, -2.5, -2.0, -1.5]
BASELINE_ENERGIES = [-3.5, -3.0, -2.2, -1.3]
LDOS_ENERGY = -2.0
LDOS_GATE_STRENGTHS = [0.25, 0.75]

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_potential(gate: float) -> np.ndarray:
    return quantum_point_contact_potential(
        LX,
        LY,
        gate,
        longitudinal_sigma=LONGITUDINAL_SIGMA,
        transverse_strength=TRANSVERSE_STRENGTH,
        onsite_background=ONSITE_ENERGY,
    )


def _device_hamiltonian(potential: np.ndarray) -> np.ndarray:
    return rectangular_device_hamiltonian(LX, LY, potential, HOPPING)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    lead_mode_energies, _ = transverse_modes(LY, ONSITE_ENERGY, HOPPING)
    open_channels = open_mode_count(ENERGY_GRID, lead_mode_energies, HOPPING)
    device_by_gate = {
        gate: _device_hamiltonian(_make_potential(gate)) for gate in GATE_STRENGTHS
    }
    transmissions = np.empty((len(GATE_STRENGTHS), ENERGY_GRID.size), dtype=np.float64)
    conductances = np.empty_like(transmissions)
    local_mode_energies: dict[float, np.ndarray] = {}
    local_open_counts: dict[float, np.ndarray] = {}
    center_x = (LX - 1) // 2

    for gate_index, gate in enumerate(GATE_STRENGTHS):
        potential = _make_potential(gate)
        central_slice = tight_binding_hamiltonian(
            LY, ONSITE_ENERGY + potential[center_x], HOPPING
        )
        local_mode_energies[gate] = np.linalg.eigvalsh(central_slice)
        local_open_counts[gate] = open_mode_count(
            ENERGY_GRID, local_mode_energies[gate], HOPPING
        )

    for energy_index, energy in enumerate(ENERGY_GRID):
        sigma_left, sigma_right = two_dimensional_self_energy_matrices(
            float(energy), LX, LY, ONSITE_ENERGY, HOPPING, CONTACT_HOPPING
        )
        gamma_left = broadening_matrix(sigma_left)
        gamma_right = broadening_matrix(sigma_right)
        for gate_index, gate in enumerate(GATE_STRENGTHS):
            green = retarded_2d_device_green_function(
                float(energy), device_by_gate[gate], sigma_left, sigma_right
            )
            value = multichannel_transmission(green, gamma_left, gamma_right)
            transmissions[gate_index, energy_index] = value
            conductances[gate_index, energy_index] = physical_conductance(value)

    if not np.all(np.isfinite(transmissions)):
        raise RuntimeError("QPC transmission contains non-finite values")
    if np.any(transmissions < 0.0):
        raise RuntimeError("QPC transmission contains negative values")
    incoming_excess = transmissions - open_channels[None, :]
    maximum_channel_excess = float(np.max(incoming_excess))
    if maximum_channel_excess > 1e-8:
        raise RuntimeError(
            "QPC transmission exceeds the number of incoming lead channels: "
            f"maximum excess={maximum_channel_excess:.3e}"
        )

    transmission_rows = []
    for gate_index, gate in enumerate(GATE_STRENGTHS):
        for energy_index, energy in enumerate(ENERGY_GRID):
            transmission_rows.append({
                "energy": float(energy),
                "gate_strength": gate,
                "transmission_T": float(transmissions[gate_index, energy_index]),
                "N_open_lead": int(open_channels[energy_index]),
                "N_local_open_centre": int(local_open_counts[gate][energy_index]),
                "dimensionless_conductance_g": float(transmissions[gate_index, energy_index]),
                "conductance_S": float(conductances[gate_index, energy_index]),
                "local_subband_energies": ";".join(
                    f"{value:.12g}" for value in local_mode_energies[gate]
                ),
            })
    _write_csv(
        RESULTS_DIR / "stage5_qpc_transmission.csv",
        ["energy", "gate_strength", "transmission_T", "N_open_lead",
         "N_local_open_centre", "dimensionless_conductance_g", "conductance_S",
         "local_subband_energies"],
        transmission_rows,
    )

    gate_scan_rows = []
    for energy in GATE_SCAN_ENERGIES:
        index = int(np.argmin(np.abs(ENERGY_GRID - energy)))
        if abs(ENERGY_GRID[index] - energy) > 1e-12:
            raise RuntimeError(f"Gate scan energy {energy} is not on the energy grid")
        for gate_index, gate in enumerate(GATE_STRENGTHS):
            gate_scan_rows.append({
                "energy": energy,
                "gate_strength": gate,
                "transmission_T": float(transmissions[gate_index, index]),
                "N_open_lead": int(open_channels[index]),
                "N_local_open_centre": int(local_open_counts[gate][index]),
                "conductance_S": float(conductances[gate_index, index]),
            })
    _write_csv(
        RESULTS_DIR / "stage5_qpc_gate_scan.csv",
        ["energy", "gate_strength", "transmission_T", "N_open_lead",
         "N_local_open_centre", "conductance_S"],
        gate_scan_rows,
    )

    baseline_errors = []
    for energy in BASELINE_ENERGIES:
        index = int(np.argmin(np.abs(ENERGY_GRID - energy)))
        predicted = int(open_channels[index])
        numerical = float(transmissions[0, index])
        baseline_errors.append({
            "energy": energy,
            "N_open": predicted,
            "T_clean": numerical,
            "absolute_error": abs(numerical - predicted),
        })
    baseline_max_error = max(row["absolute_error"] for row in baseline_errors)
    if baseline_max_error > 1e-8:
        raise RuntimeError(
            f"Clean gate=0 baseline failed Stage 4 channel comparison: {baseline_max_error:.3e}"
        )

    ldos_arrays: dict[float, np.ndarray] = {}
    ldos_rows = []
    for gate in LDOS_GATE_STRENGTHS:
        potential = _make_potential(gate)
        hamiltonian = _device_hamiltonian(potential)
        sigma_left, sigma_right = two_dimensional_self_energy_matrices(
            LDOS_ENERGY, LX, LY, ONSITE_ENERGY, HOPPING, CONTACT_HOPPING
        )
        green = retarded_2d_device_green_function(
            LDOS_ENERGY, hamiltonian, sigma_left, sigma_right
        )
        density = local_density_of_states(green).reshape(LX, LY)
        if not np.all(np.isfinite(density)) or np.min(density) < -1e-10:
            raise RuntimeError(f"LDOS contains non-finite or significantly negative values for gate={gate}")
        ldos_arrays[gate] = density
        for x in range(LX):
            for y in range(LY):
                ldos_rows.append({
                    "x": x,
                    "y": y,
                    "gate_strength": gate,
                    "energy": LDOS_ENERGY,
                    "ldos_per_energy": float(density[x, y]),
                })
    _write_csv(
        RESULTS_DIR / "stage5_qpc_ldos.csv",
        ["x", "y", "gate_strength", "energy", "ldos_per_energy"],
        ldos_rows,
    )

    # Store the plotted potential profile as a separate derived table.
    potential_rows = []
    plotted_potential = _make_potential(max(GATE_STRENGTHS))
    for x in range(LX):
        for y in range(LY):
            potential_rows.append({"x": x, "y": y, "onsite_energy": float(plotted_potential[x, y])})
    _write_csv(
        RESULTS_DIR / "stage5_qpc_potential.csv",
        ["x", "y", "onsite_energy"], potential_rows
    )

    gate_scan_monotonic = {}
    for energy in GATE_SCAN_ENERGIES:
        index = int(np.argmin(np.abs(ENERGY_GRID - energy)))
        values = transmissions[:, index]
        gate_scan_monotonic[str(energy)] = bool(np.all(np.diff(values) <= 1e-10))

    metadata = {
        "stage": "Stage 5: quantum point contact and channel filtering",
        "geometry": {"Lx": LX, "Ly": LY, "site_ordering": "index(x,y)=x*Ly+y",
                     "boundaries": "open device boundaries; hard-wall transverse semi-infinite leads"},
        "tight_binding": {"hopping_t": HOPPING, "onsite_epsilon_0": ONSITE_ENERGY,
                          "contact_hopping_t_c": CONTACT_HOPPING,
                          "convention": "nearest-neighbour x/y Hamiltonian elements are -t"},
        "qpc_potential": {
            "equation": "epsilon(x,y)=epsilon_0+Vg*exp(-(x-xc)^2/(2*sigma_x^2))*(1+alpha*((y-yc)/y_scale)^2)",
            "gate_strengths": GATE_STRENGTHS,
            "longitudinal_sigma_sites": LONGITUDINAL_SIGMA,
            "transverse_strength_alpha": TRANSVERSE_STRENGTH,
            "center_x": (LX - 1) / 2.0,
            "center_y": (LY - 1) / 2.0,
            "y_scale": max((LY - 1) / 2.0, 0.5),
            "mirror_symmetric_in_x_and_y": True,
        },
        "energy_grid": {"minimum": float(ENERGY_GRID[0]), "maximum": float(ENERGY_GRID[-1]),
                        "points": int(ENERGY_GRID.size), "spacing": float(ENERGY_GRID[1]-ENERGY_GRID[0]),
                        "units": "tight-binding energy units"},
        "gate_scan_energies": GATE_SCAN_ENERGIES,
        "lead_transverse_mode_energies": [float(value) for value in lead_mode_energies],
        "lead_lower_mode_thresholds": [float(value - 2.0 * HOPPING) for value in lead_mode_energies],
        "local_centre_subbands_by_gate": {
            str(gate): {
                "transverse_energies": [float(value) for value in local_mode_energies[gate]],
                "longitudinal_lower_thresholds": [float(value - 2.0 * HOPPING) for value in local_mode_energies[gate]],
            }
            for gate in GATE_STRENGTHS
        },
        "local_channel_interpretation": "N_local_open_centre counts central-slice eigenmodes satisfying |E-epsilon_local,n|<=2t; it is an energetic local-mode indicator, not a conserved channel count of the nonuniform constriction.",
        "conductance": {"dimensionless": "g=T", "physical": "G=(2e^2/h)T in siemens",
                        "conductance_quantum_S": conductance_quantum(),
                        "constants_source": PHYSICAL_CONSTANTS_SOURCE},
        "baseline_clean_stage4_check": {
            "sampled_energies": baseline_errors,
            "maximum_absolute_T_minus_N_open": baseline_max_error,
            "tolerance": 1e-8,
        },
        "transmission_bound_check": {"maximum_T_minus_N_open_across_all_gates_and_energies": maximum_channel_excess,
                                     "tolerance": 1e-8, "transmission_clipped": False},
        "gate_scan_nonincreasing_at_selected_energies": gate_scan_monotonic,
        "ldos": {"energy": LDOS_ENERGY, "gate_strengths": LDOS_GATE_STRENGTHS,
                 "definition": "rho_i(E)=-Im(G^r_ii)/pi from Stage 2; raw values, no normalization", 
                 "data_file": "results/stage5_qpc_ldos.csv"},
        "solver": "Dense linear solve for a 240-site device; no artificial energy broadening.",
    }
    (RESULTS_DIR / "stage5_qpc_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    # Potential map for the strongest configured gate.
    figure, axis = plt.subplots(figsize=(8, 3.8))
    image = axis.imshow(plotted_potential.T, origin="lower", aspect="auto",
                        extent=(-0.5, LX - 0.5, -0.5, LY - 0.5), cmap="magma")
    axis.set(xlabel="Longitudinal slice x", ylabel="Transverse site y",
             title=f"Smooth QPC onsite potential, Vg={max(GATE_STRENGTHS):g}")
    figure.colorbar(image, ax=axis, label="Onsite energy (t units)")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "stage5_qpc_potential.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    for index, gate in enumerate(GATE_STRENGTHS):
        axis.plot(ENERGY_GRID, transmissions[index], label=f"Vg={gate:g}")
    axis.step(ENERGY_GRID, open_channels, where="mid", color="black", linestyle=":",
              linewidth=1.2, label=r"Lead $N_{open}$")
    axis.set(xlabel="Energy (tight-binding units)", ylabel="Transmission T(E)",
             title="QPC channel filtering")
    axis.legend(ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "stage5_qpc_transmission.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    for index, gate in enumerate(GATE_STRENGTHS):
        axes[0].plot(ENERGY_GRID, transmissions[index], label=f"Vg={gate:g}")
        axes[1].plot(ENERGY_GRID, conductances[index] * 1e6, label=f"Vg={gate:g}")
    axes[0].set_ylabel(r"Dimensionless $g=T$")
    axes[1].set_ylabel(r"Conductance ($\mu$S)")
    axes[1].set_xlabel("Energy (tight-binding units)")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(ncol=2)
    figure.suptitle(r"QPC conductance: $G=(2e^2/h)T$")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "stage5_qpc_conductance.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    for energy in GATE_SCAN_ENERGIES:
        index = int(np.argmin(np.abs(ENERGY_GRID - energy)))
        axis.plot(GATE_STRENGTHS, transmissions[:, index], marker="o",
                  label=f"E={energy:g}, Nopen={int(open_channels[index])}")
    axis.set(xlabel="Gate strength Vg (t units)", ylabel="Transmission T",
             title="Transmission versus gate strength")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "stage5_qpc_gate_scan.png", dpi=180)
    plt.close(figure)

    # Both LDOS panels use the same colour normalization; stored values remain raw.
    maximum_ldos = max(float(np.max(values)) for values in ldos_arrays.values())
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, constrained_layout=True)
    for axis, gate in zip(axes, LDOS_GATE_STRENGTHS):
        image = axis.imshow(ldos_arrays[gate].T, origin="lower", aspect="auto",
                            extent=(-0.5, LX - 0.5, -0.5, LY - 0.5),
                            cmap="viridis", vmin=0.0, vmax=maximum_ldos)
        axis.set(xlabel="Longitudinal slice x", title=f"Vg={gate:g}")
    axes[0].set_ylabel("Transverse site y")
    figure.suptitle(f"Raw site LDOS at E={LDOS_ENERGY:g} (shared colour scale)")
    figure.colorbar(image, ax=axes, label=r"$-Im(G^r_{ii})/\pi$ (per energy)", shrink=0.86)
    figure.savefig(FIGURES_DIR / "stage5_qpc_ldos.png", dpi=180)
    plt.close(figure)

    e_index = int(np.argmin(np.abs(ENERGY_GRID - (-2.0))))
    print(f"QPC geometry: Lx={LX}, Ly={LY}, sites={LX * LY}")
    print(f"Energy range: {ENERGY_GRID[0]:g} to {ENERGY_GRID[-1]:g} ({ENERGY_GRID.size} points)")
    print("Gate strengths:", GATE_STRENGTHS)
    print("Lead transverse energies:", ", ".join(f"{value:.6f}" for value in lead_mode_energies))
    print(f"Clean baseline max |T-N_open| at selected energies: {baseline_max_error:.3e}")
    print(f"Maximum T-N_open over all gate/energy points: {maximum_channel_excess:.3e}")
    print(f"At E=-2: N_open={int(open_channels[e_index])}; gate scan T=" +
          ", ".join(f"{value:.6g}" for value in transmissions[:, e_index]))
    for gate in LDOS_GATE_STRENGTHS:
        print(f"LDOS gate={gate:g}: min={np.min(ldos_arrays[gate]):.6g}, max={np.max(ldos_arrays[gate]):.6g}")
    print(f"Wrote results/stage5_qpc_transmission.csv, stage5_qpc_gate_scan.csv, stage5_qpc_metadata.json")
    print("Wrote Stage 5 QPC potential, transmission, conductance, gate-scan, and LDOS figures")


if __name__ == "__main__":
    main()
