"""Run the finite clean-chain and single-barrier Stage 1 benchmark."""

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
from quantum_transport.greens import retarded_device_green_function
from quantum_transport.hamiltonian import tight_binding_hamiltonian
from quantum_transport.leads import lead_self_energy
from quantum_transport.transmission import landauer_transmission


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def _device_transmission(
    energy: float,
    onsite_energies: float | np.ndarray,
    *,
    n_sites: int,
    hopping: float,
    lead_onsite: float,
    contact_hopping: float,
) -> float:
    """Evaluate one energy through the public Part 2–5 functions."""
    device = tight_binding_hamiltonian(n_sites, onsite_energies, hopping)
    sigma_left = lead_self_energy(energy, lead_onsite, hopping, contact_hopping)
    sigma_right = lead_self_energy(energy, lead_onsite, hopping, contact_hopping)
    green = retarded_device_green_function(energy, device, sigma_left, sigma_right)
    gamma_left = left_broadening_matrix(n_sites, sigma_left)
    gamma_right = right_broadening_matrix(n_sites, sigma_right)
    return landauer_transmission(green, gamma_left, gamma_right)


def main() -> None:
    n_sites = 10
    hopping = 1.0
    epsilon_0 = 0.0
    contact_hopping = 1.0
    barrier_strength = 1.5
    barrier_site_index = 4  # zero-based site 5 of 10
    energy_min = epsilon_0 - 2.2 * hopping
    energy_max = epsilon_0 + 2.2 * hopping
    n_energies = 443
    energies = np.linspace(energy_min, energy_max, n_energies)

    clean_onsite: float | np.ndarray = epsilon_0
    barrier_onsite = np.full(n_sites, epsilon_0, dtype=np.float64)
    barrier_onsite[barrier_site_index] += barrier_strength

    clean = np.array(
        [
            _device_transmission(
                energy,
                clean_onsite,
                n_sites=n_sites,
                hopping=hopping,
                lead_onsite=epsilon_0,
                contact_hopping=contact_hopping,
            )
            for energy in energies
        ],
        dtype=np.float64,
    )
    barrier = np.array(
        [
            _device_transmission(
                energy,
                barrier_onsite,
                n_sites=n_sites,
                hopping=hopping,
                lead_onsite=epsilon_0,
                contact_hopping=contact_hopping,
            )
            for energy in energies
        ],
        dtype=np.float64,
    )

    if clean.shape != energies.shape or barrier.shape != energies.shape:
        raise RuntimeError("transmission arrays do not match the energy grid")
    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(barrier)):
        raise RuntimeError("transmission contains non-finite values")
    if np.any(clean < 0.0) or np.any(barrier < 0.0):
        raise RuntimeError("transmission contains negative values")
    np.testing.assert_allclose(clean, clean[::-1], atol=1e-10, rtol=0.0)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "stage1_transmission.csv"
    metadata_path = RESULTS_DIR / "stage1_metadata.json"
    figure_path = FIGURES_DIR / "stage1_transmission.png"

    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["energy", "clean_transmission", "barrier_transmission"])
        writer.writerows(zip(energies, clean, barrier))

    metadata = {
        "stage": "Stage 1 finite-device Landauer benchmark",
        "N": n_sites,
        "hopping_t": hopping,
        "lead_and_clean_device_onsite_epsilon_0": epsilon_0,
        "contact_hopping_t_c": contact_hopping,
        "barrier_strength_U": barrier_strength,
        "barrier_site_index_zero_based": barrier_site_index,
        "energy_range": [float(energy_min), float(energy_max)],
        "number_of_energy_points": n_energies,
        "units": "consistent tight-binding energy units with hopping magnitude t=1",
        "boundary_conditions": "open for the finite device; semi-infinite 1D leads",
        "device_hamiltonian_convention": "H_ii=epsilon_i; H_i,i+1=H_i+1,i=-t",
        "lead_band": [epsilon_0 - 2.0 * hopping, epsilon_0 + 2.0 * hopping],
        "lead_surface_green_function": "retarded branch; negative imaginary part inside the band",
        "lead_self_energy": "Sigma^r=t_c^2*g_s^r, applied to the first and last device sites",
        "device_green_function": "G^r=[E I-H_device-Sigma_L-Sigma_R]^-1; no additional broadening",
        "broadening": "Gamma=i(Sigma^r-Sigma^{r dagger})",
        "transmission": "T=Tr[Gamma_L G^r Gamma_R (G^r)^dagger]",
        "barrier_definition": "single onsite increase U at zero-based device site 4",
        "transmission_processing": "no normalization or upper clipping; only tiny roundoff negatives are set to zero",
    }
    with metadata_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)
        output_file.write("\n")

    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.plot(energies, clean, color="black", linewidth=1.8, label="Clean device")
    axis.plot(energies, barrier, color="#2864a5", linewidth=1.7, label="Single onsite barrier")
    axis.axvline(epsilon_0 - 2.0 * hopping, color="0.55", linestyle="--", linewidth=1.0)
    axis.axvline(epsilon_0 + 2.0 * hopping, color="0.55", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Energy E (tight-binding units)")
    axis.set_ylabel("Landauer transmission T(E)")
    axis.set_title("Finite 1D tight-binding transport (N=10)")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.savefig(figure_path, dpi=300)
    plt.close(figure)

    interior = np.abs(energies - epsilon_0) <= 1.5 * hopping
    outside = np.abs(energies - epsilon_0) > 2.0 * hopping
    min_clean = float(np.min(clean[interior]))
    max_deviation = float(np.max(np.abs(clean[interior] - 1.0)))
    outside_max = float(max(np.max(clean[outside]), np.max(barrier[outside])))
    print(f"Clean T minimum for |E-epsilon_0| <= 1.5t: {min_clean:.12g}")
    print(f"Clean max |T-1| in that region: {max_deviation:.12g}")
    for representative_energy in (0.0, 0.5, 1.0):
        index = int(np.argmin(np.abs(energies - representative_energy)))
        print(
            f"E={energies[index]:.3f}: clean T={clean[index]:.12g}, "
            f"barrier T={barrier[index]:.12g}"
        )
    print(f"Maximum transmission outside the lead band: {outside_max:.12g}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()


