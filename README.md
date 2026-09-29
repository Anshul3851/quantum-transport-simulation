# Quantum Transport in Tight-Binding Systems

A computational study of coherent quantum transport in finite tight-binding systems using Green functions and the Landauer formalism. The model uses nearest-neighbour hopping with off-diagonal elements -t, open device boundaries, and retarded semi-infinite lead self-energies. Energies are reported in hopping units unless stated otherwise.

## Project progression

1. **Finite 1D benchmark** — matched-chain transmission and a single onsite barrier.
2. **Resonance and local density of states** — finite wells, double barriers, and resonant tunnelling.
3. **Disorder and length dependence** — seeded onsite-disorder ensembles with mean, median, typical, and mean-log transmission summaries.
4. **Clean multi-channel strip** — 2D hard-wall transverse modes and conductance versus open-channel count.
5. **Quantum point contact** — a smooth gated constriction, channel filtering, and local spectral density.
6. **Final synthesis** — saved-data robustness checks and cross-stage overview; no new transport calculation.

## Core conventions

The finite-device Hamiltonian is

H_ii = epsilon_i,    H_(i,i+1) = H_(i+1,i) = -t,    t > 0.

The 1D lead dispersion is E(k) = epsilon_0 - 2t cos(k), with propagating band epsilon_0 ± 2t. The retarded lead self-energy is Sigma^r = t_c^2 g_s^r, and the device Green function is G^r = [E I - H - Sigma_L^r - Sigma_R^r]^-1. The broadening convention is Gamma = i(Sigma^r - Sigma^(r dagger)); transmission is evaluated using the Landauer/Caroli expression. No arbitrary energy broadening is added, and transmissions are not normalized or upper-clipped; routines apply only their documented roundoff handling.

## Findings and validation

The saved clean matched-chain and matched-strip benchmarks reproduce their open-channel counts to floating-point precision away from band or mode thresholds. The finite 1D barrier suppresses transmission, the double barrier has a resolved resonance with enhanced central-well LDOS, and disorder lowers transmission statistics as disorder and device length increase in the sampled ensembles. The gated QPC shows monotone transmission suppression across its five saved gate scans. These are finite-grid, finite-size numerical results; they do not establish exact conductance quantization or thermodynamic localization behavior.

The full pytest suite checks the Hamiltonian, lead self-energy, Green functions, broadening, transmission, disorder, multi-channel modes, QPC, and Stage 6 saved-data synthesis. Stage 3 uses recorded seeds and finite ensembles; Stage 4 excludes a documented threshold neighborhood for its channel comparison. QPC LDOS is a local spectral quantity, not current density.

## Project material

- Detailed methods, measured values, and limitations: [report/stage1.md](report/stage1.md) through [report/stage6.md](report/stage6.md).
- Final cross-stage summary: [results/final_stage_summary.csv](results/final_stage_summary.csv) and [results/final_stage_summary.json](results/final_stage_summary.json).
- Overview figure: [figures/final_project_overview.png](figures/final_project_overview.png).
- Stage walkthroughs: [notebooks/](notebooks/), including [notebooks/06_final_synthesis.ipynb](notebooks/06_final_synthesis.ipynb).

The final report distinguishes sampled numerical observations from analytic reference behavior and describes the finite-size and sampling limits of each stage.
