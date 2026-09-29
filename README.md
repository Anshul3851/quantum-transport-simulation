# Quantum Transport in Tight-Binding Systems

## Stage 1: finite-device transport

This project studies coherent, single-particle transport through a finite one-dimensional tight-binding device connected to semi-infinite one-dimensional leads. The Stage 1 question is how the device's onsite energies affect its energy-dependent Landauer transmission, first for a clean matched chain and then for a chain with a single onsite barrier.

All energies use consistent tight-binding units, with the hopping magnitude set to `t = 1` in the benchmark. The device has open boundaries. Its Hamiltonian convention is

\[
H_{ii}=\epsilon_i,\qquad H_{i,i+1}=H_{i+1,i}=-t,\quad t>0.
\]

## Leads and Green functions

Each lead is a semi-infinite nearest-neighbour chain with onsite energy `epsilon_0` and hopping `-t`. Its bulk dispersion is

\[
E(k)=\epsilon_0-2t\cos(k),
\]

with propagating band `[epsilon_0 - 2t, epsilon_0 + 2t]`. The retarded surface Green function uses the branch with negative imaginary part inside this band. The lead self-energy is `Sigma^r = t_c^2 g_s^r`, where `t_c` is the device-lead contact hopping.

For scalar left and right self-energies applied at the first and last sites, respectively, the device Green function is

\[
G^r(E)=\left[E I-H_{\rm device}-\Sigma_L^r(E)-\Sigma_R^r(E)\right]^{-1},\qquad G^a=(G^r)^\dagger.
\]

The broadening is `Gamma = i(Sigma^r - Sigma^{r dagger})`; for a scalar self-energy this is `-2 Im(Sigma^r)`. The transmission is calculated with the Caroli/Landauer expression

\[
T(E)=\operatorname{Tr}\left[\Gamma_L G^r \Gamma_R G^a\right].
\]

The Green matrix is computed by solving a linear system against the identity. No additional imaginary energy broadening is inserted. The transmission routine returns a real scalar, rejects significant imaginary or negative results, and only treats values within its stated roundoff tolerance as numerical noise.

## Stage 1 benchmarks

The clean benchmark uses `N = 10`, `t = 1`, `epsilon_0 = 0`, and `t_c = 1`. The device and leads are identical. Analytically, a clean matched chain is expected to have unit transmission in the propagating band away from its edges. The benchmark evaluates 443 energies from `-2.2` to `2.2`, so it spans the band and nearby evanescent regions without sampling the exactly singular band-edge points.

The barrier benchmark adds an onsite energy `U = 1.5` at zero-based device site 4 (the fifth site). It uses the same leads and energy grid. The generated CSV contains both transmission curves; a JSON file records model and numerical conventions. The figure is a finite-device benchmark and does not establish a thermodynamic result.

## Validation

The pytest suite checks the Hamiltonian, retarded lead branch and self-energy, device Green-function equations, broadening placement and sign, and transmission behavior. Stage 1 checks include a matched-chain benchmark, barrier suppression, deterministic evaluation, and zero transmission outside the propagating band. The benchmark script also checks finite data and clean-chain energy-reflection symmetry before writing its outputs.

## Current limitations

The transport models are coherent and non-interacting, with ideal semi-infinite nearest-neighbour leads. Stages 1?3 study 1D devices, including static onsite disorder; Stage 4 adds a clean finite-width 2D strip. The project does not include inelastic scattering, finite temperature, magnetic fields, spin-orbit coupling, interactions, or self-consistent potentials. Exact band-edge energies are excluded from the benchmark grid because the lead broadening vanishes there and the effective matrix can be singular; no artificial broadening is added to regularize them.

## Stage 2: barriers, wells, and local density of states

Stage 2 extends the clean-chain and single-barrier calculation to finite wells and symmetric double barriers. It compares their transmission spectra, examines resonant transmission together with the total and site-resolved density of states, and records the parameter choices in `results/stage2_metadata.json`. The generated figures and numerical tables are stored under `figures/` and `results/`; the walkthrough is in `notebooks/02_resonant_tunnelling.ipynb`.

## Stage 3: disorder ensembles and length dependence

Stage 3 samples independent onsite profiles `epsilon_i = epsilon_0 + w_i`, with `w_i ~ Uniform(-W/2, W/2)`, and measures coherent transmission for finite 1D devices. The reproducible run uses `N = 40`, `t = epsilon_0 = 0`, `t_c = 1`, 100 realizations per disorder/energy condition, and a recorded master seed. It reports arithmetic mean, median, sample standard deviation, standard error, mean log-transmission, and typical transmission, plus a separate length-dependence study for `N = 10, 20, 40, 80`.

Run `python scripts/run_transport_stage3.py` from the project root to regenerate Stage 3 tables, metadata, and figures. The analysis and finite-size interpretation are documented in `report/stage3.md`; the walkthrough is `notebooks/03_disorder_and_localization.ipynb`. This finite ensemble demonstrates disorder-dependent suppression and length trends for the sampled conditions; it is not a thermodynamic localization-length determination.


## Stage 4: 2D multi-channel transport

Stage 4 extends the 1D framework to a finite-width 2D tight-binding strip with open hard-wall transverse boundaries and multiple propagating modes. The clean matched-strip benchmark compares numerical Landauer transmission with the analytic number of open transverse channels and reports conductance as both `g = T` and `G = (2e^2/h) T`.

Run `python scripts/run_transport_stage4.py` from the project root. Stage 4 data and metadata are saved as `results/stage4_multichannel.csv`, `stage4_modes.csv`, and `stage4_metadata.json`; figures are saved under `figures/`. The derivation, measured comparison, and finite-size numerical limitations are in `report/stage4.md`, with an API-based walkthrough in `notebooks/04_multichannel_transport.ipynb`.


## Stage 5: quantum point contact and channel filtering

Stage 5 introduces a smooth, symmetric quantum point contact in the finite-width 2D strip and studies how gate strength filters transverse channels. It compares transmission with clean-lead and local constriction mode information, reports dimensionless and physical conductance, and visualizes the unnormalized spatial LDOS. The analysis and limitations are in `report/stage5.md`; the walkthrough is `notebooks/05_quantum_point_contact.ipynb`.
