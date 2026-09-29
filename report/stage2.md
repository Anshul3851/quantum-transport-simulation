# Stage 2: barriers, wells, resonant tunnelling, and LDOS

## Physical motivation

Stage 2 extends the clean and single-site barrier benchmark to spatially extended wells and symmetric double barriers. The study asks how these onsite structures change coherent transmission and whether selected transmission peaks are accompanied by density concentrated in the central well. These are calculations for a finite device connected to ideal one-dimensional leads; they are not comparisons with experiment or thermodynamic claims.

## Model and onsite profiles

The finite device has `N = 60` sites, hopping magnitude `t = 1`, background and lead onsite energy `epsilon_0 = 0`, and contact hopping `t_c = 1`. The open-boundary device convention is `H[i,i] = epsilon_i` and `H[i,i+1] = H[i+1,i] = -t`.

Four profiles were evaluated:

- Clean: all sites have onsite energy `0`.
- Single barrier: a one-site onsite increase `U = 1.5` at zero-based site 29.
- Finite well: depth `V = 0.8` on the half-open interval `[23, 37)`.
- Symmetric double barrier: two 3-site regions with onsite increase `1.5`, at sites 20–22 and 37–39, surrounding a 14-site well on `[23, 37)` with onsite energy `-0.8`.

The energy grid has 2,401 equally spaced points from `-1.98` to `1.98`, with step approximately `0.00165`. It stays inside the lead band edges at `-2` and `2` to avoid their zero-broadening singular points.

## Mathematical definitions

The ideal lead dispersion is `E(k) = epsilon_0 - 2 t cos(k)`, so its propagating band is `[epsilon_0 - 2t, epsilon_0 + 2t]`. The retarded surface Green function uses the negative-imaginary branch inside the band, and the self-energy is `Sigma^r = t_c^2 g_s^r` at the device endpoints.

The device Green functions, broadenings, and transmission are

\[
G^r=[E I-H_{\mathrm{device}}-\Sigma_L^r-\Sigma_R^r]^{-1},\qquad G^a=(G^r)^\dagger,
\]
\[
\Gamma_{L/R}=i(\Sigma_{L/R}^r-\Sigma_{L/R}^{r\dagger}),\qquad
T(E)=\operatorname{Tr}[\Gamma_L G^r\Gamma_R G^a].
\]

The local and total densities of states are

\[
\rho_i(E)=-\frac{1}{\pi}\operatorname{Im}G^r_{ii}(E),\qquad
\rho(E)=-\frac{1}{\pi}\operatorname{Im}\operatorname{Tr}G^r(E)=\sum_i\rho_i(E).
\]

All quantities use consistent tight-binding energy units. No extra imaginary energy shift is added. The transmission routine tolerates only roundoff-scale imaginary or negative residues; it does not normalize or cap transmission.

## Numerical observations

**Clean benchmark.** In the safe interior `|E| <= 1.5t`, the minimum clean transmission was `1.0`, with maximum absolute deviation from one of `1.4322e-14`. This is consistent with the analytical matched-device expectation for energies away from the band edges.

At the grid energy nearest `E = 0.5`, namely `E = 0.49995`, the clean, single-barrier, and finite-well transmissions were `1.000000`, `0.625003`, and `0.940519`, respectively. Over all four structures and sampled energies, the maximum transmission was `1.0`; the minimum was non-negative.

**Detected double-barrier transmission peaks.** A fixed-window local-maximum rule was used: estimated prominence at least `0.02`, separation at least 20 grid points, and energies satisfying `|E| < 1.9t`. Among the ten saved peaks, the three highest transmissions were:

| Energy | Transmission | Total DOS | Central-well DOS fraction |
|---:|---:|---:|---:|
| `0.39105` | `0.99999987` | `11.4757` | `0.2916` |
| `-0.73095` | `0.99999105` | `83.6340` | `0.8378` |
| `0.80025` | `0.99997846` | `16.1154` | `0.4935` |

**Resonance interpretation.** The highest-transmission peak by itself (`E = 0.39105`) has only about 29% of its total DOS in the central well, so it is not labeled a well-localized resonance from transmission alone. The barrier-region onsite energy is `1.5`, giving its local band bottom `1.5 - 2t = -0.5`. The detected peak at `E = -0.73095` lies below this edge, where propagation in the barrier material is evanescent, while remaining inside the central well's local band. At this peak, the transmission is `0.999991`, total DOS is `83.6340`, and `83.8%` of the DOS lies on the central well sites.

At nearby detuned grid energies `-0.77055` and `-0.69135`, total DOS was `8.2974` and `7.9359`; the central-well fractions were `8.6%` and `11.1%`. The increased DOS and spatial concentration at the selected peak support interpreting it as a well-associated resonant tunnelling state in this finite model. The peak energy is specific to these chosen parameters and grid; it is not a universal resonance energy.

The maximum absolute difference between total DOS calculated from the Green-function trace and from summing site LDOS was `2.842e-14` over the scan.

## Validation

The Stage 2 tests check profile construction and validation, mirror symmetry, transmission reciprocity under profile reversal, LDOS and total-DOS identities, clean-chain transmission, single-channel transmission bounds, barrier suppression, deterministic calculations, and a representative double-barrier resonance with enhanced central-well LDOS.

Python compilation passed. The focused Stage 2 suite passed with **15 tests**; the full Stage 1 and Stage 2 suite passed with **110 tests**. The simulation script completed, and the saved CSV values were checked for shape and finite values. The four PNGs were decoded and checked for valid image dimensions. Jupyter is unavailable in the environment; the notebook JSON was parsed and all nine code cells were executed directly in Python.

## Peak detection and implementation notes

SciPy was not installed in the runtime. Rather than install a dependency, the analysis script uses a small NumPy local-maximum detector with a fixed 80-point window on each side for its prominence estimate. The threshold, separation, window, and safe-band filter are recorded in the metadata. The LDOS resonance is selected as the highest-transmission detected peak below the barrier-region band bottom; its central-well DOS fraction is then measured independently.

## Limitations

The model is coherent, non-interacting, one-dimensional, and single-channel, with ideal semi-infinite leads and scalar endpoint couplings. The potential profiles are static and deterministic. Resonance positions and widths depend on the finite device size, barrier/well choices, and energy-grid resolution. The study does not include disorder averages, localization, dephasing, interactions, finite-temperature effects, self-consistent potentials, higher-dimensional systems, or experimental comparison.

## Generated Stage 2 artifacts

- `results/stage2_transmission.csv`
- `results/stage2_dos.csv`
- `results/stage2_resonances.csv`
- `results/stage2_resonance_ldos.csv`
- `results/stage2_metadata.json`
- `figures/stage2_potential_profiles.png`
- `figures/stage2_transmission.png`
- `figures/stage2_dos.png`
- `figures/stage2_resonance_ldos.png`
- `notebooks/02_resonant_tunnelling.ipynb`
