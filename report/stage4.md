# Stage 4: 2D multi-channel transport and conductance quantization

## Model and site ordering

The device is a finite rectangular strip of length `Lx = 30` and width `Ly = 4` with `120` sites. Sites are indexed by `index(x,y) = x*Ly + y`, so each longitudinal slice is contiguous. All nearest-neighbour matrix elements in both x and y directions are `-t`, with `t = 1` and onsite energy `epsilon_0 = 0`. The device has open boundaries along both axes. The leads are semi-infinite along x with the same hard-wall transverse boundaries and parameters; the contact hopping is `t_c = 1`. The device and leads are therefore perfectly matched.

## Transverse modes and thresholds

The transverse slice Hamiltonian is an open chain with eigenenergies

`epsilon_n = epsilon_0 - 2 t cos(n pi/(Ly+1))`, for `n=1,...,Ly`.

For `Ly=4`, the numerical energies are `[-1.61803399, -0.61803399, 0.61803399, 1.61803399]`, matching the analytic open-chain spectrum. Each mode has longitudinal dispersion `E_n(k) = epsilon_n - 2t cos(k)` and propagates when `|E-epsilon_n| <= 2t`. Its lower edge is the opening threshold and upper edge is where it ceases to propagate.

| Mode | `epsilon_n` | Opening threshold `epsilon_n-2t` | Upper edge `epsilon_n+2t` |
|---:|---:|---:|---:|
| 1 | -1.618034 | -3.618034 | 0.381966 |
| 2 | -0.618034 | -2.618034 | 1.381966 |
| 3 | 0.618034 | -1.381966 | 2.618034 |
| 4 | 1.618034 | -0.381966 | 3.618034 |

These analytic mode energies and band edges define `N_open(E)` independently of the numerical transmission calculation.

## Semi-infinite lead and transport construction

Diagonalize the transverse slice Hamiltonian as `H_y = U diag(epsilon_n) U^dagger`. For each transverse mode, evaluate the Stage 3 semi-infinite 1D retarded surface Green function `g_n^r(E)` using onsite `epsilon_n` and longitudinal hopping `t`. Transform to the transverse site basis with `g_surface = U diag(g_n^r) U^dagger`. This is the exact semi-infinite strip surface construction; no finite lead approximation is used.

The lead-device coupling is `-t_c` times the transverse identity, so the boundary-slice self-energy is `Sigma_slice = t_c^2 g_surface`. Embed this block only on the first or last longitudinal slice to obtain `Sigma_L` and `Sigma_R`. The dense retarded device Green function is calculated by solving `(E I - H_device - Sigma_L - Sigma_R) G^r = I`. Broadening uses `Gamma_alpha = i(Sigma_alpha - Sigma_alpha^dagger)`, and transmission is the unnormalized Caroli trace `Tr[Gamma_L G^r Gamma_R (G^r)^dagger]`.

Dimensionless conductance is `g=T`. Physical conductance is `G=(2e^2/h)T`; the factor two follows the requested spin-degenerate conductance quantum. SciPy was not installed in the execution environment, so the implementation used the documented exact SI defining constants as a fallback, yielding `2e^2/h = 7.748091729863649e-5 S`. No temperature dependence is included.

## Numerical transmission versus analytic channel count

The script evaluates 751 energies uniformly from `-3.75` to `3.75`, producing open-channel counts from zero to four and visible steps at the analytic mode edges. Of those energy samples, 671 lie at least `0.05` tight-binding energy units from every mode threshold. For those samples, the maximum absolute difference `|T-N_open|` is `7.11e-15`; the median (typical) absolute difference is `8.88e-16` and the mean is `1.20e-15`. The maximum across the full sweep, including points close to thresholds, is `2.18e-14` for this particular grid. Transmission is computed directly: it is not clipped or normalized to the channel count.

The analytic prediction is the mode count from the transverse spectrum; `T(E)` is the numerical finite-device Green-function/Landauer result. Their near-machine-precision agreement away from thresholds is the expected clean matched-strip behavior. The analysis does not use this agreement to assert exact quantization at threshold energies, where the lead square-root branch changes rapidly and matrix conditioning can be delicate.

## Outputs and numerical limitations

- `results/stage4_multichannel.csv`: energy, numerical transmission, `N_open`, dimensionless conductance, physical conductance, and distance to the nearest threshold.
- `results/stage4_modes.csv`: transverse eigenenergies, mode vectors, and lower/upper thresholds.
- `results/stage4_metadata.json`: model, equations, constants source, threshold exclusion, and measured comparison statistics.
- `figures/stage4_mode_thresholds.png`, `stage4_multichannel_transport.png`, and `stage4_conductance.png`.

The full device Green matrix is dense and the calculation is intended for moderate strip sizes; the benchmark has only 120 device sites. The model is clean, coherent, single-particle, and non-interacting. It omits disorder, finite temperature, magnetic fields, spin-orbit coupling, interactions, and constrictions. This finite-width channel-count check is not a study of localization or a thermodynamic limit.
