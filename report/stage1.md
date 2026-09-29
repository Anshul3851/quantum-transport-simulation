# Stage 1: finite tight-binding Landauer benchmark

## Model

The finite device is a one-dimensional open chain with `N = 10`, nearest-neighbour hopping magnitude `t = 1`, and uniform clean onsite energy `epsilon_0 = 0`. The matrix convention is `H[i,i] = epsilon_i` and `H[i,i+1] = H[i+1,i] = -t`. Each semi-infinite lead has the same onsite energy and hopping. The device-lead contact hopping is `t_c = 1`.

The barrier case adds `U = 1.5` to zero-based site 4 (the fifth device site), leaving the other onsite energies unchanged.

## Equations and numerical conventions

The infinite lead dispersion is `E(k) = epsilon_0 - 2 t cos(k)`, with propagating band `[epsilon_0 - 2t, epsilon_0 + 2t]`. The retarded surface Green function uses a negative imaginary part inside the band and the decaying real branch outside. The self-energy is `Sigma^r = t_c^2 g_s^r`.

The device Green functions and broadenings are

\[
G^r=[E I-H_{\mathrm{device}}-\Sigma_L^r-\Sigma_R^r]^{-1},\quad
G^a=(G^r)^\dagger,\quad
\Gamma_{L/R}=i(\Sigma_{L/R}^r-\Sigma_{L/R}^{r\dagger}).
\]

Each lead self-energy and broadening acts only at its endpoint. Transmission uses

\[
T(E)=\operatorname{Tr}[\Gamma_L G^r\Gamma_R G^a].
\]

The Green matrix is obtained with `numpy.linalg.solve` against the identity, rather than an explicit matrix inverse. There is no added imaginary energy shift. The scan contains 443 equally spaced energies from `-2.2` to `2.2`; exact band edges are not sampled. The output data preserve the unnormalized transmission values.

## Clean benchmark

**Analytically known expectation:** a finite device identical to ideal leads has unit transmission through the propagating band, away from the band edges.

**Numerical observation:** for the safe interior region `|E - epsilon_0| <= 1.5t`, the minimum clean transmission was `1.0`, and the maximum absolute deviation from one was `1.7763568394e-15`. The clean curve also passed an energy-reflection symmetry check about `epsilon_0 = 0`.

## Barrier benchmark

**Numerical observations:** the barrier transmission was `0.640000` at `E = 0`, approximately `0.625141` at the nearest grid point `E = 0.497738`, and approximately `0.572164` at the nearest grid point `E = 0.995475`. Each is below the corresponding clean value, which was one at those energies.

Outside the lead band, the lead self-energies are real and the broadenings vanish. The maximum clean or barrier transmission at sampled energies with `|E - epsilon_0| > 2t` was zero.

## Validation checks

The automated tests cover the Part 2–5 components and the transmission calculation. They check scalar output, real and non-negative physical transmission, use of the Hermitian conjugate, matched-chain transmission, barrier suppression, determinism, invalid dimensions, and zero broadening. The benchmark checks output shapes, finite and non-negative transmission values, and clean-chain energy-reflection symmetry.

Compilation passed. The full pytest suite passed with 95 tests. The benchmark script completed and wrote the CSV, metadata, and figure listed below.

## Band-edge issue

An initial benchmark grid sampled exactly `E = +/-2t`. At those points the lead broadening is zero, and the effective device matrix was singular under the no-extra-broadening convention. The grid was changed to 443 points over the same range, retaining points both inside and outside the band while avoiding the exact edges. No imaginary shift or transmission clipping was introduced to hide the singularity.

## Interpretation and limitations

The numerical results validate the implementation against the clean matched-chain expectation and show reduced transmission for this single finite onsite barrier at the reported energies. They are observations for a coherent, non-interacting, finite one-dimensional model; they do not imply a thermodynamic or universal transport result.

This stage uses ideal semi-infinite single-channel leads, scalar endpoint contact hoppings, and one static onsite perturbation. It does not include disorder, localization, interactions, dephasing, finite-temperature averaging, self-consistent electrostatics, or higher-dimensional transport. The exactly singular band-edge points are omitted from the scan.

## Generated artifacts

- `results/stage1_transmission.csv` — sampled energy and clean/barrier transmission columns.
- `results/stage1_metadata.json` — model parameters and numerical conventions.
- `figures/stage1_transmission.png` — clean and barrier transmission curves with lead-band edges.
- `notebooks/01_landauer_foundation.ipynb` — concise walkthrough using the package functions.
