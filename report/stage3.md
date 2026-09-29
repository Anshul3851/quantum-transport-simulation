# Stage 3: disorder and localization in coherent quantum transport

## Model and numerical conventions

The finite device retains the open-boundary 1D Hamiltonian with diagonal `epsilon_i = epsilon_0 + w_i` and nearest-neighbour matrix elements `-t`. Each onsite shift is independently sampled from `Uniform(-W/2, W/2)`. Ideal semi-infinite leads have onsite `epsilon_0`, hopping `-t`, and contact hopping `t_c`; transmission uses the existing retarded Green-function and Landauer convention. No transmission clipping or normalization is applied.

The run uses `t = 1`, `epsilon_0 = 0`, `t_c = 1`, master seed `20260317`, and NumPy `SeedSequence` child streams with distinct recorded seeds. For the single-realization spectra, `N = 40`, `W = 0, 0.5, 1, 2`, four profiles per W, and 181 energies from `-1.8` to `1.8` are used. The N=40 ensemble has 100 independently seeded profiles for each W and evaluates `E = 0, 0.5, 1`. Length scaling uses `N = 10, 20, 40, 80`, `W = 0.5, 1, 2`, 100 independent profiles per `(N,W)`, and `E = 0, 0.5` (the same profiles are evaluated at both energies).

Reported standard deviation is the sample standard deviation (`ddof=1`); standard error is that value divided by `sqrt(100)`. `mean_log_T` averages `ln(T)` and `typical_T = exp(mean_log_T)`. A floor of `1e-12` is applied only inside the logarithm. The CSV records per-summary counts below that floor; all counts in this run were zero. Raw transmissions are retained. The selected energies are inside the lead band and away from its edges.

## Ensemble results at N=40

Each entry below is mean transmission; the corresponding CSV also contains medians, sample SDs, SEMs, mean log-transmission, typical transmission, log-floor counts, and all realization seeds.

| W | E=0 | E=0.5 | E=1 |
|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 1.0000 |
| 0.5 | 0.8867 | 0.8399 | 0.7917 |
| 1 | 0.5796 | 0.4838 | 0.3860 |
| 2 | 0.1935 | 0.1632 | 0.0673 |

For W=0, transmission is unity to numerical precision, as expected for a clean device matched to the leads. For the sampled disordered ensembles, increasing W lowers mean transmission at all three energies. At W=2, the mean log-transmission is `-2.646`, `-3.167`, and `-4.598` at E=0, 0.5, and 1, respectively; corresponding typical transmissions are `0.0710`, `0.0421`, and `0.0101`. These are finite-sample summaries, not asymptotic values.

## Length dependence

Mean log-transmission decreases and typical transmission decreases at each sampled disorder and energy as N grows from 10 to 80. At E=0, typical transmission changes as follows:

| W | N=10 | N=20 | N=40 | N=80 |
|---:|---:|---:|---:|---:|
| 0.5 | 0.9483 | 0.9051 | 0.8098 | 0.7114 |
| 1 | 0.8586 | 0.7136 | 0.4631 | 0.2331 |
| 2 | 0.4613 | 0.2076 | 0.0574 | 0.00171 |

The mean log-transmission also decreases monotonically over these sampled lengths at E=0.5: for W=0.5 it changes from `-0.0582` to `-0.4026`, for W=1 from `-0.2140` to `-1.7452`, and for W=2 from `-0.8674` to `-7.0700`. This trend is consistent with disorder-induced suppression in the finite 1D samples. It does not establish an infinite-system localization law by itself.

As a descriptive check only, ordinary least-squares fits of mean `ln(T)` versus N were retained when the slope was negative and R? was at least 0.95 over the four lengths. All six `(W,E)` combinations passed. Slopes (per site) were `-0.00410` and `-0.00494` for W=0.5 at E=0 and 0.5; `-0.01868` and `-0.02205` for W=1; and `-0.07967` and `-0.08896` for W=2. Under the explicitly exploratory convention `xi = -1/slope`, these imply roughly 244, 203, 53.5, 45.4, 12.6, and 11.2 sites, respectively. These estimates are based on four finite lengths and 100 disorder samples per point; they are not thermodynamic localization lengths.

## Outputs and limitations

- `results/stage3_ensemble.csv`: 12 N=40 `(W,E)` summary rows.
- `results/stage3_length_scaling.csv`: 24 `(N,W,E)` summary rows.
- `results/stage3_disorder_realizations.csv` and `stage3_single_realization.csv`: individual profiles and spectra.
- `results/stage3_metadata.json`: model parameters, seed-stream convention, statistics, log-floor policy, and exploratory fit coefficients.
- `figures/stage3_disorder_realizations.png`, `stage3_transmission_disorder.png`, `stage3_ensemble_statistics.png`, and `stage3_length_dependence.png`.

The results are coherent, single-channel finite-device calculations for one deterministic ensemble seed. The SEM describes variation of the sample mean under independent disorder realizations; it does not cover model or finite-size uncertainty. The exploratory fits and finite ensemble do not establish a universal localization length or thermodynamic limit.
