# Stage 5: quantum point contact and channel filtering

## Geometry and gate potential

Stage 5 uses a fixed-width `Lx=40`, `Ly=6` strip with 240 sites, open device boundaries, transverse hard walls, and the Stage 4 ordering `index(x,y)=x*Ly+y`. The hopping is `-t` in both directions with `t=1`; the background/device-lead onsite energy is `epsilon_0=0` and contact hopping is `t_c=1`.

The QPC potential is

`epsilon(x,y) = epsilon_0 + Vg exp[-(x-xc)^2/(2 sigma_x^2)] [1 + alpha ((y-yc)/y_scale)^2]`,

with `xc=(Lx-1)/2`, `yc=(Ly-1)/2`, `y_scale=max((Ly-1)/2, 1/2)`, `sigma_x=6` sites and `alpha=1.5`. It is smooth and symmetric about the longitudinal centre and transverse midpoint. Its Gaussian envelope tends to zero toward the leads; at the center it is weakest on the two central rows and stronger toward the outer rows. The studied gate strengths are `Vg = 0, 0.25, 0.5, 0.75, 1` in units of `t`.

## Analytic channel information

For the clean lead width `Ly=6`, the open hard-wall transverse slice has mode energies

`epsilon_n = epsilon_0 - 2t cos(n pi/(Ly+1))`.

The six energies are approximately `[-1.80194, -1.24698, -0.44504, 0.44504, 1.24698, 1.80194]`. The corresponding lower longitudinal band edges are `[-3.80194, -3.24698, -2.44504, -1.55496, -0.75302, -0.19806]`; `N_open(E)` counts clean-lead modes satisfying `|E-epsilon_n| <= 2t`. The energy sweep `[-3.6,-1.2]` therefore covers openings of the second, third, and fourth lead modes.

At the QPC centre, a separate transverse slice Hamiltonian is built using the onsite values from the middle profile. Its eigenvalues shift with gate strength. The resulting local subband lower edges for the first three modes are:

| `Vg` | Local lower edges, first three modes |
|---:|---|
| 0 | -3.802, -3.247, -2.445 |
| 0.25 | -3.466, -2.802, -1.975 |
| 0.5 | -3.143, -2.375, -1.512 |
| 0.75 | -2.829, -1.964, -1.058 |
| 1 | -2.523, -1.567, -0.616 |

`N_local_open_centre` applies the same 1D band criterion to these centre-slice eigenvalues. This is an energetic indicator, not an exact channel count of the nonuniform QPC: longitudinal variation causes reflection and mode coupling, and finite barriers permit evanescent tunnelling.

## Numerical transmission and conductance

The script evaluates 121 energies uniformly between `-3.6` and `-1.2` for each of the five gate strengths. The gate-zero device reproduces the Stage 4 clean matched strip at the validation energies: `(E,N_open,T)=(-3.5,1,1)`, `(-3.0,2,2)`, `(-2.2,3,3)`, and `(-1.3,4,4)`, with maximum absolute error `3.55e-15`.

At `E=-2`, the clean lead has three incoming modes. The measured transmissions as gate strength increases are:

| `Vg` | `N_local_open_centre` | Numerical `T(E=-2)` |
|---:|---:|---:|
| 0 | 3 | 3.00000 |
| 0.25 | 2 | 2.27813 |
| 0.5 | 2 | 1.99994 |
| 0.75 | 1 | 1.28142 |
| 1 | 1 | 1.00006 |

At the selected gate-scan energies `-3.5`, `-3.0`, `-2.5`, `-2.0`, and `-1.5`, transmission decreased monotonically across the tested gate values. The curves show near-integer plateaus in some intervals?e.g. around `T=2` near `E=-2`?with smooth intermediate values as the gate changes. This is evidence of channel filtering, but not exact quantization over the full scan. At `E=-3.5`, `Vg=0.25` has no locally propagating centre mode by the slice criterion yet gives `T=0.172`; that finite transmission is consistent with tunnelling through a finite smooth barrier and demonstrates why the local mode count is not a hard bound.

For every simulated energy and gate, numerical transmission stayed below the clean-lead incoming mode count up to roundoff. The largest measured `T-N_open` was `6.22e-15`; the implementation did not clip transmission. Dimensionless conductance is `g=T`; physical conductance is `G=(2e^2/h)T`. SciPy was unavailable in the run environment, so the conductance quantum used the exact SI defining constants fallback, `2e^2/h = 7.748091729863649e-5 S`. No temperature dependence is included.

## LDOS and spatial interpretation

The Stage 2 LDOS function computes the raw site quantity `rho_i(E)=-Im(G^r_ii(E))/pi`. Maps are shown at `E=-2` for `Vg=0.25` and `Vg=0.75` on a shared colour scale; no numerical normalization is applied. The sum over device sites falls from `28.56` to `24.63` between these examples. At the centre slice, the LDOS on each outermost transverse row changes from about `0.1428` at `Vg=0.25` to `0.0352` at `Vg=0.75`, while the inner rows retain relatively more weight. The LDOS is spectral weight, not a current-density map, so it identifies spatial redistribution but does not by itself quantify local flow.

## Outputs and limitations

- `results/stage5_qpc_transmission.csv`: energy/gate transmission, clean-lead and local-centre mode counts, dimensionless and physical conductance, local subband energies.
- `results/stage5_qpc_gate_scan.csv`: transmission at five selected energies versus gate.
- `results/stage5_qpc_ldos.csv` and `stage5_qpc_potential.csv`: raw spatial LDOS and the strongest-gate onsite profile.
- `results/stage5_qpc_metadata.json`: parameters, formula, analytic/local modes, baseline and channel-bound checks, conductance convention.
- `figures/stage5_qpc_potential.png`, `stage5_qpc_transmission.png`, `stage5_qpc_conductance.png`, `stage5_qpc_gate_scan.png`, and `stage5_qpc_ldos.png`.

The device calculation uses a dense solve for 240 sites. The local transverse-mode picture is approximate for a finite, spatially varying constriction. The model is coherent and non-interacting; disorder, finite temperature, magnetic fields, spin-orbit coupling, and electron interactions are omitted. The results show finite-device channel filtering, not a thermodynamic or universal quantization claim.
