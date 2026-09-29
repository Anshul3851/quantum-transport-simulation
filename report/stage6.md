# Stage 6: final synthesis and robustness review

## Scope and audit

Stage 6 reviews the saved results from Stages 1–5. It does not rerun or modify any earlier transport calculation. The expected source modules, scripts, tests, notebooks, reports, generated CSV/JSON results, and figures for all five stages were present. Research outputs are included in version control; Python caches and pytest caches are ignored. The project uses the stated Hamiltonian convention (nearest-neighbour matrix elements -t), open device boundaries, retarded lead self-energies, and Landauer transmission consistently across the reviewed metadata and reports.

The prior README described Stages 1–4 but omitted the Stage 5 QPC from its project progression and placed an incomplete stage summary among the limitations. It has been replaced with a concise overview of the complete implemented project. No inconsistent numerical claims were found in the saved stage reports that change the interpretation below. Stage 5 metadata correctly describes the central-slice local mode count as an energetic indicator for a nonuniform constriction, and describes LDOS as raw local spectral density rather than current density.

## Stage 1: matched 1D chain and barrier

Using only the existing 443-row transmission table, the clean matched chain was checked in three interior energy windows. For |E| <= 1.5 (301 points), the maximum absolute deviation of clean T from one is 1.78e-15. The corresponding deviations remain below 1.78e-15 for |E| <= 1 and below 1.45e-15 for |E| <= 0.5. This shows that the clean reference is not dependent on using the widest safe interior window.

At the saved energy E=0, the one-site barrier transmission is 0.64. Across |E| <= 1.5, its saved transmission ranges from 0.4403 to 0.64. These window checks reuse the original energy grid and do not sample the singular band edges.

## Stage 2: resonance and LDOS selectivity

The saved double-barrier transmission has a resonance at E=-0.73095 with T=0.999991. Five consecutive grid points from E=-0.73425 to -0.72765 have transmissions 0.6095, 0.8619, 1.0000, 0.8673, and 0.6203, so the peak is resolved over multiple samples rather than appearing as a single isolated grid value.

For the saved site-resolved LDOS samples, the central-well fraction is 0.8378 at the resonance and 0.0857 and 0.1110 at the two detuned energies -0.77055 and -0.69135. The site-summed LDOS at the resonance is 83.63, compared with 8.297 and 7.936 at those detuned samples. The site sums agree with the independently saved total-DOS values to roundoff. The resonance-centered LDOS file contains only three energy samples; this is a selectivity check, not a full parameter or linewidth study.

## Stage 3: disorder and length summaries

For N=40 and E=0, increasing W from 0.5 to 2 reduces the saved ensemble mean transmission from 0.8867 to 0.1935, the median from 0.9081 to 0.09176, and the typical transmission from 0.8814 to 0.07096. The mean log-transmission changes from -0.1262 to -2.6456. Each ensemble condition contains 100 independently seeded disorder realizations, with no samples below the recorded logarithm floor for the representative values reported here.

Across all six sampled (W,E) groups in the length-scaling table, mean, median, typical, and mean-log transmission each decrease monotonically over N=10, 20, 40, and 80. The decreasing typical and mean-log values show why the arithmetic mean alone is insufficient for these skewed transmission distributions. These finite ensembles and four lengths do not establish a thermodynamic localization length; reported fits remain exploratory finite-size summaries.

## Stage 4: multi-channel consistency

The Stage 4 metadata specifies excluding energies within 0.05 of any mode threshold. Reapplying that same rule to the saved transport CSV retains 671 of 751 energy points. The recomputed maximum absolute difference between T and N_open is 7.11e-15; the median absolute difference is 8.88e-16. This reproduces the clean matched-strip channel-count check using the actual table and its saved exclusion convention.

The comparison is for a clean, matched, finite-width strip. It does not test disorder, contact mismatch, or threshold behavior inside the excluded neighborhoods.

## Stage 5: QPC gate scan and spectral-density interpretation

Transmission is nonincreasing with gate strength at all five saved gate-scan energies (-3.5, -3.0, -2.5, -2.0, and -1.5). At E=-2, the lead supports three open modes while T changes from 3.0000 at zero gate to 2.2781, 1.9999, 1.2814, and 1.0001 as the gate increases from 0 to 1. The central-slice local open-mode indicator changes from 3 to 2, 2, 1, and 1. These values describe a finite smooth constriction, so they should not be read as exact quantized plateaus.

Across all 605 saved gate/energy transmission rows, the largest positive T-N_open is 6.22e-15, consistent with the incoming-channel bound to numerical precision. The largest absolute difference is larger because a gate can suppress transmission below the lead mode count; that is expected and is not a bound violation.

At E=-2, the raw full-device LDOS sums for gate strengths 0.25 and 0.75 are 28.56 and 24.63. These values describe local spectral weight. LDOS alone is not a current density and is not evidence of a local flux distribution. The saved maps use only two gate strengths and one energy.

## Integrated overview

The overview figure places one representative comparison from each implemented stage side by side. Each panel uses previously saved data; the Stage 2 LDOS markers are available at only three sampled energies. The accompanying CSV/JSON summary records the principal result, validation metric, and limitation for each stage. The synthesis is generated by python scripts/run_transport_stage6.py.

The new notebook notebooks/06_final_synthesis.ipynb loads the saved Stage 1–5 result tables and metadata, checks mode counts and the saved QPC potential using existing source modules, prints the derived robustness summary, and regenerates the Stage 6 summary artifacts.

## Validation

Python compilation checks completed successfully with python -m compileall -q src scripts tests. The focused Stage 6 suite passed all 8 tests, and the full project suite passed all 188 tests. The notebook parsed as valid nbformat JSON; all four Python cells compiled and executed in order. nbclient was unavailable, so execution used a shared Python namespace without installing dependencies. The notebook confirmed that existing Stage 4 mode counts match all 751 saved energies and that the Stage 5 source module reproduces the saved 240-site potential. The final scan found no secret-like values, machine-specific absolute paths, .env files, or temporary files. Git status confirmed no modifications to prior Stage 1–5 source, tests, figures, reports, notebooks, metadata, or result tables.