# Results index for the revised manuscript

This index maps the current 17-page manuscript `plos2 (12).pdf` to its three main tables, eight main figures, numerical inputs, and reproduction code. All paths below are relative to the release root. The current manuscript numbering in this document takes precedence over historical figure numbers in earlier reproduction maps, reports, or source-description notes. The obsolete feature-ablation caption after the Conclusion is not a ninth figure; Table 2 contains the ablation evidence.

The release preserves the data, fitted prediction outputs, experiment implementations, and numerical result reproduction. Plotting scripts and figure rendering are excluded. The data described here are generated experimental observations, not individual institutional student records.

## Reproduce and check numerical results

From the release root, install the recorded environment and run:

```bash
python -m pip install -r requirements-complete-lock.txt
python reproduce_results.py --output reproduced
```

The wrapper recomputes summaries from the saved cohort records in a separate output directory. It also recalculates the denominator and prospective-sampling diagnostics, writes numerical paper tables, exports the final figure source CSVs, and compares regenerated outputs with the archived values. It does not retrain every model. Use a different output directory for another run; existing output directories are not overwritten. The `--verify` option adds checks of saved predictions and numerical calculations.

To export and compare only the numerical data behind the final eight figures:

```bash
python code/export_figure_data.py --output exported_figure_data --compare-to results/figure_source_data
```

This exporter reads aggregated result tables and the saved baseline seed-42 predictions. It recreates 15 CSV files, using the original row order and exact plotted subsets, without importing a plotting library. The comparison checks schema, text and row order exactly and numeric values with absolute tolerance `1e-12` and relative tolerance zero. The descriptive `figure_descriptions.json` is archival metadata, not an input used to recreate the numbers.

The numerical workflow uses these scripts:

| Script | Numerical role |
| --- | --- |
| `code/run_r2.py` | Generate and fit the original HGB benchmark, ablations, slope settings, stresses, and Spy grid; save per-cohort results and predictions. |
| `code/run_complete_r2.py` | Fit the added logistic and propensity methods on the same cohort/test identities; save added predictions and diagnostics. |
| `code/propensity_audit.py` | Fit the training-only numerator/calibration variants and calculate simulator-assisted conditional quantities. |
| `code/diagnostics_complete_r2.py` | Calculate denominator comparisons and the prospective-verification sampling simulation. |
| `code/summarize_r2.py` | Aggregate the original benchmark; calculate its paired contrasts, ablations, diagnostics, capacity intervals, and slope intervals. |
| `code/summarize_completion.py` | Aggregate the expanded primary comparison and its paired contrasts, denominator diagnostics, and sampling-design outcomes. |
| `code/make_numerical_tables.py` | Format paper-table CSVs and the Spy-grid summary without plotting. |
| `code/export_figure_data.py` | Export the exact final figure data selections from regenerated numerical summaries and seed-42 predictions. |
| `code/aggregate_optimizer_audit.py` and `code/summarize_numerical_sensitivity.py` | Summarize the separately reported optimization audits; they do not replace the primary fits. |

The training scripts are supplied for a full computational rerun. Summary regeneration and saved-prediction checks are distinct from rerunning all training jobs. See the release README for the full rerun workflow and dependency requirements.

## Metric populations and uncertainty

- `AP_full` evaluates average precision on the complete test population. `AP_U` evaluates it only among test students with recorded label `S = 0`.
- `P_U_10`, `HR_U_5` and `HR_U_10` use shortlists restricted to `U`. A capacity is a percentage of the number of unlabeled test students. Previously recorded positives do not take shortlist places.
- `HR_full_5`, `HR_full_10` and `HR_full_15` are separate full-population shortlist comparisons. They must not be substituted for the main U-only outcomes.
- `mean` and `sd` describe cohort means and sample standard deviations. `ci_low` and `ci_high` are bootstrap limits. In paired files, `difference` means `method_a` minus `method_b`. In `paired_ablation.csv`, the contrast is ablated minus full-feature Spy.
- Bootstrap resampling uses cohorts, not individual students, and 10,000 resamples. Intervals are unadjusted; capacity and slope intervals are pointwise.
- Baseline, each mechanism, each slope, each ablation, and the alternative generator use 100 cohorts. Each of the ten stress settings uses 50; each Spy-grid configuration uses 20. The same seeds reused across settings do not create additional independent samples from a single population.

Two deterministic bootstrap implementations were retained in the executed analysis. Their means agree for overlapping results, but their finite-bootstrap interval endpoints can differ slightly. **Fig 2, Fig 3, Fig 7, Table 2, and Table 3 use the original aggregation stream. Fig 4 and Fig 5 use the expanded-comparison stream.** Do not interchange the similarly named files when reproducing the displayed intervals. The exporter follows these exact selections.

## Main tables

| Manuscript item | Exact source and selection | Numerical reproduction |
| --- | --- | --- |
| **Table 1 — Baseline ranking and screening** | `results/tables/complete_main_for_paper.csv`, all 14 primary methods. The manuscript retains `AP_full`, `AP_U`, `P_U_10`, `HR_U_5`, and `HR_U_10`; the CSV also retains the other evaluated capacities. Full-precision values are in `complete_summary_metrics.csv`, `scenario = baseline`. | `summarize_completion.py` combines `results/cohorts/baseline/{seed}.json` with `results/completion/baseline/{seed}.json`; `make_numerical_tables.py` creates the formatted CSV. |
| **Table 2 — Feature ablation** | `results/tables/summary_metrics.csv`: model `Spy`, scenarios `baseline`, `drop_consumption_level`, `drop_consumption_structure`, `drop_behavioral_regularity`, and `drop_noncard_records`; metrics `AP_full` and `HR_U_10`. The paired recall difference and interval come from `results/tables/paired_ablation.csv`, `metric = HR_U_10`. | `summarize_r2.py` pairs each `drop_*` scenario with the full-feature baseline at the same seed before bootstrapping. The full-feature row equals the Table 1 Spy baseline. |
| **Table 3 — Eight labeling mechanisms** | `results/tables/mechanism_results_for_paper.csv`, all eight rows. Retain `Naive_HR_U10`, `Spy_HR_U10`, and `paired_HR_U10_difference_95CI`; the source also contains the full-test AP contrast. Full-precision sources are `summary_metrics.csv` and `paired_method_differences.csv`. | `summarize_r2.py` creates the original-stream means/paired intervals; `make_numerical_tables.py` formats the eight-row table. `complete_mechanisms_for_paper.csv` is a broader additional-method summary, not the direct source of this three-column main comparison. |

All table paths in the above table are under `results/tables/`. The `drop_noncard_records` scenario is displayed as “Records beyond the card.” Model display names are mapped below.

## Main figures

Final plotted data are archived under `results/figure_source_data/`. The listed parent tables are under `results/tables/`; their reproduction scripts are in `code/`.

| Manuscript item | Final numerical file and selection | Parent numerical source and reproduction |
| --- | --- | --- |
| **Fig 1 — Precision–recall curves** | `Fig1_seed42_predictions_U.csv`: 3,344 unlabeled seed-42 test observations, with generated ID, true label and 11 displayed score columns. `Fig1_AP_validation.csv` gives the 11 legend AP values and counts. This is one fixed cohort, not a 100-cohort mean. | Saved scores in `results/cohorts/baseline/42.npz` and `results/completion/baseline/42.npz`; align by `ids`, check `y`/`s`, restrict to `s == 0`. The exporter recalculates `AP_U` using `average_precision_score`. Precision–recall coordinates can be obtained directly from each score column and `y`; no rendering code is required. |
| **Fig 2 — Capacity and yield** | `Fig2_capacity_summary.csv`, 180 rows: six HGB/low-spending procedures × capacities 1–30%, `scenario = baseline`. Panel A uses `HR_mean`, `HR_ci_low`, and `HR_ci_high`; Panel B uses `P_mean` without an uncertainty band. | **`capacity_summary.csv`**, not `complete_capacity_summary.csv`. Recreated by `summarize_r2.py`; export filters baseline and the six specified methods. |
| **Fig 3 — Spy HGB across mechanisms** | `Fig3_paired_differences.csv`, 16 rows: eight mechanisms × `AP_full` and `HR_U_10`, with `method_a = Spy`, `method_b = Naive`. Both mean differences and paired interval endpoints are retained. | **`paired_method_differences.csv`**, not `complete_paired_differences.csv`. Recreated by `summarize_r2.py`; exporter preserves original row order. |
| **Fig 4 — Added methods versus naive logistic** | `Fig4_plotted_values.csv`, 64 rows: eight mechanisms × four methods (`Spy_LR`, `SAR_EM`, `R1_IndexEM`, `LBE_LF`) × two metrics (`AP_U`, `HR_U_10`). Every contrast uses `method_b = Naive_LR`. | `complete_paired_differences.csv`, recreated by `summarize_completion.py`. The exporter retains the panel, method and mechanism order used in the final figure. |
| **Fig 5 — Stress settings and alternative generator** | `Fig5_paired_stress_values.csv`, 11 rows: ten stress settings, plus the alternative generator; `Spy` minus `Naive`; **`AP_U` only**. This is not `AP_full` and is not a ranking of Spy against the EM methods. | `complete_paired_differences.csv`, recreated by `summarize_completion.py`. `stress_absolute_values.csv` additionally preserves `AP_U` and `HR_U_10` for Spy, the three propensity methods and the logistic true-label reference. |
| **Fig 6 — Denominator diagnostics** | `Fig6_plotted_values.csv`, eight rows: latent `q` and conditional `e(X)` denominator choices under raw, capped raw, calibrated, and known-population numerator configurations. Every displayed outcome is **`AP_full`**, over 100 baseline cohorts. | `denominator_audit_summary.csv`; `propensity_audit.py` supplies numerator/conditional arrays, `diagnostics_complete_r2.py` recalculates denominator results, and `summarize_completion.py` aggregates them. The exporter selects only the eight displayed valid diagnostic rows. |
| **Fig 7 — Inverse-labeling slope** | `summary_metrics.csv`, filtered to `beta_0.0`, `beta_0.3`, `beta_0.62`, `beta_0.9`; models `Naive`, `ElkanNoto`, `Spy`, `TrueLabel`; metrics `HR_U_5` and `HR_U_10`. These are 32 plotted observations, with cohort mean and pointwise intervals. | **`summary_metrics.csv`**, from `summarize_r2.py`, not the completed benchmark's similarly named file. The full original summary is copied unchanged by the exporter so the exact filters remain inspectable. |
| **Fig 8 — Verification sampling design** | `prospective_sampling_simulation_summary.csv`: select `probability_observe_no_missed_exact` for Panel A and `recall_hat_rmse` for Panel B, at sample sizes 50, 100, 200 and 400. Each sample size has 100 cohorts, each with 1,000 sampling draws. | `diagnostics_complete_r2.py` writes `prospective_sampling_simulation_raw.csv` using saved baseline Spy scores; `summarize_completion.py` creates the summary. The exporter copies the summary including the additional diagnostic outcomes. |

For Fig 1, the dotted reference value is the mean of `y` in the U-only CSV; it is the realized hardship prevalence in that fixed cohort. Fig 1 intentionally shows the 11 score columns supplied in its CSV, while Table 1 retains the complete 14-method comparison.

For Fig 6, fitted raw numerators use the full training split, while the calibrated configuration fits on 80% and calibrates on 20%. Their difference cannot isolate calibration alone. Rows beginning `INVALID_truth_dependent_` occur in the full audit for diagnostic transparency, but use true labels to alter a denominator and are excluded from Fig 6 and the practical-method comparison. Known-population quantities are simulator-assisted references, not deployable methods. Gray links between denominator choices in the figure are not confidence intervals for paired differences.

For Fig 7, `beta_0.0` is exactly the constant-labeling mechanism reused in Table 3. The slope-0.62 setting has intercept 0.891 and bounds 0.10–0.92; it is distinct from the main `baseline` rule with intercept 0.88 and bounds 0.12–0.88.

For Fig 8, all top-10%-of-U students are verified and nonselected students are sampled without replacement. Panel A is the exact hypergeometric probability of detecting no missed positive, not a Monte Carlo proportion. Panel B is the mean of cohort-specific recall-estimator RMSEs. The simulation assumes complete participation and correct verification and is not an institutional audit.

## Other numerical results cited in the text

| Result | Numerical records |
| --- | --- |
| Baseline contrasts against learner-specific references | `results/tables/complete_reference_contrasts_for_paper.csv` and full-precision `complete_paired_differences.csv`, `scenario = baseline`. |
| Reliable-negative retention, purity and hidden-positive contamination | `results/tables/method_diagnostics_raw.csv` and `method_diagnostics_summary.csv`, baseline Spy. |
| Nine Spy parameter configurations | `results/tables/spy_grid_summary.csv`, produced by `make_numerical_tables.py` from the original per-cohort metrics/diagnostics. Each cell uses 20 seeds; the reference cell reuses baseline seeds 42–61. |
| Primary optimization stopping | `results/tables/em_convergence_raw.csv` and `em_convergence_summary.csv`. Primary iteration-limit outputs remain in the main comparisons. |
| Extended-budget audit | `results/tables/convergence_sensitivity_raw.csv` and `convergence_sensitivity_summary.csv`, backed by `results/convergence_sensitivity/`. These are separate audit results. |
| Same-objective LBE optimization audit | `results/tables/lbe_optimizer_audit.csv`, `lbe_optimizer_policy_raw.csv`, and `lbe_optimizer_policy_summary.csv`, backed by `results/lbe_optimizer_audit/`. Replacing capped fits for an audit does not redefine the primary benchmark. |
| Positive-label-noise discussion | `results/tables/complete_summary_metrics.csv`, `scenario = false_positive_rate_0.03`, `metric = AP_full`, primary model keys `Spy`, `SAR_EM`, `R1_IndexEM`, `LBE_LF`, each with `n = 50`. Means round to 0.8837, 0.8013, 0.5899, and 0.9645. These full-test AP values differ from the `AP_U` values in `stress_absolute_values.csv`. |
| Full-population shortlist comparisons | `results/tables/complete_summary_metrics.csv` and `complete_paired_differences.csv`, metrics `HR_full_5`, `HR_full_10`, and `HR_full_15`; retain the different population definition. |

## Model and scenario keys

| Stored model key | Manuscript name |
| --- | --- |
| `Naive` | Naive HGB |
| `CostSensitive` | Weighted HGB |
| `ElkanNoto` | Elkan–Noto HGB |
| `Spy` | Spy HGB |
| `TrueLabel` | True-label HGB |
| `LowSpending` | Low spending |
| `Naive_LR` | Naive logistic |
| `CostSensitive_LR` | Weighted logistic |
| `ElkanNoto_LR` | Elkan–Noto logistic |
| `Spy_LR` | Spy logistic |
| `TrueLabel_LR` | True-label logistic |
| `SAR_EM` | SAR-EM |
| `R1_IndexEM` | Index-EM |
| `LBE_LF` | LBE-LF |

Keys ending `_UPosterior` represent additional derived score variants. They are not additional independently fitted primary learners and are not substituted for the corresponding main-table score.

The eight mechanism keys, in manuscript display order, are `beta_0.0` (constant), `baseline` (linear inverse), `logistic`, `threshold`, `heterogeneous` (two-college), `reverse`, `nonmonotonic`, and `unobserved` (unobserved factor).

The eleven Fig 5 keys, in display order, are `poor_rate_0.05`, `poor_rate_0.25`, `frugal_rate_0.0`, `frugal_rate_0.25`, `overlap_0.5`, `overlap_1.0`, `noise_scale_1.5`, `noise_scale_2.0`, `false_positive_rate_0.01`, `false_positive_rate_0.03`, and `alternative_factors`. False-positive probabilities are conditional on `Y = 0`, not percentages of recorded positives.
