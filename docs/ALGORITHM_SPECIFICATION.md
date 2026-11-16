# Complete specification of the R2 algorithms and numerical diagnostics

This document describes the executable R2 procedures in `code/models_r2.py`, `code/models_propensity_r2.py`, their runners, and diagnostic scripts. It replaces the algorithm details formerly referred to as S1 Text. It does not claim recovery of missing first-revision implementations, empirical identification of hardship, or a universal performance ceiling.

## Sources and implementation identity

- SAR-EM follows the selected-at-random EM procedure of Bekker, Robberechts and Davis, *Beyond the Selected Completely at Random Assumption for Learning from Positive and Unlabeled Data*, ECML PKDD 2019 proceedings, published by Springer in 2020, pp. 71--85, DOI `10.1007/978-3-030-46147-8_5`. Author implementation: <https://github.com/ML-KULeuven/SAR-PU>. The original release records inspected commit `6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7`; the applicable MIT license is retained in `protocol/SAR_PU_LICENSE.txt`.
- LBE-LF is the explicit linear logistic implementation of Gong et al., *Instance-Dependent Positive and Unlabeled Learning With Labeling Bias Estimation*, IEEE TPAMI 44(8), 2022, pp. 4163--4177, DOI `10.1109/TPAMI.2021.3061456`. It uses the likelihood and EM/Adam construction rather than calling the authors' original software.
- Index-EM is this study's restricted-index adaptation. The code identifier `R1_IndexEM` is retained for compatibility with saved results. It must not be relabeled as Gong et al.'s LBE.
- The source citation in the inherited `protocol/additional_methods.txt` combines a preliminary workshop title with the later SAR authors/conference. The full three-author SAR reference above is the corrected source attribution.

## Common inputs, splits, and scores

The 14 observed features and their fixed order are specified in `GENERATOR_SPECIFICATION.md`. Each cohort is split 70/30 using `random_state=7`, jointly stratified by simulated `Y` and observed `S`. This use of truth balances experimental cells only. Model fitting, scaling, threshold construction, and label-frequency estimation use training rows. Practical fits do not receive latent typicality, true propensity, college, frugal membership, or test labels.

True-label reference models replace training `S` by training `Y`. They otherwise use their own specified learner and are learner-specific references. The low-spending baseline uses the negative of `monthly_total_spend`; it has no fitting stage. Original generated row indices break ties without consulting truth.

For standard logistic models, `StandardScaler` is fitted in each training pipeline. Sample weights are passed to `LogisticRegression`, not to `StandardScaler`; the scaler therefore uses the unweighted fitting rows. In the Elkan--Noto second stage, those rows include the duplicated unlabeled observations. The propensity estimators fit one standardizer to the original training matrix and retain it throughout their optimization.

Every primary propensity-estimator ranking uses its class score `f(X)`. Separately saved `*_UPosterior` scores are

\[
\frac{f(X)(1-e(X))}{\max\{1-f(X)e(X),10^{-15}\}}.
\]

They are additional model-based diagnostics, not the primary benchmark score and not known population conditional probabilities.

## Base learner settings

The full effective estimator settings are retained in `model_parameters.json`; locked package versions are in `requirements-complete-lock.txt`. The recorded environment uses Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, and SciPy 1.17.0. Numerical library thread counts are set to one in the experiment runners.

| HGB setting | Value |
|---|---|
| Estimator | `HistGradientBoostingClassifier` |
| loss | `log_loss` |
| max_iter / learning_rate | 300 / 0.08 |
| random_state | 0 |
| max_leaf_nodes / min_samples_leaf | 31 / 20 |
| max_depth | None |
| max_bins / max_features | 255 / 1.0 |
| l2_regularization | 0 |
| early_stopping | `auto` |
| validation_fraction | 0.1 |
| scoring / n_iter_no_change / tol | `loss` / 10 / `1e-7` |
| class_weight | None; any weights are supplied explicitly |
| categorical_features | `from_dtype`; model inputs are numeric NumPy arrays |
| monotonic_cst / interaction_cst | None / None |
| warm_start / verbose | False / 0 |

Only `max_iter`, `learning_rate`, and `random_state` are explicitly passed by the inherited HGB factory; the table records the remaining effective defaults in the executed environment. Automatic early stopping can activate after Elkan--Noto duplicates rows beyond 10,000, whereas an original baseline training split has 8,400 rows. Sharing the HGB family does not mean identical fitted iteration counts or optimization behavior.

| Standard logistic setting | Value |
|---|---|
| Pipeline | `StandardScaler`, then `LogisticRegression` |
| C / max_iter / random_state | 1 / 1000 / 0 |
| solver / tol | `lbfgs` / `1e-4` |
| fit_intercept / intercept_scaling | True / 1 |
| class_weight | None |
| dual / warm_start / verbose | False / False / 0 |
| l1_ratio | 0 |
| penalty metadata | `deprecated`, as returned by scikit-learn 1.8.0 |
| n_jobs | None |

Do not substitute a different library release's defaults when seeking exact numerical reproduction.

## Naive, class-weighted, and true-label procedures

Naive fitting treats observed `S` as the binary response. Class-weighted fitting assigns weight `n0/n1` to recorded positives and 1 to unlabeled rows, where `n0` and `n1` are original training counts. This equalizes total observed-class weights and preserves the exact implemented scaling. It is not SMOTE or oversampling.

The true-label reference fits on training `Y`. `fit_model` requires both response classes and raises an error otherwise. The returned score is always the positive-column `predict_proba` output.

## Two-stage Spy procedure

1. Initialize `default_rng(0)` and identify original training positives.
2. Sample without replacement `max(floor(n_positive*spy_fraction),1)` spies. The reference fraction is 0.15. A count exhausting all positives raises an error.
3. Temporarily change spy labels to zero and fit the first classifier on all original training rows.
4. Obtain first-stage positive scores for all training rows. Set the initial threshold to percentile 5 of spy scores using NumPy's `method='linear'` interpolation.
5. Select reliable negatives only among original `S=0` rows whose score is **strictly less than** the threshold.
6. If fewer than 50 reliable negatives are selected, replace the threshold by percentile 30 of scores among the original unlabeled training rows, again using linear interpolation and a strict inequality. This fallback is not a rule forcing exactly 50 retained rows. If no reliable negative remains, raise an explicit error.
7. Fit a new classifier using all original recorded positives plus retained reliable negatives. The spies regain their original positive labels.

The grid crosses fractions `{0.05,0.15,0.30}` with percentiles `{1,5,10}`; every cell is retained. The reference grid cell reuses the first 20 baseline cohorts. No replacement setting is selected by test performance.

Stored diagnostics include number of spies, original and final thresholds, fallback status, number and fraction of original unlabeled training observations retained, retained-set purity, and fraction of hidden training positives retained. The two truth-dependent diagnostic ratios are evaluated only after fitting; they do not affect threshold selection or training.

## Weighted Elkan--Noto procedure

Using `default_rng(0)`, hold out `max(floor(0.20*n_positive),1)` recorded training positives without replacement. Fit the first classifier to all remaining training rows. For held-out set `H`, let `h_i` be the first fitted labeling score, and calculate

\[
\hat c=\operatorname{clip}\left(\frac{1}{|H|}\sum_{i\in H}h_i,0.001,0.999\right),\qquad
\hat g_i=\operatorname{clip}(h_i,10^{-6},1-10^{-6}).
\]

For every originally unlabeled training row,

\[
\tilde w_i=\frac{1-\hat c}{\hat c}\frac{\hat g_i}{1-\hat g_i},\qquad
w_i=\operatorname{clip}(\tilde w_i,0,1).
\]

The second training matrix consists, in order, of all original recorded positives with response 1 and weight 1, all original unlabeled rows with response 1 and weight `w`, and the same unlabeled rows with response 0 and weight `1-w`. Held-out positives return in this second stage. The method returns the second classifier's probability, not a scalar post-hoc rescaling of the naive score. Stored diagnostics include `c_hat`, hold-out count, and the fraction of unlabeled weights equal to one.

## Propensity model inputs and parameterization

Let `z` denote training-standardized features and let `A=[1,z]` have 15 columns. The class model is `f=sigmoid(A theta)` in all three methods. The propensity model is `e=sigmoid(B gamma)`.

SAR-EM uses `B=[1,z_J]`, where the following nine feature names and signs are listed in code order:

| Feature | Zero-based original feature index | Index-EM sign |
|---|---:|---:|
| monthly_total_spend | 0 | -1 |
| avg_meal_price | 1 | -1 |
| meals_per_day | 2 | +1 |
| breakfast_rate | 3 | +1 |
| weekend_on_campus | 4 | +1 |
| canteen_ratio | 5 | +1 |
| snack_drink_ratio | 6 | -1 |
| low_price_window_ratio | 7 | +1 |
| supermarket_spend | 8 | -1 |

Index-EM uses only `B=[1,H]`, where `H` is the sum of the nine signed standardized values divided by 9. These directions are fixed from the generator design, not inferred from true individual labels or chosen using test outcomes. Both the propensity intercept and index coefficient are freely fitted; there is no sign constraint.

LBE-LF uses `B=A`, so both its class and propensity models receive all 14 features. College and latent variables are not added to these matrices.

## Logistic subproblem shared by the propensity implementations

For design matrix `D`, fractional targets `t`, row weights `w`, and penalty multiplier `lambda`, the subproblem minimizes

\[
\sum_i w_i[\log(1+\exp(D_i b))-t_iD_i b]
+\frac{\lambda}{2}\sum_{j>0}b_j^2.
\]

The intercept is unpenalized. Missing row weights mean all ones. Fractional targets implement the same weighted binary logistic objective as duplication into positive and negative labels with weights `t` and `1-t`.

The code uses `scipy.optimize.minimize` with analytic gradient, `method='L-BFGS-B'`, `maxiter=200`, `ftol=1e-12`, `gtol=1e-7`, and `maxls=40`. Initial coefficients are zero unless warm-start coefficients are supplied. A nonfinite objective raises an error. An unsuccessful optimizer status also raises an error if its final infinity-norm gradient exceeds `1e-3`; an unsuccessful status with a sufficiently small gradient is accepted by the code.

## Common posterior and likelihood calculations

With observed `S=s`, the E-step is

\[
r=s+(1-s)\frac{f(1-e)}{\max(1-fe,10^{-15})}.
\]

The recorded-label mean log likelihood is calculated with `g=clip(fe,1e-15,1-1e-15)`:

\[
\ell=\operatorname{mean}\{s\log g+(1-s)\log(1-g)\}.
\]

SAR's monitored expected-complete expression is evaluated at the current `f,e` and their current `r`:

\[
Q=\operatorname{mean}\left[s\log\max(fe,10^{-15})+
(1-s)\{r\log\max(f(1-e),10^{-15})+(1-r)\log\max(1-f,10^{-15})\}\right].
\]

The logistic linear predictors are not hard-clipped, and coefficients are not constrained to finite ranges. The explicit floors/clips above stabilize probability calculations; they do not establish model correctness or identifiability.

## SAR-EM and Index-EM initialization and stopping

Let `p=mean(s)` on the training split. Initialize the class model by the logistic subproblem with target `s`, weights `s*(1-p)+(1-s)*p`, and `lambda=1/C=1`. Compute `f`, then initialize the propensity model with target `s`, weights `s+(1-s)*f`, and the same penalty.

For each iteration:

1. Compute the posterior `r` from the current `f,e`.
2. Fit propensity coefficients with target `s`, weights `r`, penalty 1, warm-started from the previous propensity coefficients.
3. Fit class coefficients with fractional target `r`, equal row weights, penalty 1, warm-started from the previous class coefficients.
4. Recompute `f,e`, recorded-label likelihood, `Q`, and the propensity vector on original unlabeled training rows.

The primary maximum is 500 iterations. A convergence check begins at iteration 12 and uses a window of 10 updates. For each unlabeled row, regress its last 10 propensity values on centered iteration positions `(-4.5,...,4.5)`; take the mean absolute slope across rows. The second statistic is `max(last 10 Q values) - first Q value in that window`, exactly as in the code. Stop only if both statistics are below `1e-4`.

The final inverse-propensity classifier refit is disabled, corresponding to `refit_classifier=False` in the authors' SAR code. Training standardization, an unpenalized intercept, the explicit propensity-feature restriction, and the current high-accuracy optimizer are disclosed choices. This is not a claim of bitwise equivalence to the historical author implementation.

## LBE-LF initialization, full-batch Adam, and stopping

Initialize `theta` by naive logistic fitting to `s` using the subproblem above with `lambda=1/initial_C=1`. Initialize every propensity coefficient in `gamma` to zero, giving `e=0.5`. Initialize all first and second Adam moments to zero.

At each update, compute `r` from the current model and the normalized gradients

\[
g_\theta=A^\top(f-r)/n,\qquad g_\gamma=B^\top(re-s)/n.
\]

Update both parameter vectors using full-batch Adam with learning rate `0.001`, `beta1=0.9`, `beta2=0.999`, and `epsilon=1e-8`. Moments use standard exponential averages and bias corrections `1-beta1^iteration` and `1-beta2^iteration`. There is no L2 penalty after initialization. Although the configuration records `l2=0`, this value is not a tunable penalty used by the implemented Adam gradients.

The primary maximum is 20,000 updates. Starting at update 1,000, check every 100 updates. Stop when both the absolute mean observed-log-likelihood change over the preceding 100 updates is less than `1e-7` and the largest absolute entry across the two normalized gradient vectors is less than `1e-5`. For exact reproduction, the stopping check uses gradients calculated immediately before the current Adam update and the likelihood calculated after that update. Final reported gradient diagnostics are recomputed at the final coefficients.

Adam is not guaranteed to increase the likelihood at every update. The number of decreases exceeding `1e-8`, likelihood traces, final coefficients, scaling values, number of iterations, convergence status, and stopping statistics are saved. The main comparison retains all primary results, including fits that reach their iteration limits.

## Iteration-budget and same-objective optimization audits

`code/convergence_sensitivity.py` revisits every primary propensity fit whose convergence flag is false. It restarts the same procedure from its specified initialization with maximum 2,000 iterations for SAR-EM/Index-EM or 60,000 updates for LBE-LF. These extended-budget results are stored separately and do not replace primary scores.

`code/polish_capped_lbe.py` conducts an additional audit on every capped primary LBE-LF fit. It starts from that fit's stored final Adam coefficients and minimizes the same unpenalized mean observed-label negative log likelihood using joint L-BFGS-B optimization. Settings are `maxiter=5000`, `ftol=1e-14`, `gtol=1e-9`, `maxls=50`, and `maxcor=30`. A separate stationarity flag records whether the final gradient infinity norm is below `1e-6`. The audit uses stable log-sigmoid/log-add-exp expressions, does not constrain coefficients, and checks that the objective has not worsened by more than `1e-9`. The audit and primary results remain distinct; any summary that substitutes audit results only for capped fits is an explicit diagnostic policy rather than the main method comparison.

## Feature ablations

Each ablation uses the same 100 baseline cohorts and train/test splits and refits Spy HGB after removing one group:

| Group | Removed features |
|---|---|
| Consumption level | `monthly_total_spend`, `avg_meal_price`, `supermarket_spend` |
| Consumption structure | `canteen_ratio`, `snack_drink_ratio`, `low_price_window_ratio` |
| Behavioral regularity | `meals_per_day`, `breakfast_rate`, `weekend_on_campus` |
| Records beyond the card | `library_visits`, `dorm_night_out`, `score_rank_pct` |

The two noise features remain in every ablation. The contrast is ablated minus full-feature Spy performance on the same seed.

## Evaluation metrics and cohort inference

Let `U` be test rows with `S=0`; a hidden positive has `Y=1,S=0`. For capacity `p%`, select `k=floor(p*|U|/100)` rows after sorting by decreasing score and then increasing original row identifier. Recorded positives never occupy a main shortlist place. Precision is selected hidden positives divided by `k`; hidden-positive recall is selected hidden positives divided by all hidden positives in `U`. The implementation records undefined precision when `k=0` and undefined recall when no hidden positives exist. It reports capacities 5%, 10%, 15%, and curves at every integer capacity from 1% to 30%.

Average precision uses `sklearn.metrics.average_precision_score`, separately on all test rows and on `U`. It is the sum of precision weighted by changes in recall at distinct prediction-score thresholds, not trapezoidal area. Legacy full-cohort shortlist metrics are retained under separate names; their capacity denominator differs from the main `U` metrics.

Cohort summaries report arithmetic means and sample standard deviations (`ddof=1`). Contrasts pair methods or ablations by scenario and seed before resampling. `code/summarize_r2.py` uses 10,000 bootstrap samples of cohorts and percentile limits 0.025 and 0.975. The pseudorandom bootstrap seed is the integer represented by the first eight bytes of the SHA-256 digest of each comparison's string key, in little-endian order. Its `bootstrap` helper drops nonfinite values and returns undefined interval bounds if fewer than two remain. Capacity bands are pointwise, not simultaneous. The intervals are unadjusted simulation intervals; they do not estimate variability across real institutions.

## Simulator-assisted conditional propensity and calibration diagnostics

These calculations use the known clean baseline generator and are unavailable to practical learners. For every test feature vector, `code/propensity_audit.py` computes

\[
e(X)=E[q(T)\mid X,Y=1],\quad
\eta(X)=P(Y=1\mid X),\quad g(X)=\eta(X)e(X).
\]

It uses a 64-node Gauss-Jacobi rule for the beta distribution and a seed-42 comparison against 128 nodes. For each primitive feature, interior values use the normal log density; values at a finite lower or upper clipping bound use the corresponding Gaussian tail point mass. Positive likelihood integrates the product of feature likelihoods over `T`. Negative likelihood is the mixture of the nonfrugal and frugal feature likelihoods with weights 0.9 and 0.1. Bayes' rule uses hardship prior 0.13. Conditional on meal count, meal price and supermarket spending, the monthly residual contributes the same normal factor to either class; the independent noise features also cancel.

The same conditional function `e(X)` is evaluated for every test row. It does not select different denominators based on true test class. The individual latent `q(T_i)` is evaluated on all rows as a distinct diagnostic denominator, not equated with `e(X)`.

The full-training raw numerator is the original naive HGB labeling score. A separate numerator model uses 80% of the original training split; the other 20% is reserved for Platt calibration using a split with `random_state=19`, stratified by training `S`. Before the logit transformation, scores are clipped to `[1e-6,1-1e-6]`. Platt calibration is a one-input `LogisticRegression(C=1e6,max_iter=1000)` fit to recorded labels. The split-raw and calibrated scores share the same 80%-trained model. Comparing a full-training raw score with a calibrated split score changes both allocation and calibration; only the split-raw comparison holds the fitted numerator fixed.

The main conditional-denominator audit protects division by flooring `e(X)` at `1e-6`, which is inactive for this baseline's positive propensities. Separate variants cap the divided output to `[0,1]`. `code/diagnostics_complete_r2.py` additionally evaluates floors `{0.05,0.10,0.20,0.40}` and upper caps `{0.90,0.99,1.00}`, uses latent and conditional denominators, and retains two explicitly `INVALID_truth_dependent_*` diagnostics that assign a constant denominator only to true negatives. Those invalid variants use test truth, are not deployable estimators, and must not be presented as part of the main comparison. The release preserves them to make the negative-case convention auditable.

## Verification-sampling simulation

`code/diagnostics_complete_r2.py` uses baseline Spy HGB predictions. All members of the top 10% of `U` are treated as correctly verified. From the remaining `M` rows, the design samples `m` in `{50,100,200,400}` without replacement. If `F` is the actual missed-positive count, the number observed in the sample is drawn as `K ~ Hypergeometric(M,F,m)`, with 1,000 draws per cohort and sample size. The random stream is `default_rng(SeedSequence([78004,seed,m]))`.

The estimated missed count is `M*K/m`, and estimated recall is `TP/(TP+M*K/m)`, where `TP` is the verified shortlist-positive count. The code records bias and root mean squared error against each cohort's true recall and the exact zero-discovery probability `hypergeom.pmf(0,M,F,m)`. Aggregation averages cohort-specific results, rather than calculating a pooled-student recall. The simulation assumes complete participation and error-free verification. It is not an institutional audit or evidence of demographic fairness.

## Reading implementation records

`results/cohorts/` records original strategies; `results/completion/` records added logistic and propensity fits. Files record scenario, seed, generated-data hash, predictions, and method diagnostics. Additional audit directories retain their own outputs. The method configuration and array hashes identify the intended paired runs. Existing verification reports document what was checked when the experiments were produced; simply reading those reports is not a claim that all training has been rerun during packaging.
