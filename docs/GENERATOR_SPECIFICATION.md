# Complete specification of the R2 simulation generators

This document specifies the generated data used for the R2 experiments. The executable authority is `code/generator_r2.py`; `protocol/generator_parameters.csv` records the baseline coefficients and `protocol/scenario_configurations.csv` records the scenario changes. All coefficients and distributions are stipulated simulation inputs, not estimates fitted to external student records. The public campus-data context motivates feature names; it does not validate this joint distribution.

`code/synth_original.py` is retained as the inherited source and supplies the ordered feature list. Its historical introductory comments do not supersede the R2 specification or the manuscript's restricted interpretation of simulation evidence. R2 results are generated through `generator_r2.generate`, not by treating historical comments as an empirical calibration claim.

## Variables and feature order

For student index `i`, let `Y` be true hardship, `S` the recorded-positive indicator, `T` latent typicality, `F` frugal membership among students without hardship, and `C` a binary college indicator. The default cohort size is 12,000. Array position `0,...,n-1` supplies the generated student identifier used for deterministic tie breaking.

Only the following 14 columns, in this order, are model features:

1. `monthly_total_spend`
2. `avg_meal_price`
3. `meals_per_day`
4. `breakfast_rate`
5. `weekend_on_campus`
6. `canteen_ratio`
7. `snack_drink_ratio`
8. `low_price_window_ratio`
9. `supermarket_spend`
10. `library_visits`
11. `dorm_night_out`
12. `score_rank_pct`
13. `noise_a`
14. `noise_b`

The returned bookkeeping columns `y_true`, `labeled`, `typicality`, `latent_typicality_all`, `q_individual`, `college`, and `frugal` are excluded from practical model inputs. `typicality` stores `T` for `Y=1` and `NaN` for `Y=0`; `latent_typicality_all` stores `T` for every row. The shared latent normal variable in the unobserved-factor scenario is regenerated from its stream and is not returned as an observed feature.

## Baseline latent draws and random streams

The main stream is `numpy.random.default_rng(seed)`. It generates, in order:

- `Y = (Uniform(0,1) < poor_rate)`, with default `poor_rate=0.13`;
- `T ~ Beta(2.2,1.8)` for every row, independently of `Y`;
- `F = (Y=0) AND (Uniform(0,1) < frugal_rate)`, with default `frugal_rate=0.10`;
- one independent Gaussian-error vector for each of the 11 primitive features in the coefficient-table order below;
- the monthly-spending error vector;
- independent standard-normal vectors `noise_a` and `noise_b`;
- the uniform vector used to draw recorded labels among positives.

A separate stream `numpy.random.default_rng(seed + 1000003)` generates `C` uniformly from `{0,1}`, then `L ~ Normal(0,1)`. If false-positive labels are enabled, this stream subsequently generates their uniform vector. Keeping the auxiliary stream separate preserves the default main-stream feature and label draws. The college and `L` vectors are generated even when the selected scenario does not use them.

The beta mean is `E[T]=2.2/(2.2+1.8)=0.55`. In the baseline, features share dependence through `Y`, `F`, and `T`. Conditional primitive feature errors are independent; monthly spending is explicitly derived from other features. `C` has no baseline feature effect. In the unobserved-factor scenario, shared `L` adds further dependence between features and labeling.

## Primitive feature means, errors, and bounds

Define `clip(x,l,u)=min(max(x,l),u)`; a missing upper bound means no upper truncation. For primitive feature `j`, let `a_j`, `b_j`, `r_j`, `f_j`, `sigma_j`, `l_j`, and `u_j` be the table entries. At default overlap `o=0`, the conditional mean is

\[
\mu_j=\begin{cases}
T a_j+(1-T)b_j,&Y=1,\\
f_j,&Y=0,F=1,\\
r_j,&Y=0,F=0.
\end{cases}
\]

For overlap parameter `o`, the frugal mean is replaced by

\[
f_j(o)=(1-o)f_j+o(0.55a_j+0.45b_j).
\]

For error multiplier `v=noise_scale`, draw independent `epsilon_j ~ Normal(0,(v sigma_j)^2)`. The recorded primitive feature is

\[
X_j=\operatorname{clip}(\mu_j+\epsilon_j+I_{\rm unobserved}\,0.5\sigma_j L,l_j,u_j).
\]

The extra unobserved-factor term uses the original `sigma_j`, not `v sigma_j`.

| Feature | Hardship typical `a` | Hardship atypical `b` | Nonhardship nonfrugal `r` | Nonhardship frugal `f` | Gaussian SD `sigma` | Lower | Upper |
|---|---:|---:|---:|---:|---:|---:|---:|
| avg_meal_price | 5.2 | 8.5 | 10.5 | 7.5 | 1.2 | 2 | None |
| meals_per_day | 2.7 | 2.0 | 1.6 | 2.3 | 0.35 | 0.3 | 3.5 |
| breakfast_rate | 0.78 | 0.45 | 0.28 | 0.50 | 0.10 | 0 | 1 |
| weekend_on_campus | 0.85 | 0.60 | 0.42 | 0.55 | 0.12 | 0 | 1 |
| canteen_ratio | 0.88 | 0.70 | 0.55 | 0.72 | 0.08 | 0 | 1 |
| snack_drink_ratio | 0.05 | 0.15 | 0.30 | 0.15 | 0.06 | 0 | 1 |
| low_price_window_ratio | 0.55 | 0.30 | 0.12 | 0.30 | 0.10 | 0 | 1 |
| supermarket_spend | 60 | 150 | 380 | 160 | 60 | 0 | None |
| library_visits | 14 | 10 | 8 | 10 | 4 | 0 | None |
| dorm_night_out | 1.5 | 3.0 | 5.5 | 3.0 | 1.2 | 0 | None |
| score_rank_pct | 0.44 | 0.50 | 0.52 | 0.50 | 0.18 | 0 | 1 |

These are continuous generated variables; count-like measurements are not rounded to integers. Clipping creates point masses at finite bounds, rather than a truncated-and-renormalized normal density.

Monthly spending is generated after these bounds have been applied:

\[
X_{\rm spend}=30X_{\rm meals}X_{\rm price}+X_{\rm supermarket}
+\epsilon_{\rm spend},\qquad
\epsilon_{\rm spend}\sim N(0,(40v)^2).
\]

Monthly spending has no additional clipping. Finally, `noise_a` and `noise_b` are independent `Normal(0,1)` variables and are not scaled by `noise_scale`.

## Eight labeling mechanisms

Write `sigmoid(x)=1/(1+exp(-x))`. A latent probability `q` is defined for every row. The clean-label draw is `S = Y AND (Uniform(0,1) < q)`, so `q` is a labeling probability for true positives; it is not the label probability for negatives.

The code constructs a 128-node Gauss-Jacobi rule using `roots_jacobi(128,0.8,1.2)`, transforms nodes to `[0,1]`, and normalizes weights to sum to one. Expectations in the following rules use this quadrature approximation under `Beta(2.2,1.8)`.

| Manuscript mechanism | Code selection | Rule for `q` |
|---|---|---|
| Linear inverse baseline | `linear_inverse` with defaults | `clip(0.88 - 0.62*T, 0.12, 0.88)` |
| Constant / SCAR | `scar`, or the cached `beta_0.0` scenario | `0.55` |
| Logistic | `logistic` | `sigmoid(A - 4*T)` |
| Threshold | `threshold` | `0.20` if `T > h`; otherwise `0.725` |
| Two-college heterogeneous | `heterogeneous` | `clip(0.891 - 0.62*T,0.12,0.92)` for `C=0`; `0.55` for `C=1` |
| Reverse | `reverse` | `clip(0.209 + 0.62*T,0.12,0.92)` |
| Nonmonotonic | `nonmonotonic` | `0.55 + 0.30*(cos(2*pi*T) - E[cos(2*pi*T)])` |
| Unobserved factor | `unobserved` | `sigmoid(A - 4*T + 1.5*L)` |

`A` solves `E[sigmoid(A-4T)]=0.55` by `brentq` on `[-10,10]`; the code gives approximately `2.4441462432149645`. The threshold is `h=Beta(2.2,1.8).ppf(2/3)`, approximately `0.6691648448230779`. The cosine centering constant is approximately `-0.28391865652501447`. Exact computation, rather than these rounded display constants, is used during generation.

The nonmonotonic and logistic probabilities are not clipped. All mechanisms are checked for probabilities outside `[0,1]`, which produce an error. The unobserved-factor mechanism uses the same `A` as logistic labeling; it is not re-centered to force an identical mean label rate after adding `L`. The baseline latent probability declines from `0.88` to `0.26`; its `0.12` lower bound is inactive, and its population mean among positives is `0.539`.

### Slope sensitivity

For `b` in `{0,0.3,0.62,0.9}`, the slope scenario uses

\[
q_b(T)=\operatorname{clip}(0.55+0.55b-bT,0.10,0.92).
\]

The zero-slope setting supplies the constant-mechanism results. It is reused, not independently resampled. At `b=0.62`, the intercept is `0.891`, so this setting is not the main baseline with intercept `0.88` and bounds `[0.12,0.88]`.

### False-positive recorded labels

At stress probability `rho` in `{0.01,0.03}`, after the clean draw the code additionally sets `S=1` for each `Y=0` row whose auxiliary uniform draw is below `rho`. Thus `rho=P(S=1|Y=0)`, not the fraction of recorded positives that are incorrect. The stored `q_individual` remains the latent positive-label rule evaluated at every row; it is not overwritten by `rho` for negatives. These scenarios violate the clean-positive-label assumption.

## Alternative correlated-factor generator

Only the scenario `alternative_factors` calls `generate_factors(n,poor_rate,seed)`. This is a separate stipulated generator, not recovery of a previously missing implementation. The dispatcher passes only `n`, `poor_rate`, and `seed` to it: the baseline `frugal_rate`, mechanism selection, overlap, noise-scale, and false-positive parameters are not crossed with this scenario in the executed plan.

Using `numpy.random.default_rng(seed)`, draw `Z=(Z_1,Z_2,Z_3)` from a zero-mean multivariate normal with correlation matrix

\[
\Sigma=\begin{pmatrix}
1&-0.28&0.36\\
-0.28&1&0.18\\
0.36&0.18&1
\end{pmatrix}.
\]

Draw `C` uniformly from `{0,1}` from this same stream. Define `SES=Z_1+0.25(2C-1)` and `Y=I(SES<c)`. The cutoff solves

\[
0.5\Phi(c-0.25)+0.5\Phi(c+0.25)=\texttt{poor_rate}
\]

by `brentq` on `[-5,5]`. At `poor_rate=0.13`, `c` is approximately `-1.1616573492113493`. Therefore prevalence is specified in the population, not forced to exactly 13% in each realized cohort. Latent typicality is `T=1-Phi(Z_1)`. It is not beta-distributed in this alternative generator and does not use the college-shifted `SES` in its definition.

For the same 11 primitive features and clipping bounds as above, set

\[
\bar p_j=0.55a_j+0.45b_j,\quad
m_j=(\bar p_j+r_j)/2,\quad
s_j=\max\{0.65|r_j-\bar p_j|,\sigma_j\}.
\]

Then generate

\[
X_j=\operatorname{clip}\left[m_j+s_j\{\lambda_j^\top Z+0.18(2C-1)\}
+\sigma_j\sqrt{3/5}\,V_j,l_j,u_j\right],
\qquad V_j\sim t_5.
\]

The independent `t_5` errors are scaled by `sqrt(3/5)` so their standard deviation before multiplication by `sigma_j` is one. Their loadings and resulting means/scales are:

| Feature | Loading on `Z1` | `Z2` | `Z3` | `m_j` | `s_j` |
|---|---:|---:|---:|---:|---:|
| avg_meal_price | 1 | -0.15 | 0.10 | 8.5925 | 2.47975 |
| meals_per_day | -0.2 | 0.15 | 0.6 | 1.9925 | 0.51025 |
| breakfast_rate | -0.2 | 0.4 | 0.4 | 0.45575 | 0.228475 |
| weekend_on_campus | -0.15 | 0.15 | 0.6 | 0.57875 | 0.206375 |
| canteen_ratio | -0.5 | 0.1 | 0.3 | 0.6745 | 0.16185 |
| snack_drink_ratio | 0.6 | -0.1 | -0.2 | 0.1975 | 0.13325 |
| low_price_window_ratio | -0.6 | 0.1 | 0.1 | 0.27875 | 0.206375 |
| supermarket_spend | 0.8 | 0 | -0.1 | 240.25 | 181.675 |
| library_visits | -0.1 | 0.8 | 0.3 | 10.1 | 4 |
| dorm_night_out | 0.4 | -0.3 | -0.4 | 3.8375 | 2.16125 |
| score_rank_pct | 0.15 | -0.7 | 0.1 | 0.4935 | 0.18 |

Monthly spending is again `30*meals_per_day*avg_meal_price + supermarket_spend`, now with independent error `40*sqrt(3/5)*t_5` and no additional clipping. Independent standard-normal `noise_a` and `noise_b` follow. Labels use `q=clip(0.88-0.62T,0.12,0.88)` and `S=Y AND (Uniform(0,1)<q)`. All `frugal` bookkeeping values are zero; there is no separate frugal subgroup in this generator. The draw order is correlated normals, colleges, each primitive t-error vector, monthly t-error, two noise vectors, and label uniforms.

## Executed scenarios and pairing

| Analysis | Cohorts | Seeds | Change from default |
|---|---:|---|---|
| Main baseline | 100 | 42--141 | None |
| Each slope setting | 100 | 42--141 | Slope and bounds specified above |
| Each of the six additional mechanisms | 100 | 42--141 | Mechanism only |
| Low and high hardship prevalence | 50 each | 42--91 | `poor_rate=0.05` or `0.25` |
| No or larger frugal subgroup | 50 each | 42--91 | `frugal_rate=0` or `0.25` |
| Intermediate or full frugal overlap | 50 each | 42--91 | `overlap=0.5` or `1` |
| Larger primitive and monthly errors | 50 each | 42--91 | `noise_scale=1.5` or `2` |
| False-positive recorded labels | 50 each | 42--91 | `false_positive_rate=0.01` or `0.03` |
| Alternative factors | 100 | 42--141 | Separate factor generator |
| Each of four feature ablations | 100 | 42--141 | Baseline data; specified features dropped from fitting |
| Baseline logistic learner comparison | 100 | 42--141 | Baseline data; model family changed |
| Each Spy-grid cell | 20 | 42--61 | Fraction in `{0.05,0.15,0.30}` and quantile in `{1,5,10}` |

The Spy-grid reference `fraction=0.15, quantile=5` reuses the first 20 baseline fits; the plan runs only the other eight grid cells. The eight-mechanism comparison consists of baseline linear inverse, cached constant/zero-slope, and the six additional mechanisms. The expanded logistic and propensity estimators cover 1,700 main/slope/mechanism/stress/alternative scenario-seed cases; repeated seeds across different generators are paired design labels, not additional independent samples from one population.

Every scenario uses a 70% training / 30% test split, with `train_test_split(..., random_state=7, stratify=[str(Y_i)+str(S_i)])`. This use of simulation truth is disclosed experimental stratification; it is not an operational splitting procedure. Identical scenario, seed, generated-array hash, and student identifiers establish method pairing. Training uses only the allowed feature columns and recorded labels, except explicitly named true-label reference models. See `ALGORITHM_SPECIFICATION.md` for the fitting procedures and evaluation definitions.
