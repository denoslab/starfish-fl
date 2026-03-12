# Diagnostics Reference

All regression tasks include a `diagnostics` sub-object in their mid-artifact output. These diagnostics are computed locally at each site and provide privacy-safe summary statistics (no individual-level data is shared).

## Common Diagnostics

| Field | Description |
|-------|-------------|
| `diagnostics.vif` | Variance Inflation Factor for each feature -- values > 10 suggest multicollinearity |
| `diagnostics.residual_summary` | Summary statistics of residuals: mean, std, min, q25, median, q75, max |
| `diagnostics.cooks_distance` | Cook's distance summary: max, mean, n_influential (count > 4/n threshold) |

## OLS / Linear Regression

| Field | Description |
|-------|-------------|
| `diagnostics.shapiro_wilk` | Shapiro-Wilk normality test on residuals (statistic, p_value) |
| `diagnostics.prediction_intervals` | Summary of CI and PI widths (ci_width_mean, pi_width_mean, etc.) |

## Logistic Regression

| Field | Description |
|-------|-------------|
| `diagnostics.hosmer_lemeshow` | Hosmer-Lemeshow goodness-of-fit test (statistic, p_value, df) |
| `diagnostics.deviance_residual_summary` | Summary of deviance residuals |
| `diagnostics.prediction_intervals` | Summary of predicted probability CI widths |

## Poisson / Negative Binomial

| Field | Description |
|-------|-------------|
| `diagnostics.overdispersion` | Overdispersion ratio and p_value -- ratio > 1 suggests overdispersion |
| `diagnostics.deviance_residual_summary` | Summary of deviance residuals |
| `diagnostics.pearson_residual_summary` | Summary of Pearson residuals |
| `diagnostics.prediction_intervals` | Summary of predicted rate CI widths |

## Cox Proportional Hazards

| Field | Description |
|-------|-------------|
| `diagnostics.proportional_hazards_test` | Schoenfeld residuals test -- per-feature test_statistic and p_value |
| `diagnostics.deviance_residual_summary` | Summary of deviance residuals |

## Censored Regression (Tobit)

| Field | Description |
|-------|-------------|
| `diagnostics.vif` | Variance Inflation Factor for each feature |
| `diagnostics.residual_summary` | Residual summary for observed (uncensored) data points |
| `diagnostics.shapiro_wilk` | Normality test on residuals of observed data |
| `diagnostics.censoring_summary` | Counts: n_observed, n_right_censored, n_left_censored, pct_censored |
| `diagnostics.aic` | Akaike Information Criterion |
| `diagnostics.bic` | Bayesian Information Criterion |

## Multiple Imputation (MICE)

| Field | Description |
|-------|-------------|
| `diagnostics.vif` | VIF computed on first imputed dataset |
| `diagnostics.residual_summary` | Residual summary from first imputed dataset's OLS fit |
| `diagnostics.shapiro_wilk` | Normality test on first imputed dataset's residuals |

## R Tasks

All R-based tasks include equivalent diagnostics via the shared `r_diagnostics_utils.R` module. The output fields follow the same structure as their Python counterparts.
