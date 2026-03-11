# Task Configuration Guide for Starfish Controller

## Understanding the Tasks Field

When creating a new project, the **Tasks** field requires a JSON array that defines the federated learning workflow. Each task represents a step in the training process.

## Task Structure

Each task must have:
- **seq**: Sequential number (starting from 1)
- **model**: The ML model class name (e.g., "LogisticRegression")
- **config**: Configuration dictionary for the task

## Valid Task Examples

### Example 1: Single Task (Logistic Regression)

```json
[
  {
    "seq": 1,
    "model": "LogisticRegression",
    "config": {
      "total_round": 5,
      "current_round": 1,
      "local_epochs": 1,
      "learning_rate": 0.01
    }
  }
]
```

### Example 2: Multiple Sequential Tasks

```json
[
  {
    "seq": 1,
    "model": "LogisticRegression",
    "config": {
      "total_round": 3,
      "current_round": 1,
      "local_epochs": 1
    }
  },
  {
    "seq": 2,
    "model": "LogisticRegression",
    "config": {
      "total_round": 5,
      "current_round": 1,
      "local_epochs": 2
    }
  }
]
```

## Validation Rules

The system validates:

1. **At least one task** must be provided
2. **seq must start at 1** and be consecutive (1, 2, 3...)
3. **Each task must have** `seq`, `model`, and `config` keys
4. **seq must be** a non-negative integer
5. **model must exist** in `starfish/controller/tasks/`
6. **config must be** a non-empty dictionary

## Currently Available Models

### Logistic Regression

**Description:** Binary classification using logistic regression

**Use Case:** Predicting binary outcomes (Yes/No, 0/1, True/False)

**File Location:** `starfish/controller/tasks/logistic_regression.py`

**Dataset Requirements:** CSV with features in all columns except last, binary label (0 or 1) in last column

### Linear Regression

**Description:** Continuous value prediction using linear regression

**Use Case:** Predicting continuous numerical outcomes (e.g., prices, temperatures, life expectancy)

**File Location:** `starfish/controller/tasks/linear_regression.py`

**Dataset Requirements:** CSV with features in all columns except last, continuous target value in last column

### Statistical Logistic Regression

**Description:** Statistical logistic regression with inference outputs (coefficients, p-values, confidence intervals)

**Use Case:** Binary classification with focus on statistical significance

**File Location:** `starfish/controller/tasks/stats_models/logistic_regression_stats.py`

**Dataset Requirements:** CSV with features in all columns except last, binary outcome (0 or 1) in last column. Minimum 30 samples required.

**Statistical Outputs:**
- Coefficients (Log-Odds) with standard errors, p-values, confidence intervals
- Odds Ratios (exponentiated coefficients)
- Pseudo R-squared (McFadden's)
- Likelihood Ratio Chi-Squared statistic

### ANCOVA

**Description:** Analysis of Covariance - tests group differences while controlling for continuous covariates

**Use Case:** Comparing group means while accounting for continuous variables (e.g., treatment effects controlling for age)

**File Location:** `starfish/controller/tasks/stats_models/ancova.py`

**Dataset Requirements:** CSV with:
- First K columns: group indicators (one-hot encoded)
- Middle columns: continuous covariates
- Last column: continuous outcome variable

Minimum 30 samples required.

**Statistical Outputs:**
- Coefficients with standard errors, p-values, confidence intervals
- F-statistics for group effects
- Partial eta-squared (effect size)
- Adjusted group means

### Ordinal Logistic Regression

**Description:** Proportional Odds Model for ordered categorical outcomes (e.g., "Low", "Medium", "High")

**Use Case:** When your outcome has natural ordering but isn't continuous (e.g., disease severity levels, satisfaction ratings 1-5, education levels)

**File Location:** `starfish/controller/tasks/stats_models/ordinal_logistic_regression.py`

**Dataset Requirements:** CSV with:
- First N-1 columns: predictor variables (continuous or dummy-encoded)
- Last column: ordinal outcome variable (integer-coded: 0, 1, 2, ..., K-1)

Minimum 30 samples required. Minimum 3 categories required.

**Statistical Outputs:**
- Coefficients (Log-Odds) with standard errors, p-values, confidence intervals
- Threshold/Cut-point parameters (K-1 thresholds for K categories)
- Odds Ratios (exponentiated coefficients)
- Pseudo R-squared (McFadden's)
- AIC/BIC for model comparison

### Mixed Effects Logistic Regression

**Description:** Multilevel/hierarchical logistic regression for clustered binary data with random intercepts

**Use Case:** When observations are grouped/nested within clusters (e.g., patients within hospitals, students within schools, repeated measures within subjects)

**File Location:** `starfish/controller/tasks/stats_models/mixed_effects_logistic_regression.py`

**Dataset Requirements:** CSV with:
- First column: group/cluster identifier (e.g., hospital_id, site_id)
- Middle columns: predictor variables (continuous or dummy-encoded)
- Last column: binary outcome variable (0 or 1)

Minimum 30 samples required. Minimum 5 groups recommended.

**Configuration Options:**
```json
{
  "seq": 1,
  "model": "MixedEffectsLogisticRegression",
  "config": {
    "total_round": 1,
    "current_round": 1,
    "vcp_p": 1.0,
    "fe_p": 2.0
  }
}
```

- `vcp_p`: Prior standard deviation for variance component (default: 1.0)
- `fe_p`: Prior standard deviation for fixed effects (default: 2.0)

**Statistical Outputs:**
- Fixed Effects: Coefficients (Log-Odds), standard errors, p-values, confidence intervals
- Odds Ratios with confidence intervals
- Random Effects: Variance components, group-specific intercepts
- ICC (Intraclass Correlation Coefficient) - proportion of variance due to groups

**Example Dataset:** `examples/mixed_effects_logistic_sample.csv`

### Cox Proportional Hazards

**Description:** Time-to-event analysis using Cox Proportional Hazards regression

**Use Case:** Clinical trials, oncology, cardiology — any study with time-to-event endpoints (survival, disease progression, treatment failure)

**File Location:** `starfish/controller/tasks/cox_proportional_hazards/`

**Dataset Requirements:** CSV with features in all columns except the last two. Second-to-last column is time, last column is event indicator (1=event, 0=censored).

**Python Library:** `lifelines.CoxPHFitter`

**Example Configuration:**
```json
[
  {
    "seq": 1,
    "model": "CoxProportionalHazards",
    "config": {
      "total_round": 1,
      "current_round": 1
    }
  }
]
```

**Statistical Outputs:**
- Coefficients (log-hazard ratios) with standard errors, p-values, 95% CI
- Hazard ratios (exponentiated coefficients)
- Concordance index (C-statistic)

**Aggregation:** Inverse-variance weighted meta-analysis of log-hazard ratios

### R Cox Proportional Hazards

**Description:** Time-to-event analysis using R's `survival::coxph()`

**Use Case:** Same as Cox PH above, for researchers who prefer R

**File Location:** `starfish/controller/tasks/r_cox_proportional_hazards/`

**Dataset Requirements:** Same as Cox PH above

**R Dependencies:** `survival` (installed automatically in Docker)

**Example Configuration:**
```json
[
  {
    "seq": 1,
    "model": "RCoxProportionalHazards",
    "config": {
      "total_round": 1,
      "current_round": 1
    }
  }
]
```

### Kaplan-Meier

**Description:** Non-parametric survival estimation with log-rank test for group comparisons

**Use Case:** Visualizing survival curves, comparing survival between treatment groups, preliminary analysis before Cox regression

**File Location:** `starfish/controller/tasks/kaplan_meier/`

**Dataset Requirements:** CSV with group column (1st), feature columns (middle), time column (2nd-to-last), event column (last, 1=event, 0=censored)

**Python Library:** `lifelines.KaplanMeierFitter`, `lifelines.statistics.logrank_test`

**Example Configuration:**
```json
[
  {
    "seq": 1,
    "model": "KaplanMeier",
    "config": {
      "total_round": 1,
      "current_round": 1
    }
  }
]
```

**Statistical Outputs:**
- Survival function (time points + probabilities) per group
- Median survival time with confidence intervals
- Log-rank test statistic and p-value (for 2-group comparisons)
- At-risk tables for federated pooling

**Aggregation:** Pool at-risk tables across sites, recompute KM estimate from combined counts

### R Kaplan-Meier

**Description:** Non-parametric survival estimation using R's `survival::survfit()` and `survdiff()`

**Use Case:** Same as Kaplan-Meier above, for researchers who prefer R

**File Location:** `starfish/controller/tasks/r_kaplan_meier/`

**Dataset Requirements:** Same as Kaplan-Meier above

**R Dependencies:** `survival` (installed automatically in Docker)

**Example Configuration:**
```json
[
  {
    "seq": 1,
    "model": "RKaplanMeier",
    "config": {
      "total_round": 1,
      "current_round": 1
    }
  }
]
```

### R Logistic Regression

**Description:** Binary classification using logistic regression implemented in R (`glm(family=binomial)`)

**Use Case:** Researchers with R/biostatistics backgrounds who want to leverage R's statistical ecosystem for federated binary classification

**File Location:** `starfish/controller/tasks/r_logistic_regression/`

**Dataset Requirements:** CSV with features in all columns except last, binary label (0 or 1) in last column

**R Dependencies:** `jsonlite` (installed automatically in Docker)

**Example Configuration:**
```json
[
  {
    "seq": 1,
    "model": "RLogisticRegression",
    "config": {
      "total_round": 5,
      "current_round": 1
    }
  }
]
```

**Statistical Outputs:**
- Coefficients and intercept (from `glm`)
- Accuracy, AUC, Sensitivity, Specificity, NPV, PPV

### Poisson Regression

**Description:** Count data regression using Poisson GLM for modeling event rates

**Use Case:** Modeling count outcomes (e.g., number of hospital visits, disease incidence rates, adverse events per exposure time)

**File Location:** `starfish/controller/tasks/poisson_regression/`

**Dataset Requirements:** CSV with features in all columns except last two. Second-to-last column is offset (log-exposure), last column is count (non-negative integer).

**Example Configuration:**
```json
[
  {
    "seq": 1,
    "model": "PoissonRegression",
    "config": {
      "total_round": 1,
      "current_round": 1
    }
  }
]
```

**Statistical Outputs:**
- Coefficients (log-rate ratios) with standard errors, z-values, p-values, 95% CI
- Rate Ratios (exponentiated coefficients)
- Deviance, Pearson Chi-squared, AIC

**Aggregation:** Inverse-variance weighted meta-analysis of log-rate ratios

### R Poisson Regression

**Description:** Count data regression using R's `glm(family=poisson)`

**Use Case:** Same as Poisson above, for researchers who prefer R

**File Location:** `starfish/controller/tasks/r_poisson_regression/`

**Dataset Requirements:** Same as Poisson above

**R Dependencies:** `jsonlite` (installed automatically in Docker)

**Example Configuration:**
```json
[
  {
    "seq": 1,
    "model": "RPoissonRegression",
    "config": {
      "total_round": 1,
      "current_round": 1
    }
  }
]
```

### Negative Binomial Regression

**Description:** Overdispersed count data regression using Negative Binomial model

**Use Case:** When count data shows more variance than Poisson assumes (overdispersion), common in healthcare event counts, insurance claims, ecological abundance data

**File Location:** `starfish/controller/tasks/negative_binomial_regression/`

**Dataset Requirements:** Same as Poisson (features, offset, count)

**Example Configuration:**
```json
[
  {
    "seq": 1,
    "model": "NegativeBinomialRegression",
    "config": {
      "total_round": 1,
      "current_round": 1
    }
  }
]
```

**Statistical Outputs:**
- Coefficients (log-rate ratios) with standard errors, z-values, p-values, 95% CI
- Rate Ratios (exponentiated coefficients)
- Dispersion parameter (alpha)
- Log-likelihood, AIC

**Aggregation:** Inverse-variance weighted meta-analysis of log-rate ratios; pool dispersion parameter via weighted average

### R Negative Binomial Regression

**Description:** Overdispersed count data regression using R's `MASS::glm.nb()`

**Use Case:** Same as Negative Binomial above, for researchers who prefer R

**File Location:** `starfish/controller/tasks/r_negative_binomial_regression/`

**Dataset Requirements:** Same as Poisson/NB above

**R Dependencies:** `MASS` (included with R), `jsonlite`

**Example Configuration:**
```json
[
  {
    "seq": 1,
    "model": "RNegativeBinomialRegression",
    "config": {
      "total_round": 1,
      "current_round": 1
    }
  }
]
```

### Multiple Imputation (MICE)

**Description:** Multiple Imputation by Chained Equations for handling missing data in federated analysis

**Use Case:** Real-world clinical and epidemiological data nearly always has missing values. MICE creates multiple plausible imputed datasets, fits a linear regression on each, and pools results using Rubin's rules. Essential for any multi-site study with incomplete data.

**File Location:** `starfish/controller/tasks/multiple_imputation/`

**Dataset Requirements:**
- CSV file (no header row)
- All columns except the last: feature columns (may contain missing values as empty cells/NaN)
- Last column: continuous outcome variable
- Missing values represented as empty cells in CSV

**Example Configuration:**
```json
[
  {
    "seq": 1,
    "model": "MultipleImputation",
    "config": {
      "total_round": 1,
      "current_round": 1,
      "m": 5,
      "max_iter": 10
    }
  }
]
```

**Config Parameters:**
- **m**: Number of imputations (default: 5). More imputations reduce Monte Carlo error.
- **max_iter**: Maximum MICE iterations per imputation (default: 10)

**Statistical Outputs:**
- Pooled coefficients with standard errors, t-values, p-values, 95% CI
- Within-imputation and between-imputation variance components
- Adjusted degrees of freedom (Barnard-Rubin)
- Missingness fractions per variable
- Number of complete cases

**Aggregation:** Inverse-variance weighted meta-analysis of pooled coefficients across sites, with between-site variance incorporated via Rubin's rules

### R Multiple Imputation (MICE)

**Description:** Multiple Imputation using R's `mice` package

**Use Case:** Same as Multiple Imputation above, for researchers who prefer R. Uses `mice::mice()` for imputation, `lm()` for analysis, and `mice::pool()` for Rubin's rules.

**File Location:** `starfish/controller/tasks/r_multiple_imputation/`

**Dataset Requirements:** Same as Multiple Imputation above

**R Dependencies:** `mice`, `jsonlite` (installed automatically in Docker)

**Example Configuration:**
```json
[
  {
    "seq": 1,
    "model": "RMultipleImputation",
    "config": {
      "total_round": 1,
      "current_round": 1,
      "m": 5,
      "max_iter": 10
    }
  }
]
```

## Writing R-Based Tasks

The framework supports FL tasks written in R via the `AbstractRTask` base class. This allows researchers to use existing R algorithms within the federated learning pipeline.

### Architecture

R tasks use a Python-R bridge:
1. A Python wrapper class (extending `AbstractRTask`) handles the FL lifecycle
2. Core ML logic lives in R scripts invoked via `Rscript` subprocess
3. Communication between Python and R uses JSON files

### Creating a New R Task

1. Create a directory under `starfish/controller/tasks/` named with the snake_case version of your class name (e.g., `r_my_model/`)
2. Create three R scripts in a `scripts/` subdirectory:
   - `prepare_data.R` — validate data, return `{"valid": true, "sample_size": N}`
   - `training.R` — fit model, return coefficients and metrics as JSON
   - `aggregate.R` — aggregate mid-artifacts from all sites, return averaged model
3. Create `task.py` with a class extending `AbstractRTask` that sets `r_script_dir`
4. Create an empty `__init__.py`

### R Script Interface

Each R script receives two command-line arguments:
- `input.json` — input data (config, data paths, previous model, mid-artifacts)
- `output.json` — path where the script must write its JSON result

**Input JSON structure:**
```json
{
  "run_id": 42,
  "data_path": "/path/to/dataset",
  "config": {"total_round": 5, "current_round": 1},
  "sample_size": 200,
  "previous_model": null,
  "mid_artifacts": null
}
```

**Output JSON structure (training):**
```json
{
  "sample_size": 200,
  "coef_": [[1.2, -0.5, 0.8]],
  "intercept_": [0.3],
  "metric_acc": 0.85,
  "metric_auc": 0.88
}
```

### Example Task Class

```python
import os
from starfish.controller.tasks.abstract_r_task import AbstractRTask

class RMyModel(AbstractRTask):
    def __init__(self, run):
        self.r_script_dir = os.path.join(
            os.path.dirname(__file__), 'scripts')
        super().__init__(run)
```

## Model Diagnostics and Prediction Intervals

All regression tasks now include a `diagnostics` sub-object in their mid-artifact output. These diagnostics are computed locally at each site and provide privacy-safe summary statistics (no individual-level data is shared).

### Common Diagnostics (all regression models)

| Field | Description |
|-------|-------------|
| `diagnostics.vif` | Variance Inflation Factor for each feature — values > 10 suggest multicollinearity |
| `diagnostics.residual_summary` | Summary statistics of residuals: mean, std, min, q25, median, q75, max |
| `diagnostics.cooks_distance` | Cook's distance summary: max, mean, n_influential (count > 4/n threshold) |

### OLS / Linear Regression Diagnostics

| Field | Description |
|-------|-------------|
| `diagnostics.shapiro_wilk` | Shapiro-Wilk normality test on residuals (statistic, p_value) |
| `diagnostics.prediction_intervals` | Summary of confidence and prediction interval widths (ci_width_mean, pi_width_mean, etc.) |

### Logistic Regression Diagnostics

| Field | Description |
|-------|-------------|
| `diagnostics.hosmer_lemeshow` | Hosmer-Lemeshow goodness-of-fit test (statistic, p_value, df) |
| `diagnostics.deviance_residual_summary` | Summary of deviance residuals |
| `diagnostics.prediction_intervals` | Summary of predicted probability CI widths |

### Poisson / Negative Binomial Diagnostics

| Field | Description |
|-------|-------------|
| `diagnostics.overdispersion` | Overdispersion ratio (Pearson chi2 / df_resid) and p_value — ratio > 1 suggests overdispersion |
| `diagnostics.deviance_residual_summary` | Summary of deviance residuals |
| `diagnostics.pearson_residual_summary` | Summary of Pearson residuals |
| `diagnostics.prediction_intervals` | Summary of predicted rate CI widths |

### Cox Proportional Hazards Diagnostics

| Field | Description |
|-------|-------------|
| `diagnostics.proportional_hazards_test` | Schoenfeld residuals test for PH assumption — per-feature test_statistic and p_value |
| `diagnostics.deviance_residual_summary` | Summary of deviance residuals |

### Multiple Imputation (MICE) Diagnostics

| Field | Description |
|-------|-------------|
| `diagnostics.vif` | VIF computed on first imputed dataset |
| `diagnostics.residual_summary` | Residual summary from first imputed dataset's OLS fit |
| `diagnostics.shapiro_wilk` | Normality test on first imputed dataset's residuals |

### R Tasks

All R-based tasks include equivalent diagnostics via the shared `r_diagnostics_utils.R` utility module. The diagnostics fields follow the same structure as their Python counterparts.

## Configuration Parameters Explained

### Required Parameters

- **total_round**: Total number of federated learning rounds (how many times models are aggregated)
- **current_round**: The round to start with (in most cases we start with 1)

### Optional Parameters

- **local_epochs**: Number of epochs each site trains locally (default: 1 in code)
- **test_size**: Proportion of data used for testing (default: 0.2 in code)
- **learning_rate**: Learning rate for training (if applicable)
- **n_group_columns**: (ANCOVA only) Number of columns representing group membership
- **vcp_p**: (Mixed Effects Logistic Regression only) Prior SD for variance components (default: 1.0)
- **fe_p**: (Mixed Effects Logistic Regression only) Prior SD for fixed effects (default: 2.0)
- **m**: (Multiple Imputation only) Number of imputed datasets to create (default: 5)
- **max_iter**: (Multiple Imputation only) Max MICE iterations per imputation (default: 10)
- **description**: Optional description of what this task does

## Step-by-Step: Creating a Project

1. **Go to**: http://localhost:8001/controller/projects/new/
2. **Project Name**: Enter "Test Project"
3. **Project Description**: Enter "Testing federated learning"
4. **Tasks**: Copy and paste this:
   ```json
   [{"seq":1,"model":"LogisticRegression","config":{"total_round":5,"current_round":1}}]
   ```
5. **Click Submit**
