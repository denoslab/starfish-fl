# Starfish Controller - User Guide

## Web Interface Guide

### Initial Setup - Register Your Site

When you first access the controller at `http://localhost:8001/controller/`, you'll see for example:

```
Site e5c7cc7c-9b8b-4cde-8465-d67aa499f0b8 is currently NOT registered.
```

**Action Required:**
1. Enter a **Site Name** (e.g., "Hospital A", "Research Lab 1", "Site Alpha")
2. Enter a **Site Description** (e.g., "Cardiac imaging data site", "Cancer research facility")
3. Click **"Register Site"** or **"Submit"** button

After registration, your site will be able to participate in federated learning projects.

---

## Main Navigation

### Home
**Purpose:** Central dashboard for your site

**What you'll see:**
- Current site registration status
- List of all projects you're participating in
- Quick actions for each project
- Site management options

**Actions available:**
- Update site name/description
- Deregister site
- View project details
- Leave projects

---

### Create Project (Coordinator Role)

**Purpose:** Create a new federated learning project where you act as the coordinator

**When to use:**
- You want to begin a new federated learning collaboration
- You have a machine learning task that requires data from multiple sites
- You need to manage and control the overall project setup

**Required information:**
- **Project Name**: Descriptive name (e.g., "Diabetes Prediction Study")
- **Project Description**: What the project is about
- **Task Configuration**: See [TASK_GUIDE.md](TASK_GUIDE.md) for all 18 available models

**Process:**
1. Fill in project details
2. Submit to create
3. Wait for participants to join
4. Upload your local dataset
5. Start training runs
6. Monitor progress
7. View aggregated results

---

### Join Project (Participant Role)

**Purpose:** Join an existing federated learning project as a participant

**When to use:**
- You want to contribute your local data to a project coordinated by someone else
- You want your model to benefit from collaborative training with other participants

**Process:**
1. Browse available projects
2. Review project requirements
3. Click "Join" on a project
4. Upload your dataset (matching project schema)
5. Participate in training rounds
6. View results

---

## Dataset Requirements

### General Format

All datasets must be uploaded as **CSV files** with the following specifications:

- **Format**: CSV (Comma-Separated Values)
- **Header**: **NO HEADER ROW** - Data should start from the first line
- **Structure**: Features in columns, with the **label/target in the last column**
- **Encoding**: UTF-8

### LogisticRegression Dataset Format

For binary classification tasks using LogisticRegression:

**Structure:**
```
feature1,feature2,feature3,...,featureN,label
value1,value2,value3,...,valueN,0
value1,value2,value3,...,valueN,1
...
```

**Requirements:**
- All columns except the last are **features** (predictors/independent variables)
- The **last column** must be the **binary label** (0 or 1)
- **No headers** - the file should contain only numeric data
- **No missing values** - ensure data is clean and complete
- **Numeric values only** - all features and labels must be numbers

**Example CSV content:**
```
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
6.2,3.4,5.4,2.3,1
5.9,3.0,5.1,1.8,1
```

In this example:
- First 4 columns are features
- Last column (0 or 1) is the binary classification label

### LinearRegression Dataset Format

For continuous value prediction using LinearRegression:

**Structure:**
```
feature1,feature2,feature3,...,featureN,target
value1,value2,value3,...,valueN,continuous_value
...
```

**Requirements:**
- All columns except the last are **features**
- The **last column** must be a **continuous numerical value** (e.g., 45.2, 1250.75)
- **No headers** - data only
- **No missing values**
- **Numeric values only**

**Example use cases:** Predicting house prices, life expectancy, temperature, sales volume

### LogisticRegressionStats Dataset Format

For statistical binary classification with inference outputs:

**Structure:** Same as LogisticRegression (binary label in last column)

**Additional Requirements:**
- **Minimum 30 samples** required for statistical validity
- Features should be meaningful for interpretation (avoid high-cardinality dummy encoding)

**What's Different:**
- Outputs include statistical significance (p-values, confidence intervals)
- Provides odds ratios for understanding feature effects
- Typically run for 1 round (statistical meta-analysis, not iterative training)

**Example outputs:** Coefficients with p-values, 95% confidence intervals, pseudo R-squared

### Ancova Dataset Format

For Analysis of Covariance (comparing groups while controlling for covariates):

**Structure:**
```
group1,group2,...,covariate1,covariate2,...,outcome
0,1,...,25.3,1.2,...,45.7
1,0,...,30.1,1.8,...,50.2
...
```

**Requirements:**
- **First K columns**: Group indicators (one-hot encoded, 0 or 1)
- **Middle columns**: Continuous covariates (control variables)
- **Last column**: Continuous outcome variable
- **Minimum 30 samples** required

**Example:** Testing treatment effects (groups) while controlling for age and baseline measurements (covariates)

### Preprocessing Guidelines

**Critical:** All sites must have **identical feature sets** (same columns in same order)

**Why preprocessing matters:**
- One-hot encoding creates different columns if done separately per site
- Example: Site 1 has categories A,B,C → creates columns for A,B,C
- Site 2 has categories A,B,D → creates columns for A,B,D (mismatch!)

**Solution:** Use `starfish/preprocess_dataset.py`
1. Run preprocessing on the full dataset once
2. It will return the preprocessed data into two site-specific csv files
3. Upload each file to the respective site

This ensures all sites have identical feature columns.

### Data Privacy Note

- Datasets are stored **locally** on your site only
- Raw data never leaves your site
- Only model weights/statistics are shared during federated learning

---

## Understanding Your Results

After a successful federated learning run, you can download three types of files that contain different information about the training process.

### Output Files Overview

| File | Purpose | Who Gets It | Contains |
|------|---------|-------------|----------|
| **logs.txt** | Training process details | All participants | Execution logs, debugging info |
| **artifacts** | Final aggregated results | All participants | Final model + metrics |
| **mid-artifacts** | Intermediate results | Coordinator only | Local model before aggregation |

### 1. Logs (logs.txt)

**Purpose:** Records the entire training process step-by-step

### 2. Artifacts (Final Results)

**Purpose:** Contains the final trained model and performance metrics after aggregation

#### For Classification Models (LogisticRegression)

**JSON Structure:**
(Note: it might differ between models)
```json
{
  "sample_size": 569,
  "coef_": [[-0.566, -0.668, ...]],
  "intercept_": [0.561],
  "metric_acc": 0.973,
  "metric_auc": 0.969,
  "metric_sensitivity": 0.985,
  "metric_specificity": 0.953,
  "metric_npv": 0.976,
  "metric_ppv": 0.972
}
```

**Field Explanations:**

| Field | Meaning | Interpretation |
|-------|---------|----------------|
| **sample_size** | Total number of samples used | How much data contributed to training |
| **coef_** | Model coefficients (weights) | Learned feature importance (one per feature) |
| **intercept_** | Model bias/intercept | Baseline prediction value |
| **metric_acc** | Accuracy | Overall correctness (0-1, higher is better) |
| **metric_auc** | Area Under ROC Curve | Model's ability to distinguish classes (0-1) |
| **metric_sensitivity** | True Positive Rate (Recall) | % of actual positives correctly identified |
| **metric_specificity** | True Negative Rate | % of actual negatives correctly identified |
| **metric_npv** | Negative Predictive Value | When model predicts negative, how often is it right? |
| **metric_ppv** | Positive Predictive Value (Precision) | When model predicts positive, how often is it right? |


**Performance Guide:**
- **90%+ accuracy**: Excellent model
- **80-90% accuracy**: Good model
- **70-80% accuracy**: Acceptable model
- **<70% accuracy**: May need improvement

#### For Regression Models (LinearRegression)

**JSON Structure:**
(Note: it might differ between models)
```json
{
  "sample_size": 450,
  "coef_": [[0.234, -0.567, 1.234, ...]],
  "intercept_": [45.23],
  "metric_mse": 12.45,
  "metric_mae": 2.87,
  "metric_r2": 0.89
}
```

**Field Explanations:**

| Field | Meaning | Interpretation |
|-------|---------|----------------|
| **metric_mse** | Mean Squared Error | Average squared difference between predicted and actual (lower is better) |
| **metric_mae** | Mean Absolute Error | Average absolute difference (lower is better) |
| **metric_r2** | R-squared (Coefficient of Determination) | Proportion of variance explained (0-1, higher is better) |

**R² Performance Guide:**
- **0.9+**: Excellent fit
- **0.7-0.9**: Good fit
- **0.5-0.7**: Moderate fit
- **<0.5**: Weak fit, may need feature engineering

#### For Statistical Models (LogisticRegressionStats, Ancova)

**JSON Structure:**
(Note: it might differ between models)
```json
{
  "sample_size": 250,
  "coefficients": [0.523, -0.234, 1.456],
  "std_errors": [0.112, 0.089, 0.234],
  "p_values": [0.001, 0.008, 0.000],
  "conf_int_lower": [0.303, -0.408, 0.997],
  "conf_int_upper": [0.743, -0.060, 1.915],
  "odds_ratios": [1.687, 0.791, 4.289],
  "pseudo_r2": 0.234,
  "chi2_statistic": 45.67,
  "aic": 234.56
}
```

**Field Explanations:**

| Field | Meaning | Statistical Interpretation |
|-------|---------|---------------------------|
| **coefficients** | Regression coefficients | Log-odds (LogisticRegressionStats) or effect sizes (Ancova) |
| **std_errors** | Standard errors | Uncertainty in coefficient estimates |
| **p_values** | Statistical significance | P < 0.05 indicates significant effect |
| **conf_int_lower/upper** | 95% Confidence Intervals | Range of plausible coefficient values |
| **odds_ratios** | (LogisticRegressionStats only) | Multiplicative effect on odds of outcome |
| **pseudo_r2** | Pseudo R-squared | Model fit (similar to R² for regression) |
| **chi2_statistic** | Chi-squared test statistic | Overall model significance |
| **aic** | Akaike Information Criterion | Model comparison metric (lower is better) |

**Interpreting p-values:**
- **p < 0.001**: Highly significant (****)
- **p < 0.01**: Very significant (***)
- **p < 0.05**: Significant (**)
- **p >= 0.05**: Not statistically significant

**Interpreting Odds Ratios (LogisticRegressionStats):**
- **OR = 1**: No effect
- **OR > 1**: Increases odds of outcome (e.g., OR=2.5 means 2.5x higher odds)
- **OR < 1**: Decreases odds of outcome (e.g., OR=0.5 means 50% lower odds)

#### For Survival Models (CoxProportionalHazards)

**JSON Structure:**
```json
{
  "sample_size": 300,
  "coef": [-0.234, 0.567],
  "se": [0.112, 0.089],
  "hazard_ratio": [0.791, 1.763],
  "p_values": [0.037, 0.001],
  "ci_lower": [-0.454, 0.392],
  "ci_upper": [-0.014, 0.742],
  "concordance_index": 0.72,
  "diagnostics": { ... }
}
```

| Field | Meaning | Interpretation |
|-------|---------|----------------|
| **hazard_ratio** | Hazard ratios | HR > 1 = higher risk, HR < 1 = lower risk |
| **concordance_index** | C-statistic | Discrimination ability (0.5 = random, 1.0 = perfect) |

#### For Count Data Models (PoissonRegression, NegativeBinomialRegression)

**JSON Structure:**
```json
{
  "sample_size": 500,
  "coef": [0.523, -0.234],
  "se": [0.112, 0.089],
  "rate_ratios": [1.687, 0.791],
  "deviance": 234.56,
  "pearson_chi2": 245.12,
  "diagnostics": { ... }
}
```

| Field | Meaning | Interpretation |
|-------|---------|----------------|
| **rate_ratios** | Incidence rate ratios | RR > 1 = higher rate, RR < 1 = lower rate |
| **deviance** | Deviance statistic | Model fit (closer to df_resid = good fit) |

#### Understanding Model Diagnostics

All regression tasks now include a `diagnostics` sub-object with privacy-safe summary statistics:

| Diagnostic | What It Tells You | Concerning Values |
|-----------|-------------------|-------------------|
| **VIF** | Multicollinearity between features | VIF > 10 suggests problems |
| **Shapiro-Wilk p-value** | Are residuals normally distributed? | p < 0.05 = non-normal |
| **Hosmer-Lemeshow p-value** | Is the logistic model well-calibrated? | p < 0.05 = poor fit |
| **Overdispersion ratio** | Is there more variance than expected? | ratio >> 1 = overdispersed |
| **Cook's distance n_influential** | How many outliers affect the model? | Many influential points = investigate |
| **PH test p-value** (Cox) | Is the proportional hazards assumption met? | p < 0.05 = assumption violated |

See [TASK_GUIDE.md](TASK_GUIDE.md) for the complete diagnostics field reference.

### 3. Mid-Artifacts (Intermediate Results)

**Purpose:** Contains your local model BEFORE aggregation with other participants

**Who gets this:** Coordinator only (for aggregation purposes)

**Content:** Same structure as artifacts, but represents only YOUR site's local model. Includes model diagnostics.

**Use case:**
- Compare local vs. aggregated model performance
- Understand how federated learning improved results
- Debug if aggregation isn't working as expected
- Review model diagnostics (VIF, residuals, goodness-of-fit) per site

### Understanding File Names:**
- Format: `{run_id}-{task_seq}-{round_number}-{type}`
- Example: `7-1-6-artifacts` = Run 7, Task 1, Round 6, final artifacts
- Example: `7-1-6-logs.txt` = Run 7, Task 1, Round 6, logs

### Performance Metrics Explained

**Confusion Matrix Context:**
```
                    Predicted
                 Negative  Positive
Actual Negative     TN       FP
       Positive     FN       TP
```

- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative  
- **FP (False Positive)**: Incorrectly predicted positive
- **FN (False Negative)**: Incorrectly predicted negative

**Calculated Metrics:**
- **Accuracy** = (TP + TN) / Total
- **Sensitivity (Recall)** = TP / (TP + FN) - "How many actual positives did we catch?"
- **Specificity** = TN / (TN + FP) - "How many actual negatives did we identify?"
- **PPV (Precision)** = TP / (TP + FP) - "When we predict positive, how often are we right?"
- **NPV** = TN / (TN + FN) - "When we predict negative, how often are we right?"

---

## Quick Reference

### Project Roles

| Role | Capabilities |
|------|-------------|
| **Coordinator** | Create project, approve participants, start runs, aggregate models, also trains on their own data |
| **Participant** | Join projects, upload data, train locally, view results |

**Note:** The coordinator is also a participant! They:
- Upload their own dataset
- Train a local model on their data
- Send their weights to the router (like other participants)
- **PLUS**: Download all participants' weights and aggregate them
- **PLUS**: Calculate final metrics and distribute the global model

**Think of it as:** Coordinator = Participant + Aggregation duties

### Task Types

| Task | Description | Dataset Format |
|------|-------------|----------------|
| **LogisticRegression** | Binary classification | CSV with features + binary label (0/1) |
| **LinearRegression** | Continuous value prediction | CSV with features + continuous target |
| **LogisticRegressionStats** | Statistical binary classification | CSV with features + binary label, min 30 samples |
| **Ancova** | Group comparison with covariates | CSV with groups + covariates + outcome |
| **OrdinalLogisticRegression** | Ordered categorical outcomes | CSV with features + ordinal label (0,1,2,...) |
| **MixedEffectsLogisticRegression** | Clustered binary data | CSV with group ID + features + binary label |
| **CoxProportionalHazards** | Survival analysis | CSV with features + time + event (0/1) |
| **KaplanMeier** | Non-parametric survival | CSV with group + features + time + event |
| **PoissonRegression** | Count data (event rates) | CSV with features + offset + count |
| **NegativeBinomialRegression** | Overdispersed count data | CSV with features + offset + count |
| **MultipleImputation** | Missing data handling (MICE) | CSV with features (may have NaN) + outcome |

R versions of LogisticRegression, CoxPH, KaplanMeier, Poisson, NegBinomial, and MultipleImputation are also available (prefix model name with `R`, e.g., `RCoxProportionalHazards`).

### Run States

| State | Description |
|-------|-------------|
| **Pending** | Waiting to start |
| **Standby** | Ready, waiting for trigger |
| **Preparing** | Loading datasets, validating |
| **Running** | Active training |
| **Aggregating** | Coordinator combining models |
| **Completed** | Finished successfully |
| **Failed** | Error occurred |

---
