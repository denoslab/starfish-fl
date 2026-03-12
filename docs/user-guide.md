# User Guide

This guide covers the Controller web interface for managing federated learning projects.

For the full user guide with dataset formats, result interpretation, and performance metrics, see the [complete Controller User Guide](https://github.com/denoslab/starfish-fl/blob/main/controller/USER_GUIDE.md).

## Register Your Site

When you first access the controller at `http://localhost:8001/controller/`, you'll need to register:

1. Enter a **Site Name** (e.g., "Hospital A")
2. Enter a **Site Description** (e.g., "Cardiac imaging data site")
3. Click **Register Site**

## Create a Project (Coordinator)

1. Go to `http://localhost:8001/controller/projects/new/`
2. Enter project name and description
3. Configure tasks (see [Task Configuration](tasks/configuration.md))
4. Submit to create
5. Wait for participants to join
6. Upload your local dataset
7. Start training runs

## Join a Project (Participant)

1. Browse available projects
2. Review project requirements
3. Click "Join"
4. Upload your dataset (matching project schema)
5. Participate in training rounds

## Dataset Requirements

- **Format**: CSV (comma-separated values)
- **Header**: No header row -- data starts from the first line
- **Encoding**: UTF-8
- **Values**: Numeric only, no missing values (except MICE tasks)

!!! warning "Critical"
    All sites must have **identical feature sets** (same columns in same order). Use `starfish/preprocess_dataset.py` to ensure consistency.

## Understanding Results

After a successful run, you can download:

| File | Purpose | Who Gets It |
|------|---------|-------------|
| **logs.txt** | Training process details | All participants |
| **artifacts** | Final aggregated results | All participants |
| **mid-artifacts** | Intermediate local results | Coordinator only |

### Key Metrics by Model Type

| Model Type | Key Metrics |
|-----------|-------------|
| Classification | Accuracy, AUC, Sensitivity, Specificity, NPV, PPV |
| Linear Regression | MSE, MAE, R-squared |
| Statistical Models | Coefficients, p-values, CI, odds ratios, pseudo R-squared |
| Survival Models | Hazard ratios, concordance index |
| Censored Regression | Coefficients, sigma, log-likelihood, censoring summary |
| Count Data | Rate ratios, deviance, dispersion parameter |

### Model Diagnostics

All regression tasks include diagnostics in their output:

| Diagnostic | What It Tells You | Concerning Values |
|-----------|-------------------|-------------------|
| **VIF** | Multicollinearity between features | VIF > 10 |
| **Shapiro-Wilk** | Are residuals normally distributed? | p < 0.05 |
| **Hosmer-Lemeshow** | Is the logistic model well-calibrated? | p < 0.05 |
| **Overdispersion** | More variance than expected? | ratio >> 1 |
| **Cook's distance** | How many outliers affect the model? | Many influential points |
| **PH test** (Cox) | Proportional hazards assumption met? | p < 0.05 |
| **Censoring summary** | Breakdown of censored observations | High % censored may reduce power |

## Project Roles

| Role | Capabilities |
|------|-------------|
| **Coordinator** | Create project, approve participants, start runs, aggregate models, also trains locally |
| **Participant** | Join projects, upload data, train locally, view results |

The coordinator is also a participant -- they train on their own data **and** aggregate all participants' models.
