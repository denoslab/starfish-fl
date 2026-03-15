# Task Configuration

When creating a new project, the **Tasks** field requires a JSON array that defines the federated learning workflow.

For the complete task configuration reference with all model-specific options, see the [full TASK_GUIDE](https://github.com/denoslab/starfish-fl/blob/main/controller/TASK_GUIDE.md).

## Task Structure

Each task must have:

- **seq**: Sequential number (starting from 1)
- **model**: The ML model class name
- **config**: Configuration dictionary

```json
[
  {
    "seq": 1,
    "model": "LogisticRegression",
    "config": {
      "total_round": 5,
      "current_round": 1
    }
  }
]
```

## Available Models

### Classification & Regression

| Model Name | Description | Dataset Format |
|-----------|-------------|----------------|
| `LogisticRegression` | Binary classification | Features + binary label (0/1) |
| `RLogisticRegression` | R version of logistic regression | Same |
| `LogisticRegressionStats` | Statistical logistic with inference | Features + binary label, min 30 samples |
| `LinearRegression` | Continuous value prediction | Features + continuous target |
| `SvmRegression` | Support Vector Machine regression | Features + continuous target |
| `Ancova` | Analysis of Covariance | Groups + covariates + outcome |
| `OrdinalLogisticRegression` | Ordered categorical outcomes | Features + ordinal label (0,1,2,...) |
| `MixedEffectsLogisticRegression` | Clustered binary data | Group ID + features + binary label |

### Survival Analysis & Censored Outcomes

| Model Name | Description | Dataset Format |
|-----------|-------------|----------------|
| `CoxProportionalHazards` | Time-to-event regression | Features + time + event (0/1) |
| `RCoxProportionalHazards` | R version (survival::coxph) | Same |
| `KaplanMeier` | Non-parametric survival estimation | Group + features + time + event |
| `RKaplanMeier` | R version (survival::survfit) | Same |
| `CensoredRegression` | Tobit Type I (left/right censoring) | Features + outcome + censoring (-1/0/1) |
| `RCensoredRegression` | R version (survival::survreg) | Same |

### Count Data Models

| Model Name | Description | Dataset Format |
|-----------|-------------|----------------|
| `PoissonRegression` | Count data with rate ratios | Features + offset + count |
| `RPoissonRegression` | R version (glm family=poisson) | Same |
| `NegativeBinomialRegression` | Overdispersed count data | Features + offset + count |
| `RNegativeBinomialRegression` | R version (MASS::glm.nb) | Same |

### Missing Data

| Model Name | Description | Dataset Format |
|-----------|-------------|----------------|
| `MultipleImputation` | MICE with Rubin's rules | Features (may have NaN) + outcome |
| `RMultipleImputation` | R version (mice::mice) | Same |

### Image Segmentation

| Model Name | Description | Dataset Format |
|-----------|-------------|----------------|
| `FederatedUNet` | UNet with FedAvg aggregation | Zip of `images/` + `masks/` directories |

## Required Config Parameters

| Parameter | Description |
|-----------|-------------|
| `total_round` | Total number of federated learning rounds |
| `current_round` | Starting round (usually 1) |

## Optional Config Parameters

| Parameter | Applies To | Description |
|-----------|-----------|-------------|
| `local_epochs` | Classification/Regression | Local training epochs per round |
| `learning_rate` | Classification/Regression | Training learning rate |
| `n_group_columns` | ANCOVA | Number of group indicator columns |
| `vcp_p` | Mixed Effects Logistic | Prior SD for variance components (default: 1.0) |
| `fe_p` | Mixed Effects Logistic | Prior SD for fixed effects (default: 2.0) |
| `m` | Multiple Imputation | Number of imputed datasets (default: 5) |
| `max_iter` | Multiple Imputation | Max MICE iterations (default: 10) |
| `patch_size` | FederatedUNet | Image resize dimension (default: 64) |
| `architecture` | FederatedUNet | Encoder backbone, e.g. `resnet50`, `mobilenetv2` (default: resnet50) |
| `type_Unet` | FederatedUNet | Segmentation architecture type (default: unet) |
| `batch_size` | FederatedUNet | Training mini-batch size (default: 8) |

## Example Configurations

=== "Logistic Regression"

    ```json
    [{"seq": 1, "model": "LogisticRegression", "config": {"total_round": 5, "current_round": 1}}]
    ```

=== "Cox PH (R)"

    ```json
    [{"seq": 1, "model": "RCoxProportionalHazards", "config": {"total_round": 1, "current_round": 1}}]
    ```

=== "Censored Regression"

    ```json
    [{"seq": 1, "model": "CensoredRegression", "config": {"total_round": 1, "current_round": 1}}]
    ```

=== "MICE"

    ```json
    [{"seq": 1, "model": "MultipleImputation", "config": {"total_round": 1, "current_round": 1, "m": 5, "max_iter": 10}}]
    ```

=== "FederatedUNet"

    ```json
    [{"seq": 1, "model": "FederatedUNet", "config": {"total_round": 1, "current_round": 1, "local_epochs": 1, "architecture": "resnet50", "type_Unet": "unet", "patch_size": 64, "batch_size": 1, "learning_rate": 0.0001}}]
    ```

## Writing R-Based Tasks

R tasks use a Python-R bridge via `AbstractRTask`. Each R task needs:

1. A Python wrapper class that sets `r_script_dir`
2. Three R scripts in a `scripts/` subdirectory:
    - `prepare_data.R` -- validate data
    - `training.R` -- fit model
    - `aggregate.R` -- meta-analyze across sites

See [Architecture > Adding a New ML Task](../architecture.md#adding-a-new-ml-task) for details.
