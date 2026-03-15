# Starfish-FL

**An Agentic Federated Learning Framework**

[![Controller Tests](https://github.com/denoslab/starfish-fl/actions/workflows/controller-tests.yml/badge.svg)](https://github.com/denoslab/starfish-fl/actions/workflows/controller-tests.yml)
[![Router Tests](https://github.com/denoslab/starfish-fl/actions/workflows/router-tests.yml/badge.svg)](https://github.com/denoslab/starfish-fl/actions/workflows/router-tests.yml)
[![E2E Tests](https://github.com/denoslab/starfish-fl/actions/workflows/e2e-tests.yml/badge.svg)](https://github.com/denoslab/starfish-fl/actions/workflows/e2e-tests.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/denoslab/starfish-fl/blob/main/LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![R 4.x](https://img.shields.io/badge/R-4.x-276DC3.svg)](https://www.r-project.org/)

Starfish-FL is an agentic federated learning (FL) framework that is native to AI agents. It is an essential component of the STARFISH project. It focuses on federated learning and analysis for the Analysis Mandate function of STARFISH.

Starfish-FL also offers a friendly user interface for easy use in domains including healthcare, computing resource allocation, and finance. Starfish-FL enables secure, privacy-preserving collaborative machine learning across multiple sites without centralizing sensitive data.

## Use Cases

**Biostatistics & Clinical Research** -- Starfish-FL supports the methods biostatisticians use daily -- logistic regression, Cox proportional hazards, Kaplan-Meier survival curves, Poisson and negative binomial models for count data, censored regression (Tobit) for detection-limit outcomes, MICE for missing data, and more -- all federated out of the box with proper inverse-variance weighted meta-analysis and built-in diagnostics (VIF, residuals, goodness-of-fit tests). Every task is available in **both Python and R**, so researchers can work in their preferred language. Hospitals and research institutions can collaboratively build models on patient data without sharing sensitive records.

**Carbon-Aware Computing** -- Starfish-FL enables predicting energy consumption and carbon footprints for containerized workloads across distributed infrastructure. By training regression models federally across edge and cloud sites, organizations can forecast resource energy demands and make carbon-conscious scheduling decisions -- all without centralizing sensitive operational data.

## Components

Starfish-FL is a mono-repo with three components:

- **[Controller](https://github.com/denoslab/starfish-fl/tree/main/controller)** -- Installed on each site; handles local ML training, Celery task queuing, and a web UI (port 8001)
- **[Router](https://github.com/denoslab/starfish-fl/tree/main/router)** -- Central coordination server; manages global state, message forwarding, and artifact storage (port 8000)
- **[CLI](https://github.com/denoslab/starfish-fl/tree/main/cli)** -- Typer-based CLI (`starfish` command) for human and AI agent use

## Supported Tasks (Python & R)

| Task | Python | R |
|------|:------:|:-:|
| Logistic Regression | `LogisticRegression` | `RLogisticRegression` |
| Statistical Logistic Regression | `LogisticRegressionStats` | -- |
| Linear Regression | `LinearRegression` | -- |
| SVM Regression | `SvmRegression` | -- |
| ANCOVA | `Ancova` | -- |
| Ordinal Logistic Regression | `OrdinalLogisticRegression` | -- |
| Mixed Effects Logistic Regression | `MixedEffectsLogisticRegression` | -- |
| Cox Proportional Hazards | `CoxProportionalHazards` | `RCoxProportionalHazards` |
| Kaplan-Meier | `KaplanMeier` | `RKaplanMeier` |
| Censored Regression (Tobit) | `CensoredRegression` | `RCensoredRegression` |
| Poisson Regression | `PoissonRegression` | `RPoissonRegression` |
| Negative Binomial Regression | `NegativeBinomialRegression` | `RNegativeBinomialRegression` |
| Multiple Imputation (MICE) | `MultipleImputation` | `RMultipleImputation` |
| Federated UNet (Image Segmentation) | `FederatedUNet` | -- |

## Quick Links

- [Getting Started](getting-started.md) -- Setup and first project
- [Task Configuration](tasks/configuration.md) -- How to configure FL tasks
- [User Guide](user-guide.md) -- Web interface walkthrough
- [CLI Reference](cli.md) -- Command-line interface for human and AI agent use
- [API Reference](api/abstract-task.md) -- Python API docs
- [Architecture](architecture.md) -- System design and data flow

## Citation

```bibtex
@software{starfish,
  title = {Starfish-FL: A Federated Learning System},
  author = {DENOS Lab},
  year = {2026},
  url = {https://github.com/denoslab/starfish-fl}
}
```
