# Starfish-FL

**An Agentic Federated Learning Framework**

[![Controller Tests](https://github.com/denoslab/starfish-fl/actions/workflows/controller-tests.yml/badge.svg)](https://github.com/denoslab/starfish-fl/actions/workflows/controller-tests.yml)
[![Router Tests](https://github.com/denoslab/starfish-fl/actions/workflows/router-tests.yml/badge.svg)](https://github.com/denoslab/starfish-fl/actions/workflows/router-tests.yml)
[![E2E Tests](https://github.com/denoslab/starfish-fl/actions/workflows/e2e-tests.yml/badge.svg)](https://github.com/denoslab/starfish-fl/actions/workflows/e2e-tests.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/denoslab/starfish-fl/blob/main/LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![R 4.x](https://img.shields.io/badge/R-4.x-276DC3.svg)](https://www.r-project.org/)

Starfish-FL is an agentic federated learning (FL) framework that is native to AI agents. It enables secure, privacy-preserving collaborative machine learning across multiple sites without centralizing sensitive data.

## Built for Biostatisticians

Starfish-FL is designed with biostatisticians and clinical researchers in mind. Every supported analysis method is available in **both Python and R**, so researchers can work in their preferred language without learning a new toolchain.

The task library covers the methods biostatisticians use daily:

| Category | Methods |
|----------|---------|
| **Regression** | Linear, Logistic, Ordinal Logistic, Mixed Effects Logistic, SVM, ANCOVA |
| **Survival Analysis** | Cox Proportional Hazards, Kaplan-Meier |
| **Censored Outcomes** | Tobit Type I (left/right censoring) |
| **Count Data** | Poisson, Negative Binomial |
| **Missing Data** | Multiple Imputation (MICE) with Rubin's rules |

All methods include federated aggregation via inverse-variance weighted meta-analysis and built-in diagnostics (VIF, residuals, goodness-of-fit tests).

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
