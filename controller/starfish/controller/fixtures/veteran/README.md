# Veteran Dataset (Veterans' Administration Lung Cancer Trial)

## Source

Kalbfleisch, J.D. and Prentice, R.L. (1980). *The Statistical Analysis of
Failure Time Data*. Wiley, New York. Included in R's `survival` package as
`veteran`.

Downloaded from the Rdatasets repository:
https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/veteran.csv

## Description

Randomised trial of two treatment regimens for lung cancer. 137 observations,
128 events (deaths), 9 right-censored.

## Preprocessing

The raw dataset contains one categorical variable (`celltype` with four
levels). The following transformations produce the numeric CSV fixtures:

1. `celltype` dummy-encoded with `squamous` as reference:
   `celltype_smallcell`, `celltype_adeno`, `celltype_large`
2. `trt` recoded: 1 (standard) -> 0, 2 (test) -> 1 as `trt_test`
3. `prior` recoded: 0 -> 0, 10 -> 1 as `prior_yes`

## File Formats

| File | Columns | Description |
|------|---------|-------------|
| `veteran_py.csv` | 10 (8 features, time, event) | Python Cox PH task format |
| `veteran_r.csv` | 11 (group=0, 8 features, time, event) | R Cox PH task format |

Feature order: `trt_test`, `celltype_smallcell`, `celltype_adeno`,
`celltype_large`, `karno`, `diagtime`, `age`, `prior_yes`

## Reference Values

From `lifelines.CoxPHFitter` (equivalent to R's `coxph()`):

| Feature | Coef | SE | HR | p-value | Significant |
|---------|------|----|----|---------|-------------|
| trt_test | 0.2946 | 0.2076 | 1.3426 | 0.156 | No |
| celltype_smallcell | 0.8616 | 0.2753 | 2.3668 | 0.002 | Yes |
| celltype_adeno | 1.1961 | 0.3009 | 3.3071 | <0.001 | Yes |
| celltype_large | 0.4013 | 0.2827 | 1.4938 | 0.156 | No |
| karno | -0.0328 | 0.0055 | 0.9677 | <0.001 | Yes |
| diagtime | 0.0001 | 0.0091 | 1.0001 | 0.993 | No |
| age | -0.0087 | 0.0093 | 0.9913 | 0.349 | No |
| prior_yes | 0.0716 | 0.2323 | 1.0742 | 0.758 | No |

Concordance index: 0.736

## Regeneration

Run `generate_reference.R` (requires R with `survival` package) or use
Python with `lifelines` and `pandas` to download and preprocess the dataset.
