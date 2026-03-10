#!/usr/bin/env Rscript
# training.R — Fit Poisson Regression using glm(family=poisson).
#
# Usage: Rscript training.R <input.json> <output.json>

library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path)
data_path   <- input$data_path
sample_size <- input$sample_size

# Read CSV (no header)
df <- read.csv(data_path, header = FALSE)
n_cols <- ncol(df)

# Last column = count, second-to-last = offset (log-exposure), rest = features
count_col  <- df[, n_cols]
offset_col <- df[, n_cols - 1]
features   <- df[, 1:(n_cols - 2), drop = FALSE]

n_features <- ncol(features)
feature_names <- paste0("x", seq_len(n_features))
colnames(features) <- feature_names

model_df <- data.frame(y = count_col, offset_val = offset_col, features)

# Fit Poisson GLM
formula <- as.formula(paste("y ~", paste(feature_names, collapse = " + ")))
model <- glm(formula, data = model_df, family = poisson(),
             offset = offset_val)

# Extract results
s <- summary(model)
coef_vals  <- as.numeric(s$coefficients[, "Estimate"])
se_vals    <- as.numeric(s$coefficients[, "Std. Error"])
z_vals     <- as.numeric(s$coefficients[, "z value"])
p_vals     <- as.numeric(s$coefficients[, "Pr(>|z|)"])
ci         <- confint.default(model)
ci_lower   <- as.numeric(ci[, 1])
ci_upper   <- as.numeric(ci[, 2])
rate_ratios <- exp(coef_vals)

deviance    <- jsonlite::unbox(as.numeric(s$deviance))
pearson_chi2 <- jsonlite::unbox(sum(residuals(model, type = "pearson")^2))
aic         <- jsonlite::unbox(as.numeric(s$aic))

coef_names <- c("const", feature_names)

result <- list(
  sample_size   = jsonlite::unbox(as.integer(sample_size)),
  coef          = coef_vals,
  se            = se_vals,
  z_values      = z_vals,
  p_values      = p_vals,
  ci_lower      = ci_lower,
  ci_upper      = ci_upper,
  rate_ratios   = rate_ratios,
  deviance      = deviance,
  pearson_chi2  = pearson_chi2,
  aic           = aic,
  feature_names = coef_names
)

write(toJSON(result, digits = 10), output_path)
