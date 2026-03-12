#!/usr/bin/env Rscript
# training.R — Fit Tobit (censored regression) using survival::survreg().
#
# Usage: Rscript training.R <input.json> <output.json>

library(jsonlite)
library(survival)

# Source shared diagnostics utilities
this_script <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))
source(file.path(dirname(this_script), "..", "..", "r_diagnostics_utils.R"))

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path)
data_path   <- input$data_path
sample_size <- input$sample_size

# Read CSV (no header)
df <- read.csv(data_path, header = FALSE)
n_cols <- ncol(df)

# Last column = censoring, second-to-last = outcome, rest = features
censor  <- df[, n_cols]
outcome <- df[, n_cols - 1]
features <- df[, 1:(n_cols - 2), drop = FALSE]

n_features <- ncol(features)
feature_names <- paste0("x", seq_len(n_features))
colnames(features) <- feature_names

# Build Surv object for survreg
# survreg uses Surv(time, event) where event=1 means observed
# For left-censored: Surv(time, time2=NA, type="left")
# We use interval2 type which handles all censoring types
# type="left": left-censored, type="right": right-censored

# Convert our censoring convention to Surv objects
# 0 = observed, 1 = right-censored, -1 = left-censored
# For survreg with type="right": event=1 means observed, event=0 means right-censored
# For left-censored we use interval censoring: Surv(left, right, type="interval2")
#   observed: left == right
#   right-censored: right = Inf
#   left-censored: left = -Inf

has_left <- any(censor == -1)

if (has_left) {
  # Use interval2 censoring to handle all types
  left_bound  <- ifelse(censor == -1, -Inf, outcome)
  right_bound <- ifelse(censor == 1, Inf, outcome)
  surv_obj <- Surv(left_bound, right_bound, type = "interval2")
} else {
  # Simple right-censoring: event = 1 - censor (observed=1, right-censored=0)
  surv_obj <- Surv(outcome, 1 - censor)
}

surv_df <- data.frame(outcome = outcome, features)
formula <- as.formula(paste("surv_obj ~", paste(feature_names, collapse = " + ")))

# Fit Tobit model (survreg with Gaussian distribution)
model <- survreg(formula, data = surv_df, dist = "gaussian")

# Extract results
s <- summary(model)
coef_table <- s$table

# Coefficients (including intercept)
all_names <- rownames(coef_table)
# Exclude "Log(scale)" row
param_rows <- all_names[all_names != "Log(scale)"]
coef_vals <- as.numeric(coef_table[param_rows, "Value"])
se_vals   <- as.numeric(coef_table[param_rows, "Std. Error"])
z_vals    <- as.numeric(coef_table[param_rows, "z"])
p_vals    <- as.numeric(coef_table[param_rows, "p"])

ci_lower <- coef_vals - 1.96 * se_vals
ci_upper <- coef_vals + 1.96 * se_vals

sigma_val <- model$scale

# Log-likelihood
log_lik <- as.numeric(logLik(model))

# Diagnostics
diag <- list()

# VIF on features (no intercept)
tryCatch({
  diag$vif <- compute_vif(as.matrix(features))
}, error = function(e) {})

# Residuals for observed data
tryCatch({
  resid_vals <- residuals(model, type = "deviance")
  diag$residual_summary <- residual_summary(resid_vals)
  diag$shapiro_wilk <- shapiro_wilk_summary(resid_vals)
}, error = function(e) {})

# Censoring summary
diag$censoring_summary <- list(
  n_observed        = jsonlite::unbox(sum(censor == 0)),
  n_right_censored  = jsonlite::unbox(sum(censor == 1)),
  n_left_censored   = jsonlite::unbox(sum(censor == -1)),
  pct_censored      = jsonlite::unbox(mean(censor != 0) * 100)
)

# AIC / BIC
diag$aic <- jsonlite::unbox(AIC(model))
diag$bic <- jsonlite::unbox(BIC(model))

# Feature names for output: intercept + feature names
out_names <- c("(Intercept)", feature_names)

result <- list(
  sample_size     = jsonlite::unbox(as.integer(sample_size)),
  coef            = coef_vals,
  se              = se_vals,
  sigma           = jsonlite::unbox(sigma_val),
  p_values        = p_vals,
  ci_lower        = ci_lower,
  ci_upper        = ci_upper,
  log_likelihood  = jsonlite::unbox(log_lik),
  feature_names   = out_names,
  diagnostics     = diag
)

write(toJSON(result, digits = 10), output_path)
