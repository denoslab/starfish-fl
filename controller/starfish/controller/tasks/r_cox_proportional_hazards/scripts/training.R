#!/usr/bin/env Rscript
# training.R — Fit Cox PH using survival::coxph().
#
# Usage: Rscript training.R <input.json> <output.json>

library(jsonlite)
library(survival)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path)
data_path   <- input$data_path
sample_size <- input$sample_size

# Read CSV (no header)
df <- read.csv(data_path, header = FALSE)
n_cols <- ncol(df)

# Last column = event, second-to-last = time, rest = features
# Skip first column (group) for Cox PH — use features only
event <- df[, n_cols]
time_col <- df[, n_cols - 1]
features <- df[, 2:(n_cols - 2), drop = FALSE]

n_features <- ncol(features)
feature_names <- paste0("x", seq_len(n_features))
colnames(features) <- feature_names

surv_df <- data.frame(time = time_col, event = event, features)

# Fit Cox PH model
surv_obj <- Surv(surv_df$time, surv_df$event)
formula <- as.formula(paste("surv_obj ~", paste(feature_names, collapse = " + ")))
model <- coxph(formula, data = surv_df)

# Extract results
s <- summary(model)
coef_vals   <- as.numeric(s$coefficients[, "coef"])
se_vals     <- as.numeric(s$coefficients[, "se(coef)"])
hr_vals     <- as.numeric(s$coefficients[, "exp(coef)"])
p_vals      <- as.numeric(s$coefficients[, "Pr(>|z|)"])
ci_lower    <- as.numeric(s$conf.int[, "lower .95"])
ci_upper    <- as.numeric(s$conf.int[, "upper .95"])
concordance <- jsonlite::unbox(as.numeric(s$concordance[1]))

result <- list(
  sample_size      = jsonlite::unbox(as.integer(sample_size)),
  coef             = coef_vals,
  se               = se_vals,
  hazard_ratio     = hr_vals,
  p_values         = p_vals,
  ci_lower         = log(ci_lower),
  ci_upper         = log(ci_upper),
  concordance_index = concordance,
  feature_names    = feature_names
)

write(toJSON(result, digits = 10), output_path)
