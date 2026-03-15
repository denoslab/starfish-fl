#!/usr/bin/env Rscript
# training.R — Fit Cox PH using survival::coxph().
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

# Diagnostics
diag <- list()

# Schoenfeld residuals / proportional hazards test
tryCatch({
  ph_test <- cox.zph(model)
  diag$proportional_hazards_test <- list(
    test_statistic = as.numeric(ph_test$table[, "chisq"]),
    p_value        = as.numeric(ph_test$table[, "p"]),
    feature_names  = rownames(ph_test$table)
  )
}, error = function(e) {
  diag$proportional_hazards_test <<- list(test_performed = FALSE)
})

# VIF on features
tryCatch({
  diag$vif <- compute_vif(as.matrix(features))
}, error = function(e) {})

# Deviance residuals
tryCatch({
  dev_resid <- residuals(model, type = "deviance")
  diag$deviance_residual_summary <- residual_summary(dev_resid)
}, error = function(e) {})

result <- list(
  sample_size      = jsonlite::unbox(as.integer(sample_size)),
  coef             = coef_vals,
  se               = se_vals,
  hazard_ratio     = hr_vals,
  p_values         = p_vals,
  ci_lower         = log(ci_lower),
  ci_upper         = log(ci_upper),
  concordance_index = concordance,
  feature_names    = feature_names,
  diagnostics      = diag
)

write(toJSON(result, digits = 10), output_path)
