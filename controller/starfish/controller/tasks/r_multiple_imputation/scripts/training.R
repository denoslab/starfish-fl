#!/usr/bin/env Rscript
# training.R — Run MICE imputation and fit linear regression on each
#              imputed dataset, then pool using Rubin's rules.
#
# Usage: Rscript training.R <input.json> <output.json>

library(jsonlite)
library(mice)

# Source shared diagnostics utilities
this_script <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))
source(file.path(dirname(this_script), "..", "..", "r_diagnostics_utils.R"))

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path)
data_path   <- input$data_path
sample_size <- input$sample_size
config      <- input$config

# Read config
m        <- ifelse(is.null(config$m), 5L, as.integer(config$m))
max_iter <- ifelse(is.null(config$max_iter), 10L, as.integer(config$max_iter))

# Read CSV (no header)
df <- read.csv(data_path, header = FALSE)
n_cols <- ncol(df)

# Replace empty strings with NA and convert to numeric
for (i in seq_len(n_cols)) {
  df[, i] <- as.numeric(df[, i])
}

# Last column = outcome, rest = features
n_features <- n_cols - 1
feature_names <- paste0("x", seq_len(n_features))
outcome_name <- "y"
colnames(df) <- c(feature_names, outcome_name)

# Compute missingness diagnostics
missing_mask <- is.na(df)
missingness_fractions <- colMeans(missing_mask)
complete_cases <- sum(complete.cases(df))

# Run MICE
imp <- mice(df, m = m, maxit = max_iter, printFlag = FALSE, seed = 42)

# Fit linear regression on each imputed dataset and pool
formula <- as.formula(paste(outcome_name, "~", paste(feature_names, collapse = " + ")))
analyses <- lapply(seq_len(m), function(i) {
  d <- complete(imp, i)
  lm(formula, data = d)
})
fit <- as.mira(analyses)
pooled <- pool(fit)
s <- summary(pooled, conf.int = TRUE)

coef_vals <- as.numeric(s$estimate)
se_vals   <- as.numeric(s$std.error)
t_vals    <- as.numeric(s$statistic)
p_vals    <- as.numeric(s$p.value)
ci_lower  <- as.numeric(s$`2.5 %`)
ci_upper  <- as.numeric(s$`97.5 %`)
df_vals   <- as.numeric(s$df)

# Within and between variance from pool object
within_var  <- as.numeric(pooled$pooled$ubar)
between_var <- as.numeric(pooled$pooled$b)

coef_names <- c("const", feature_names)

# Diagnostics from first imputed dataset
diag <- list()
tryCatch({
  first_complete <- complete(imp, 1)
  first_model <- lm(formula, data = first_complete)
  diag <- lm_diagnostics(first_model, as.matrix(first_complete[, feature_names]))
}, error = function(e) {})

result <- list(
  sample_size           = jsonlite::unbox(as.integer(sample_size)),
  complete_cases        = jsonlite::unbox(as.integer(complete_cases)),
  m                     = jsonlite::unbox(as.integer(m)),
  coef                  = coef_vals,
  se                    = se_vals,
  within_var            = within_var,
  between_var           = between_var,
  t_values              = t_vals,
  p_values              = p_vals,
  ci_lower              = ci_lower,
  ci_upper              = ci_upper,
  df                    = df_vals,
  missingness_fractions = as.numeric(missingness_fractions),
  feature_names         = coef_names,
  diagnostics           = diag
)

write(toJSON(result, digits = 10), output_path)
