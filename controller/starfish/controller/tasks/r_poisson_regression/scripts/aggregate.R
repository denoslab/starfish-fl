#!/usr/bin/env Rscript
# aggregate.R — Inverse-variance weighted meta-analysis of Poisson coefficients.
#
# Usage: Rscript aggregate.R <input.json> <output.json>

library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path, simplifyVector = FALSE)
mid_artifacts <- input$mid_artifacts

n_coefs <- length(mid_artifacts[[1]]$coef)
weighted_coef <- rep(0, n_coefs)
weight_sum    <- rep(0, n_coefs)
total_samples <- 0
total_deviance <- 0
total_pearson  <- 0

for (art in mid_artifacts) {
  coef_vals <- unlist(art$coef)
  se_vals   <- unlist(art$se)
  w <- 1.0 / (se_vals^2 + 1e-10)
  weighted_coef <- weighted_coef + w * coef_vals
  weight_sum    <- weight_sum + w
  total_samples <- total_samples + art$sample_size
  total_deviance <- total_deviance + art$deviance
  total_pearson  <- total_pearson + art$pearson_chi2
}

pooled_coef <- weighted_coef / weight_sum
pooled_se   <- sqrt(1.0 / weight_sum)
pooled_rr   <- exp(pooled_coef)
z_values    <- pooled_coef / pooled_se
p_values    <- 2 * (1 - pnorm(abs(z_values)))
ci_lower    <- pooled_coef - 1.96 * pooled_se
ci_upper    <- pooled_coef + 1.96 * pooled_se

result <- list(
  sample_size   = jsonlite::unbox(as.integer(total_samples)),
  coef          = as.numeric(pooled_coef),
  se            = as.numeric(pooled_se),
  z_values      = as.numeric(z_values),
  p_values      = as.numeric(p_values),
  ci_lower      = as.numeric(ci_lower),
  ci_upper      = as.numeric(ci_upper),
  rate_ratios   = as.numeric(pooled_rr),
  deviance      = jsonlite::unbox(total_deviance),
  pearson_chi2  = jsonlite::unbox(total_pearson),
  feature_names = mid_artifacts[[1]]$feature_names
)

write(toJSON(result, digits = 10), output_path)
