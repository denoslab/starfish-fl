#!/usr/bin/env Rscript
# aggregate.R — Inverse-variance weighted meta-analysis of MICE-pooled
#               coefficients across federated sites.
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
total_complete <- 0

site_coefs <- list()
k <- length(mid_artifacts)

for (i in seq_along(mid_artifacts)) {
  art <- mid_artifacts[[i]]
  coef_vals <- unlist(art$coef)
  se_vals   <- unlist(art$se)
  w <- 1.0 / (se_vals^2 + 1e-10)
  weighted_coef <- weighted_coef + w * coef_vals
  weight_sum    <- weight_sum + w
  total_samples <- total_samples + art$sample_size
  total_complete <- total_complete + ifelse(is.null(art$complete_cases),
                                            art$sample_size, art$complete_cases)
  site_coefs[[i]] <- coef_vals
}

pooled_coef <- weighted_coef / weight_sum
pooled_se   <- sqrt(1.0 / weight_sum)

# Add between-site variance if multiple sites
if (k > 1) {
  site_mat <- do.call(rbind, site_coefs)
  between_site_var <- apply(site_mat, 2, var)
  total_var <- pooled_se^2 + (1 + 1.0 / k) * between_site_var
  pooled_se <- sqrt(total_var)
}

t_values <- pooled_coef / (pooled_se + 1e-10)
p_values <- 2 * (1 - pnorm(abs(t_values)))
ci_lower <- pooled_coef - 1.96 * pooled_se
ci_upper <- pooled_coef + 1.96 * pooled_se

# Aggregate missingness fractions as weighted average
agg_missing <- rep(0, length(mid_artifacts[[1]]$missingness_fractions))
for (art in mid_artifacts) {
  fracs <- unlist(art$missingness_fractions)
  if (length(fracs) > 0) {
    agg_missing <- agg_missing + fracs * art$sample_size
  }
}
if (total_samples > 0 && length(agg_missing) > 0) {
  agg_missing <- agg_missing / total_samples
}

result <- list(
  sample_size           = jsonlite::unbox(as.integer(total_samples)),
  complete_cases        = jsonlite::unbox(as.integer(total_complete)),
  n_sites               = jsonlite::unbox(as.integer(k)),
  coef                  = as.numeric(pooled_coef),
  se                    = as.numeric(pooled_se),
  t_values              = as.numeric(t_values),
  p_values              = as.numeric(p_values),
  ci_lower              = as.numeric(ci_lower),
  ci_upper              = as.numeric(ci_upper),
  missingness_fractions = as.numeric(agg_missing),
  feature_names         = mid_artifacts[[1]]$feature_names
)

write(toJSON(result, digits = 10), output_path)
