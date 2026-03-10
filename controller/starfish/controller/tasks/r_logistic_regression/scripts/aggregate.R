#!/usr/bin/env Rscript
# aggregate.R — Federated weighted averaging of logistic regression coefficients.
#
# Usage: Rscript aggregate.R <input.json> <output.json>

library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path, simplifyVector = FALSE)
mid_artifacts <- input$mid_artifacts

total_samples <- 0
coef_sum      <- NULL
intercept_sum <- NULL

for (art in mid_artifacts) {
  n   <- art$sample_size
  # coef_ is [[...]], get the inner list
  c_vals <- unlist(art$coef_[[1]])
  i_vals <- unlist(art$intercept_)

  weighted_coef      <- c_vals * n
  weighted_intercept <- i_vals * n

  if (is.null(coef_sum)) {
    coef_sum      <- weighted_coef
    intercept_sum <- weighted_intercept
  } else {
    coef_sum      <- coef_sum + weighted_coef
    intercept_sum <- intercept_sum + weighted_intercept
  }
  total_samples <- total_samples + n
}

avg_coef      <- coef_sum / total_samples
avg_intercept <- intercept_sum / total_samples

result <- list(
  sample_size = as.integer(total_samples),
  coef_       = list(as.numeric(avg_coef)),
  intercept_  = as.numeric(avg_intercept)
)

write(toJSON(result, auto_unbox = FALSE, digits = 10), output_path)
