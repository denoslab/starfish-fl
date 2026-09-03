#!/usr/bin/env Rscript
# aggregate.R — Federated inverse-variance pooling of the site-level
# proportion differences, with heterogeneity diagnostics.
#
# Usage: Rscript aggregate.R <input.json> <output.json>

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path, simplifyVector = FALSE)
mid_artifacts <- input$mid_artifacts

# Only the transmitted payload is read. The per-arm counts carried in the
# site artifacts are not used here and never reach the pooled estimate.
delta <- vapply(mid_artifacts, function(a) as.numeric(a$delta_per), numeric(1))
se    <- vapply(mid_artifacts, function(a) as.numeric(a$se_delta),  numeric(1))
n     <- vapply(mid_artifacts, function(a) as.integer(a$sample_size), integer(1))

fe <- rma.uni(yi = delta, vi = se^2, method = "FE")
dl <- rma.uni(yi = delta, vi = se^2, method = "DL")

result <- list(
  sample_size   = jsonlite::unbox(as.integer(sum(n))),
  k             = jsonlite::unbox(as.integer(fe$k)),
  estimate      = jsonlite::unbox(as.numeric(fe$beta)),
  se            = jsonlite::unbox(as.numeric(fe$se)),
  ci_lb         = jsonlite::unbox(as.numeric(fe$ci.lb)),
  ci_ub         = jsonlite::unbox(as.numeric(fe$ci.ub)),
  zval          = jsonlite::unbox(as.numeric(fe$zval)),
  pval          = jsonlite::unbox(as.numeric(fe$pval)),
  Q             = jsonlite::unbox(as.numeric(fe$QE)),
  Q_df          = jsonlite::unbox(as.integer(fe$k - 1)),
  Q_pval        = jsonlite::unbox(as.numeric(fe$QEp)),
  I2            = jsonlite::unbox(as.numeric(fe$I2)),
  H2            = jsonlite::unbox(as.numeric(fe$H2)),
  tau2          = jsonlite::unbox(as.numeric(dl$tau2)),
  re_estimate   = jsonlite::unbox(as.numeric(dl$beta)),
  re_ci_lb      = jsonlite::unbox(as.numeric(dl$ci.lb)),
  re_ci_ub      = jsonlite::unbox(as.numeric(dl$ci.ub))
)

write(toJSON(result, digits = 10), output_path)
