#!/usr/bin/env Rscript
# training.R — Local estimation of a site's proportion difference and its
# standard error for one outcome.
#
# Usage: Rscript training.R <input.json> <output.json>

library(jsonlite)
library(metafor)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input  <- fromJSON(input_path)
config <- input$config

df <- read.csv(input$data_path, header = TRUE)

# Analysis set, as defined in prepare_data.R.
keep <- switch(config$analysis_set,
  mrs         = !is.na(df$fup_mrs_0_and_1),
  safety      = df$admin_thromb == "Administered at this hospital",
  safety_vital = df$admin_thromb == "Administered at this hospital" &
                 !is.na(df$fup_mrs_0_and_1),
  randomized  = rep(TRUE, nrow(df))
)
df <- df[keep & df$p_drugtype %in% c(0, 1), , drop = FALSE]

y <- df[[config$outcome_var]]
if (identical(config$analysis_set, "randomized")) y[is.na(y)] <- 0L

alt <- df$p_drugtype == 0
ten <- df$p_drugtype == 1

n_alt <- sum(alt)
x_alt <- sum(y[alt])
n_ten <- sum(ten)
x_ten <- sum(y[ten])

# Continuity correction. 0.5 is added to each of the four cells when any one of
# them is zero, which is metafor's add = 0.5, to = "only0". Without it a site
# with both arms degenerate has a standard error of zero and infinite weight.
corrected <- min(x_alt, n_alt - x_alt, x_ten, n_ten - x_ten) == 0
to <- if (isFALSE(config$continuity_correction)) "none" else "only0"

esc <- escalc(measure = "RD",
              ai = x_ten, bi = n_ten - x_ten,
              ci = x_alt, di = n_alt - x_alt,
              add = 0.5, to = to)

# Only delta_per and se_delta are transmitted to the coordinating centre. The
# per-arm counts stay at the site and are reported in Table 1 as a declared
# exception to the disclosure-control policy. aggregate.R reads neither.
result <- list(
  sample_size = jsonlite::unbox(as.integer(input$sample_size)),
  site_id     = jsonlite::unbox(as.integer(unique(df$p_siteid))),
  delta_per   = jsonlite::unbox(as.numeric(esc$yi) * 100),
  se_delta    = jsonlite::unbox(sqrt(as.numeric(esc$vi)) * 100),
  n_alt       = jsonlite::unbox(as.integer(n_alt)),
  x_alt       = jsonlite::unbox(as.integer(x_alt)),
  n_ten       = jsonlite::unbox(as.integer(n_ten)),
  x_ten       = jsonlite::unbox(as.integer(x_ten)),
  corrected   = jsonlite::unbox(corrected && to == "only0")
)

write(toJSON(result, digits = 10), output_path)
