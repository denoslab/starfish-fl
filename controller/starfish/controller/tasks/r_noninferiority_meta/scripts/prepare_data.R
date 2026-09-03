#!/usr/bin/env Rscript
# prepare_data.R — Validate the site partition for the site-stratified
# non-inferiority meta-analysis and report its analysis-set size.
#
# Usage: Rscript prepare_data.R <input.json> <output.json>

library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input  <- fromJSON(input_path)
config <- input$config

df <- tryCatch(
  read.csv(input$data_path, header = TRUE),
  error = function(e) NULL
)

invalid <- function() {
  write(toJSON(list(valid = FALSE, sample_size = 0L), auto_unbox = TRUE),
        output_path)
  quit(status = 0)
}

if (is.null(df) || nrow(df) == 0) invalid()

required <- c("p_siteid", "p_drugtype", config$outcome_var)
if (!all(required %in% names(df))) invalid()

# Analysis set. The mRS set is defined once, by a recorded 90 to 120 day mRS,
# so the two mRS outcomes are analysed on identical records. The safety set is
# every participant who received either thrombolytic; death additionally
# requires a recorded 90-day outcome. The randomized set retains every record
# and scores a missing outcome as a non-event.
keep <- switch(config$analysis_set,
  mrs         = !is.na(df$fup_mrs_0_and_1),
  safety      = df$admin_thromb == "Administered at this hospital",
  safety_vital = df$admin_thromb == "Administered at this hospital" &
                 !is.na(df$fup_mrs_0_and_1),
  randomized  = rep(TRUE, nrow(df)),
  NULL
)
if (is.null(keep)) invalid()

df <- df[keep & df$p_drugtype %in% c(0, 1), , drop = FALSE]
if (nrow(df) == 0) invalid()

# Both treatment arms must be present for a site-level difference to exist.
if (length(unique(df$p_drugtype)) < 2) invalid()

y <- df[[config$outcome_var]]
if (!identical(config$analysis_set, "randomized") && any(is.na(y))) invalid()
if (!all(stats::na.omit(unique(y)) %in% c(0, 1))) invalid()

result <- list(
  valid       = TRUE,
  sample_size = nrow(df)
)

write(toJSON(result, auto_unbox = TRUE), output_path)
