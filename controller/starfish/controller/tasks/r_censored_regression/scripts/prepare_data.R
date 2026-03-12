#!/usr/bin/env Rscript
# prepare_data.R — Validate censored regression data.
#
# Data format: CSV with no header.
#   Columns 1..(n-2) = features
#   Column (n-1)     = outcome (continuous)
#   Column n         = censoring indicator (0=observed, 1=right-censored, -1=left-censored)
#
# Usage: Rscript prepare_data.R <input.json> <output.json>

library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path)
data_path <- input$data_path

df <- tryCatch(
  read.csv(data_path, header = FALSE),
  error = function(e) NULL
)

if (is.null(df) || nrow(df) == 0) {
  result <- list(valid = jsonlite::unbox(FALSE), sample_size = jsonlite::unbox(0L))
  write(toJSON(result), output_path)
  quit(status = 0)
}

n_cols <- ncol(df)
censor <- df[, n_cols]
outcome <- df[, n_cols - 1]

# Validate: censoring must be in {-1, 0, 1}, outcome must be numeric
if (!all(censor %in% c(-1, 0, 1)) || !is.numeric(outcome)) {
  result <- list(valid = jsonlite::unbox(FALSE), sample_size = jsonlite::unbox(0L))
  write(toJSON(result), output_path)
  quit(status = 0)
}

result <- list(
  valid       = jsonlite::unbox(TRUE),
  sample_size = jsonlite::unbox(nrow(df))
)

write(toJSON(result), output_path)
