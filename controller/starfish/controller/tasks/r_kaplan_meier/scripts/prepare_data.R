#!/usr/bin/env Rscript
# prepare_data.R — Validate survival data for Kaplan-Meier.
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
event <- df[, n_cols]
time_col <- df[, n_cols - 1]

if (!all(event %in% c(0, 1)) || any(time_col < 0)) {
  result <- list(valid = jsonlite::unbox(FALSE), sample_size = jsonlite::unbox(0L))
  write(toJSON(result), output_path)
  quit(status = 0)
}

result <- list(
  valid       = jsonlite::unbox(TRUE),
  sample_size = jsonlite::unbox(nrow(df))
)

write(toJSON(result), output_path)
