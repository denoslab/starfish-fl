#!/usr/bin/env Rscript
# prepare_data.R — Validate and inspect the dataset for R logistic regression.
#
# Usage: Rscript prepare_data.R <input.json> <output.json>

library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path)
data_path <- input$data_path

# Read CSV (no header, last column is label)
df <- tryCatch(
  read.csv(data_path, header = FALSE),
  error = function(e) NULL
)

if (is.null(df) || nrow(df) == 0) {
  result <- list(valid = FALSE, sample_size = 0L)
  write(toJSON(result, auto_unbox = TRUE), output_path)
  quit(status = 0)
}

n_cols <- ncol(df)
y <- df[, n_cols]

# Validate binary outcome
unique_vals <- sort(unique(y))
if (!all(unique_vals %in% c(0, 1))) {
  result <- list(valid = FALSE, sample_size = 0L)
  write(toJSON(result, auto_unbox = TRUE), output_path)
  quit(status = 0)
}

result <- list(
  valid       = TRUE,
  sample_size = nrow(df)
)

write(toJSON(result, auto_unbox = TRUE), output_path)
