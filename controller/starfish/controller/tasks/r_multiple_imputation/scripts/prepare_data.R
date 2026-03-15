#!/usr/bin/env Rscript
# prepare_data.R — Validate data with missing values for Multiple Imputation.
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

# Replace empty strings with NA (in case CSV has blanks)
df[df == ""] <- NA
for (i in seq_len(ncol(df))) {
  df[, i] <- as.numeric(df[, i])
}

# Must have at least some observed values in outcome (last column)
outcome <- df[, ncol(df)]
if (all(is.na(outcome))) {
  result <- list(valid = jsonlite::unbox(FALSE), sample_size = jsonlite::unbox(0L))
  write(toJSON(result), output_path)
  quit(status = 0)
}

result <- list(
  valid       = jsonlite::unbox(TRUE),
  sample_size = jsonlite::unbox(nrow(df))
)

write(toJSON(result), output_path)
