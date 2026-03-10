#!/usr/bin/env Rscript
# aggregate.R — Pool KM estimates by combining at-risk tables across sites.
#
# Usage: Rscript aggregate.R <input.json> <output.json>

library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path, simplifyVector = FALSE)
mid_artifacts <- input$mid_artifacts

total_samples <- 0
for (art in mid_artifacts) {
  total_samples <- total_samples + art$sample_size
}

# Collect all group names
all_groups <- c()
for (art in mid_artifacts) {
  all_groups <- unique(c(all_groups, names(art$at_risk_table)))
}
all_groups <- sort(all_groups)

pooled_km <- list()

for (group in all_groups) {
  # Merge event times across sites
  all_times <- c()
  for (art in mid_artifacts) {
    if (group %in% names(art$at_risk_table)) {
      all_times <- c(all_times, unlist(art$at_risk_table[[group]]$times))
    }
  }
  all_times <- sort(unique(all_times))

  # Pool events and at-risk counts
  pooled_events  <- rep(0, length(all_times))
  pooled_at_risk <- rep(0, length(all_times))

  for (art in mid_artifacts) {
    if (!(group %in% names(art$at_risk_table))) next
    tbl <- art$at_risk_table[[group]]
    site_times  <- unlist(tbl$times)
    site_events <- unlist(tbl$events)
    site_risk   <- unlist(tbl$at_risk)
    for (i in seq_along(all_times)) {
      idx <- which(site_times == all_times[i])
      if (length(idx) > 0) {
        pooled_events[i]  <- pooled_events[i] + site_events[idx[1]]
        pooled_at_risk[i] <- pooled_at_risk[i] + site_risk[idx[1]]
      }
    }
  }

  # Recompute KM curve
  survival <- numeric(length(all_times))
  s <- 1.0
  for (i in seq_along(all_times)) {
    n <- pooled_at_risk[i]
    d <- pooled_events[i]
    if (n > 0) s <- s * (1.0 - d / n)
    survival[i] <- s
  }

  # Median survival
  median_surv <- NULL
  for (i in seq_along(survival)) {
    if (survival[i] <= 0.5) {
      median_surv <- jsonlite::unbox(all_times[i])
      break
    }
  }

  total_events <- sum(pooled_events)

  pooled_km[[group]] <- list(
    timeline              = all_times,
    survival_probability  = survival,
    median_survival       = median_surv,
    n_events              = jsonlite::unbox(as.integer(total_events))
  )
}

result <- list(
  sample_size = jsonlite::unbox(as.integer(total_samples)),
  km_results  = pooled_km
)

write(toJSON(result, digits = 10), output_path)
