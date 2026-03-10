#!/usr/bin/env Rscript
# training.R — Kaplan-Meier estimation and log-rank test using survival package.
#
# Usage: Rscript training.R <input.json> <output.json>

library(jsonlite)
library(survival)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path)
data_path   <- input$data_path
sample_size <- input$sample_size

df <- read.csv(data_path, header = FALSE)
n_cols <- ncol(df)

group    <- df[, 1]
time_col <- df[, n_cols - 1]
event    <- df[, n_cols]

surv_obj <- Surv(time_col, event)
groups   <- sort(unique(group))

# KM results per group
km_results <- list()
at_risk_table <- list()

for (g in groups) {
  mask <- group == g
  fit  <- survfit(Surv(time_col[mask], event[mask]) ~ 1)

  group_name <- paste0("group_", as.integer(g))

  km_results[[group_name]] <- list(
    timeline              = fit$time,
    survival_probability  = fit$surv,
    ci_lower              = fit$lower,
    ci_upper              = fit$upper,
    median_survival       = if (is.na(fit$table["median"])) NULL
                            else jsonlite::unbox(as.numeric(fit$table["median"])),
    n_observations        = jsonlite::unbox(sum(mask)),
    n_events              = jsonlite::unbox(sum(event[mask]))
  )

  # Build at-risk table for federated pooling
  event_times <- sort(unique(time_col[mask & event == 1]))
  tbl_events  <- integer(length(event_times))
  tbl_at_risk <- integer(length(event_times))
  for (i in seq_along(event_times)) {
    t <- event_times[i]
    tbl_at_risk[i] <- sum(time_col[mask] >= t)
    tbl_events[i]  <- sum(time_col[mask] == t & event[mask] == 1)
  }
  at_risk_table[[group_name]] <- list(
    times   = event_times,
    events  = tbl_events,
    at_risk = tbl_at_risk
  )
}

# Log-rank test (if exactly 2 groups)
logrank <- NULL
if (length(groups) == 2) {
  lr <- survdiff(surv_obj ~ group)
  logrank <- list(
    test_statistic = jsonlite::unbox(as.numeric(lr$chisq)),
    p_value        = jsonlite::unbox(1 - pchisq(lr$chisq, df = 1))
  )
}

result <- list(
  sample_size    = jsonlite::unbox(as.integer(sample_size)),
  km_results     = km_results,
  logrank        = logrank,
  at_risk_table  = at_risk_table
)

write(toJSON(result, digits = 10), output_path)
