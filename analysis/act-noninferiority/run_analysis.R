#!/usr/bin/env Rscript
# run_analysis.R — Regenerate Table 1, Table 2 and Table 3 of the manuscript in
# a single run, by executing the r_noninferiority_meta task scripts across the
# emulated 21-site network, and write the execution log for the run.
#
# Usage: Rscript run_analysis.R <act_extract.csv> <output_dir>

library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: Rscript run_analysis.R <act_extract.csv> <output_dir>", call. = FALSE)
}
extract_path <- args[1]
output_dir   <- args[2]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
# The task scripts are resolved from the location of this file rather than from
# the working directory, so the run gives the same result wherever it is
# started. This file sits two levels below the repository root.
this_file  <- sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))[1]
repo_root  <- normalizePath(file.path(dirname(this_file), "..", ".."),
                            mustWork = TRUE)
script_dir <- normalizePath(file.path(repo_root, "controller", "starfish",
                                      "controller", "tasks",
                                      "r_noninferiority_meta", "scripts"),
                            mustWork = TRUE)
log_path <- file.path(output_dir, "execution-log.txt")
cat("", file = log_path)

log_line <- function(...) {
  line <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "  ", paste0(..., collapse = ""))
  cat(line, "\n", sep = "")
  cat(line, "\n", sep = "", file = log_path, append = TRUE)
}

log_line("run_analysis.R starting")
log_line("R version      : ", R.version.string)
log_line("metafor version: ", as.character(packageVersion("metafor")))
# Neither the path nor the file name of the input is logged. The deposited log
# must not carry the directory layout of the machine the analysis was run on,
# the name of the restricted trial extract, or a fingerprint of it. Provenance
# of the input is established below, by checking the derived counts against
# those published by the parent trial, which identifies the dataset without
# disclosing anything about it.
log_line("task           : ", basename(dirname(script_dir)))
log_line("input          : AcT trial extract, identified below by its derived counts")

# Partition the extract by site. Each site file stands in for the records held
# behind one institution's firewall; the analysis set is applied locally.
data <- read.csv(extract_path, header = TRUE)
data <- data[!is.na(data$p_siteid) & data$p_drugtype %in% c(0, 1), , drop = FALSE]
sites <- sort(unique(data$p_siteid))
# The partitions hold patient-level records and are written to a temporary
# directory, so the deposited output carries only the tables and this log.
site_dir <- file.path(tempdir(), "site_partitions")
dir.create(site_dir, showWarnings = FALSE)
on.exit(unlink(site_dir, recursive = TRUE), add = TRUE)
for (s in sites) {
  write.csv(data[data$p_siteid == s, , drop = FALSE],
            file.path(site_dir, paste0("site_", s, ".csv")), row.names = FALSE)
}
log_line("partitioned ", nrow(data), " records into ", length(sites), " sites")

run_script <- function(name, payload) {
  in_path  <- tempfile(fileext = ".json")
  out_path <- tempfile(fileext = ".json")
  write(toJSON(payload, auto_unbox = TRUE, digits = 10), in_path)
  status <- system2("Rscript", c("--vanilla", file.path(script_dir, name),
                                 in_path, out_path), stdout = FALSE, stderr = FALSE)
  if (status != 0) stop(name, " failed for payload ", toJSON(payload$config, auto_unbox = TRUE))
  result <- fromJSON(out_path, simplifyVector = FALSE)
  unlink(c(in_path, out_path))
  result
}

# One federated pass: local estimation at every site, then aggregation.
run_pass <- function(outcome_var, analysis_set, correction) {
  config <- list(outcome_var = outcome_var, analysis_set = analysis_set,
                 continuity_correction = correction)
  log_line("pass: outcome=", outcome_var, " set=", analysis_set,
           " continuity_correction=", correction)
  artifacts <- list()
  for (s in sites) {
    data_path <- file.path(site_dir, paste0("site_", s, ".csv"))
    prep <- run_script("prepare_data.R",
                       list(config = config, data_path = data_path))
    if (!isTRUE(prep$valid)) stop("site ", s, " failed validation")
    artifacts[[length(artifacts) + 1]] <- run_script("training.R",
      list(config = config, data_path = data_path, sample_size = prep$sample_size))
  }
  pooled <- run_script("aggregate.R", list(config = config, mid_artifacts = artifacts))
  log_line("  k=", pooled$k, " estimate=", signif(pooled$estimate, 6),
           " SE=", signif(pooled$se, 6), " Q=", signif(pooled$Q, 6),
           " I2=", signif(pooled$I2, 4))
  list(sites = do.call(rbind, lapply(artifacts, function(a) data.frame(
         site = a$site_id, delta_per = a$delta_per, se_delta = a$se_delta,
         n_alt = a$n_alt, x_alt = a$x_alt, n_ten = a$n_ten, x_ten = a$x_ten,
         corrected = a$corrected))),
       pooled = pooled)
}

# Crude overall difference, pooling counts across sites before differencing.
# It cannot be formed from the transmitted payload and is computed here from
# the extract as the reference reproduction of the AcT primary analysis.
crude <- function(sites_df) {
  na <- sum(sites_df$n_alt); xa <- sum(sites_df$x_alt)
  nt <- sum(sites_df$n_ten); xt <- sum(sites_df$x_ten)
  pa <- xa / na; pt <- xt / nt
  d  <- (pt - pa) * 100
  se <- 100 * sqrt(pa * (1 - pa) / na + pt * (1 - pt) / nt)
  z  <- qnorm(0.975)
  list(estimate = d, se = se, ci_lb = d - z * se, ci_ub = d + z * se,
       n_alt = na, x_alt = xa, n_ten = nt, x_ten = xt)
}

outcomes <- list(
  list(index = 1, var = "fup_mrs_0_and_1", set = "mrs"),
  list(index = 2, var = "fup_mrs_0_to_2",  set = "mrs"),
  list(index = 3, var = "portal_sae_sich", set = "safety"),
  list(index = 4, var = "p_deceased",      set = "safety_vital")
)
published <- list(c(266, 765, 296, 802), c(425, 765, 452, 802),
                  c(24, 763, 27, 800),   c(117, 758, 122, 796))

passes <- lapply(outcomes, function(o) run_pass(o$var, o$set, TRUE))

# The run stops here unless every outcome reproduces the counts published by
# the parent trial, so a stale column cannot reach a table.
for (i in seq_along(outcomes)) {
  cr <- crude(passes[[i]]$sites)
  got <- c(cr$x_alt, cr$n_alt, cr$x_ten, cr$n_ten)
  log_line("outcome (", outcomes[[i]]$index, ") counts: alteplase ", got[1], "/", got[2],
           ", tenecteplase ", got[3], "/", got[4],
           "  published ", published[[i]][1], "/", published[[i]][2],
           ", ", published[[i]][3], "/", published[[i]][4])
  if (!identical(as.numeric(got), as.numeric(published[[i]])))
    stop("outcome (", outcomes[[i]]$index, ") does not reproduce the published counts")
}
log_line("all four outcomes reproduce the published AcT counts")

# Table 1, site-level results.
table1 <- data.frame(site = passes[[1]]$sites$site)
for (i in seq_along(outcomes)) {
  s <- passes[[i]]$sites
  idx <- match(table1$site, s$site)
  table1[[paste0("delta_per_", i)]] <- signif(s$delta_per[idx], 3)
  table1[[paste0("se_delta_", i)]]  <- signif(s$se_delta[idx], 3)
  table1[[paste0("alt_", i)]]       <- paste0(s$x_alt[idx], "/", s$n_alt[idx])
  table1[[paste0("ten_", i)]]       <- paste0(s$x_ten[idx], "/", s$n_ten[idx])
  table1[[paste0("corrected_", i)]] <- s$corrected[idx]
}
write.csv(table1, file.path(output_dir, "Table1.csv"), row.names = FALSE)
log_line("wrote Table1.csv")

# Table 2, pooled results under the three estimators.
table2 <- do.call(rbind, lapply(seq_along(outcomes), function(i) {
  p <- passes[[i]]$pooled; cr <- crude(passes[[i]]$sites)
  data.frame(outcome = outcomes[[i]]$index,
    crude_estimate = cr$estimate, crude_ci_lb = cr$ci_lb, crude_ci_ub = cr$ci_ub,
    strat_estimate = p$estimate, strat_ci_lb = p$ci_lb, strat_ci_ub = p$ci_ub,
    re_estimate = p$re_estimate, re_ci_lb = p$re_ci_lb, re_ci_ub = p$re_ci_ub,
    tau2 = p$tau2, I2 = p$I2, H2 = p$H2, Q = p$Q, Q_df = p$Q_df, Q_pval = p$Q_pval,
    se = p$se, zval = p$zval, pval = p$pval)
}))
write.csv(table2, file.path(output_dir, "Table2.csv"), row.names = FALSE)
log_line("wrote Table2.csv")

# Table 3, the primary outcome under ten specifications.
rows <- list()
for (set in c("mrs", "randomized")) {
  label <- if (set == "mrs") "Complete case" else "Full randomized"
  base <- if (set == "mrs") passes[[1]] else run_pass("fup_mrs_0_and_1", set, TRUE)
  cr <- crude(base$sites)
  rows[[length(rows) + 1]] <- data.frame(estimator = "Crude", analysis_set = label,
    continuity_correction = "not applicable", model = "not applicable",
    estimate = cr$estimate, ci_lb = cr$ci_lb, ci_ub = cr$ci_ub)
  for (corr in c(TRUE, FALSE)) {
    p <- if (corr) base$pooled else run_pass("fup_mrs_0_and_1", set, FALSE)$pooled
    cc <- if (corr) "applied" else "not applied"
    rows[[length(rows) + 1]] <- data.frame(estimator = "Site-stratified",
      analysis_set = label, continuity_correction = cc, model = "Fixed effect",
      estimate = p$estimate, ci_lb = p$ci_lb, ci_ub = p$ci_ub)
    rows[[length(rows) + 1]] <- data.frame(estimator = "Site-stratified",
      analysis_set = label, continuity_correction = cc, model = "Random effects",
      estimate = p$re_estimate, ci_lb = p$re_ci_lb, ci_ub = p$re_ci_ub)
  }
}
table3 <- do.call(rbind, rows)
write.csv(table3, file.path(output_dir, "Table3.csv"), row.names = FALSE)
log_line("wrote Table3.csv")
log_line("lowest confidence bound across all ", nrow(table3), " specifications: ",
         signif(min(table3$ci_lb), 4))
unlink(site_dir, recursive = TRUE)
log_line("removed the temporary site partitions")
log_line("run_analysis.R complete")
