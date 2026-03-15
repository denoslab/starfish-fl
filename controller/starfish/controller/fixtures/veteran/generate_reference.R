#!/usr/bin/env Rscript
# generate_reference.R — Produce reference values and CSV fixtures from the
# veteran dataset (Kalbfleisch & Prentice, 1980) for regression testing.
#
# Usage:  Rscript generate_reference.R
#
# Outputs:
#   veteran_py.csv  — 10 columns (8 features, time, event)
#   veteran_r.csv   — 11 columns (group=0, 8 features, time, event)
#   reference coefficients printed to stdout

library(survival)

data(veteran)

# Dummy-encode celltype (reference = squamous)
veteran$celltype_smallcell <- as.integer(veteran$celltype == "smallcell")
veteran$celltype_adeno     <- as.integer(veteran$celltype == "adeno")
veteran$celltype_large     <- as.integer(veteran$celltype == "large")

# Binarise trt (1=standard→0, 2=test→1) and prior (0→0, 10→1)
veteran$trt_test  <- as.integer(veteran$trt == 2)
veteran$prior_yes <- as.integer(veteran$prior == 10)

# Fit reference model
model <- coxph(Surv(time, status) ~ trt_test + celltype_smallcell +
               celltype_adeno + celltype_large + karno + diagtime +
               age + prior_yes, data = veteran)

cat("=== Reference Cox PH coefficients (veteran dataset) ===\n\n")
print(summary(model))
cat("\n=== Concordance ===\n")
cat(summary(model)$concordance, "\n")

# Build numeric matrix in feature order matching the model formula
features <- veteran[, c("trt_test", "celltype_smallcell", "celltype_adeno",
                         "celltype_large", "karno", "diagtime", "age",
                         "prior_yes")]

# Python format: features, time, event  (10 cols)
py_mat <- cbind(features, time = veteran$time, event = veteran$status)
write.table(py_mat, file = "veteran_py.csv", sep = ",", row.names = FALSE,
            col.names = FALSE)

# R format: group(=0), features, time, event  (11 cols)
r_mat <- cbind(group = 0, features, time = veteran$time,
               event = veteran$status)
write.table(r_mat, file = "veteran_r.csv", sep = ",", row.names = FALSE,
            col.names = FALSE)

cat("\nFixtures written: veteran_py.csv, veteran_r.csv\n")

# Print compact reference values for embedding in test code
s <- summary(model)
cat("\n=== Values for test_regression_cox_ph.py ===\n")
cat("FEATURE_NAMES =", paste0("[", paste(shQuote(rownames(s$coefficients), type="cmd"), collapse=", "), "]"), "\n")
cat("REF_COEF =", paste0("[", paste(round(s$coefficients[,"coef"], 6), collapse=", "), "]"), "\n")
cat("REF_SE =", paste0("[", paste(round(s$coefficients[,"se(coef)"], 6), collapse=", "), "]"), "\n")
cat("REF_HR =", paste0("[", paste(round(s$coefficients[,"exp(coef)"], 6), collapse=", "), "]"), "\n")
cat("REF_P =", paste0("[", paste(round(s$coefficients[,"Pr(>|z|)"], 6), collapse=", "), "]"), "\n")
cat("REF_CONCORDANCE =", round(s$concordance[1], 6), "\n")
