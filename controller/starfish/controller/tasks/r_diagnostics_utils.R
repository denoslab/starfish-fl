## r_diagnostics_utils.R — Shared diagnostic functions for R-based FL tasks.
##
## Source this file from any R training script:
##   this_script <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))
##   source(file.path(dirname(this_script), "..", "..", "r_diagnostics_utils.R"))

compute_vif <- function(X) {
  # Compute VIF for each column of design matrix (without intercept).
  # X: numeric matrix (n x p), no intercept column.
  p <- ncol(X)
  if (is.null(p) || p == 0) return(numeric(0))
  vifs <- numeric(p)
  for (j in seq_len(p)) {
    y_j <- X[, j]
    others <- X[, -j, drop = FALSE]
    if (ncol(others) == 0) {
      vifs[j] <- 1.0
    } else {
      fit <- lm.fit(cbind(1, others), y_j)
      ss_res <- sum(fit$residuals^2)
      ss_tot <- sum((y_j - mean(y_j))^2)
      r_sq <- 1 - ss_res / (ss_tot + 1e-10)
      vifs[j] <- 1 / (1 - r_sq + 1e-10)
    }
  }
  vifs
}

residual_summary <- function(r) {
  # Privacy-safe summary of a residual vector.
  list(
    mean   = mean(r),
    std    = if (length(r) > 1) sd(r) else 0,
    min    = min(r),
    q25    = as.numeric(quantile(r, 0.25)),
    median = median(r),
    q75    = as.numeric(quantile(r, 0.75)),
    max    = max(r)
  )
}

cooks_distance_summary <- function(model) {
  # Cook's distance summary from a fitted lm/glm object.
  cd <- cooks.distance(model)
  n <- length(cd)
  threshold <- 4 / n
  list(
    max           = max(cd),
    mean          = mean(cd),
    n_influential = sum(cd > threshold),
    threshold     = threshold
  )
}

shapiro_wilk_summary <- function(residuals, max_n = 5000) {
  # Shapiro-Wilk normality test on residuals.
  r <- residuals
  if (length(r) < 3) return(list(statistic = NULL, p_value = NULL))
  if (length(r) > max_n) {
    set.seed(42)
    r <- sample(r, max_n)
  }
  test <- shapiro.test(r)
  list(statistic = as.numeric(test$statistic),
       p_value   = as.numeric(test$p.value))
}

hosmer_lemeshow_test <- function(y_true, y_prob, n_groups = 10) {
  # Hosmer-Lemeshow goodness-of-fit test.
  n <- length(y_true)
  if (n < n_groups * 2) {
    return(list(statistic = NULL, p_value = NULL, df = NULL))
  }
  ord <- order(y_prob)
  y_true <- y_true[ord]
  y_prob <- y_prob[ord]
  groups <- cut(seq_len(n), breaks = n_groups, labels = FALSE)
  chi2 <- 0
  for (g in seq_len(n_groups)) {
    idx <- which(groups == g)
    obs_1 <- sum(y_true[idx])
    obs_0 <- length(idx) - obs_1
    exp_1 <- sum(y_prob[idx])
    exp_0 <- length(idx) - exp_1
    if (exp_1 > 0) chi2 <- chi2 + (obs_1 - exp_1)^2 / exp_1
    if (exp_0 > 0) chi2 <- chi2 + (obs_0 - exp_0)^2 / exp_0
  }
  df <- n_groups - 2
  p_value <- pchisq(chi2, df, lower.tail = FALSE)
  list(statistic = chi2, p_value = p_value, df = df)
}

overdispersion_test <- function(model) {
  # Overdispersion ratio for Poisson/NB GLM.
  pearson_chi2 <- sum(residuals(model, type = "pearson")^2)
  df_resid <- model$df.residual
  if (df_resid <= 0) return(list(ratio = NULL, p_value = NULL))
  ratio <- pearson_chi2 / df_resid
  p_value <- pchisq(pearson_chi2, df_resid, lower.tail = FALSE)
  list(ratio = ratio, p_value = p_value)
}

prediction_interval_summary <- function(ci_lower, ci_upper,
                                        pi_lower = NULL, pi_upper = NULL) {
  # Summarise CI/PI widths (privacy-safe).
  ci_width <- ci_upper - ci_lower
  result <- list(
    ci_width_mean   = mean(ci_width),
    ci_width_std    = if (length(ci_width) > 1) sd(ci_width) else 0,
    ci_width_median = median(ci_width)
  )
  if (!is.null(pi_lower) && !is.null(pi_upper)) {
    pi_width <- pi_upper - pi_lower
    result$pi_width_mean   <- mean(pi_width)
    result$pi_width_std    <- if (length(pi_width) > 1) sd(pi_width) else 0
    result$pi_width_median <- median(pi_width)
  }
  result
}

lm_diagnostics <- function(model, X_no_intercept) {
  # Full diagnostics for lm() models.
  # model: fitted lm object
  # X_no_intercept: feature matrix without intercept column
  diag <- list()
  diag$vif <- compute_vif(X_no_intercept)
  diag$residual_summary <- residual_summary(residuals(model))
  diag$cooks_distance <- cooks_distance_summary(model)
  diag$shapiro_wilk <- shapiro_wilk_summary(residuals(model))

  # Prediction intervals
  pred_ci <- predict(model, interval = "confidence")
  pred_pi <- predict(model, interval = "prediction")
  diag$prediction_intervals <- prediction_interval_summary(
    pred_ci[, "lwr"], pred_ci[, "upr"],
    pred_pi[, "lwr"], pred_pi[, "upr"]
  )
  diag
}

glm_diagnostics <- function(model, X_no_intercept) {
  # Diagnostics for glm() models (logistic, Poisson, NB).
  # model: fitted glm object
  # X_no_intercept: feature matrix without intercept column
  diag <- list()
  diag$vif <- compute_vif(X_no_intercept)

  # Deviance residuals
  dev_resid <- residuals(model, type = "deviance")
  diag$deviance_residual_summary <- residual_summary(dev_resid)

  # Pearson residuals
  pear_resid <- residuals(model, type = "pearson")
  diag$pearson_residual_summary <- residual_summary(pear_resid)

  # Cook's distance
  diag$cooks_distance <- cooks_distance_summary(model)

  # Prediction CI (on link scale, transformed to response)
  pred <- predict(model, type = "response", se.fit = TRUE)
  ci_lower <- pred$fit - 1.96 * pred$se.fit
  ci_upper <- pred$fit + 1.96 * pred$se.fit
  diag$prediction_intervals <- prediction_interval_summary(ci_lower, ci_upper)

  diag
}
