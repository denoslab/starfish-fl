#!/usr/bin/env Rscript
# training.R — Fit logistic regression using glm(family=binomial).
#
# Usage: Rscript training.R <input.json> <output.json>

library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
input_path  <- args[1]
output_path <- args[2]

input <- fromJSON(input_path)
data_path   <- input$data_path
sample_size <- input$sample_size

# Read CSV (no header, last column is label)
df <- read.csv(data_path, header = FALSE)
n_cols <- ncol(df)

X <- as.matrix(df[, 1:(n_cols - 1)])
y <- df[, n_cols]

# Train/test split (80/20, deterministic)
set.seed(42)
n <- nrow(df)
idx <- sample(seq_len(n), size = floor(0.8 * n))
X_train <- X[idx, , drop = FALSE]
y_train <- y[idx]
X_test  <- X[-idx, , drop = FALSE]
y_test  <- y[-idx]

# Scale features
train_means <- colMeans(X_train)
train_sds   <- apply(X_train, 2, sd)
# Avoid division by zero
train_sds[train_sds == 0] <- 1

X_train_scaled <- scale(X_train, center = train_means, scale = train_sds)
X_test_scaled  <- scale(X_test,  center = train_means, scale = train_sds)

# Build data frames for glm
train_df <- data.frame(y = y_train, X_train_scaled)
test_df  <- data.frame(y = y_test,  X_test_scaled)

# Fit logistic regression with optional warm start
start_vals <- NULL
if (!is.null(input$previous_model)) {
  prev_coef      <- unlist(input$previous_model$coef_)
  prev_intercept <- unlist(input$previous_model$intercept_)
  start_vals     <- c(prev_intercept, prev_coef)
}

model <- glm(y ~ ., data = train_df, family = binomial,
             start = start_vals, maxit = 100)

# Extract coefficients
coef_all  <- coef(model)
intercept <- as.numeric(coef_all[1])
coef_vals <- as.numeric(coef_all[-1])

# Predictions on test set
prob_test <- predict(model, newdata = test_df, type = "response")
pred_test <- ifelse(prob_test > 0.5, 1, 0)

# Confusion matrix
tp <- sum(pred_test == 1 & y_test == 1)
tn <- sum(pred_test == 0 & y_test == 0)
fp <- sum(pred_test == 1 & y_test == 0)
fn <- sum(pred_test == 0 & y_test == 1)

accuracy    <- (tp + tn) / (tp + tn + fp + fn)
sensitivity <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
ppv         <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
npv         <- ifelse((tn + fn) > 0, tn / (tn + fn), 0)

# AUC (manual calculation to avoid extra dependencies)
compute_auc <- function(probs, labels) {
  pos <- probs[labels == 1]
  neg <- probs[labels == 0]
  if (length(pos) == 0 || length(neg) == 0) return(0.5)
  auc <- 0
  for (p in pos) {
    auc <- auc + sum(p > neg) + 0.5 * sum(p == neg)
  }
  auc / (length(pos) * length(neg))
}
auc <- compute_auc(prob_test, y_test)

result <- list(
  sample_size          = as.integer(sample_size),
  coef_                = list(coef_vals),
  intercept_           = c(intercept),
  metric_acc           = accuracy,
  metric_auc           = auc,
  metric_sensitivity   = sensitivity,
  metric_specificity   = specificity,
  metric_npv           = npv,
  metric_ppv           = ppv
)

write(toJSON(result, auto_unbox = FALSE, digits = 10), output_path)
