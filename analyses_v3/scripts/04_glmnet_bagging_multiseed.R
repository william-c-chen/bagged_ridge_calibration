#!/usr/bin/env Rscript
# 100 independent seed-runs of FFPE-trained glmnet bagging.
# Purpose: characterize the distribution of ||β̄||, LOO-CV, |Δslope|
# across bagging replicates with the correct FFPE training data.
library(glmnet)
dat <- read.csv("analyses/data/ffpe_paired_training_pc.csv", stringsAsFactors = FALSE)
y <- dat$y; X <- as.matrix(dat[, -1])
cat("FFPE paired training: N=", nrow(X), "\n")

# Lambda grid from full-data cv.glmnet
set.seed(0)
cv0 <- cv.glmnet(X, y, alpha = 0, nfolds = 10)
lambda_seq <- cv0$lambda

M_SEEDS <- 100; B <- 500
all_beta_bars <- matrix(0, ncol(X), M_SEEDS)
all_intercepts <- numeric(M_SEEDS)
all_l2 <- numeric(M_SEEDS)
all_cv_const <- numeric(M_SEEDS)
all_med_const <- numeric(M_SEEDS)
t0 <- Sys.time()
for (m in seq_len(M_SEEDS)) {
  seed_m <- 42 + m * 100
  set.seed(seed_m)
  coefs <- matrix(0, ncol(X), B); intcs <- numeric(B)
  for (b in seq_len(B)) {
    idx <- sample(nrow(X), nrow(X), replace = TRUE)
    cvfit <- cv.glmnet(X[idx, ], y[idx], alpha = 0, lambda = lambda_seq, nfolds = 10)
    cv_ <- as.vector(coef(cvfit, s = cvfit$lambda.1se))
    intcs[b] <- cv_[1]; coefs[, b] <- cv_[-1]
  }
  beta <- rowMeans(coefs); all_beta_bars[, m] <- beta
  all_intercepts[m] <- mean(intcs)
  all_l2[m] <- sqrt(sum(beta^2))
  norms <- apply(coefs, 2, function(x) sqrt(sum(x^2)))
  all_cv_const[m] <- sd(norms)/mean(norms)
  all_med_const[m] <- median(norms)
  if (m %% 10 == 0) cat(sprintf("  %d/%d  elapsed %.0fs\n", m, M_SEEDS,
                                as.numeric(Sys.time() - t0, units = "secs")))
}
cat(sprintf("Total: %.0fs\n", as.numeric(Sys.time() - t0, units = "secs")))

cat(sprintf("\n||β̄||_2: min=%.4f  median=%.4f  max=%.4f  (deployed: 0.1012)\n",
            min(all_l2), median(all_l2), max(all_l2)))
cat(sprintf("median constituent norm: median=%.4f  (deployed: 0.1204)\n", median(all_med_const)))
cat(sprintf("constituent CV:          median=%.3f  (deployed: 0.280)\n", median(all_cv_const)))

# Save
gene_names <- colnames(X)
df <- as.data.frame(all_beta_bars); colnames(df) <- paste0("seed_", 42 + seq_len(M_SEEDS) * 100)
df <- cbind(gene = gene_names, df)
write.csv(df, "analyses/data/glmnet_ffpe_multiseed_betas.csv", row.names = FALSE)
write.csv(data.frame(seed = 42 + seq_len(M_SEEDS) * 100,
                     intercept = all_intercepts, l2 = all_l2,
                     cv_const = all_cv_const, med_const = all_med_const),
          "analyses/data/glmnet_ffpe_multiseed_meta.csv", row.names = FALSE)
cat("Saved to analyses/data/glmnet_ffpe_multiseed_*.csv\n")
