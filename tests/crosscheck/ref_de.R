#!/usr/bin/env Rscript
# Cross-check worker: runs the lab's ACTUAL IdeaBio.R DE functions
# (foldchange.R compute_volcano / compute_anova_tukey) and base R phyper, and
# writes outputs the Python side diffs against.
suppressMessages({ library(assertthat) })

args <- commandArgs(trailingOnly = TRUE)
d <- args[1]
src <- args[2]
source(file.path(src, "helpers.R"))      # .extract_pairwise_rows
source(file.path(src, "foldchange.R"))

x <- as.matrix(read.csv(file.path(d, "de_x.csv"), row.names = 1))   # samples x features
grp <- read.csv(file.path(d, "de_groups.csv"))$group

# compute_volcano: fold change on linear, t-test on log10, bonferroni FDR
v <- compute_volcano(x, logx = TRUE, rgrp = grp, fdr_method = "bonferroni")
write.csv(v, file.path(d, "r_volcano.csv"), row.names = FALSE)

# hypergeometric upper tail (ORA statistic) via base R phyper
# P(X >= k) = phyper(k-1, K, N-K, n, lower.tail = FALSE)
hyper <- data.frame(
  k = c(3, 5, 1),
  K = c(20, 50, 10),
  N = c(200, 200, 200),
  n = c(30, 40, 15)
)
hyper$p <- phyper(hyper$k - 1, hyper$K, hyper$N - hyper$K, hyper$n, lower.tail = FALSE)
write.csv(hyper, file.path(d, "r_phyper.csv"), row.names = FALSE)

cat("DE reference outputs written.\n")
