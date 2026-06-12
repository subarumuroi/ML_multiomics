#!/usr/bin/env Rscript
# imputePCA cross-check worker: runs missMDA::imputePCA on a whole matrix and the
# lab's impute_matrix_by_group (per-group imputePCA) on a grouped matrix.
suppressMessages({
  library(missMDA)
  library(assertthat)
})

args <- commandArgs(trailingOnly = TRUE)
d <- args[1]
src <- args[2]
source(file.path(src, "imputations.R"))   # impute_matrix_by_group(), split_matrix_by_group()

xm <- as.matrix(read.csv(file.path(d, "ipca_x.csv"), row.names = 1))
grp <- read.csv(file.path(d, "ipca_groups.csv"))$group

# whole-matrix imputePCA (defaults: ncp=2, scale=TRUE, Regularized)
res <- imputePCA(xm)
write.csv(res$completeObs, file.path(d, "r_ipca_whole.csv"))

# lab per-group imputation
bg <- impute_matrix_by_group(xm, grp)
write.csv(bg, file.path(d, "r_ipca_bygroup.csv"))

cat("imputePCA reference outputs written.\n")
