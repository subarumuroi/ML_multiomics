#!/usr/bin/env Rscript
# Faithful cross-check worker: sources the lab's ACTUAL functions from
# StandardOmicAnalyses (IdeaBio.R) and runs them on the shared input, writing
# outputs the Python side will diff against.
suppressMessages({
  library(assertthat)
  library(missMDA)
})

args <- commandArgs(trailingOnly = TRUE)
d <- args[1]                                   # working dir with inputs
src <- args[2]                                 # path to StandardOmicAnalyses/R

source(file.path(src, "normalisations.R"))     # zscore()
source(file.path(src, "imputations.R"))        # impute_missing_metaboanalyst(), impute_matrix_by_group()

read_mat <- function(f) as.matrix(read.csv(file.path(d, f), row.names = 1))

xc <- read_mat("x_complete.csv")
xm <- read_mat("x_missing.csv")
grp <- read.csv(file.path(d, "groups.csv"))$group

# z-score per column (dim = 2) — the lab's actual zscore()
write.csv(zscore(xc, 2), file.path(d, "r_zscore_complete.csv"))
write.csv(zscore(xm, 2), file.path(d, "r_zscore_missing.csv"))

# log10 — the lab's DE transform convention
write.csv(log10(xc), file.path(d, "r_log10_complete.csv"))

# MetaboAnalyst imputation — the lab's actual function
write.csv(impute_missing_metaboanalyst(xm), file.path(d, "r_metaboanalyst.csv"))

# imputePCA per group — the lab's DEFAULT imputation (missMDA)
ip <- tryCatch(impute_matrix_by_group(xm, grp),
               error = function(e) { cat("imputePCA error:", conditionMessage(e), "\n"); NULL })
if (!is.null(ip)) write.csv(ip, file.path(d, "r_imputepca.csv"))

cat("R reference outputs written.\n")
