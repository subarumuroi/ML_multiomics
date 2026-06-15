#!/usr/bin/env Rscript
# Parity probe: run mixOmics::splsda (the reference) on a pre-z-scored matrix
# (scale=FALSE so we don't double-scale) and write its variates + selected
# feature indices for comparison against our native SparsePLSDA.
suppressMessages(library(mixOmics))

args <- commandArgs(trailingOnly = TRUE)
d <- args[1]
X <- as.matrix(read.csv(file.path(d, "sp_X.csv"), row.names = 1, check.names = FALSE))
y <- factor(read.csv(file.path(d, "sp_y.csv"))$y)
keepX <- as.integer(args[2])

res <- splsda(X, y, ncomp = 2, keepX = c(keepX, keepX), scale = FALSE)

write.csv(res$variates$X, file.path(d, "r_splsda_variates.csv"), row.names = FALSE)
# selected = non-zero loadings per component; write 1-based column indices
for (k in 1:2) {
  idx <- which(res$loadings$X[, k] != 0)
  writeLines(as.character(idx), file.path(d, paste0("r_splsda_sel", k, ".txt")))
}
cat("mixOmics splsda done\n")
