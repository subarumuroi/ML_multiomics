#!/usr/bin/env Rscript
# R bridge for DIABLO — calls mixOmics::block.splsda (the reference implementation).
# Reads a work dir containing config.json, block_<name>.csv (samples x features,
# row names = sample IDs), and y.csv; writes per-block variates, loadings,
# selected features, and (optionally) leave-one-out CV error from mixOmics perf().
suppressMessages({
  library(mixOmics)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
d <- args[1]
cfg <- fromJSON(file.path(d, "config.json"))

blocks <- as.character(cfg$blocks)
X <- list()
for (b in blocks) {
  X[[b]] <- as.matrix(read.csv(file.path(d, paste0("block_", b, ".csv")),
                               row.names = 1, check.names = FALSE))
}
Y <- factor(read.csv(file.path(d, "y.csv"))$y)
ncomp <- as.integer(cfg$ncomp)

K <- length(blocks)
design <- matrix(as.numeric(cfg$design), K, K)
diag(design) <- 0

# keepX: named list block -> int|vector, or absent for non-sparse (full)
kx <- list()
for (b in blocks) {
  v <- if (!is.null(cfg$keepX) && !is.null(cfg$keepX[[b]])) cfg$keepX[[b]] else ncol(X[[b]])
  if (length(v) == 1) v <- rep(as.integer(v), ncomp)
  kx[[b]] <- as.integer(v)
}

model <- block.splsda(X, Y, ncomp = ncomp, keepX = kx, design = design, scale = FALSE)

for (b in blocks) {
  write.csv(model$variates[[b]], file.path(d, paste0("variates_", b, ".csv")), row.names = FALSE)
  write.csv(model$loadings[[b]], file.path(d, paste0("loadings_", b, ".csv")))
  for (k in seq_len(ncomp)) {
    sel <- rownames(model$loadings[[b]])[model$loadings[[b]][, k] != 0]
    writeLines(sel, file.path(d, paste0("selected_", b, "_c", k, ".txt")))
  }
}

if (isTRUE(cfg$cv)) {
  pf <- tryCatch(perf(model, validation = "loo"),
                 error = function(e) { cat("perf error:", conditionMessage(e), "\n"); NULL })
  if (!is.null(pf)) {
    write_json(pf$error.rate, file.path(d, "cv_error.json"), auto_unbox = TRUE, digits = 8)
  }
}

cat("diablo.R done (", K, "blocks, ncomp", ncomp, ")\n")
