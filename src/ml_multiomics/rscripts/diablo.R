#!/usr/bin/env Rscript
# R bridge for DIABLO -- the reference mixOmics implementation.
#   * classification (nominal/ordinal target) -> block.splsda
#   * regression     (continuous target)       -> block.spls   (mode="regression")
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
target_type <- if (!is.null(cfg$target_type)) as.character(cfg$target_type) else "nominal"
regression <- identical(target_type, "continuous")
yraw <- read.csv(file.path(d, "y.csv"))$y
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

if (regression) {
  Y <- matrix(as.numeric(yraw), ncol = 1)          # univariate continuous response
  colnames(Y) <- "y"
  rownames(Y) <- rownames(X[[blocks[1]]])          # mixOmics matches rownames across X and Y
  keepY <- if (!is.null(cfg$keepY)) as.integer(cfg$keepY) else ncol(Y)
  ky <- rep(keepY, ncomp)
  model <- block.spls(X, Y = Y, ncomp = ncomp, keepX = kx, keepY = ky,
                      design = design, mode = "regression", scale = FALSE)
} else {
  Y <- factor(yraw)
  model <- block.splsda(X, Y, ncomp = ncomp, keepX = kx, design = design, scale = FALSE)
}

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
    # classification -> $error.rate; regression (block.spls) -> $measures (MSEP/R2/Q2)
    payload <- if (!is.null(pf$error.rate)) pf$error.rate
               else if (!is.null(pf$measures)) pf$measures
               else list(note = "perf ran; no standard field for this model")
    tryCatch(
      write_json(payload, file.path(d, "cv_error.json"),
                 auto_unbox = TRUE, digits = 8, force = TRUE),
      error = function(e) writeLines(
        jsonlite::toJSON(list(note = paste("perf serialise error:", conditionMessage(e))),
                         auto_unbox = TRUE), file.path(d, "cv_error.json")))
  }
}

cat("diablo.R done (", K, "blocks, ncomp", ncomp, ")\n")
