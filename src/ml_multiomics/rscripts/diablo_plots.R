#!/usr/bin/env Rscript
# Native mixOmics DIABLO figures (the iconic plots the tables can't replace).
# Re-fits block.s/plsda from the SAME work-dir contract as diablo.R
# (config.json, block_<name>.csv, y.csv) -- a separate, isolated step so a
# finicky plot (e.g. circosPlot) can fail without touching the fitted results.
# Each plot is guarded; a manifest of what was produced is written as JSON.
# PNGs are written to cfg$plotdir (persistent) if given, else the work dir.
suppressMessages({
  library(mixOmics)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
d <- args[1]
cfg <- fromJSON(file.path(d, "config.json"))
outdir <- if (!is.null(cfg$plotdir)) cfg$plotdir else d
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
prefix <- if (!is.null(cfg$prefix)) cfg$prefix else "diablo"

blocks <- as.character(cfg$blocks)
X <- list()
for (b in blocks) {
  X[[b]] <- as.matrix(read.csv(file.path(d, paste0("block_", b, ".csv")),
                               row.names = 1, check.names = FALSE))
}
target_type <- if (!is.null(cfg$target_type)) as.character(cfg$target_type) else "nominal"
regression <- identical(target_type, "continuous")
yraw <- read.csv(file.path(d, "y.csv"))$y
ncomp <- max(2L, as.integer(cfg$ncomp))   # >=2 components so the iconic plots have an axis to draw

K <- length(blocks)
design <- matrix(as.numeric(cfg$design), K, K); diag(design) <- 0

kx <- list()
for (b in blocks) {
  v <- if (!is.null(cfg$keepX) && !is.null(cfg$keepX[[b]])) cfg$keepX[[b]] else min(20L, ncol(X[[b]]))
  if (length(v) == 1) v <- rep(as.integer(v), ncomp)
  kx[[b]] <- as.integer(v)
}

if (regression) {
  Y <- matrix(as.numeric(yraw), ncol = 1); colnames(Y) <- "y"
  rownames(Y) <- rownames(X[[blocks[1]]])
  keepY <- if (!is.null(cfg$keepY)) as.integer(cfg$keepY) else 1L
  model <- block.spls(X, Y = Y, ncomp = ncomp, keepX = kx, keepY = rep(keepY, ncomp),
                      design = design, mode = "regression", scale = FALSE)
  # a categorical grouping for colour only: tertiles of the continuous target
  grp <- cut(as.numeric(yraw), breaks = quantile(as.numeric(yraw), c(0, 1/3, 2/3, 1)),
             include.lowest = TRUE, labels = c("low", "mid", "high"))
} else {
  Y <- factor(yraw)
  model <- block.splsda(X, Y, ncomp = ncomp, keepX = kx, design = design, scale = FALSE)
  grp <- Y
}

made <- c()
png_open <- function(name, w = 1400, h = 1200) {
  fp <- file.path(outdir, paste0(prefix, "_", name, ".png"))
  png(fp, width = w, height = h, res = 150); fp
}
guard <- function(name, expr, w = 1400, h = 1200) {
  fp <- png_open(name, w, h)
  ok <- tryCatch({ force(expr); TRUE },
                 error = function(e) { cat(name, "failed:", conditionMessage(e), "\n"); FALSE })
  dev.off()
  if (ok) made[[length(made) + 1]] <<- basename(fp) else unlink(fp)
}

# plotIndiv: per-block sample projection (the "do the blocks agree?" view)
guard("plotIndiv", {
  plotIndiv(model, ind.names = FALSE, legend = TRUE, group = grp,
            title = "DIABLO sample projection", ellipse = !regression)
})

# plotLoadings: top contributing features on component 1
guard("plotLoadings", {
  if (regression) plotLoadings(model, comp = 1, ndisplay = 15, size.name = 0.7,
                               title = "DIABLO loadings (comp 1)")
  else plotLoadings(model, comp = 1, contrib = "max", method = "median", ndisplay = 15,
                    size.name = 0.7, title = "DIABLO loadings (comp 1)")
})

# plotVar: cross-block variable correlation circle (works for both modes) --
# the regression-safe cousin of circosPlot.
if (K >= 2) {
  guard("plotVar", {
    plotVar(model, var.names = FALSE, cutoff = 0.5, legend = TRUE,
            title = "DIABLO variable correlation (comp 1-2)")
  })
}

# circosPlot: the iconic cross-block correlation circle. Reliable for
# classification (block.splsda); fragile for block.spls regression, so
# classification-only (guarded -- absence is non-fatal).
if (!regression && K >= 2) {
  guard("circos", {
    circosPlot(model, cutoff = 0.7, line = TRUE, size.variables = 0.6, size.labels = 1.0)
  }, w = 1500, h = 1500)
  guard("plotDiablo", { plotDiablo(model, ncomp = 1) })
}

write_json(list(plots = made, regression = regression, n_blocks = K),
           file.path(outdir, paste0(prefix, "_manifest.json")), auto_unbox = TRUE)
cat("diablo_plots.R done;", length(made), "plot(s)\n")
