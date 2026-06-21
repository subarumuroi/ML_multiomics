#!/usr/bin/env Rscript
# Native WGCNA figures: the soft-threshold scale-free-fit plot, the module
# dendrogram (plotDendroAndColors), and -- when a trait y.csv is present -- the
# module-trait relationship heatmap (labeledHeatmap of cor(eigengene, trait)).
# Re-runs the same network construction as wgcna.R from the SAME work-dir
# contract (config.json, block.csv, optional y.csv). Each plot is guarded.
suppressMessages({
  library(WGCNA)
  library(jsonlite)
})
options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
d <- args[1]
cfg <- fromJSON(file.path(d, "config.json"))
outdir <- if (!is.null(cfg$plotdir)) cfg$plotdir else d
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
prefix <- if (!is.null(cfg$prefix)) cfg$prefix else "wgcna"

X <- as.matrix(read.csv(file.path(d, "block.csv"), row.names = 1, check.names = FALSE))
networkType <- if (!is.null(cfg$network_type)) cfg$network_type else "unsigned"
minModuleSize <- if (!is.null(cfg$min_module_size)) as.integer(cfg$min_module_size) else 20L
mergeCutHeight <- if (!is.null(cfg$merge_cut_height)) as.numeric(cfg$merge_cut_height) else 0.25

made <- c()
guard <- function(name, expr, w = 1400, h = 1100) {
  fp <- file.path(outdir, paste0(prefix, "_", name, ".png"))
  png(fp, width = w, height = h, res = 150)
  ok <- tryCatch({ force(expr); TRUE },
                 error = function(e) { cat(name, "failed:", conditionMessage(e), "\n"); FALSE })
  dev.off()
  if (ok) made[[length(made) + 1]] <<- basename(fp) else unlink(fp)
}

sft <- pickSoftThreshold(X, powerVector = 1:20, networkType = networkType, verbose = 0)
power <- if (!is.null(cfg$power)) as.integer(cfg$power) else sft$powerEstimate
if (is.na(power)) power <- if (networkType == "unsigned") 6L else 12L
power <- as.integer(power)

# soft-threshold scale-free-fit + mean-connectivity (the "why this power" plot)
guard("softthreshold", {
  par(mfrow = c(1, 2))
  fit <- -sign(sft$fitIndices[, 3]) * sft$fitIndices[, 2]
  plot(sft$fitIndices[, 1], fit, type = "n", xlab = "soft-threshold power",
       ylab = "scale-free topology R^2", main = "Scale-free fit")
  text(sft$fitIndices[, 1], fit, labels = sft$fitIndices[, 1], col = "red")
  abline(h = 0.8, col = "blue", lty = 2)
  plot(sft$fitIndices[, 1], sft$fitIndices[, 5], type = "n", xlab = "soft-threshold power",
       ylab = "mean connectivity", main = "Mean connectivity")
  text(sft$fitIndices[, 1], sft$fitIndices[, 5], labels = sft$fitIndices[, 1], col = "red")
  abline(v = power, col = "blue", lty = 2)
}, w = 1700, h = 850)

adj <- adjacency(X, power = power, type = networkType)
TOM <- TOMsimilarity(adj, TOMType = networkType, verbose = 0)
diss <- 1 - TOM
tree <- hclust(as.dist(diss), method = "average")
mods <- cutreeDynamic(dendro = tree, distM = diss, deepSplit = 2,
                      pamRespectsDendro = FALSE, minClusterSize = minModuleSize)
colors0 <- labels2colors(mods)
merged <- mergeCloseModules(X, colors0, cutHeight = mergeCutHeight, verbose = 0)
colors <- merged$colors
MEs <- merged$newMEs

# module dendrogram with colour bands (the iconic WGCNA figure)
guard("dendrogram", {
  plotDendroAndColors(tree, cbind(colors0, colors),
                      c("dynamic", "merged"), dendroLabels = FALSE, addGuide = TRUE,
                      hang = 0.03, guideHang = 0.05, main = "Feature dendrogram & modules")
})

# module-trait relationship heatmap (needs a trait)
ypath <- file.path(d, "y.csv")
if (file.exists(ypath)) {
  yraw <- read.csv(ypath)$y
  trait <- suppressWarnings(as.numeric(yraw))
  if (!all(is.na(trait)) && length(unique(trait[!is.na(trait)])) > 1) {
    guard("module_trait", {
      MEo <- orderMEs(MEs)
      mtc <- cor(MEo, trait, use = "p")
      mtp <- corPvalueStudent(mtc, nrow(X))
      txt <- paste0(signif(mtc, 2), "\n(", signif(mtp, 1), ")")
      dim(txt) <- dim(mtc)
      par(mar = c(6, 9, 3, 3))
      labeledHeatmap(Matrix = mtc, xLabels = "trait", yLabels = names(MEo),
                     ySymbols = names(MEo), colorLabels = FALSE, colors = blueWhiteRed(50),
                     textMatrix = txt, setStdMargins = FALSE, cex.text = 0.7, zlim = c(-1, 1),
                     main = "Module-trait relationships")
    }, w = 700, h = 1100)
  }
}

write_json(list(plots = made, power = power,
                n_modules = length(setdiff(unique(colors), "grey"))),
           file.path(outdir, paste0(prefix, "_manifest.json")), auto_unbox = TRUE)
cat("wgcna_plots.R done; power", power, ";", length(made), "plot(s)\n")
