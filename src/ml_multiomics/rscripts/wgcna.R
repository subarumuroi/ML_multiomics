#!/usr/bin/env Rscript
# R bridge for WGCNA — calls the reference WGCNA package (Langfelder & Horvath).
# Reads work dir with config.json + block.csv (samples x features, row names =
# sample IDs); writes feature->module assignments and module eigengenes
# (samples x modules) = the reduced representation. Uses the real dynamic tree
# cut + close-module merge, which the native Python port only approximated.
suppressMessages({
  library(WGCNA)
  library(jsonlite)
})
options(stringsAsFactors = FALSE)

args <- commandArgs(trailingOnly = TRUE)
d <- args[1]
cfg <- fromJSON(file.path(d, "config.json"))

X <- as.matrix(read.csv(file.path(d, "block.csv"), row.names = 1, check.names = FALSE))
networkType <- if (!is.null(cfg$network_type)) cfg$network_type else "unsigned"
minModuleSize <- if (!is.null(cfg$min_module_size)) as.integer(cfg$min_module_size) else 20L
mergeCutHeight <- if (!is.null(cfg$merge_cut_height)) as.numeric(cfg$merge_cut_height) else 0.25

power <- cfg$power
if (is.null(power)) {
  sft <- pickSoftThreshold(X, powerVector = 1:20, networkType = networkType, verbose = 0)
  power <- sft$powerEstimate
  if (is.na(power)) power <- if (networkType == "unsigned") 6L else 12L
}
power <- as.integer(power)

adj <- adjacency(X, power = power, type = networkType)
TOM <- TOMsimilarity(adj, TOMType = networkType, verbose = 0)
diss <- 1 - TOM
tree <- hclust(as.dist(diss), method = "average")
mods <- cutreeDynamic(dendro = tree, distM = diss, deepSplit = 2,
                      pamRespectsDendro = FALSE, minClusterSize = minModuleSize)
colors <- labels2colors(mods)
merged <- mergeCloseModules(X, colors, cutHeight = mergeCutHeight, verbose = 0)
colors <- merged$colors
MEs <- merged$newMEs            # samples x modules (ME<color>), grey = unassigned

write.csv(data.frame(feature = colnames(X), module = colors),
          file.path(d, "modules.csv"), row.names = FALSE)
write.csv(MEs, file.path(d, "eigengenes.csv"))   # row names = sample IDs

cat("wgcna.R done; power", power, "; modules",
    length(setdiff(unique(colors), "grey")), "\n")
