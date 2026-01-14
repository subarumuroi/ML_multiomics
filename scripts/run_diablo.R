#!/usr/bin/env Rscript

# DIABLO Multi-Block Integration using mixOmics
# Reads preprocessed omics blocks and performs DIABLO analysis

suppressPackageStartupMessages({
  library(mixOmics)
  library(jsonlite)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_diablo.R <input_dir> <output_dir>")
}

input_dir <- args[1]
output_dir <- args[2]

cat("Starting DIABLO analysis...\n")
cat("Input directory:", input_dir, "\n")
cat("Output directory:", output_dir, "\n")

# Create output directory if it doesn't exist
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# LOAD DATA
# ============================================================================

cat("\n=== Loading Data ===\n")

# Load block data (CSV files: block_name.csv)
block_files <- list.files(input_dir, pattern = "^block_.*\\.csv$", full.names = TRUE)
block_names <- gsub("^block_|\\.csv$", "", basename(block_files))

X <- list()
for (i in seq_along(block_files)) {
  block_name <- block_names[i]
  cat("Loading block:", block_name, "\n")
  X[[block_name]] <- as.matrix(read.csv(block_files[i], row.names = 1))
}

# Load labels
y_file <- file.path(input_dir, "labels.csv")
y_df <- read.csv(y_file, row.names = 1)
Y <- as.factor(y_df$label)

cat("Loaded", length(X), "blocks with", nrow(X[[1]]), "samples\n")
cat("Classes:", levels(Y), "\n")

# ============================================================================
# DESIGN MATRIX
# ============================================================================

cat("\n=== Creating Design Matrix ===\n")

# Design matrix: controls correlation between blocks
# 0.1 = exploratory (weak connections), 1.0 = predictive (strong connections)
# For small sample sizes (n < 20), use 0.1 to avoid overfitting

n_blocks <- length(X)
design <- matrix(0.1, nrow = n_blocks, ncol = n_blocks, 
                dimnames = list(names(X), names(X)))
diag(design) <- 0

cat("Design matrix (off-diagonal = 0.1 for exploratory analysis):\n")
print(design)

# ============================================================================
# TUNING: FIND OPTIMAL PARAMETERS
# ============================================================================

cat("\n=== Tuning DIABLO Parameters ===\n")

# Test different numbers of features to keep per block
# Adjust based on your data size
test_keepX <- list()
for (block_name in names(X)) {
  n_features <- ncol(X[[block_name]])
  
  if (n_features < 20) {
    test_keepX[[block_name]] <- c(5, 10, n_features)
  } else if (n_features < 100) {
    test_keepX[[block_name]] <- c(5, 10, 20, 30)
  } else {
    test_keepX[[block_name]] <- c(10, 20, 50, 100)
  }
  
  cat("Block", block_name, "- testing keepX:", test_keepX[[block_name]], "\n")
}

# Tune with leave-one-out CV (appropriate for small n)
cat("\nRunning leave-one-out cross-validation...\n")

tune_result <- tryCatch({
  tune.block.splsda(
    X = X,
    Y = Y,
    ncomp = 3,  # Test up to 3 components
    test.keepX = test_keepX,
    design = design,
    validation = 'loo',
    dist = 'centroids.dist',
    progressBar = FALSE
  )
}, error = function(e) {
  cat("Tuning failed, using defaults\n")
  cat("Error:", e$message, "\n")
  NULL
})

# Extract optimal parameters
if (!is.null(tune_result)) {
  optimal_ncomp <- tune_result$choice.ncomp$ncomp
  optimal_keepX <- tune_result$choice.keepX
  
  cat("\nOptimal number of components:", optimal_ncomp, "\n")
  cat("Optimal features per block:\n")
  print(optimal_keepX)
} else {
  # Fallback to reasonable defaults
  optimal_ncomp <- 2
  optimal_keepX <- lapply(X, function(block) {
    min(15, ncol(block))
  })
  cat("\nUsing default parameters (tuning failed)\n")
  cat("Components:", optimal_ncomp, "\n")
  cat("Features per block:\n")
  print(optimal_keepX)
}

# ============================================================================
# FINAL MODEL
# ============================================================================

cat("\n=== Training Final DIABLO Model ===\n")

final_model <- block.splsda(
  X = X,
  Y = Y,
  ncomp = optimal_ncomp,
  keepX = optimal_keepX,
  design = design
)

cat("Model trained successfully\n")

# ============================================================================
# PERFORMANCE EVALUATION
# ============================================================================

cat("\n=== Evaluating Performance ===\n")

perf_result <- perf(
  final_model,
  validation = 'loo',
  progressBar = FALSE
)

cat("\nOverall error rate (LOO):\n")
print(perf_result$error.rate$overall)

# ============================================================================
# EXTRACT RESULTS
# ============================================================================

cat("\n=== Extracting Results ===\n")

# Sample projections (variates)
variates <- lapply(final_model$variates, function(v) {
  as.data.frame(v)
})

# Feature loadings
loadings <- lapply(final_model$loadings, function(l) {
  as.data.frame(l)
})

# Variable selection (selected features per component)
selected_vars <- selectVar(final_model, comp = 1)

# Save selected features for each block
for (block_name in names(X)) {
  if (!is.null(selected_vars[[block_name]])) {
    sel_df <- as.data.frame(selected_vars[[block_name]]$value)
    sel_df$feature <- rownames(sel_df)
    write.csv(sel_df, 
             file.path(output_dir, paste0("selected_features_", block_name, ".csv")),
             row.names = FALSE)
  }
}

# ============================================================================
# SAVE RESULTS
# ============================================================================

cat("\n=== Saving Results ===\n")

# Save variates (sample coordinates)
for (block_name in names(variates)) {
  write.csv(variates[[block_name]], 
           file.path(output_dir, paste0("variates_", block_name, ".csv")),
           row.names = TRUE)
}

# Save loadings (feature contributions)
for (block_name in names(loadings)) {
  write.csv(loadings[[block_name]], 
           file.path(output_dir, paste0("loadings_", block_name, ".csv")),
           row.names = TRUE)
}

# Save performance metrics
perf_df <- data.frame(
  component = paste0("comp", 1:optimal_ncomp),
  overall_error = perf_result$error.rate$overall[, "centroids.dist", 1:optimal_ncomp],
  balanced_error = perf_result$error.rate.class$centroids.dist[, 1:optimal_ncomp]
)
write.csv(perf_df, 
         file.path(output_dir, "performance_metrics.csv"),
         row.names = FALSE)

# Save model summary as JSON
model_summary <- list(
  n_components = optimal_ncomp,
  n_samples = nrow(X[[1]]),
  blocks = lapply(names(X), function(b) {
    list(
      name = b,
      n_features = ncol(X[[b]]),
      n_selected = optimal_keepX[[b]]
    )
  }),
  performance = list(
    overall_error_comp1 = perf_result$error.rate$overall["Overall.ER", "centroids.dist", 1],
    overall_error_comp2 = if(optimal_ncomp >= 2) perf_result$error.rate$overall["Overall.ER", "centroids.dist", 2] else NA
  )
)

write_json(model_summary, 
          file.path(output_dir, "model_summary.json"),
          pretty = TRUE, auto_unbox = TRUE)

cat("\n=== DIABLO Analysis Complete ===\n")
cat("Results saved to:", output_dir, "\n")