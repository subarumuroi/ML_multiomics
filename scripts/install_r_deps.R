#!/usr/bin/env Rscript

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Function to install if missing
install_if_missing <- function(pkg, bioc = FALSE) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    if (bioc) {
      if (!require("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
      }
      BiocManager::install(pkg, ask = FALSE, update = FALSE)
    } else {
      install.packages(pkg)
    }
  } else {
    cat(sprintf("%s already installed\n", pkg))
  }
}

# Install dependencies
install_if_missing("BiocManager")
install_if_missing("mixOmics", bioc = TRUE)
install_if_missing("jsonlite")

cat("All R dependencies ready\n")