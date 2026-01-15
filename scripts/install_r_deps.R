#!/usr/bin/env Rscript

# Install required R packages with specific versions
install.packages('BiocManager')
BiocManager::install('mixOmics')  # Version 6.26.0 tested
install.packages('jsonlite')      # Version 1.8.8 tested

cat("R dependencies installed successfully\n")