#!/usr/bin/env Rscript
#
# DIABLO Visualization Functions
# 
# This script provides publication-quality visualization functions for DIABLO
# multi-omics integration analysis using mixomics R package.
#
# Author: ML Multiomics Framework
# Date: 2024

# Suppress warnings
suppressWarnings(library(mixOmics))
suppressWarnings(library(igraph))

#' Plot DIABLO Sample Scores with Block Overlay
#' 
#' Creates a sample score plot showing how samples cluster across all blocks
#' with overlaid block-specific representations.
#' 
#' @param diablo_model DIABLO fitted model object
#' @param y Group labels (factor)
#' @param comp_x Component for X-axis (default: 1)
#' @param comp_y Component for Y-axis (default: 2)
#' @param output_file Path to save PNG file
#' @param width Figure width in pixels (default: 800)
#' @param height Figure height in pixels (default: 700)
plot_diablo_samples <- function(diablo_model, y, comp_x = 1, comp_y = 2, 
                               output_file = NULL, width = 800, height = 700) {
  
  if (!is.null(output_file)) {
    png(filename = output_file, width = width, height = height, res = 100)
  }
  
  # Enhanced plotDiablo with all blocks overlaid
  # This shows sample agreement across blocks
  plotDiablo(diablo_model, 
            ncomp = max(comp_x, comp_y),
            legend = TRUE,
            legend.position = 'right',
            title = paste('DIABLO Sample Plot (Comp', comp_x, 'vs', comp_y, ')'),
            size.xlabel = 13,
            size.ylabel = 13)
  
  if (!is.null(output_file)) {
    dev.off()
    cat(paste("Saved:", output_file, "\n"))
  }
}

#' Plot Individual Component Scores with Convex Hulls
#' 
#' Creates sample score plots for each component with group-specific
#' convex hulls to show cluster separation.
#' 
#' @param diablo_model DIABLO fitted model object
#' @param y Group labels (factor)
#' @param comp_x Component for X-axis (default: 1)
#' @param comp_y Component for Y-axis (default: 2)
#' @param output_file Path to save PNG file
#' @param width Figure width in pixels (default: 1000)
#' @param height Figure height in pixels (default: 700)
plot_diablo_indiv <- function(diablo_model, y, comp_x = 1, comp_y = 2,
                             output_file = NULL, width = 1000, height = 700) {
  
  if (!is.null(output_file)) {
    png(filename = output_file, width = width, height = height, res = 100)
  }
  
  # Plot for each block with convex hulls
  n_blocks <- length(diablo_model$X)
  block_names <- names(diablo_model$X)
  
  # Create color palette
  colors <- color.mixo(1:nlevels(y))
  
  # Plot
  plotIndiv(diablo_model,
           comp = c(comp_x, comp_y),
           ind.names = FALSE,
           ellipse = TRUE,
           legend = TRUE,
           legend.position = 'right',
           title = 'DIABLO Individual Scores with Confidence Ellipses',
           size.xlabel = 13,
           size.ylabel = 13)
  
  if (!is.null(output_file)) {
    dev.off()
    cat(paste("Saved:", output_file, "\n"))
  }
}

#' Plot Variable Loadings Across Blocks
#' 
#' Creates a heatmap showing feature loadings for each component across blocks.
#' Useful for understanding which features drive the multi-omics separation.
#' 
#' @param diablo_model DIABLO fitted model object
#' @param comp Component to visualize (default: 1)
#' @param output_file Path to save PNG file
#' @param width Figure width in pixels (default: 1200)
#' @param height Figure height in pixels (default: 800)
plot_diablo_var <- function(diablo_model, comp = 1, 
                           output_file = NULL, width = 1200, height = 800) {
  
  if (!is.null(output_file)) {
    png(filename = output_file, width = width, height = height, res = 100)
  }
  
  plotVar(diablo_model,
         comp = c(comp, comp),  # comp for sample and variable space
         var.names = TRUE,
         style = 'graphics',
         legend = TRUE,
         title = paste('DIABLO Variable Loadings (Comp', comp, ')'),
         size.xlabel = 13,
         size.ylabel = 13)
  
  if (!is.null(output_file)) {
    dev.off()
    cat(paste("Saved:", output_file, "\n"))
  }
}

#' Plot Loadings Comparison Across Blocks
#' 
#' Creates a barplot of loading weights for top features across different blocks,
#' allowing comparison of feature importance between omics layers.
#' 
#' @param diablo_model DIABLO fitted model object
#' @param comp Component to visualize (default: 1)
#' @param contrib Contribution type: 'max' or 'median' (default: 'max')
#' @param output_file Path to save PNG file
#' @param width Figure width in pixels (default: 1200)
#' @param height Figure height in pixels (default: 800)
plot_diablo_loadings <- function(diablo_model, comp = 1, contrib = 'max',
                                output_file = NULL, width = 1200, height = 800) {
  
  if (!is.null(output_file)) {
    png(filename = output_file, width = width, height = height, res = 100)
  }
  
  plotLoadings(diablo_model,
              comp = comp,
              contrib = contrib,
              method = 'mean',
              legend = TRUE,
              legend.position = 'right',
              title = paste('DIABLO Loadings Comparison (Comp', comp, ')'),
              size.xlabel = 13,
              size.ylabel = 13)
  
  if (!is.null(output_file)) {
    dev.off()
    cat(paste("Saved:", output_file, "\n"))
  }
}

#' Plot Clustered Image Map of DIABLO Results
#' 
#' Creates a clustered heatmap showing top features and sample clustering
#' across all omics blocks.
#' 
#' @param diablo_model DIABLO fitted model object
#' @param y Group labels (factor)
#' @param output_file Path to save PNG file
#' @param width Figure width in pixels (default: 1000)
#' @param height Figure height in pixels (default: 900)
#' @param n_features Number of top features to display (default: 50)
plot_diablo_cim <- function(diablo_model, y, output_file = NULL, 
                           width = 1000, height = 900, n_features = 50) {
  
  if (!is.null(output_file)) {
    png(filename = output_file, width = width, height = height, res = 100)
  }
  
  # Create clustered image map
  cimDiablo(diablo_model,
           color.grad = c('blue', 'white', 'red'),
           legend.position = 'right',
           margins = c(10, 10),
           title = 'DIABLO Clustered Image Map')
  
  if (!is.null(output_file)) {
    dev.off()
    cat(paste("Saved:", output_file, "\n"))
  }
}

#' Plot Network of Feature Correlations
#' 
#' Creates a network graph showing correlations between features across blocks,
#' useful for understanding feature relationships in multi-omics context.
#' 
#' @param diablo_model DIABLO fitted model object
#' @param comp Component to visualize (default: 1)
#' @param output_file Path to save PNG file
#' @param threshold Correlation threshold for edge drawing (default: 0.3)
#' @param width Figure width in pixels (default: 1000)
#' @param height Figure height in pixels (default: 1000)
plot_diablo_network <- function(diablo_model, comp = 1, threshold = 0.3,
                               output_file = NULL, width = 1000, height = 1000) {
  
  if (!is.null(output_file)) {
    png(filename = output_file, width = width, height = height, res = 100)
  }
  
  network(diablo_model,
         comp = comp,
         threshold = threshold,
         color.node = color.mixo(1:length(diablo_model$X)),
         shape.node = c('rectangle', 'circle', 'triangle'),
         title = paste('Feature Correlation Network (Comp', comp, ')'))
  
  if (!is.null(output_file)) {
    dev.off()
    cat(paste("Saved:", output_file, "\n"))
  }
}

#' Plot Enhanced Arrow Plot with All Samples
#' 
#' Creates an arrow plot showing agreement between omics blocks for each sample.
#' Arrows point from individual block positions to their centroid.
#' 
#' @param diablo_model DIABLO fitted model object
#' @param y Group labels (factor)
#' @param comp_x Component for X-axis (default: 1)
#' @param comp_y Component for Y-axis (default: 2)
#' @param output_file Path to save PNG file
#' @param width Figure width in pixels (default: 900)
#' @param height Figure height in pixels (default: 800)
plot_diablo_arrow <- function(diablo_model, y, comp_x = 1, comp_y = 2,
                             output_file = NULL, width = 900, height = 800) {
  
  if (!is.null(output_file)) {
    png(filename = output_file, width = width, height = height, res = 100)
  }
  
  plotArrow(diablo_model,
           comp = c(comp_x, comp_y),
           legend = TRUE,
           legend.position = 'right',
           title = paste('DIABLO Arrow Plot - Block Agreement (Comp', 
                        comp_x, 'vs', comp_y, ')'),
           size.xlabel = 13,
           size.ylabel = 13)
  
  if (!is.null(output_file)) {
    dev.off()
    cat(paste("Saved:", output_file, "\n"))
  }
}

#' Plot Enhanced Circos Plot
#' 
#' Creates a circos plot showing block relationships and feature connections
#' with colors representing block-specific information.
#' 
#' @param diablo_model DIABLO fitted model object
#' @param comp Component to visualize (default: 1)
#' @param output_file Path to save PNG file
#' @param width Figure width in pixels (default: 900)
#' @param height Figure height in pixels (default: 900)
#' @param threshold Correlation threshold (default: 0.5)
plot_diablo_circos <- function(diablo_model, comp = 1, threshold = 0.5,
                              output_file = NULL, width = 900, height = 900) {
  
  if (!is.null(output_file)) {
    png(filename = output_file, width = width, height = height, res = 100)
  }
  
  # Create circos plot with block-specific colors
  circosPlot(diablo_model,
            cutoff = threshold,
            ncol.legend = 2,
            size.legend = 1,
            comp = comp)
  
  if (!is.null(output_file)) {
    dev.off()
    cat(paste("Saved:", output_file, "\n"))
  }
}

#' Generate All DIABLO Visualizations
#' 
#' Convenience function to generate all standard DIABLO plots at once.
#' 
#' @param diablo_model DIABLO fitted model object
#' @param y Group labels (factor)
#' @param output_dir Directory to save all plots
#' @param comp_x Component for X-axis (default: 1)
#' @param comp_y Component for Y-axis (default: 2)
#' @return List of file paths to generated plots
generate_all_diablo_plots <- function(diablo_model, y, output_dir = 'results/multi_omics',
                                     comp_x = 1, comp_y = 2) {
  
  # Create output directory if needed
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  generated_files <- list()
  
  cat("\nGenerating DIABLO visualizations...\n")
  cat("="*60, "\n")
  
  # 1. Sample Plot
  file1 <- file.path(output_dir, 'diablo_samples.png')
  plot_diablo_samples(diablo_model, y, comp_x, comp_y, file1)
  generated_files$samples <- file1
  
  # 2. Individual Scores
  file2 <- file.path(output_dir, 'diablo_indiv.png')
  plot_diablo_indiv(diablo_model, y, comp_x, comp_y, file2)
  generated_files$indiv <- file2
  
  # 3. Variable Loadings
  file3 <- file.path(output_dir, 'diablo_var.png')
  plot_diablo_var(diablo_model, comp_x, file3)
  generated_files$var <- file3
  
  # 4. Loadings Comparison
  file4 <- file.path(output_dir, 'diablo_loadings.png')
  plot_diablo_loadings(diablo_model, comp_x, output_file = file4)
  generated_files$loadings <- file4
  
  # 5. Clustered Image Map
  file5 <- file.path(output_dir, 'diablo_cim.png')
  plot_diablo_cim(diablo_model, y, file5)
  generated_files$cim <- file5
  
  # 6. Network Plot
  file6 <- file.path(output_dir, 'diablo_network.png')
  plot_diablo_network(diablo_model, comp_x, output_file = file6)
  generated_files$network <- file6
  
  # 7. Arrow Plot
  file7 <- file.path(output_dir, 'diablo_arrow.png')
  plot_diablo_arrow(diablo_model, y, comp_x, comp_y, file7)
  generated_files$arrow <- file7
  
  # 8. Circos Plot
  file8 <- file.path(output_dir, 'diablo_circos.png')
  plot_diablo_circos(diablo_model, comp_x, output_file = file8)
  generated_files$circos <- file8
  
  cat("="*60, "\n")
  cat("All plots generated successfully!\n\n")
  
  return(generated_files)
}

# Main execution (if called directly)
if (!interactive()) {
  # For command line execution
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) > 0) {
    cat("DIABLO Visualization Script\n")
    cat("This script should be sourced/called by Python via r_interface.py\n")
    cat("Individual plotting functions available:\n")
    cat("  - plot_diablo_samples()\n")
    cat("  - plot_diablo_indiv()\n")
    cat("  - plot_diablo_var()\n")
    cat("  - plot_diablo_loadings()\n")
    cat("  - plot_diablo_cim()\n")
    cat("  - plot_diablo_network()\n")
    cat("  - plot_diablo_arrow()\n")
    cat("  - plot_diablo_circos()\n")
    cat("  - generate_all_diablo_plots()\n")
  }
}
