# ML_multiomics Analysis Outputs Guide

## Output Directory Structure

When you run `examples/example_complete_analysis.py`, the following structure is created:

```
results/
├── single_omics/                    # Individual omics layer analyses
│   ├── amino_acids/
│   │   ├── pca_variance_explained.csv
│   │   ├── plsda_vip_scores.csv
│   │   ├── pca_scree.png
│   │   ├── pca_scores.png
│   │   ├── plsda_scores.png
│   │   ├── plsda_vip.png
│   │   └── confusion_matrix.png
│   │
│   ├── central_carbon/             # Same structure as above
│   ├── aromatics/                  # Same structure as above
│   └── proteomics/                 # Same structure as above
│
├── multi_omics/                     # Multi-omics integration results
│   ├── method_comparison.csv       ⭐ KEY: Compare 3 integration methods
│   ├── diablo_vips.csv            # DIABLO feature importance
│   ├── diablo_correlations.csv    # Block-to-block correlations
│   ├── concatenation_importance.csv
│   ├── ensemble_importance_amino_acids.csv
│   ├── ensemble_importance_central_carbon.csv
│   ├── ensemble_importance_aromatics.csv
│   ├── ensemble_importance_proteomics.csv
│   ├── diablo_samples.png         # Sample projections
│   ├── diablo_correlations.png    # Block correlation heatmap
│   ├── diablo_arrow.png           # Arrow plot
│   └── diablo_circos.png          # Circos plot
│
├── overview/                        # Summary visualizations
│   ├── sample_distribution.png     # Sample counts per group
│   ├── method_comparison.png       ⭐ KEY: Visual comparison of 3 methods
│   └── single_omics_performance.png
│
└── analysis_summary.txt            ⭐ KEY: Comprehensive text report

```

## Key Files to Check

### 1. **method_comparison.csv** (Multi-omics integration results)
Location: `results/multi_omics/method_comparison.csv`

Shows performance of 3 integration methods:
- DIABLO (Joint Integration)
- Concatenation (Early Fusion)
- Block-wise Ensemble (Late Fusion)

Columns: Method, Accuracy, Std, Type

### 2. **method_comparison.png** (Visual comparison)
Location: `results/overview/method_comparison.png`

Bar chart comparing the 3 methods side-by-side.

### 3. **analysis_summary.txt** (Comprehensive report)
Location: `results/analysis_summary.txt`

Contains:
- Single omics results (PCA variance, PLS-DA accuracy, top features)
- Multi-omics method comparison table
- DIABLO block correlations
- Ensemble and concatenation performance
- Top features per block (both DIABLO and Ensemble)

### 4. **Ensemble feature importance** (Per-block results)
Location: `results/multi_omics/ensemble_importance_*.csv`

One file per omics block showing:
- Feature name
- Importance score (from Random Forest)

### 5. **DIABLO visualizations**
Location: `results/multi_omics/diablo_*.png`

Four plots:
- `diablo_samples.png` - Sample projections on latent components
- `diablo_correlations.png` - Heatmap of block correlations
- `diablo_arrow.png` - Arrow plot showing sample trajectories
- `diablo_circos.png` - Circos plot of feature correlations

## Quick Start: What to Look At First

1. **Start here:** `results/analysis_summary.txt`
   - Read the method comparison table
   - Check which method performed best

2. **Visual overview:** `results/overview/method_comparison.png`
   - See all 3 methods compared visually

3. **Deep dive:** `results/multi_omics/method_comparison.csv`
   - Exact accuracy values for each method

4. **Feature importance:**
   - DIABLO: `results/multi_omics/diablo_vips.csv`
   - Ensemble: `results/multi_omics/ensemble_importance_*.csv` (one per block)
   - Concatenation: `results/multi_omics/concatenation_importance.csv`

5. **Visualizations:** `results/multi_omics/*.png` and `results/overview/*.png`
   - All plots are publication-ready (300 DPI)

## Understanding the Methods

### DIABLO (Joint Integration)
- Finds shared variation across all omics layers simultaneously
- Maximizes correlation between blocks
- Good for: Understanding relationships between omics layers

### Concatenation (Early Fusion)
- Simply stacks all features together
- Trains one model on combined data
- Good for: Baseline comparison, simple approach

### Block-wise Ensemble (Late Fusion)
- Trains separate model on each omics layer
- Combines predictions via voting
- Good for: Small sample sizes, interpretable block contributions

## Notes

- All cross-validation uses Leave-One-Out (LOO) due to small sample size
- High accuracy (>90%) is expected with small n - results are exploratory
- Focus on feature importance and biological interpretation
- Validate findings in independent cohort when possible
