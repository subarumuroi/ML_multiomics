# Multi-Omics Machine Learning Framework

A comprehensive, **reusable Python framework** for analyzing and integrating multi-omics datasets. While demonstrated here with banana ripening data, the framework is designed to work with **any multi-omics study**.

## Why This Framework?

- **Plug-and-play with your data**: Bring any combination of omics layers (metabolomics, proteomics, transcriptomics, lipidomics, volatiles, etc.)
- **Configurable preprocessing**: Each omics layer can have its own transformation, scaling, and imputation strategy
- **Three integration approaches**: Compare DIABLO (joint), concatenation (early fusion), and ensemble (late fusion) methods
- **Consensus feature discovery**: Identify robust biomarkers that appear across multiple methods
- **Small sample ready**: Built-in LOO cross-validation and permutation testing for n<30 studies
- **Publication-quality outputs**: Automated R/mixOmics visualizations (circos plots, loadings, etc.)

## Features

### Single Omics Analysis
- **PCA (Principal Component Analysis)**: Exploratory visualization and variance analysis
- **PLS-DA (Partial Least Squares Discriminant Analysis)**: Supervised classification with VIP scores
- **Omics-specific preprocessing**: Tailored pipelines for metabolomics, proteomics, and volatiles

### Multi-Omics Integration
- **DIABLO**: Multi-block integration maximizing correlation between omics layers (with LOO CV tuning)
- **Concatenation baseline**: Simple early-fusion approach for comparison
- **Block-wise Ensemble**: Late-fusion ensemble with independent block classifiers

### Validation & Utilities
- **Cross-validation**: Leave-One-Out CV built into all methods
- **Permutation testing**: Statistical validation that performance > random chance
- **Consensus feature identification**: Venn diagram showing features important across all 3 methods
- **Method comparison**: Automated performance comparison across integration approaches
- **Overview visualizations**: Sample distribution and method comparison plots
- **Publication-quality plots**: Comprehensive visualization suite using R/mixomics

## Scientific Methods & Implementation

### Variable Importance Calculation

#### Single Omics (PLS-DA)
- **VIP Scores**: Calculated using mixOmics native `vip()` function
- **Interpretation**: Features with VIP > 1.0 are considered important
- **Scale**: Typically ranges 1-4+ depending on feature discriminative power

#### Multi-Omics Integration (DIABLO)
- **Loadings**: DIABLO produces normalized loadings on [-1, 1] scale
- **Importance Metric**: Features ranked by percentile (0-100) based on absolute loading magnitude
- **Importance Threshold**: Features in top 50% percentile (Percentile ≥ 50) flagged as important
- **Scientific Rationale**: Per Rohart et al. (2017) mixOmics methodology:
  - DIABLO doesn't support VIP scores (only PLS/SPLSDA have vip() function)
  - Uses raw absolute loadings on native scale, ranked by percentile within each block
  - Percentile-based importance is more robust than arbitrary thresholds
  - Loadings naturally range 0-1; percentile ranking enables fair comparison across blocks

### DIABLO Integration
- **R/mixOmics**: Direct integration with mixOmics package for statistical rigor
- **Visualization**: Publication-quality plots generated in R/mixOmics:
  - `plotDiablo()`: Sample scores overlaid on all block coordinates
  - `plotIndiv()`: Individual sample scores with confidence ellipses per block
  - `plotLoadings()`: Feature loading weights (importance) per block
  - `circosPlot()`: Feature correlations across blocks with correlation strength indicators
  - `plotArrow()`: Block agreement visualization (requires ≥10 samples; skipped for small cohorts)
- **Block Correlation**: Measures how well different omics blocks agree on sample discrimination

## Installation

### Prerequisites

**R and R Packages Required:**

DIABLO integration uses R's mixOmics package. Install R dependencies first:
```bash
# Check R is installed
Rscript --version

# Quick install - run the provided script
Rscript scripts/install_r_deps.R

# Or install manually
Rscript -e "install.packages('BiocManager')"
Rscript -e "BiocManager::install('mixOmics')"
Rscript -e "install.packages('jsonlite')"
```

**Python Installation:**
```bash
# Create conda environment (recommended)
conda create -n ml_multiomics python=3.9
conda activate ml_multiomics

# Clone the repository
git clone https://github.com/subarumuroi/ml_multiomics.git
cd ml_multiomics

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e .[dev]
```

## Package Structure

```
ml_multiomics/
├── src/
│   └── ml_multiomics/
│       ├── preprocessing/
│       │   ├── base_preprocessor.py
│       │   ├── omics_preprocessor.py
│       │   ├── integrator.py
│       │   └── __init__.py
│       │
│       ├── methods/
│       │   ├── single_omics/
│       │   │   ├── pca.py
│       │   │   ├── plsda.py
│       │   │   └── __init__.py
│       │   │
│       │   └── multi_omics/
│       │       ├── diablo.py
│       │       ├── concatenation_baseline.py│       │       ├── ensemble.py│       │       └── __init__.py
│       │
│       ├── workflows/
│       │   ├── single_omics_workflow.py
│       │   ├── multi_omics_workflow.py
│       │   └── __init__.py
│       │
│       ├── utils/
│       │   ├── validation.py
│       │   ├── visualization.py
│       │   ├── r_interface.py
│       │   └── __init__.py
│       │
│       └── __init__.py
│
├── scripts/
│   ├── run_diablo.R
│   └── install_r_deps.R 
│
├── examples/
│   ├── example_complete_analysis.py     # Main analysis script
│   ├── example_with_permutation_tests.py
│   └── add_permutation_tests.py
│
├── tests/
│
├── setup.py
└── README.md
```

## Data Files

The framework expects the following data files:
- `badata-amino-acids.csv` - Amino acids metabolomics (21 amino acids)
- `badata-metabolomics.csv` - Central carbon metabolism (33 metabolites)
- `badata-aromatics.csv` - Volatile compounds/aromatics (109 features)
- `badata-proteomics-imputed.csv` - Proteomics data (5,975 proteins)

## Quick Start

### Single Omics Analysis

```python
from ml_multiomics.workflows import SingleOmicsWorkflow
import pandas as pd

# Load your data
df = pd.read_csv("data/badata-amino-acids.csv")

# Initialize workflow
workflow = SingleOmicsWorkflow(omics_type='metabolomics')

# Run complete analysis
results = workflow.run_full_analysis(
    df=df,
    group_col='Groups',
    n_pca_components=5,
    n_plsda_components=2
)

# Save results
workflow.save_results("results/amino_acids")
workflow.display_results()
```

### Multi-Omics Integration

```python
from ml_multiomics.workflows import MultiOmicsWorkflow
import pandas as pd

# Load all omics layers
data_dict = {
    'amino_acids': pd.read_csv("data/badata-amino-acids.csv"),
    'central_carbon': pd.read_csv("data/badata-metabolomics.csv"),
    'aromatics': pd.read_csv("data/badata-aromatics.csv"),
    'proteomics': pd.read_csv("data/badata-proteomics-imputed.csv")
}

omics_types = {
    'amino_acids': 'metabolomics',
    'central_carbon': 'metabolomics',
    'aromatics': 'volatiles',
    'proteomics': 'proteomics'
}

# Initialize and run workflow
workflow = MultiOmicsWorkflow()
results = workflow.run_full_integration(
    data_dict=data_dict,
    omics_types=omics_types,
    group_col='Groups',
    n_components=2
)

# Save results
workflow.save_results("results/multi_omics")
```

## Using Your Own Data

The framework is **not limited to the included banana dataset**. To use with your own multi-omics data:

### 1. Prepare your CSVs
Each omics layer needs a CSV with a `Groups` column (your class labels) and feature columns:

```csv
Groups,Feature_1,Feature_2,...
ClassA,0.523,1.234,...
ClassA,0.612,1.189,...
ClassB,1.234,2.456,...
```

### 2. Define your blocks and omics types
```python
# Your data - any number of omics layers
data_dict = {
    'transcriptomics': pd.read_csv("your_rnaseq.csv"),
    'metabolomics': pd.read_csv("your_metabolites.csv"),
    'lipidomics': pd.read_csv("your_lipids.csv"),
}

# Map each block to a preprocessing type
# Available types: 'metabolomics', 'proteomics', 'volatiles'
omics_types = {
    'transcriptomics': 'proteomics',   # Uses log2, stricter filtering
    'metabolomics': 'metabolomics',    # Uses log, pareto scaling
    'lipidomics': 'metabolomics',      # Similar to metabolomics
}
```

### 3. Run the workflow
```python
workflow = MultiOmicsWorkflow()
results = workflow.run_full_integration(
    data_dict=data_dict,
    omics_types=omics_types,
    group_col='Groups',  # Or whatever your class column is called
    n_components=2
)
workflow.save_results("results/my_study")
```

### Supported Omics Types (Preprocessing Presets)
| Type | Best For | Transform | Scaling | Notes |
|------|----------|-----------|---------|-------|
| `metabolomics` | Metabolomics, lipidomics | log | Pareto | Half-min imputation |
| `proteomics` | Proteomics, transcriptomics | log2 | Pareto | Stricter filtering |
| `volatiles` | GC-MS volatiles, aromatics | log | Pareto | TSN normalization |

All preprocessing is **fully configurable** - see [Preprocessing Configurations](#preprocessing-configurations) below.

## Data Format

Your CSV files should have this structure:

```csv
Groups,Feature_1,Feature_2,Feature_3,...
Green,0.523,1.234,0.891,...
Green,0.612,1.189,0.923,...
Ripe,1.234,2.456,1.567,...
Ripe,1.198,2.389,1.534,...
Overripe,2.345,3.678,2.234,...
```

- **Groups column**: Contains class labels (e.g., Green, Ripe, Overripe)
- **Feature columns**: Numeric values for each feature
- Missing values are handled automatically

## Preprocessing Configurations

Each omics type has default configurations that can be customized:

```python
# Custom configuration for metabolomics
custom_config = {
    'drop_threshold': 0.5,    # Drop features with >50% missing
    'imputation': 'half_min', # 'half_min' or 'zero' or 'group_median'
    'transform': 'log',       # 'log', 'log2', or None
    'scaling': 'pareto',      # 'pareto', 'standard', or 'minmax'
}

workflow = SingleOmicsWorkflow(
    omics_type='metabolomics',
    config=custom_config
)
```

### Default Configurations

**Metabolomics:**
- Drop threshold: 50%
- Imputation: Half-minimum (better for log transform than zero-fill)
- Transform: Log
- Scaling: Pareto
- Handles negative values: Auto-shifts to positive range

**Volatiles:**
- Drop threshold: 60% (more lenient for sparse data)
- **Total Sum Normalization (TSN)**: Applied to reduce sample-to-sample variability
- Imputation: Half-minimum
- Transform: Log
- Scaling: Pareto

**Proteomics:**
- Drop threshold: 30% (stricter)
- Imputation: Group-wise median, then half-minimum fallback
- Transform: Log2
- Scaling: Pareto

## Output Files

Running `examples/example_complete_analysis.py` generates the following structure:

```
results/
├── analysis_summary.txt            # Comprehensive text report
│
├── single_omics/                   # Individual omics layer analyses
│   ├── amino_acids/
│   │   ├── pca_variance_explained.csv
│   │   ├── plsda_vip_scores.csv
│   │   ├── pca_scree.png
│   │   ├── pca_scores.png
│   │   ├── plsda_scores.png
│   │   ├── plsda_vip.png
│   │   └── confusion_matrix.png
│   ├── central_carbon/             # Same structure
│   ├── aromatics/                  # Same structure
│   └── proteomics/                 # Same structure
│
├── multi_omics/                    # Multi-omics integration
│   ├── method_comparison.csv       # ⭐ Compare 3 integration methods
│   ├── consensus_features_by_block.csv  # ⭐ Features consistent across methods (per block)
│   ├── feature_venn.png            # ⭐ Venn diagram of feature overlap
│   ├── diablo_vips.csv             # DIABLO feature importance
│   ├── diablo_correlations.csv     # Block correlations
│   ├── diablo_correlations.png     # Correlation heatmap
│   ├── diablo_samples.png          # Sample projections
│   ├── diablo_circos.png           # Feature correlations
│   ├── diablo_arrow.png            # Block agreement (if n≥10)
│   ├── concatenation_importance.csv
│   ├── ensemble_importance_*.csv   # Per-block importance (4 files)
│   ├── permutation_tests.csv       # Statistical validation (if run)
│   └── diablo_output_[timestamp]/  # R mixOmics outputs
│
└── overview/                       # Summary visualizations
    ├── sample_distribution.png
    ├── method_comparison.png       # ⭐ Visual comparison
    ├── single_omics_performance.png
    └── top_features_summary.csv    # ⭐ Top features from all methods
```

**Key Files:**
- `method_comparison.csv` - Performance of 3 integration methods
- `analysis_summary.txt` - Complete text report with all results
- `consensus_features_by_block.csv` - Features important across methods (per omics block)
- `top_features_summary.csv` - Side-by-side comparison of top features from each method
- `feature_venn.png` - Visual overlap between methods

## Statistical Validation: Permutation Testing

For small sample sizes (n<30), cross-validation alone may not detect overfitting. Permutation testing provides additional validation by testing whether model performance is significantly better than chance.

### Quick Start

Add permutation testing to your analysis:

```python
# After running integration
multi_block = workflow.integrator.get_multi_block()
perm_results = workflow.run_permutation_tests(
    multi_block=multi_block,
    n_permutations=1000,  # 1000 for POC, 5000+ for publication
    random_state=42
)

# Results automatically saved with other outputs
workflow.save_results("results/multi_omics")
```

Or use the standalone script:

```bash
# Quick validation (1000 permutations)
python examples/add_permutation_tests.py

# Publication quality (5000 permutations)
python examples/add_permutation_tests.py --n-permutations 5000
```

### Understanding Results

- **P-value < 0.05**: Model significantly better than random ✓
- **P-value ≥ 0.05**: Cannot distinguish from random chance ✗
- **With n<15**: Results are exploratory/POC only ⚠️

### Consensus Features & Venn Diagram

Identify features that are consistently important across all three integration methods:

```python
# Automatically included in run_full_integration(), or run separately:
feature_sets = workflow.identify_consensus_features(top_n=20, plot=True)

# Returns: {'concatenation': set, 'ensemble': set, 'diablo': set, 'consensus': set}
# Saves: feature_venn.png and consensus_features.csv
```

The Venn diagram visually shows overlap between methods, helping identify robust biomarker candidates that are consistently selected regardless of integration approach.

## Advanced Usage

### Comparing Integration Methods

```python
from ml_multiomics.workflows import MultiOmicsWorkflow

# Run all three integration methods with cross-validation
workflow = MultiOmicsWorkflow()

# Preprocess and integrate data
workflow.preprocess_all_layers(data_dict, omics_types, group_col='Groups')
multi_block = workflow.integrator.create_multiblock_data(align=True)

# Run DIABLO (includes LOO CV during tuning)
workflow.run_diablo(multi_block, n_components=2)

# Run concatenation baseline (with LOO CV)
workflow.run_concatenation_baseline(multi_block, cv=True)

# Run block-wise ensemble (with LOO CV)
workflow.run_ensemble(multi_block, cv=True)

# Compare methods
comparison_df = workflow.compare_methods()
print(comparison_df)
```

### Overview Visualizations

```python
from ml_multiomics.utils import OmicsPlotter

plotter = OmicsPlotter()

# Sample distribution across omics layers
data_dict = {name: pd.read_csv(path) for name, path in data_paths.items()}
fig, _ = plotter.plot_sample_overview(data_dict, group_col='Groups')

# Performance comparison
fig, _ = plotter.plot_performance_comparison(performance_dict, metric='accuracy')
```

## Citation

If you use this framework in your research, please cite:

```
[TBA]
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

Apache License 2.0 - see LICENSE file for details.

## Contact

For questions or issues, please contact [k.muroi@uq.edu.au].

## Acknowledgments

This framework implements methods from:
- DIABLO: Singh et al. (2019) DIABLO: an integrative approach for identifying key molecular drivers from multi-omics assays
- PLS-DA: Wold et al. (1983) The multivariate calibration problem in chemistry

## Recent Implementation Notes (January 2026)

### Preprocessing Improvements (Latest)

**Half-Minimum Imputation (Commit TBD)**
- **Previous**: Zero-fill for missing values → log(0+ε) = -23 (artificial floor)
- **New**: Half-minimum imputation → fills with half of smallest positive value
- **Benefit**: Keeps imputed values in realistic range for log transform

**Total Sum Normalization for Volatiles**
- **Issue**: Aromatics data had 73% CV in sample totals (injection variability)
- **Solution**: TSN normalizes each sample to median total before log transform
- **Result**: Removes technical variability, features now ranked by relative abundance

**Proteomics Fallback Imputation**
- Group-wise median imputation with half-minimum fallback
- Handles proteins missing in entire groups (where median would be NaN)

### Scientific Methodology Corrections

**Proper DIABLO Importance Calculation (Commits 904b5c7, e708e99)**
- **Previous Error**: Scaled DIABLO loadings to [0, 1.5] range to match VIP > 1 threshold
- **Issue**: This was data-fitting; not aligned with official mixOmics protocol
- **Correct Method** (per Rohart et al. 2017):
  - Use raw absolute loadings on native scale (0-1 range)
  - Flag importance by percentile rank within each block (0-100 scale)
  - Top 50% (Percentile ≥ 50) marked as important
  - Do NOT apply arbitrary VIP > 1 threshold (PLS-DA specific)
- **Result**: DIABLO methodology now scientifically rigorous and reproducible

**DIABLO Visualization Improvements (Commit e708e99)**
- **circosPlot()**: Updated with proper mixOmics parameters:
  - cutoff = 0.7 (higher threshold for clearer feature correlations)
  - line = TRUE (connects correlated features across blocks)
  - Distinct block colors and correlation line colors
  - size.labels = 1.5 for publication quality
- **plotArrow()**: Added intelligent sample size checking:
  - Requires ≥10 samples for meaningful visualization
  - Automatically skipped for small cohorts (n < 10)
  - Documentation explains purpose: shows centroid (tail) vs block-specific positions (tips)

### Bug Fixes & Prior Improvements

**Data Format Handling (Commit 698c6a6)**
- Fixed: R interface now handles both pandas DataFrame and numpy array inputs
- Solution: Added type checking and automatic conversion in `r_interface.py`

**R Visualization Integration (Commit 698c6a6)**
- Implemented: Direct R/mixOmics visualization generation
- Output: Publication-quality plots (plotDiablo, plotIndiv, plotLoadings, circosPlot)
- Location: `results/multi_omics/diablo_output_[timestamp]/diablo_output/publication_plots/`

**Code Cleanup (Commit 71b22c4)**
- Removed: Obsolete matplotlib visualization approximations
- Kept: Active R visualization generation for publication-ready output
