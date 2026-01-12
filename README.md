# Multi-Omics Machine Learning Framework

A comprehensive Python framework for analyzing and integrating multi-omics datasets from banana ripening studies, specifically designed for metabolomics (amino acids, central carbon metabolism), proteomics, and volatile compounds (aromatics) data.

## Features

### Single Omics Analysis
- **PCA (Principal Component Analysis)**: Exploratory visualization and variance analysis
- **PLS-DA (Partial Least Squares Discriminant Analysis)**: Supervised classification with VIP scores
- **Omics-specific preprocessing**: Tailored pipelines for metabolomics, proteomics, and volatiles

### Multi-Omics Integration
- **DIABLO**: Multi-block integration maximizing correlation between omics layers
- **Concatenation baseline**: Simple concatenation approach for comparison
- **Weighted integration**: Balance contributions from different omics layers

### Validation & Utilities
- **Cross-validation**: Leave-One-Out and K-Fold strategies
- **Permutation testing**: Statistical validation
- **Feature stability**: Assess robustness of feature selection
- **Publication-quality plots**: Comprehensive visualization suite

## Installation

```bash
# Create conda environment (recommended)
conda create -n multi_omics python=3.9
conda activate multi_omics

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## Package Structure

```
ml_multiomics/
├── preprocessing/
│   ├── base_preprocessor.py           # Abstract base class
│   ├── metabolomics_preprocessor.py   # Metabolomics, volatiles, proteomics
│   ├── integrator.py                  # Multi-omics data integration
│   └── __init__.py
│
├── methods/
│   ├── single_omics/
│   │   ├── pca.py                     # PCA analysis
│   │   ├── plsda.py                   # PLS-DA classification
│   │   └── __init__.py
│   │
│   └── multi_omics/
│       ├── diablo.py                  # DIABLO integration
│       ├── concatenation_baseline.py  # Simple baseline methods
│       └── __init__.py
│
├── workflows/
│   ├── single_omics_workflow.py       # Single omics pipeline
│   ├── multi_omics_workflow.py        # Integration pipeline
│   └── __init__.py
│
├── utils/
│   ├── validation.py                  # Cross-validation & testing
│   ├── visualization.py               # Plotting utilities
│   └── __init__.py
│
├── examples/
│   └── example_complete_analysis.py   # Full analysis example
│
├── README.md
└── requirements.txt
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
from workflows.single_omics_workflow import SingleOmicsWorkflow
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
from workflows.multi_omics_workflow import MultiOmicsWorkflow
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
    'fill_value': 0,          # Fill remaining NaNs with 0
    'transform': 'log',       # Log transformation
    'scaling': 'pareto',      # Pareto scaling
}

workflow = SingleOmicsWorkflow(
    omics_type='metabolomics',
    config=custom_config
)
```

### Default Configurations

**Metabolomics:**
- Drop threshold: 50%
- Imputation: Group-wise median
- Transform: Log
- Scaling: Pareto

**Volatiles:**
- Drop threshold: 60% (more lenient for sparse data)
- Imputation: Conservative group-wise median
- Transform: Log
- Scaling: Standard

**Proteomics:**
- Drop threshold: 30% (stricter)
- Imputation: Minimal (assumes pre-imputed)
- Transform: Log2
- Scaling: Standard

## Output Files

### Single Omics Analysis

```
results/amino_acids/
├── pca_variance_explained.csv      # Variance per component
├── pca_scree.png                   # Scree plot
├── pca_scores.png                  # Scores plot
├── pca_loadings.png                # Loadings plot
├── pca_biplot.png                  # Biplot
├── plsda_vip_scores.csv            # VIP scores
├── plsda_scores.png                # PLS-DA scores
├── plsda_vip.png                   # VIP plot
├── plsda_loadings.png              # Loadings plot
└── plsda_cm.png                    # Confusion matrix
```

### Multi-Omics Integration

```
results/multi_omics/
├── diablo_correlations.csv         # Block correlations
├── diablo_vips.csv                 # Important features per block
├── diablo_samples.png              # Sample projection
├── diablo_correlations.png         # Correlation heatmap
├── diablo_arrow.png                # Block agreement plot
├── diablo_circos.png               # Circos-style plot
├── concatenation_importance.csv    # Feature importance
└── method_comparison.csv           # Performance comparison
```

## Advanced Usage

### Custom Validation

```python
from utils.validation import CrossValidator, PermutationTest

# Leave-One-Out validation
validator = CrossValidator(strategy='loo')
cv_results = validator.validate_model(model, X, y)

# Permutation test
perm_test = PermutationTest(n_permutations=1000)
test_results = perm_test.test_model(model, X, y)
print(f"P-value: {test_results['p_value']:.4f}")
```

### Custom Visualizations

```python
from utils.visualization import OmicsPlotter

plotter = OmicsPlotter()

# Confidence ellipses
fig, ax = plotter.plot_confidence_ellipses(
    X=scores, y=labels, 
    comp_x=0, comp_y=1, 
    confidence=0.95
)

# Volcano plot
fig, ax = plotter.plot_volcano(
    fold_changes=fc, 
    p_values=pvals,
    feature_names=names
)
```

## Citation

If you use this framework in your research, please cite:

```
[Your paper citation here]
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

Apache License 2.0 - see LICENSE file for details.

## Contact

For questions or issues, please contact [your contact information].

## Acknowledgments

This framework implements methods from:
- DIABLO: Singh et al. (2019) DIABLO: an integrative approach for identifying key molecular drivers from multi-omics assays
- PLS-DA: Wold et al. (1983) The multivariate calibration problem in chemistry