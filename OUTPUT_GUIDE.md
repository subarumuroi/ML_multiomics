# Multi-Omics Analysis Output Guide

## Complete Guide to Understanding Analysis Results

This document explains each output file and visualization from the multi-omics analysis framework.

---

## Directory Structure

```
results/
├── analysis_summary.txt          # Text summary of all results
├── single_omics/                 # Individual omics layer analyses
│   ├── amino_acids/
│   ├── central_carbon/
│   ├── aromatics/
│   └── proteomics/
├── multi_omics/                  # Multi-omics integration results
│   ├── diablo_vips.csv          # Important features across all blocks
│   ├── diablo_correlations.csv  # How blocks correlate
│   ├── diablo_*.png             # Main visualizations
│   ├── concatenation_*.csv      # Baseline method results
│   └── diablo_output_[timestamp]/
│       └── diablo_output/       # Detailed R results
└── overview/                     # Summary visualizations
```

---

## 1. SINGLE OMICS ANALYSIS OUTPUTS

### Location: `results/single_omics/[omics_type]/`

Each omics layer gets its own folder with the same set of files:

#### **A. PCA (Principal Component Analysis)**

**pca_variance_explained.csv**
- Shows how much variance each principal component captures
- **Columns:**
  - `Component`: PC1, PC2, PC3, etc.
  - `Variance`: Percentage of variance explained by this component
  - `Cumulative_Variance`: Running total of variance explained
  
- **How to read:**
  ```
  If PC1 = 45% and PC2 = 25%, together they explain 70% of variation
  ```
  - Higher cumulative variance in fewer PCs = simpler data structure
  - Usually need 2-3 PCs to explain 70%+ of variance

**pca_scree.png**
- Line plot showing variance per component
- **Interpretation:**
  - Steep drop = few key components drive variation
  - Gradual slope = variation distributed across many components
  - "Elbow" point suggests optimal number of components

**pca_biplot.png**
- Shows:
  - **Points** = samples, colored by group (Green/Ripe/Overripe)
  - **Arrows** = features (metabolites/proteins), pointing toward high values
  - **Position** = sample's position in PC1 vs PC2 space
- **What to look for:**
  - Clear separation of groups = features differ between ripening stages
  - Overlapping groups = features don't strongly discriminate
  - Arrow direction shows which features drive each PC

**pca_scores.png**
- Shows just the samples (points) colored by group
- **Interpretation:** Group clustering indicates how well omics layer separates ripening stages

#### **B. PLS-DA (Partial Least Squares Discriminant Analysis)**

**plsda_vip_scores.csv**
- Ranks features by importance for classification
- **Columns:**
  - `Feature`: Metabolite, protein, or volatile name
  - `VIP`: Variable Importance for Projection (1-4 scale typical)
  - `Important`: TRUE if VIP > 1 (considered important)
- **VIP Interpretation:**
  - VIP > 1.0 = Important for discriminating groups
  - VIP < 0.5 = Not important, can be ignored
  - Higher VIP = stronger influence on class separation

**plsda_vip.png**
- Bar plot of top VIP scores
- **What it shows:**
  - Height = importance for discrimination
  - Top features have largest bars
  - Features above VIP = 1 line are highlighted

**plsda_cm.png** (Confusion Matrix)
- Shows Leave-One-Out Cross-Validation predictions
- **Rows** = true class, **Columns** = predicted class
- **Perfect classification:** All counts on diagonal (100% accuracy)
- **Misclassifications:** Off-diagonal entries indicate errors

---

## 2. MULTI-OMICS (DIABLO) OUTPUTS

### Location: `results/multi_omics/`

#### **A. diablo_vips.csv**

**The Most Important Output** - Complete feature importance across all omics blocks

- **Columns:**
  - `Feature`: Feature name (metabolite/protein ID)
  - `VIP`: Raw absolute loading value (0-1 scale)
  - `Percentile`: Rank within block as percentage (0-100)
  - `Block`: Which omics layer (amino_acids, aromatics, etc.)
  - `Important`: TRUE if Percentile ≥ 50 (top half of selected features)

- **How to interpret:**
  ```
  Example row:
  Feature: VAL (Valine)
  VIP: 0.72
  Percentile: 0.0
  Block: amino_acids
  Important: False
  
  → Valine is the lowest-loading feature in amino acids block
  → Not in top 50% of features for discriminating ripening
  ```

- **Percentile-Based Importance (NEW):**
  - Per official mixOmics methodology
  - Percentile = feature's rank among selected features in its block
  - Top 50% (Percentile ≥ 50) = "Important"
  - **Why not VIP > 1?** DIABLO can't compute true VIP scores; percentile is more appropriate

#### **B. diablo_correlations.csv**

- Shows how well different omics blocks correlate/agree
- **Values range 0-1:**
  - 0.9+ = Very high agreement (blocks agree strongly)
  - 0.5-0.7 = Moderate agreement
  - <0.5 = Weak agreement (blocks show different patterns)

- **Example:**
  ```
  If amino_acids ↔ proteomics correlation = 0.98
  → These blocks are highly correlated
  → They're highlighting similar biological signals
  ```

#### **C. Visualization Files**

**diablo_samples.png**
- Shows all samples plotted in DIABLO space (Component 1 vs 2)
- **What to look for:**
  - Color = ripening group (Green/Ripe/Overripe)
  - Clear separation = strong discrimination
  - Overlapping = groups are harder to distinguish
  - **Centroid** = average position of each group

**diablo_correlations.png**
- Heatmap showing block correlations
- **Color scale:**
  - Dark red = very high correlation (0.95-1.0)
  - Light/white = low correlation
- **Reading:** Find the block pairs and their correlation strength

**diablo_circos.png** (Circos Plot)
- **Most complex but informative visualization**
- **Structure:**
  - **Outer segments** = features in each block (different colors per block)
  - **Inner connections** = correlations between features across blocks
  - **Connection thickness/intensity** = correlation strength
  - **Color** = correlation direction (chocolate3 for positive, grey20 for negative)

- **How to interpret:**
  - **Thick lines** = strong cross-block feature correlations
  - **Many lines between blocks** = high block agreement
  - **Few lines** = blocks are independent
  - Look for features from different blocks that connect → these are biologically related

**diablo_output_[timestamp]/diablo_output/publication_plots/**

Contains detailed per-block visualizations:
- `01_DIABLO_samples.png` - Sample overlay on different blocks
- `02_DIABLO_indiv.png` - Individual sample scores with confidence ellipses
- `04_DIABLO_loadings.png` - Feature loadings per block
- `06_DIABLO_circos.png` - Feature correlations (same as main one)

#### **D. Detailed CSV Files in diablo_output/[timestamp]/diablo_output/**

**selected_features_[block_name].csv**
- Features selected by DIABLO for each omics block
- **Columns:**
  - `Feature`: Feature name
  - `VIP`: Absolute loading value (0-1 range)
  - `Percentile`: Rank percentile (0-100)
  - `Loading_Signed`: Signed loading (shows direction: + or -)

- **Example interpretation:**
  ```
  amino_acids block selected 3 features:
  VAL (Percentile=0, VIP=0.72)    → Lowest loading
  ASP (Percentile=50, VIP=0.69)   → Middle loading
  GLU (Percentile=100, VIP=0.06)  → Highest loading (despite low absolute value)
  
  → Percentile shows ranking, not absolute magnitude!
  ```

**loadings_[block_name].csv**
- All features (not just selected) with their loadings
- Useful for detailed feature-level interpretation
- Shows which features have positive vs negative contributions

**variates_[block_name].csv**
- Sample scores in each block's latent space
- Used for generating plots
- Each row = sample, each column = component

---

## 3. BASELINE COMPARISON OUTPUTS

### Location: `results/multi_omics/`

**concatenation_importance.csv**
- Feature importance from simple concatenation baseline
- For comparison with DIABLO (should show DIABLO is better)

**method_comparison.csv**
- Summary: Accuracy of DIABLO vs concatenation
- **Interpretation:**
  - If DIABLO > Concatenation = integration is beneficial
  - If similar = simple concatenation sufficient
  - If DIABLO < Concatenation = check for overfitting (likely with n<30 samples)

---

## 4. SUMMARY FILES

### Location: `results/`

**analysis_summary.txt**
- Text summary of entire analysis
- Shows:
  - PCA variance explained (first 2 components)
  - PLS-DA accuracy per omics layer
  - Top features per layer
  - DIABLO correlations between blocks
  - Method comparison (DIABLO vs concatenation)

**overview/sample_distribution.png**
- Shows sample sizes per group across omics layers
- Useful for spotting missing/different samples

**overview/performance_comparison.png**
- Bar plot comparing accuracy across methods and omics layers

---

## 5. KEY METRICS EXPLAINED

### Variable Importance Metrics

| Metric | Range | Method | When Used | Interpretation |
|--------|-------|--------|-----------|-----------------|
| **PLS-DA VIP** | 1-4 | VIP function | Single omics (PCA doesn't have this) | >1 = important; >0.5 = somewhat important |
| **DIABLO Loading** | 0-1 | selectVar() output | Multi-omics DIABLO | Relative magnitude within block |
| **DIABLO Percentile** | 0-100 | rank-based | Multi-omics DIABLO | ≥50 = important (top half); 0-50 = less important |

### Accuracy Metrics

- **Leave-One-Out (LOO) Validation:** Removes 1 sample, trains on rest, predicts held-out sample
  - Best for small datasets (n<30)
  - Unbiased but unstable with very small n
  
- **Confusion Matrix:** Shows which samples/groups are misclassified

---

## 6. PRACTICAL INTERPRETATION GUIDE

### For Your Banana Ripening Study:

**What the results tell you:**

1. **Single Omics (PLS-DA VIP scores)**
   - Which metabolites/proteins best discriminate ripening stages
   - Example: "Valine (VIP=3.2) is crucial for distinguishing Green from Ripe"

2. **Multi-Omics (DIABLO)**
   - **Block Correlations:** Do all omics layers agree on what matters?
     - High (0.9+): All layers show consistent signal → robust results
     - Low (<0.5): Layers show different patterns → need to explore why
   
   - **Circos Plot:** Which molecules work together across layers?
     - Connected features across blocks = biologically related
     - Isolated features = only important in one layer

3. **Feature Importance (Percentiles)**
   - Top 50% (Percentile ≥ 50) = use these for prediction/biomarkers
   - Bottom 50% = noise or redundant with selected features

### Red Flags

- ⚠️ **Very high accuracy (95%+) with n=9 samples** → Likely overfitting
  - Solution: Validate with independent cohort
  
- ⚠️ **Low block correlations (<0.5)** → Blocks disagreeing
  - Solution: Check data quality; different biology in different layers
  
- ⚠️ **Many features selected, few important (Percentile<50)** → Redundancy
  - Solution: DIABLO is selecting diverse features; not all equally valuable

---

## 7. HOW TO USE FOR YOUR REPORT

### Essential Information to Include

1. **Methods Section:**
   - "DIABLO selected X features per omics block"
   - "Features importance determined by percentile rank (top 50% marked as important)"
   - "Leave-One-Out cross-validation used for small sample size (n=9)"

2. **Results Section:**
   - Report top 3-5 most important features per omics
   - Show DIABLO accuracy vs concatenation baseline
   - Include top block correlation (e.g., 0.98)
   - Cite key figures: diablo_vips.csv, diablo_circos.png, method_comparison.csv

3. **Figures to Include:**
   - Single-omics: PLS-DA VIP plots (show discrimination ability)
   - Multi-omics: DIABLO sample plot (show group separation)
   - Multi-omics: Circos plot (show feature correlations)
   - Comparison: method_comparison.csv (show DIABLO benefit)

4. **Supplementary Data:**
   - Full diablo_vips.csv table (which features selected)
   - diablo_correlations.csv (block agreement)
   - selected_features_*.csv per block (detailed loadings)

---

## 8. COMMON QUESTIONS ANSWERED

**Q: Why do some features have very low VIP values (< 0.5)?**
A: They're still important for discrimination, just lower magnitude. DIABLO uses them to balance contribution across blocks.

**Q: What does "Percentile = 0" mean?**
A: The lowest-loading feature in that block among selected features. Still important because DIABLO chose it.

**Q: Why are DIABLO and single-omics VIPs different?**
A: Different metrics. PLS-DA VIP = discriminative power in one layer. DIABLO Percentile = ranking within block after integration.

**Q: Can I use DIABLO results to predict new samples?**
A: Yes! The selected features and loadings define the model. Use as coefficients for linear prediction.

**Q: Why only 2 components?**
A: With n=9, larger models overfit. 2 components capture main variation without overfitting.

---

## File Dependencies & Flow

```
Raw Data (badata-*.csv)
    ↓
Preprocessing → Normalization & Scaling
    ↓
    ├─→ Single Omics Analysis (PCA + PLS-DA)
    │    ├─ pca_*.png, pca_*.csv
    │    └─ plsda_*.png, plsda_*.csv
    │
    └─→ Multi-Omics Integration (DIABLO)
         ├─ Block selection in R (run_diablo.R)
         ├─ Feature importance calculation
         ├─ diablo_vips.csv (aggregated)
         ├─ Publication plots
         └─ diablo_output_*/
              └─ Detailed per-block CSVs
    
    Comparison & Summary
    ├─ method_comparison.csv
    └─ analysis_summary.txt
```

---

## Tips for Analysis Interpretation

1. **Start with diablo_vips.csv** - Get list of important features
2. **Look at diablo_samples.png** - Do groups separate visually?
3. **Check diablo_circos.png** - Which features connect across blocks?
4. **Review analysis_summary.txt** - Quick overview of all results
5. **Examine plsda_vip_scores.csv per layer** - Single-omics signals
6. **Compare block correlations** - Are layers consistent?

---

**All outputs are ready for inclusion in your report!**
Use this guide to select appropriate figures and interpret results accurately.
