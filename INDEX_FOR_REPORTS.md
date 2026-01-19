# Multi-Omics Analysis - Complete Documentation

## For Report Generation

Your analysis is complete and ready for report writing. Use these guides to understand and present your results.

---

## 📋 Start Here

### If you have **5 minutes** - Read:
- [RESULTS_QUICK_REFERENCE.md](RESULTS_QUICK_REFERENCE.md) - Your actual numbers and what they mean

### If you have **30 minutes** - Read:
- [RESULTS_QUICK_REFERENCE.md](RESULTS_QUICK_REFERENCE.md) - Your results
- [OUTPUT_GUIDE.md](OUTPUT_GUIDE.md) - Sections 1-3 (Understanding outputs)

### If you need **complete understanding** - Read:
- All sections of [OUTPUT_GUIDE.md](OUTPUT_GUIDE.md)
- [RESULTS_QUICK_REFERENCE.md](RESULTS_QUICK_REFERENCE.md)
- Review actual CSV files in `results/`

---

## 📊 Your Analysis Results Location

```
results/
├── analysis_summary.txt                    ← START HERE
├── multi_omics/
│   ├── diablo_vips.csv                    ← Main results (all selected features)
│   ├── diablo_correlations.csv            ← Block agreement metrics
│   ├── diablo_samples.png                 ← Main visualization
│   ├── diablo_circos.png                  ← Feature correlations
│   └── diablo_output_20260119_101609/
│       └── diablo_output/
│           ├── selected_features_*.csv    ← Features per block
│           ├── publication_plots/         ← High-quality figures
│           └── loadings_*.csv             ← Detailed loadings
├── single_omics/
│   ├── amino_acids/
│   │   ├── plsda_vip_scores.csv          ← Single omics importance
│   │   └── plsda_*.png                    ← Single omics plots
│   ├── central_carbon/
│   ├── aromatics/
│   └── proteomics/
└── overview/
    ├── sample_distribution.png
    └── performance_comparison.png
```

---

## 🔍 What Each File Contains

### CSV Files (Data Tables)

| File | Location | What It Shows | Use For |
|------|----------|---------------|---------|
| **diablo_vips.csv** | multi_omics/ | All selected features + percentile ranks | Main results table, feature lists |
| **diablo_correlations.csv** | multi_omics/ | How blocks correlate (0-1) | Methods, figure caption |
| **plsda_vip_scores.csv** | single_omics/[layer]/ | VIP scores for single omics | Layer-specific methods, supplementary |
| **selected_features_*.csv** | diablo_output/.../diablo_output/ | Features per block + detailed loadings | Supplementary detailed table |
| **analysis_summary.txt** | results/ | Text overview of all results | Quick reference, report outline |

### Visualization Files (PNG)

| File | Location | Type | What It Shows |
|------|----------|------|---------------|
| **diablo_samples.png** | multi_omics/ | Scatter plot | Sample separation by group (key figure) |
| **diablo_circos.png** | multi_omics/ | Circos plot | Feature correlations across blocks (key figure) |
| **diablo_correlations.png** | multi_omics/ | Heatmap | Block correlation strength |
| **plsda_vip.png** | single_omics/[layer]/ | Bar plot | Top features per layer (supplementary) |
| **pca_biplot.png** | single_omics/[layer]/ | Biplot | Feature directions + samples (supplementary) |

---

## 📝 How To Use These Files In Your Report

### For Methods Section
- Reference [OUTPUT_GUIDE.md](OUTPUT_GUIDE.md) Section 5-6
- Use percentile-based importance explanation
- Cite mixOmics paper (Rohart et al. 2017)
- Mention Leave-One-Out validation for small n

### For Results Section
- Start with [RESULTS_QUICK_REFERENCE.md](RESULTS_QUICK_REFERENCE.md) "Key Numbers to Report"
- Include: accuracy, correlations, feature counts
- Reference: diablo_vips.csv (all selected features)
- Main figures: diablo_samples.png, diablo_circos.png

### For Discussion
- Use block correlations to explain layer agreement
- Interpret why aromatics are independent
- Address sample size limitations
- Discuss biological plausibility

### For Supplementary Materials
- Include: diablo_correlations.csv (matrix format)
- Include: selected_features_*.csv (per-layer detail)
- Include: plsda_vip_scores.csv (single omics detail)
- Include: publication_plots/ (high-res figures)

---

## ✅ Checklist: Before Writing Report

### Data Understanding
- [ ] Read RESULTS_QUICK_REFERENCE.md (your specific numbers)
- [ ] Understand VIP vs Percentile difference
- [ ] Know your block correlations (0.33-0.98)
- [ ] Recognize sample size limitation (n=9)

### Files Prepared
- [ ] Located all CSVs in results/
- [ ] Located all PNG figures
- [ ] Downloaded high-res versions from publication_plots/
- [ ] Reviewed analysis_summary.txt

### Key Metrics Identified
- [ ] Top 3-5 features per omics layer
- [ ] Block correlation matrix
- [ ] Classification accuracy (100% ± warning)
- [ ] Feature selection counts (3-12 per layer)

### Figures Selected
- [ ] diablo_samples.png (main figure)
- [ ] diablo_circos.png (correlation figure)
- [ ] Single omics VIP plots (supplementary)
- [ ] Correlations heatmap (supplementary)

---

## 📚 Reference Guides

### Understanding Concepts

**VIP vs Percentile:**
- **VIP (Single Omics):** Variable Importance for Projection, 1-4 scale, VIP > 1 = important
- **Percentile (Multi-Omics):** Rank within block, 0-100 scale, ≥50 = important
- **Why different:** DIABLO can't compute true VIP; percentile ranking is scientifically appropriate

**Block Correlations:**
- **0.9+:** Very high agreement (robust signal)
- **0.5-0.7:** Moderate agreement (consistent but independent)
- **<0.5:** Weak agreement (different biology in layers)

**Accuracy with n=9:**
- ✓ 100% in cross-validation is expected with small n
- ⚠️ Likely overfitting to this specific dataset
- ⚠️ Needs validation on independent cohort (n>30)

---

## 🎯 Quick Reference: Numbers to Report

From your analysis:

```
CLASSIFICATION PERFORMANCE
- Single Omics Accuracy (all layers): 100%
- DIABLO Accuracy: 100%
- Validation: Leave-One-Out cross-validation

FEATURES SELECTED
- Amino acids: 3 features (9 important)
- Central carbon: 3 features (14 important)
- Aromatics: 5 features (27 important)
- Proteomics: 12 features (~1788 important)
- Total selected: ~23 features

BLOCK AGREEMENT (Correlations)
- Amino acids ↔ Central carbon: 0.946
- Amino acids ↔ Proteomics: 0.985
- Central carbon ↔ Proteomics: 0.958
- Aromatics ↔ Others: 0.33-0.46

VARIANCE EXPLAINED (PCA, First 2 Components)
- Amino acids: 97.8%
- Central carbon: 93.6%
- Aromatics: 96.9%
- Proteomics: 59.9%

IMPORTANT FEATURES (VIP > 1 in single omics)
- Amino acids: 9/21 (43%)
- Central carbon: 14/33 (42%)
- Aromatics: 27/99 (27%)
- Proteomics: 1788/5091 (35%)
```

---

## ⚠️ Important Caveats to Address

### Sample Size
- n=9 is very small for machine learning
- Perfect accuracy is likely due to overfitting
- Solution: Validate on independent n>30 cohort
- This study is hypothesis-generating

### Aromatics Independence
- Volatile compounds show lower correlation (0.33-0.46)
- Different from metabolite/protein patterns
- Possible reason: Different chemistry/measurement
- Conclusion: Aromatics show unique ripening signature

### Feature Explosion (Proteomics)
- 1788 important features out of 5091
- Too many for biomarker panel
- Recommendation: Further feature reduction or validation

---

## 🚀 Next Steps

1. **This Week:** Read the guides, understand your results
2. **Next Week:** Draft Methods and Results sections
3. **Week 3:** Prepare figures, write Discussion
4. **Week 4:** Address reviewer questions about sample size/overfitting
5. **Validation:** Plan follow-up study with n>30

---

## 📞 Questions?

Refer to [OUTPUT_GUIDE.md](OUTPUT_GUIDE.md) Section 8 "Common Questions Answered"

Or check specific sections:
- Understanding visualizations → OUTPUT_GUIDE.md Section 6
- Metric interpretation → RESULTS_QUICK_REFERENCE.md "How to Read Each Visualization"
- Report writing → RESULTS_QUICK_REFERENCE.md "Data Interpretation Tips for Your Report"

---

## 📄 Analysis Summary (From results/analysis_summary.txt)

### Single Omics Results
```
✓ Amino acids:      9 important features (100% accuracy)
✓ Central carbon:  14 important features (100% accuracy)
✓ Aromatics:       27 important features (100% accuracy)
✓ Proteomics:    1788 important features (100% accuracy)
```

### Multi-Omics Integration (DIABLO)
```
✓ Selected 3-12 features per block (23 total)
✓ Top 50% percentile: ~12 highly important features
✓ Achieved 100% classification accuracy
✓ Block correlations: 0.946-0.985 (except aromatics: 0.33-0.46)
```

### Interpretation
```
→ All omics layers discriminate ripening stages individually
→ DIABLO successfully integrates all layers
→ Metabolites and proteins strongly coordinate (r>0.94)
→ Aromatics show independent but complementary signal
→ Results suggest coordinated ripening across layers
```

---

**Your analysis is production-ready for report writing! Use these guides to explain your findings clearly and accurately.**
