"""
ml_multiomics.analysis
======================
Standard omics analyses (differential expression, enrichment) reimplemented in
pure Python to match the lab's IdeaBio.R conventions, verified numerically
against the actual R functions (tests/crosscheck/).

  differential.compute_volcano   pairwise Welch t-test + fold change + FDR
                                  (matches IdeaBio.R compute_volcano)
  differential.anova_tukey        one-way ANOVA + Tukey HSD per feature
                                  (matches IdeaBio.R compute_anova_tukey)
  enrichment.ora                  hypergeometric over-representation analysis
                                  (matches clusterProfiler::enricher's statistic)

IMPORTANT: differential expression operates on RAW / linear abundances (it
computes its own fold change and applies log10 internally for the test) — NOT on
the z-scored preprocessed matrix. For repeated-measures designs (timepoints per
bioreactor) aggregate to one row per independent unit first; the lab's DE, like
this port, otherwise treats every sample as independent.
"""

from .differential import compute_volcano, anova_tukey
from .enrichment import ora
from .integration import detect_oversized_blocks, reduce_block, balance_blocks
from .systematic import (
    systematic_assessment, integration_assessment, integration_blocks, FoldPipeline,
)
from .pipeline import OmicsPipeline
from .standard import (
    qc_summary, aggregate_to_units, differential_expression,
    over_representation, gsea_ranked_list, univariate_association, standard_to_ml_bridge,
)
from . import figures
from .native_plots import diablo_plots, wgcna_plots

__all__ = [
    "compute_volcano", "anova_tukey", "ora",
    "detect_oversized_blocks", "reduce_block", "balance_blocks",
    "systematic_assessment", "integration_assessment", "integration_blocks", "FoldPipeline",
    "OmicsPipeline",
    "qc_summary", "aggregate_to_units", "differential_expression",
    "over_representation", "gsea_ranked_list", "univariate_association", "standard_to_ml_bridge",
    "figures", "diablo_plots", "wgcna_plots",
]
