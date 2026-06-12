# Consolidation Plan — Unified Multi-Omics ML Library

**Status:** proposed (awaiting confirmation before code migration)
**Author:** Subaru Muroi
**Goal:** Fold three fragmented repos into one clean, pip-installable, pure-Python
multi-omics ML library, then demonstrate it end-to-end through a Quarto report,
then wrap it in a skill so any operator can run the whole workflow.

---

## 0. Hard rules (non-negotiable)

1. **Single-package API.** EVERY algorithm lives in `ml_multiomics`. Downstream
   code imports nothing else for analysis — `from ml_multiomics import ...` is
   the only entry point. This means MOFA moves *fully* out of `ml_psi_mofa`;
   GSEA/ORA/DE move into `ml_multiomics/analysis/`. Project repos
   (`ml_psi_mofa`, future datasets) keep ONLY dataset-specific glue (loading,
   QC, domain config) and call the library for all methods.
2. **Pure-Python, no R / no Julia** in the default install. (R-backed DIABLO is
   an optional `[r]` extra only.) Removes dependence on the lab's gatekept
   `IdeaBio.R` / `IdeaBio.jl`.
3. **One shell: bash (MSYS2).** All commands run through MSYS2 bash, not
   PowerShell. POSIX paths (`/c/Users/...`), forward slashes, one consistent
   environment. No more Windows/Linux venv thrash.

### Environment & tooling (verified 2026-05-29)

| Tool | MSYS2 bash (chosen) | WSL (rejected) |
|------|---------------------|----------------|
| bash / POSIX | yes | yes |
| Python 3.12 | yes | yes |
| R 4.4.0 | yes | **missing** |
| Quarto | yes | **missing** |
| git / pip | yes | **missing** |

MSYS2 has the entire toolchain already; WSL would need R + Quarto + pip + git
installed first. So MSYS2 bash is the working environment. Venvs are
Windows-layout (`Scripts/`) but driven only from MSYS2 bash. **Standardize on
`.venv` naming** across all repos (currently `venv` in two, `.venv` in one).

---

## 1. Why

Three repos currently overlap and fragment the same work:

| Repo | What it has | Fate |
|------|-------------|------|
| `ml_multiomics` | Clean architecture (preprocessing class hierarchy → methods → workflows, config-driven); PCA, dense PLS-DA, R-backed DIABLO, concat/ensemble baselines, CV/permutation utils | **Becomes the library (the home/shell)** |
| `multiomics_integration` | Richer methods, all pure-Python: sparse PLS-DA, native DIABLO, ordinal regression, random forest + SHAP, WGCNA; permutation + bootstrap-stability on every method; consensus integration; strong viz suite | **Methods ported in; repo retired** |
| `ml_psi_mofa` | MOFA + stable archetypes, factor-yield regression, GSEA/ORA prep, factor-matrix export | **MOFA core lifts into library; repo becomes a project repo that depends on the library** |

Strategic note: consolidating on the **pure-Python** implementations removes all
dependence on the lab's R (`IdeaBio.R`) and Julia (`IdeaBio.jl`) tooling. The
library + skill becomes a self-contained, automatable workflow — the
demonstration that the analysis pipeline does not require a human gatekeeper.

---

## 2. Target architecture (3 layers)

```
LAYER 3 — SKILL                 "Analyze this dataset end-to-end, emit the report."
  Thin orchestration recipe + Quarto report template + dataset-config convention.
  Drives Layer 2. This is the operator interface.
        │ calls
LAYER 2 — PROJECT REPOS         one per dataset (ml_psi_mofa is the first)
  data + config + rendered report. No algorithms. Depends on Layer 1.
        │ imports
LAYER 1 — THE LIBRARY  (this repo, ml_multiomics)
  pip-installable, versioned, tested. All methods behind one consistent API.
```

### Library internal layout (target)

```
ml_multiomics/
├── preprocessing/        # unified: class hierarchy + functional pipeline
│   ├── base_preprocessor.py
│   ├── omics_preprocessor.py      # Metabolomics / Volatiles / Proteomics
│   ├── pipeline.py                # prepare_block, prepare_multiblock, encode_ordinal
│   └── integrator.py              # OmicsIntegrator, MultiBlockData
├── methods/              # ALL algorithms, one convention: fit / get_* / plot_*
│   ├── unsupervised/
│   │   ├── pca.py
│   │   ├── nmf.py                 # NEW
│   │   ├── mofa.py                # LIFTED from ml_psi_mofa
│   │   └── wgcna.py               # ported
│   ├── supervised/
│   │   ├── plsda.py               # sparse (ported) + dense (existing) variants
│   │   ├── diablo.py              # native pure-Python (ported); R-backed optional
│   │   ├── random_forest.py       # ported (+SHAP)
│   │   ├── ordinal.py             # ported
│   │   └── linear.py              # NEW: LASSO / ElasticNet
│   └── baselines/
│       ├── concatenation.py       # existing
│       └── ensemble.py            # existing
├── validation/           # CV, permutation tests, bootstrap stability
├── analysis/             # NEW: lab-standard DE/enrichment in Python
│   ├── differential.py            # volcano/fold-change, ANOVA + Tukey (port IdeaBio.R)
│   └── enrichment.py              # GSEA/ORA prep (reuse ml_psi_mofa)
├── consensus/            # consensus features + evidence integration (ported)
├── visualization/        # merged viz suite
└── workflows/            # SingleOmics / MultiOmics orchestration (existing, extended)
```

---

## 3. Method inventory (target API)

Every method exposes the same shape: `fit(...) → self`, `get_*_df()` accessors,
`plot_*()` figures, and is wired into the shared `validation/` harness
(LOO-CV, permutation test, bootstrap stability).

| Method | Type | Source | Notes |
|--------|------|--------|-------|
| PCA | unsupervised | ml_multiomics | exploratory |
| NMF | unsupervised | **new** | non-negative factorization; compare to MOFA |
| MOFA+ | unsupervised | **lift from ml_psi_mofa** | + stable-archetype ensemble |
| WGCNA | unsupervised | multiomics_integration | also usable as dim-reduction |
| PLS-DA (dense) | supervised | ml_multiomics | quick exploratory |
| sPLS-DA (sparse) | supervised | multiomics_integration | built-in feature selection (preferred) |
| DIABLO (native) | supervised multi-block | multiomics_integration | pure-Python, no R |
| DIABLO (R/mixOmics) | supervised multi-block | ml_multiomics | optional, validated cross-check |
| Random Forest | supervised | multiomics_integration | + SHAP + permutation importance |
| Ordinal regression | supervised | multiomics_integration | AT/IT/SE via `mord` |
| LASSO / ElasticNet | supervised | **new** | regularized; small-n friendly |

Shared rigor layer (applies to all): `cross_validate_*`, `permutation_test_*`
(early-stop), `stability_selection_*` (bootstrap), `find_consensus_features`,
`integrate_wgcna_evidence`.

---

## 4. Duplication resolution

- **PLS-DA:** keep BOTH but namespace clearly — `plsda.PLSDA` (dense, exploratory)
  and `plsda.SparsePLSDA` (sparse, feature-selecting, preferred). Sparse is from
  multiomics_integration; dense is the existing sklearn wrapper.
- **DIABLO:** native pure-Python (multiomics_integration) becomes the default
  `DIABLO`. R-backed version kept as `DIABLO_R` behind an optional extra
  (`pip install ml_multiomics[r]`) for reviewers who want mixOmics parity.
- **Preprocessing:** keep ml_multiomics's class hierarchy as the public surface;
  re-implement multiomics_integration's `prepare_block`/`prepare_multiblock` as
  thin functional wrappers over the classes (one source of truth for imputation
  / transform / scaling math).

---

## 5. What gets added fresh

- `methods/supervised/linear.py` — LASSO + ElasticNet with the standard API and
  rigor hooks. ElasticNet preferred for correlated omics features.
- `methods/unsupervised/nmf.py` — sklearn NMF wrapper with factor/loading
  accessors mirroring the MOFA/PCA interface for apples-to-apples comparison.
- `analysis/differential.py` — Python port of IdeaBio.R `compute_volcano`,
  `compute_anova_tukey` (Welch t-test + fold change + BH/Tukey; continuous
  p-values avoid the Wilcoxon discrete-floor problem at small n).
- `analysis/enrichment.py` — GSEA/ORA prep reused from ml_psi_mofa.

---

## 6. Execution order

1. **(plan)** this document — confirm before moving code. ← you are here
2. **scaffold** the target `methods/ validation/ analysis/ consensus/` layout
   (empty modules + registry + config schema), keep existing code working.
3. **port** multiomics_integration methods + rigor harness.
4. **add** LASSO/ElasticNet + NMF.
5. **lift** MOFA core from ml_psi_mofa.
6. **unify** preprocessing; dedupe PLS-DA/DIABLO.
7. **port** standard DE/enrichment to Python (`analysis/`).
8. **demo**: 27-PSI data → Quarto "Part 3: Omics & ML" report extending the
   IDEA Bio `fermentation_report` template.
9. **skill**: SKILL.md + recipe + report template + dataset-config convention.
10. **tests + packaging**: pyproject, `pip install -e .`, per-method smoke tests.

Each step is its own commit (or small series), verified before the next.

---

## 7. Out of scope (deliberately)

- No deep learning — at n<30 it is indefensible; reviewers reject it.
- No rewrite of the IDEA Bio fermentation front-end — we *extend* its report,
  picking up at the `.metabolics` stub. We do not replace Part 1/Part 2.
- ml_psi_mofa's psilocybin-specific QC/yield logic stays in ml_psi_mofa.

---

## 8. Data model & methodology (finalized 2026-05-29)

### Canonical container: `OmicsDataset`
- N blocks; single-omics = 1 block. Blocks may have DIFFERENT sample sets
  (banana: amino/metab/prot n=9 but aromatics n=12) → **align by sample ID,
  intersect to common samples, never by position.**
- Shared **sample-metadata table**: flexible design columns (condition,
  replicate, timepoint, phase, batch...). ID parsing is a **pluggable,
  dataset-specific** convention (NOT universal). For bioreactor data the
  `F#C#R#T#` parser applies: **`F#C#` JOINTLY = condition** (C nominally means
  "condition" but is unreliable alone), `R#` = replicate, `T#` = timepoint.
  Independent unit (for grouping) = bioreactor = `F#C#R#`. Other datasets supply
  their own metadata or parser (banana: sample = `<Stage>-<rep>`).
- **Target spec**: type ∈ {nominal, ordinal, continuous, none} + source. Target
  may be a metadata column OR derived from a SEPARATE measurement table (e.g.
  extracellular yield) aligned by (bioreactor, phase). Yield and omics live on
  different sparse timepoint grids, so alignment is explicit with a rule:
  pick-timepoint or phase-aggregate(mean).
- **Provenance log** per block: raw → imputed → transformed → scaled. Every
  step recorded; methods read it and never silently repeat a step.

### Grouping — Q1: explicit always + aggregation optional
- Container always carries the **independent unit** (psilocybin: bioreactor
  `F_C_R`; banana: replicate). Every CV / permutation / bootstrap REQUIRES a
  groups vector → leave-one-group-out, group-stratified permutation. This kills
  timepoint pseudoreplication leakage. Aggregation (one row per reactor per
  phase) is an optional preprocessing step, not forced.

### Target — Q2: yield, phase-aligned
- Default psilocybin target = **continuous yield** (yield_over_biomass), aligned
  to omics at phase 3. Yield exists at many timepoints, omics at few → align by
  (bioreactor, phase); default = phase-3 mean (matches MOFA aggregation),
  specific-timepoint configurable.
- Banana = **ordinal** (Green<Ripe<Overripe). All four target types are
  first-class; method applicability is gated by target type (regression vs
  classification vs ordinal vs unsupervised).

### Normalization — Q3: lab conventions as defaults, inside provenance framework
- DEFAULT profiles = the **lab's conventions** (they understand the data),
  reimplemented pure-Python, applied ONCE, provenance-tracked, never repeated.
- Lab conventions found in `StandardOmicAnalyses/R`:
  - Imputation default: **imputePCA per group** (missMDA). Alt: MetaboAnalyst
    `0.2 × min(positive per column)`, negatives→NA.
  - Scaling: **z-score** (`normalisations.R`).
  - Log: **log10** (in DE `compute_volcano`/`compute_anova`).
  - Feature filtering: drop features by **group-wise missing COUNT** threshold.
- ⚠️ **DIVERGENCE**: current Python repos use half-min + pareto +
  overall-fraction-drop — does NOT match the lab. Directive: match the lab.
  Pareto kept as a documented alternative profile.
- **Z-scoring worry resolved**: z-score ONCE (lab), then disable method-internal
  re-scaling (`PLSRegression(scale=False)`, no MinMaxScaler-on-scaled, MOFA gets
  centered-not-z-scored input). The bug was double-scaling, not z-scoring.
- **CONFIRMED via IdeaBio.jl/src/omics (Julia, the cross-language reference):**
  - normalisation = **ZScore only** (column-wise, missing-aware — z-scores only
    non-missing entries). No pareto.
  - transformation = **Log2 / Log10**, missing-aware, **non-positive → missing**.
  - imputation = **MetaboAnalyst** (`0.2 × min(non-missing per feature)`) or
    **RemoveAllMissing** (drop any feature with any missing). No imputePCA in
    Julia (imputePCA is the heavier R-only option).
  - `mofa_prep.py` (the user's CURRENT canonical preprocessing) matches: missing-
    aware `_log2_transform` (log2(x+1), ≤-1→NaN) + `_zscore` (per-feature on
    non-NaN) + variance/missingness filtering, NaN preserved (never imputes).
- **All three lab tools (R, Julia, MOFA-Python) agree** → no ambiguity.

### Preprocessing canon & missingness contract (FINALIZED)

- **Discard ml_multiomics's `BasePreprocessor` class hierarchy.** Canonical
  preprocessing = `ml_psi_mofa/mofa_prep.py` primitives (missing-aware log2 /
  z-score / variance filter), generalized into the library. (User: "the latest
  methods I've been using live in mofa.")
- **Three-stage pipeline, imputation method-gated:**
  1. Transform (missing-aware): log2(x+1) / log10, NaN preserved.
  2. Normalize (missing-aware): z-score per feature on non-NaN, NaN preserved.
  3. Impute (OPT-IN, LAST): runs ONLY for methods with `handles_missing=False`.
- **Every method carries a `handles_missing` capability flag.**
  - MOFA → `handles_missing=True`: receives the NaN-carrying matrix directly.
    Imputing first would destroy the missingness information MOFA models.
  - RF / PLS-DA / ordinal / LASSO / sklearn-NMF → `handles_missing=False`: get a
    just-in-time imputed copy (MetaboAnalyst 0.2×min default; RemoveAllMissing or
    imputePCA-per-group optional), recorded in provenance.
- Container's canonical state = transformed + normalized, NOT imputed.
- `spectronaut.R` is proteomics INGESTION (parsing Spectronaut report format),
  not normalization — handled in the loader/ingestion layer, not the profiles.

### Cross-check results (vs lab's actual code; tests/crosscheck/)
- Python primitives are numerically IDENTICAL to both R and Julia wherever the
  two lab tools agree: zscore (complete), log10, metaboanalyst — all 0 / 1e-15.
- **z-score on missing data — RESOLVED:** R propagates NaN over the whole feature
  (no na.rm); Julia + Python skip NaN. Python matches Julia exactly (diff 0.0).
  Skip-NaN is CANONICAL: it is required for MOFA (which gets z-scored-not-imputed
  data); R's behavior would make any partially-missing feature all-NaN. The lab's
  own R and Julia tools disagree here — we follow Julia.
- **log2 default — DECISION:** default proteomics transform = `log2(x+1)`
  (mofa_prep convention, what the user's validated pipeline uses, zero-safe).
  IdeaBio.jl uses plain `log2(x)` (zeros -> missing). Cross-check confirmed the
  ONLY difference is the +1 pseudocount (plain log2 matches Julia at 0.0).
  Plain log2 remains available as a profile option. (Flag: flip the default with
  one line if lab parity is preferred over zero-safety.)
- imputePCA-per-group still unported (Task #26).

### Reporting — addresses "hard to interpret/explain"
- Effect-size + CI first; p-values secondary and always reported WITH the
  achievable resolution given n (e.g. "6 bioreactors → finest group-permutation
  p = 1/X"). Stability framed as "selected in 87/100 resamples (robust)".
  Plain-language one-liner per result for bench-biologist readers.
