# Assumptions & Methodological Choices

This document records every non-obvious design decision in the consolidated
`ml_multiomics` library and the rationale behind it, so the pipeline can be
defended in review, cited in a methods section, and reproduced exactly. Each
choice is paired with *why* and, where relevant, the numerical evidence
(`tests/crosscheck/`).

> Scope: this is the "for posterity / publication" reference. For a hands-on
> walkthrough see [TUTORIAL.md](TUTORIAL.md); for the high-level architecture see
> [../CONSOLIDATION_PLAN.md](../CONSOLIDATION_PLAN.md).

---

## 1. Data model

**Canonical container `OmicsDataset`** holds N omics blocks (single-omics is just
N=1), a shared sample-metadata table, an optional target spec, and a per-block
provenance log.

- **Alignment is always by sample ID with intersection, never by position.**
  Rationale: blocks routinely differ in sample membership and order. In the
  banana data the aromatics block has 12 samples while the other three have 9,
  and sample order differs across files; positional alignment would silently
  mis-pair samples.
- **Metadata parsing is pluggable and dataset-specific**, not universal. For
  bioreactor data the `F#C#R#T#` convention is parsed into the four raw factors
  `F`, `C`, `R`, `T` (plus the composites `condition` = `F_C` and `bioreactor` =
  `F_C_R`). `F` (fermentation batch) and `C` (e.g. C1/C2) are both **condition
  factors** — neither is a genetic "construct" — `R#` = replicate, `T#` =
  timepoint. The helper imposes no roles: the **user declares**, via the
  `AnalysisSpec`, which factor is the grouping unit (usually `bioreactor`
  `F#C#R#`), a target, a covariate, or excluded — keeping the design flexible.
  Other datasets supply their own metadata or parser.
- **Target spec** supports four types — `nominal`, `ordinal`, `continuous`,
  `none` — and method applicability is gated by target type. The target may be a
  metadata column or derived from a separate measurement table.

---

## 2. Preprocessing

Preprocessing has **three stages, and imputation is method-gated** (see §3):

1. **Transform** (missing-aware): `log2(x+1)` (default for proteomics) or
   `log10` (default for metabolomics/volatiles). NaN is preserved; non-positive
   values outside the transform domain become NaN.
2. **Feature filtering**: drop near-zero-variance features; optional missingness
   filter.
3. **Normalize** (missing-aware): per-feature z-score computed on non-missing
   values.

The canonical primitives are a port of the MOFA pipeline's `mofa_prep.py`
(the user's current, validated preprocessing) — **not** a generic preprocessing
hierarchy. Default per-omics profiles follow the lab's conventions.

### Why these specific conventions (and where they came from)

The library's defaults match the lab's own standard-analysis tooling
(`IdeaBio.R` / `IdeaBio.jl`), reimplemented in pure Python and **verified
numerically** (§5):

| Step | Convention | Source / parity |
|------|------------|-----------------|
| transform | `log2(x+1)` (proteomics) / `log10` | mofa_prep / IdeaBio.jl |
| normalize | per-feature z-score | IdeaBio.R `zscore`, IdeaBio.jl `normalise_zscore` |
| impute (opt-in) | MetaboAnalyst `0.2 × min(positive)`, or imputePCA-per-group | IdeaBio.R / IdeaBio.jl / missMDA |
| feature filter | variance + missingness | mofa_prep |

### Documented divergences (deliberate)

- **z-score on missing data → skip-NaN.** The lab's R `zscore` has no `na.rm`,
  so a single missing value turns an entire feature into NaN; IdeaBio.jl and this
  library skip NaN. **Skip-NaN is canonical here** because it is required for
  MOFA (which receives z-scored-but-unimputed data) — R's behavior would make any
  partially-missing feature unusable. The lab's own R and Julia tools disagree on
  this point; we follow Julia.
- **`log2(x+1)` vs plain `log2(x)`.** The default uses the `+1` pseudocount
  (zero-safe, matches the validated MOFA pipeline). IdeaBio.jl uses plain
  `log2(x)` (zeros → missing). Both are available as profile options; the
  difference is *only* the pseudocount (verified, §5).
- **MetaboAnalyst min basis.** R and this library use `0.2 × min(positive)`;
  IdeaBio.jl uses `0.2 × min(non-missing)`. Identical on positive intensity data.

### No double-scaling

Because the data is z-scored exactly once in preprocessing, methods must **not**
re-scale internally (e.g. sklearn `PLSRegression` must be built with
`scale=False`; no `MinMaxScaler` on already-scaled data; MOFA receives centered
input and does its own scaling). The provenance log records the data's state so
this is checkable. This addresses a real bug class — the prior code pareto-scaled
in preprocessing and then z-scored again inside PLS-DA.

---

## 3. Missingness

**MOFA models missing values natively; most methods cannot.** This is handled by
a capability flag, not by imputing everything up front:

- Each method declares `handles_missing`.
  - `True` (MOFA, mask-aware factorizations): receives the NaN-carrying matrix
    directly. **Imputing first would destroy the information MOFA models.**
  - `False` (Random Forest, PLS-DA, LASSO, ordinal, sklearn NMF): a just-in-time
    imputed copy is produced before fitting (MetaboAnalyst default).
- The container's canonical state is **transformed + normalized but not
  imputed**. Imputation is the last, method-specific step.

Imputation strategies available: `metaboanalyst` (default), `remove_all_missing`,
`imputepca` (regularized iterative PCA; pure-Python, machine-precision parity
with missMDA — §5).

---

## 4. Statistical methodology

### Grouping-aware resampling (the central correctness choice)

Every cross-validation, permutation test, and bootstrap **requires an explicit
`groups` vector = the independent experimental unit** (bioreactor `F#C#R#` for
psilocybin; the biological replicate for banana). Splits hold out whole groups
(leave-one-group-out); a group never straddles train/test.

**Why:** repeated measures (e.g. multiple timepoints within one bioreactor) are
pseudoreplicates, not independent samples. Naive sample-level cross-validation
leaks — a timepoint in train and another timepoint of the same reactor in test —
inflating accuracy. This is the most likely cause of prior "hard to interpret"
results. The ported methods deliberately replace the original sample-level
LeaveOneOut / label shuffling with group-level equivalents.

### Honest resolution reporting

p-values are reported alongside the **finest achievable resolution** for the
design. Group-level permutation has a discrete floor: with G groups split into
label counts, only `G! / ∏ c_k!` distinct arrangements exist, so the smallest
attainable two-sided p is `~2 / n_arrangements`. `permutation_resolution()`
reports this so a p-value is never read without its floor — at small sample sizes
the floor, not the data, can be the binding constraint.

### Effect sizes first

Reporting leads with effect size + confidence interval; p-values are secondary
and always carry their resolution. Stability is reported as "selected in k/N
resamples (robust)" rather than an opaque frequency.

### Targets: regression vs classification

For psilocybin, the default target is **continuous yield** (aligned to omics by
bioreactor and phase), not 7-way condition classification — 2–4 replicates per
condition make categorical classification statistically weak. Banana is
**ordinal** (Green < Ripe < Overripe). Methods auto-resolve classifier vs
regressor from the target type.

### No deep learning

At n < ~30 (both datasets), deep models are not appropriate for the data. 
The method set is deliberately regularized / tree-based / factor-analytic:
PCA, NMF, MOFA, WGCNA (unsupervised); sparse PLS-DA, DIABLO, Random Forest,
ordinal regression, LASSO/ElasticNet (supervised).

### Leakage-free validation; permutation + stability are primary, CV is a sanity flag

All data-dependent preprocessing (detection filter, imputation, variance filter,
z-score) and any dimensionality reduction are fit **inside each CV / permutation /
bootstrap fold, on training rows only** (`FittablePreprocessor`,
`validation.leakage_free_cv`). The earlier global-z-score / pre-split-imputation
leakage is gone. Because a single leave-one-group-out score is noise at n=9–28, the
**primary inferences are the permutation test** (signal beyond chance, always read
against the design's resolution floor) **and bootstrap selection stability** (does the
signal recur). Cross-validation is reported only as a **binary overfit flag** — never
ranked, never a leaderboard. Permutation is skipped (with an explicit recorded note)
for very large naive designs where refitting is prohibitive; the reduced models carry
the signal test.

---

## 4b. Declarative analysis, block imbalance & self-documentation

- **The user declares; the package executes.** An `AnalysisSpec` records the decisions a
  human must own — the (required, explicit) grouping/independent-unit column, each layer's
  role (predictor/target/covariate/exclude), per-layer transforms, and which layers (if any)
  integrate. The engine validates, flags, records, and runs; it never parses sample IDs,
  proposes roles, or auto-combines layers. `parse_bioreactor_ids` / `parse_delimited` are
  optional helpers that *populate* the grouping column.
- **Block-imbalance reduction.** When one block dwarfs the others (proteomics ~4000 vs
  metabolites ~8–46), naive integration is dominated by it and yields an unstable
  "top-proteins" list. The engine auto-reduces such a block (WGCNA modules / PCA / NMF;
  hybrid trigger `>200 features AND >5× the median other block`) before integrating, and
  **shows the naive-vs-reduced selection-stability contrast** that justifies it. DIABLO gained
  a regression mode (`mixOmics::block.spls`) so a continuous yield can be integrated directly.
- **No alternative without a discriminator.** Reduced-vs-direct / imputation / reducer / filter
  comparisons are each resolved by stability + signal + parsimony, or declared indistinguishable
  (then prefer the simpler) — never by a raw score.
- **Upstream-state handling.** If a file arrived already transformed/normalized/imputed, the
  user declares it (`input_states`); the preprocessor skips steps already applied and flags ones
  it cannot safely redo from processed data. The legitimate "revert" is to point the spec at a
  `raw_source` (e.g. banana's unimputed proteomics) — transforms are never algorithmically inverted.
- **Self-documentation is a package feature, not report text.** Every method implements
  `describe()` / `assumptions()` / `divergences(context)` (computed against the live data) and a
  `report_card`; every data mutation is recorded in a `ProvenanceTrail` (markdown + JSON). A
  parametrized test fails if any method ships without these, so completeness is enforced by CI.

---

## 5. Numerical verification

The preprocessing layer is verified against the lab's **actual** R and Julia code
(re-runnable: `tests/crosscheck/run_all.py`; captured in
`tests/crosscheck/RESULTS.md`). Tolerance for "MATCH" is 1e-6; observed diffs are
at machine precision.

| primitive | vs IdeaBio.R | vs IdeaBio.jl | notes |
|-----------|--------------|---------------|-------|
| z-score (complete) | MATCH ~3e-15 | MATCH 0.0 | identical |
| z-score (missing) | diverges (by design) | MATCH 0.0 | follow Julia / skip-NaN (§2) |
| log10 | MATCH ~5e-15 | MATCH 0.0 | |
| log2 | — | `+1` pseudocount diff | documented default (§2) |
| MetaboAnalyst impute | MATCH 0.0 | MATCH 0.0 | |
| imputePCA / by-group | MATCH ~5e-14 | n/a | faithful missMDA port |

**Lesson:** "replicate the lab" was ambiguous — the lab's R and Julia tools are
not numerically identical to each other. Parity is established by *running* their
code (or reproducing exact function bodies) and diffing, not by reading it.

### Differential expression & enrichment (analysis module)

`analysis.compute_volcano` (pairwise Welch t-test + fold change + FDR),
`analysis.anova_tukey` (one-way ANOVA + Tukey HSD), and `analysis.ora`
(hypergeometric over-representation) are pure-Python ports of the lab's
conventions, cross-checked vs the actual R (`run_crosscheck_de.py`):

| quantity | result |
|----------|--------|
| fold change, log2fc, Welch p, q (single contrast) | MATCH ~1e-15 vs IdeaBio.R compute_volcano |
| ORA hypergeometric tail | MATCH ~4e-16 vs base R phyper |

- **DE operates on RAW/linear abundances**, not the z-scored matrix (it computes
  its own fold change and log10-transforms internally for the test) — matching
  the lab.
- **Pseudoreplication caveat (flagged for review):** the lab's DE, and this port,
  treat every sample as independent. For repeated-measures designs (timepoints
  per bioreactor) aggregate to one row per independent unit *before* DE.
- ⚠️ **Bug found in the lab's `compute_volcano`, NOT replicated.** It labels
  result features with `rep(colnames(fc), times=nrow(fc))` where it needs
  `each=nrow(fc)`; because R unrolls the matrix column-major, **feature labels
  are mis-paired with their values whenever there is more than one contrast.**
  Our Python labels correctly (verified: single-contrast outputs match the lab
  exactly; multi-contrast outputs differ only because the lab mislabels). This is
  the kind of silent error that motivates running + diffing the reference code.
- **GSEA** (running-sum / fgsea) is intentionally not reimplemented in pure
  Python — no parity is claimed; use the existing clusterProfiler (R) path.

---

## 6. Reproducibility

- Deterministic: fixed seeds (`random_state=42` for RF; seeded RNG in tests);
  imputePCA uses zero-init (no RNG). Cross-check inputs are hard-coded / seeded.
- Environment: Python 3.12; R 4.4.0 + `missMDA` and Julia 1.11.3 are needed only
  to *re-run the cross-checks*, not to use the library. Pure-Python, no R/Julia
  in the default install (an optional R extra backs a mixOmics DIABLO variant).
- Every analysis run records provenance per block and can emit its inputs and
  outputs for an audit trail.

---

## 7. Known limitations / open items

- Just-in-time imputation currently uses whole-matrix column statistics
  (matching the lab's impute-then-analyse flow); strict fold-wise imputation
  inside CV is a possible future refinement.
- Methods (done + verified): Random Forest, **XGBoost** (gradient boosting,
  conservative small-n defaults, native missing handling), sparse PLS-DA (with
  bootstrap stability selection), LASSO/ElasticNet, Ordinal (`mord`) — supervised;
  PCA, NMF — reducers (sklearn). The legacy preprocessing hierarchy, single_omics/
  multi_omics, workflows/, utils/ were removed (dedupe, Task #19) — one clean
  surface: core / preprocessing / methods / validation / analysis. Unsupervised
  "reducers" (PCA, NMF; MOFA to follow) expose a samples x factor matrix for the
  reduce->predict pattern at p >> n. NMF requires non-negative input (z-scored
  rejected) — preprocess with normalize='none'.
- **DIABLO and WGCNA use the REFERENCE R implementations** (mixOmics::block.splsda,
  WGCNA package) via a subprocess bridge — chosen because these reimplementations
  are hardest to validate and the user lacks the domain background to assess their
  parameters; the standard tools also bring validated tuning (`tune.block.splsda`).
  A sPLS-DA probe vs mixOmics showed our native NIPALS matches the *variates*
  (r=0.999) but the sparse feature *selection* only ~75-85% — good for components,
  not for exact biomarker lists. The native Python ports are kept as
  `NativeDIABLO` / `NativeWGCNA` (experimental, unvalidated). Depending on mixOmics/
  WGCNA (community-standard, citable) does NOT reintroduce dependence on the lab's
  gatekept IdeaBio.R/.jl. Caveat: parity/correctness ≠ appropriateness — these
  methods may still be ill-suited to n≈24.
- **Reports are interpretation-framed, not leaderboards** (examples/banana_report,
  examples/psilocybin_report): at small n a single CV score is noisy, so CV is a
  guardrail (beats predict-the-mean? overfitting?), reducing the big proteomics
  block is principled described preprocessing, and the panel is descriptive (no
  crowned winner). Queued: MOFA single-group lift (multiphase stays in ml_psi_mofa
  while in active development).
- MOFA core is to be lifted from `ml_psi_mofa` into the library so all algorithms
  share one import surface.
