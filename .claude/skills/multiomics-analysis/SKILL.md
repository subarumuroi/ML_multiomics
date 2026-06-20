---
name: multiomics-analysis
description: >-
  Run a multi-omics ML analysis with the ml_multiomics package for a user who is
  NOT a computational expert. Use when someone has one or more omics tables
  (proteomics / metabolomics / volatiles, as CSV or Excel) and wants to explore
  structure, find important features, or relate omics to an outcome (a category,
  an ordered category, or a number like yield) — especially the low-sample
  high-feature (low-n, high-p) case. Helps the user DECLARE the few decisions that
  matter (an AnalysisSpec), runs the whole method matrix leakage-free, and explains
  the results — signal, stability, and the stable features/modules — in plain language.
---

# Multi-omics analysis playbook

You are guiding someone who may not know machine learning. The library
(`ml_multiomics`) holds the LOGIC and enforces soundness; the user owns the
DECISIONS. Your job: help them author an `AnalysisSpec` (what each layer is for,
what the independent unit is, what to predict), run the engine, and interpret the
output. **The user decides which layers matter and how to use them; the package
never guesses or auto-combines layers.**

What the engine guarantees so you don't have to: per-fold preprocessing (no
leakage), method-aware handling, permutation + stability as the real inferences
(CV is only an overfit flag), and a provenance trail of everything done. It will
never crown a "best predictor" — small-n scores are noise.

Full rationale: `docs/ASSUMPTIONS_AND_CHOICES.md`. API details: `docs/USER_GUIDE.md`.
Worked specs: `examples/psilocybin_report/analysis.py`, `examples/banana_report/analysis.py`.

## Step 1 — Understand the data (ask, don't assume)
1. **Rows and columns of each file?** samples × features with a sample-ID column; one file per omics layer.
2. **What is ONE independent biological unit?** (each fermenter run, each fruit). This drives leave-one-out and prevents fake results. It must become an explicit metadata column — the engine never parses sample IDs.
3. **What is each layer FOR?** predictor / target / covariate / exclude. (e.g. an unreliable external-metabolite block → `exclude`.)
4. **What do they want to relate the omics to?** a category (nominal), an ordered category (ordinal), or a number (continuous) — or just explore.
5. **Has anything already been done to a file?** (already log-transformed / normalized / imputed). If so we either skip that step or revert to a raw file — see Step 4.

## Step 2 — Load into the container
```python
import pandas as pd
from ml_multiomics import OmicsDataset
ds = OmicsDataset(name=STUDY)
for layer, path, otype in LAYERS:
    ds.add_block(layer, pd.read_csv(path).set_index(SAMPLE_ID_COL), omics_type=otype)
ds.align()   # intersect to common samples by ID
```

## Step 3 — Build the grouping column (REQUIRED; the engine won't parse IDs)
Turn the "independent unit" answer into an explicit column in `ds.sample_meta`.
Helpers exist to populate it from IDs (you run them; the engine consumes the column):
```python
from ml_multiomics.core import parse_bioreactor_ids, parse_delimited
meta = parse_bioreactor_ids(ds.common_samples())     # F#C#R#T#  -> adds 'bioreactor'
# or parse_delimited(ds.common_samples(), sep="-", names=("stage","replicate"))
meta["unit"] = meta.index                            # if each sample is its own unit
ds.set_sample_metadata(meta)
```

## Step 4 — DECLARE the analysis (the AnalysisSpec)
This is where the user's decisions are recorded. Validation rejects anything ambiguous.
```python
from ml_multiomics import AnalysisSpec
spec = AnalysisSpec(
    grouping_column="bioreactor",              # the independent unit (required)
    roles={"proteomics": "predictor", "metab": "predictor", "ext": "exclude"},
    target_type="continuous",                   # nominal | ordinal | continuous
    target_column="yield",                      # OR target_values=<Series>; ordinal needs ordinal_order=[...]
    integration_groups=[["proteomics", "metab"]],  # which layers DIABLO integrates (opt-in; omit = none)
    min_obs_frac=0.5,                            # detection filter for correlation/latent methods
    # upstream-state handling (Step 1 q5):
    input_states={"proteomics": {"imputed": True}},     # flag what was already done, OR
    raw_sources={"proteomics": raw_unimputed_df},        # REVERT to a raw file for ML
)
spec.validate(ds)
print(spec.describe())   # read the declared decisions back to the user
```

## Step 5 — Run the systematic assessment
```python
from ml_multiomics.analysis import systematic_assessment, integration_assessment
sysres = systematic_assessment(ds, spec, n_permutations=99, stability_bootstrap=20)
integ  = integration_assessment(ds, spec, stability_bootstrap=20)   # DIABLO naive vs reduced
```
Raise `n_permutations` for finer p-value resolution; both are slower. (For the lab's
F#C#R#T# data, proteomics is auto-reduced before integration — nothing to set.)

## Step 6 — Read the results to the user (in plain language)
- **Is there signal?** Per approach, the permutation p-value read **against the
  resolution floor** (`...["permutation"]`): if the design can't reach 0.05, say so.
- **Does it recur?** Stability (`n_stable`, `consensus`): features stable across
  many approaches are the trustworthy hypothesis; a long unstable list is not.
- **Which approach to trust?** The `discriminators` — they prefer signal + stability
  + parsimony, and say when options are indistinguishable. Read the verdict verbatim.
- **Integration:** the naive-vs-reduced table + verdict — typically the naive
  individual-feature list is unstable and the reduced modules/factors recur; prefer
  the reduced, then expand a module to its member features for biology.
- **Per model:** its `report_card` — what it is, its assumptions, and its
  **divergences on THIS data** (e.g. "ordinal treated as nominal", "block imbalance").
- **CV is a sanity flag only** (`overfit`), never a ranking.

## Step 7 — Standard analysis (optional, for parity / DE / enrichment)
```python
from ml_multiomics.analysis import differential_expression, over_representation, gsea_ranked_list
de = differential_expression(X_raw, unit_labels=units, condition_labels=groups)  # aggregates to units
```
Run on RAW abundances; GSEA uses the external R path via `gsea_ranked_list`.

## Guardrails (do not violate)
- The grouping column is **required and explicit** — never let the engine guess the unit.
- The **user decides** layer roles; never auto-combine layers or invent a target.
- **Never present an alternative without a discriminator** — if nothing separates two
  options, say so and prefer the simpler/more interpretable one.
- Report signal (permutation vs floor) + stability, **never a bare CV leaderboard**.
- If a file was already transformed/normalized/imputed, declare it (`input_states`)
  or revert to raw (`raw_sources`) — do not double-process.
- Everything is provenance-tracked; if the user is unsure, show them the trail and
  let them revise the spec.

## Robustness & reproducibility (what the engine handles for you)
- **Deterministic.** Same `seed` -> identical panel, p-values, stability, consensus.
  Always pass a fixed `seed` so a report re-renders to the same numbers; raise
  `n_permutations` (49 -> 199-999) only to refine the p-value resolution.
- **Graceful degradation.** A method/reducer that fails on a given dataset (e.g. a
  degenerate fold, or a target a method can't take) is recorded as an `error` row and
  the rest of the assessment proceeds -- one failure never aborts the report. Check for
  `error` keys and tell the user which approaches were skipped and why.
- **Leakage-free by construction.** All preprocessing AND any reduction are refit
  inside each CV/permutation/bootstrap fold on training rows only; you don't manage this.
- **R methods are bounded.** DIABLO/WGCNA shell out to R with a timeout (default 600s)
  so a hang can't block forever. **WGCNA needs n >= ~15-20** -- omit it as a reducer for
  tiny datasets (e.g. banana n=9); the engine will otherwise record its failure.
- **Cost knobs.** Permutation is skipped (with a recorded note) for very large naive
  designs (`max_perm_features`); reduce `n_permutations`/`stability_bootstrap`/`reducers`
  for a quick exploratory run, raise them for a final one. Results stay deterministic.
- **Validate against a reference when one exists.** e.g. cross-check reducer factors
  against an established MOFA model (see tests/crosscheck/crosscheck_mofa_factors.py).
