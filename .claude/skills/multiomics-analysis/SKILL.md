---
name: multiomics-analysis
description: >-
  Run a multi-omics ML analysis with the ml_multiomics package for a user who is
  NOT a computational expert. Use when someone has one or more omics tables
  (proteomics / metabolomics / volatiles, as CSV or Excel) and wants to explore
  structure, find important features, or predict an outcome (a category, an
  ordered category, or a number like yield) — especially the low-sample
  high-feature (low-n, high-p) case. Guides the user through the few decisions
  that matter, runs the right model with leakage-free validation, and explains
  the results in plain language.
---

# Multi-omics analysis playbook

You are guiding someone who may not know machine learning. Do the technical work
for them, explain choices in plain language, and never let them fall into the
two traps that ruin omics ML: **data leakage from the wrong grouping** and
**double-scaling / wrong missing-value handling**. The library
(`ml_multiomics`) enforces the right defaults; your job is to wire it up for
their data and interpret the output.

Full rationale: `docs/ASSUMPTIONS_AND_CHOICES.md`. Decision details with code:
`docs/USER_GUIDE.md`. This skill is the operating procedure.

## Before you start
- Confirm the package is importable: `from ml_multiomics import OmicsDataset`.
  If not, `pip install -e .` in the repo (Python 3.12, MSYS2 bash).
- Optional extras only if needed: `pip install ml_multiomics[ordinal]` (ordinal
  regression), `[shap]` (SHAP importance).

## Step 1 — Understand the data (ask, don't assume)
Ask the user, in plain terms:
1. **What are the rows and columns of each file?** You need samples × features,
   with a sample-ID column. One file per omics layer.
2. **What is one independent biological unit?** (e.g. "each fermenter run", "each
   fruit"). This is the single most important question — see Step 4.
3. **What do they want to learn?** One of:
   - explore / find structure (no outcome) → unsupervised
   - predict a category, ordered category, or a number → supervised
4. **Are there missing values?** (blanks in the tables.)

## Step 2 — Load into the container
```python
import pandas as pd, numpy as np
from ml_multiomics import OmicsDataset
df = pd.read_csv(PATH).set_index(SAMPLE_ID_COL)   # or pd.read_excel
ds = OmicsDataset(name=STUDY)
ds.add_block(LAYER_NAME, df, omics_type="proteomics")  # or metabolomics/volatiles
# repeat add_block per layer; then ds.align() if multiple layers
```
Attach metadata with a parser (`parse_bioreactor_ids` for `F#C#R#T#`,
`parse_delimited` for `Green-1`) or build a small DataFrame indexed by sample ID.

## Step 3 — Preprocess (let the defaults work)
```python
from ml_multiomics import Preprocessor
Preprocessor().run(ds)   # missing-aware log + z-score, lab-matched, scaled ONCE
```
Exception: if you will run **NMF**, it needs non-negative input — preprocess with
`Preprocessor(profile=Profile(transform="log2", normalize="none")).run(ds)`.

## Step 4 — Set the grouping (prevents fake results)
Translate the user's "independent unit" answer into a `groups` vector. Every
validation call uses it so no unit splits across train/test.
- repeated measures per unit (timepoints in a reactor): `groups = ds.groups("bioreactor")`
- each sample independent (separate fruit): `groups = np.arange(n_samples)`
Tell the user, plainly, why: "samples from the same X aren't independent;
keeping them together stops the model from cheating."

## Step 5 — Pick the method (use the recipe)
Map their goal to a method (see USER_GUIDE "Workflow recipes"):
- **low-n high-p + predict** → reduce then predict: `WGCNA`/`NMF` → `RandomForest`/`Lasso`
- **multi-omics, what separates groups?** → `DIABLO`
- **ordered category** → `Ordinal`
- **a number (yield)** → `Lasso`/`ElasticNet` or `RandomForest` (regression)
- **just explore** → `NMF`/`WGCNA` (inspect factors/modules)
Set `target_type` to `nominal` / `ordinal` / `continuous` / `none` accordingly.

## Step 6 — Run with leakage-free validation
```python
from ml_multiomics import RandomForest
X, y = ds.get(LAYER), ds.sample_meta[TARGET_COL].to_numpy()
m = RandomForest().fit(X, y, target_type="nominal")
cv = m.cross_validate(X, y, groups=groups, target_type="nominal")
imp = m.importances(top_n=20)
pt  = m.permutation_test(X, y, groups=groups, n_permutations=200)
```
Reduce→predict variant: `red = WGCNA().fit(X,y).reduce(); RandomForest().fit(red, y, ...)`.

## Step 7 — Explain the results honestly
Report to the user in plain language:
- **Effect size first**: the grouped-CV accuracy / R², not a p-value.
- **The p-value WITH its limit**: `pt["resolution"]["finest_two_sided_p"]` — if
  the design can't go below e.g. 0.1, say so ("with this few groups, that's the
  best the test can show").
- **Top features** (importances / VIP / coefficients), with their real names.
- **Caveats**: small n means exploratory; cross-reference methods; don't
  over-claim.

## Guardrails (do not violate)
- Always pass `groups=` to CV/permutation. Never ungrouped CV on repeated measures.
- Never impute before MOFA (it models missingness); never feed z-scored data to NMF.
- Don't add a second scaler — preprocessing already scaled once.
- No deep learning at n < ~30. Prefer the methods above.
- If the user asks "is it significant?", answer with effect size + resolution,
  not a bare p-value.
