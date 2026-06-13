# User Guide — run your own analysis & choose the right options

This is a practical, plain-language guide for running the pipeline on **your own
dataset** and making the choices it asks of you. No machine-learning background
is assumed. For each decision you face, this guide explains *what it is*, *your
options*, *when to pick each*, and *the exact code*.

It is written so the work can be reproduced **without AI assistance**. Where a
choice depends on your data or question (and we cannot decide it for you), you
get a rule of thumb and the code for each option.

- The *why* behind the defaults: [ASSUMPTIONS_AND_CHOICES.md](ASSUMPTIONS_AND_CHOICES.md)
- A short install + verify + quickstart: [TUTORIAL.md](TUTORIAL.md)

---

## The pipeline in one picture

```
  your data (Excel/CSV, one table per omics layer)
        │   add_block()
        ▼
  OmicsDataset ── set_sample_metadata() ── set_target()
        │   Preprocessor().run()         (transform + normalize; keeps NaN)
        ▼
  a method  (RandomForest, MOFA, PLS-DA, ...)
        │   .fit() / .cross_validate(groups=...) / .permutation_test(groups=...)
        ▼
  results  (predictions, importances, p-values WITH their resolution)
```

You will make a handful of decisions along the way. They are listed below in the
order you hit them. **Decision 1 (the grouping unit) and Decision 2 (missing
values) are the two that matter most for getting trustworthy results.**

---

## Step A — Get your data in

Each omics layer is a table: **rows = samples, columns = features**, with a
column holding the sample IDs. Layers may have different samples — that is fine,
they are aligned by ID later.

```python
import pandas as pd
from ml_multiomics import OmicsDataset

prot = pd.read_csv("my_proteomics.csv").set_index("sample_id")
metab = pd.read_csv("my_metabolomics.csv").set_index("sample_id")

ds = OmicsDataset(name="my_study")
ds.add_block("proteomics", prot, omics_type="proteomics")
ds.add_block("metabolomics", metab, omics_type="metabolomics")
```

`omics_type` selects sensible preprocessing defaults (see Step C). Use
`"proteomics"`, `"metabolomics"`, `"volatiles"`, or leave it `None` for the
generic default.

### Tell the dataset about your samples (metadata)

Metadata is a table indexed by sample ID describing each sample (condition,
replicate, timepoint, batch, …). You have three options:

| Your sample IDs look like… | Use |
|----------------------------|-----|
| `F503_C1_R1_T1` (bioreactor) | `parse_bioreactor_ids(ids)` |
| `Green-1`, `Ripe-2` (group-replicate) | `parse_delimited(ids, sep="-", names=("group","replicate"))` |
| anything else | build the table yourself (below) |

```python
from ml_multiomics.core import parse_bioreactor_ids, parse_delimited

ds.set_sample_metadata(parse_bioreactor_ids(prot.index))
```

Build it yourself when your IDs don't match a parser — it's just a DataFrame:

```python
meta = pd.DataFrame({
    "condition":  [...],   # the experimental group each sample belongs to
    "replicate":  [...],   # which biological replicate
    "timepoint":  [...],   # optional
    "bioreactor": [...],   # optional: the independent unit (see Decision 1)
}, index=prot.index)
ds.set_sample_metadata(meta)
```

### Align multi-omics layers

```python
ds.align()    # subset every block + metadata to samples present in ALL blocks
```

---

## Decision 1 — What is your *independent unit*? (the grouping)  ⚠️ most important

**What it is.** When you validate a model you must never let two measurements of
the *same biological thing* land on opposite sides of a train/test split. If you
do, the model "sees the answer" and accuracy is inflated and meaningless. The
independent unit is the thing your replicates are *of*.

**How to identify yours:**

| If your data is… | The independent unit is… | Group by |
|------------------|--------------------------|----------|
| One sample per biological replicate (e.g. 3 separate fruit) | the replicate (each row is independent) | each sample its own group |
| Multiple timepoints per bioreactor run | the **bioreactor** (timepoints are repeated measures) | `bioreactor` (`F#C#R#`) |
| Technical replicates of the same sample | the biological sample | the sample ID |
| Batches that share processing | sometimes the batch | `batch` |

**How to use it.** Every validation call takes a `groups` vector:

```python
import numpy as np

# bioreactor data with repeated timepoints -> group by reactor
groups = ds.groups("bioreactor").to_numpy()

# independent replicates (e.g. banana) -> each sample is its own group
groups = np.arange(ds.blocks["proteomics"].shape[0])
```

**If you get this wrong:** grouping by the wrong (too-fine) unit causes leakage
and over-optimistic results. When unsure, pick the *coarser* unit (the thing you
could independently repeat the experiment on).

**Check the achievable resolution** before trusting any p-value:

```python
from ml_multiomics.validation import permutation_resolution
print(permutation_resolution(groups, ds.sample_meta["condition"]))
# finest_two_sided_p = the smallest p your design can ever produce
```

---

## Decision 2 — Missing values: propagate or impute?  ⚠️

**The rule depends entirely on the method you will run.**

| You will run… | Do this with NaNs | Why |
|---------------|-------------------|-----|
| **MOFA** (or any factor model that models missingness) | **PROPAGATE** — leave NaNs in place | MOFA uses the missingness pattern; imputing first *destroys* information it relies on |
| **Random Forest, PLS-DA, LASSO, ordinal, NMF** | **IMPUTE** before fitting | these need a complete matrix |

**You usually don't do anything manually** — it is automatic. Every method
declares `handles_missing`. MOFA-type methods (`handles_missing = True`) receive
the NaN-carrying matrix as-is; the others (`handles_missing = False`) impute
just-in-time when you call `.fit()`. The dataset's stored matrix always keeps
NaNs, so the same preprocessed data feeds both kinds of method correctly.

**Choosing how to impute** (for methods that need it):

| Strategy | When to use | Code |
|----------|-------------|------|
| `metaboanalyst` (default) | values missing because they're below detection limit (very common in metabolomics/proteomics) | `RandomForest(impute="metaboanalyst")` |
| `imputepca` | missingness is more random and features are correlated; matches the lab's default | `RandomForest(impute="imputepca")` |
| `remove_all_missing` | you want zero imputed values; drops any feature with a gap | `RandomForest(impute="remove_all_missing")` |

To **force-propagate** for a method that normally imputes, you generally
shouldn't — but you can filter features first so there are no NaNs to impute
(see Decision 5).

---

## Decision 3 — Transform (log)

Most omics data is right-skewed and should be log-transformed.

| Your data | Choose | Note |
|-----------|--------|------|
| Proteomics / intensities that can be 0 | `log2` (which is `log2(x+1)`) | zero-safe; the default & matches the validated MOFA pipeline |
| Metabolomite concentrations (all positive) | `log10` | values ≤ 0 become NaN |
| Already log-transformed upstream | `none` | don't double-transform |

```python
from ml_multiomics import Preprocessor, Profile

# override the transform for all blocks
Preprocessor(profile=Profile(transform="log10", normalize="zscore")).run(ds)
```

> Note: the default `log2` adds a `+1` pseudocount (zero-safe). The lab's Julia
> tool uses plain `log2` (zeros → missing). Both are valid; the difference is
> only the pseudocount. Use plain log only if you have a reason and no zeros.

---

## Decision 4 — Normalize (scale)

| Choose | When |
|--------|------|
| `zscore` (default) | general use; comparability across features; matches the lab |
| `none` | data already normalized upstream |
| pareto* | metabolomics where you want to keep more of the biological variance |

\*Pareto is available as a documented option but not the default; ask if you need
it wired into a profile.

```python
Preprocessor(profile=Profile(transform="log2", normalize="zscore")).run(ds)
```

**Important:** scaling happens **once**, here. Do not also turn on a method's own
internal scaling — that double-scales. The methods in this library are configured
not to re-scale.

---

## Decision 5 — Aggregate timepoints, or keep them?

Only relevant if you have multiple timepoints per unit (e.g. bioreactor runs).

| Your question | Do |
|---------------|----|
| "What distinguishes the conditions at a snapshot?" (cross-sectional) | aggregate to one row per unit (e.g. per bioreactor per phase) before analysis |
| "How do things change over time?" (dynamics) | keep timepoints, and **group by the unit** (Decision 1) so they don't leak |

Feature filtering thresholds (variance, missingness) can be tuned in the
`Profile` if you have very sparse data:

```python
Profile(transform="log2", normalize="zscore", variance_min=1e-8, max_missing_frac=0.5)
```

---

## Decision 6 — Target and method

### What are you predicting? (target type)

| Your outcome | target type | Example |
|--------------|-------------|---------|
| Unordered categories | `nominal` | strain A vs B vs C |
| **Ordered** categories | `ordinal` | Green < Ripe < Overripe |
| A number | `continuous` | psilocybin yield |
| Nothing (exploration) | `none` | find structure / modules |

```python
ds.set_target("ripeness", type="ordinal", column="stage",
              ordinal_order=["Green", "Ripe", "Overripe"])
# or continuous from a column / Series:
ds.set_target("yield", type="continuous", values=yield_series)
```

### Which method?

| You want to… | Use | Target |
|--------------|-----|--------|
| Explore structure, find latent factors across omics | MOFA / PCA / NMF | none |
| Find co-varying feature modules | WGCNA | none |
| Predict a category, get feature importance | **Random Forest** | nominal/ordinal |
| Predict a number (e.g. yield) | **Random Forest (regression)** | continuous |
| Classify with built-in feature selection | sparse PLS-DA | nominal |
| Integrate multiple omics for classification | DIABLO | nominal |
| Predict an ordered category | ordinal regression | ordinal |
| Regularized linear prediction (small n) | LASSO / ElasticNet | continuous |

> Currently implemented: Random Forest, sparse PLS-DA, DIABLO (multi-block),
> WGCNA (modules + reduction), LASSO / ElasticNet (regularized linear), NMF
> (parts-based reduction), and Ordinal regression (`pip install ml_multiomics[ordinal]`).
> MOFA is being ported. The patterns below apply to all of them.

### Dimensionality reduction (the "reduce → predict" pattern)

When you have far more features than samples (p ≫ n — almost always, in omics),
it is often better to **reduce first, then predict**. Unsupervised "reducer"
methods (WGCNA now; NMF, PCA, MOFA to follow) collapse thousands of features
into a handful of factors/modules, and that small matrix feeds a supervised
method:

```python
from ml_multiomics import WGCNA, RandomForest
import numpy as np

# WGCNA collapses correlated features into module eigengenes (no leakage:
# modules are built without the labels)
wg = WGCNA(corr_method="spearman").fit(X, y, target_type="ordinal")
reduced = wg.reduce(strategy="eigengenes_and_hubs")   # samples x (modules + hubs)

# now predict on the small reduced matrix
groups = np.arange(len(y))
rf = RandomForest().fit(reduced, y, target_type="nominal")
print(rf.cross_validate(reduced, y, groups=groups, target_type="nominal")["accuracy"])
```

This is exactly how the MOFA factor-yield workflow operates (reduce to factors,
then regress yield on factors). **NMF** does the same — `NMF().fit(X).reduce()`
returns its `W` scores (samples × factors); note NMF needs non-negative input, so
preprocess with `normalize="none"` (z-scored data is rejected). MOFA/PCA factor
scores will follow. You can also add `reduced` back to an OmicsDataset as a new
block.

### Workflow recipes by dataset type

This package is built for **low-n, high-p** data (few samples, many features —
typical omics). Pick the recipe that matches your situation:

**1. Low-n, high-p, predict an outcome (the common case).**
Reduce first, then predict — don't throw 5,000 features at a model with 9
samples. `WGCNA`/`NMF`/`MOFA` → factors, then `RandomForest`/`Lasso` on the
factors. Or use a method with built-in selection (`SparsePLSDA`, `Lasso`).
Always grouping-aware CV. Avoid deep learning at this n.

**2. Multiple omics layers, "what separates the groups?"**
`DIABLO` — it integrates blocks and finds shared discriminative signal, with
per-block VIP and block-correlation structure.

**3. Ordered categorical outcome (Green < Ripe < Overripe, dose levels).**
`Ordinal` regression — respects the ordering; report MAE (ordinal distance) as
well as accuracy.

**4. Continuous outcome (yield, titer, rate).**
`Lasso`/`ElasticNet` (sparse, interpretable) or `RandomForest` regression
(non-linear). For p≫n, reduce first (recipe 1).

**5. No labels — just understand the data.**
`PCA`/`NMF`/`MOFA` (factors) or `WGCNA` (modules). Inspect loadings/eigengenes to
see what drives each factor/module.

**6. Repeated measures (timepoints per bioreactor) and you want dynamics.**
Keep timepoints, group by the unit (Decision 1); for a static snapshot instead,
aggregate per unit/phase (Decision 5).

In every recipe: set the **grouping** (Decision 1), decide **propagate vs
impute** based on the method (Decision 2), and read p-values with their
**resolution**.

---

## Running a method

```python
from ml_multiomics import RandomForest

X = ds.get("proteomics")
y = ds.sample_meta["stage"].to_numpy()
groups = np.arange(len(y))          # Decision 1

rf = RandomForest().fit(X, y, target_type="nominal")

cv = rf.cross_validate(X, y, groups=groups, target_type="nominal")  # grouping-aware
print(cv["accuracy"])               # or cv["r2"] for regression

rf.importances(top_n=20)            # which features matter
pt = rf.permutation_test(X, y, groups=groups, n_permutations=200)
print(pt["p_value"], pt["resolution"]["finest_two_sided_p"])
```

Switching to regression is just the target type:

```python
rf = RandomForest().fit(X, y_continuous, target_type="continuous")
cv = rf.cross_validate(X, y_continuous, groups=groups, target_type="continuous")
print(cv["r2"], cv["rmse"])
```

---

## Reading results honestly

- **Always read a p-value next to its resolution.** If `finest_two_sided_p` is
  0.1 because you only have a few groups, a p of 0.1 is the *best possible* — not
  evidence of nothing. Small designs have hard floors.
- **Lead with effect size**, not the p-value: how big is the difference / how
  good is the R²?
- **Stability over single runs**: a feature "selected in 90/100 resamples" is
  more trustworthy than one that tops a single fit.
- **Grouped accuracy is the honest number.** If grouped-CV accuracy is much lower
  than a naive (ungrouped) number you saw elsewhere, the grouped one is right and
  the other was leaking.

---

## Two worked examples

### Example A — independent replicates, ordered categories (banana)

```python
import numpy as np, pandas as pd
from ml_multiomics import OmicsDataset, Preprocessor, RandomForest
from ml_multiomics.core import parse_delimited

df = pd.read_csv("data/badata-proteomics-imputed.csv").set_index("Sample").drop(columns=["Groups"])
ds = OmicsDataset("banana"); ds.add_block("proteomics", df, omics_type="proteomics")
ds.set_sample_metadata(parse_delimited(df.index, sep="-", names=("stage","replicate")))
Preprocessor().run(ds)                              # log2 + z-score

X, y = ds.get("proteomics"), ds.sample_meta["stage"].to_numpy()
groups = np.arange(len(y))                          # each replicate independent
rf = RandomForest().fit(X, y, target_type="nominal")
print(rf.cross_validate(X, y, groups=groups, target_type="nominal")["accuracy"])
```

### Example B — bioreactor, predict yield, repeated timepoints

```python
from ml_multiomics.core import parse_bioreactor_ids

ds = OmicsDataset("psilo"); ds.add_block("proteomics", prot, omics_type="proteomics")
ds.set_sample_metadata(parse_bioreactor_ids(prot.index))
Preprocessor().run(ds)

X = ds.get("proteomics")
y = yield_series.reindex(X.index).to_numpy()        # continuous yield, aligned by sample
groups = ds.groups("bioreactor").to_numpy()         # timepoints grouped by reactor -> no leak

rf = RandomForest().fit(X, y, target_type="continuous")
print(rf.cross_validate(X, y, groups=groups, target_type="continuous")["r2"])
# For the SAME data with MOFA: do NOT impute — MOFA takes the NaN-carrying matrix directly.
```

---

## Reproducibility checklist

- [ ] Record your **grouping unit** choice and why (Decision 1).
- [ ] Record **propagate vs impute** and the imputation strategy (Decision 2).
- [ ] Record **transform** and **normalize** choices (Decisions 3–4).
- [ ] Record whether timepoints were **aggregated** (Decision 5).
- [ ] Record the **target type** and **method** (Decision 6).
- [ ] Keep seeds fixed (defaults are deterministic) and save inputs + outputs.
- [ ] Re-run `tests/crosscheck/run_all.py` to confirm the preprocessing still
      matches the lab's reference code on your machine.
