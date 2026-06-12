# Tutorial — install, verify it yourself, run an analysis

A hands-on walkthrough of the consolidated `ml_multiomics` library. By the end
you will have (1) installed it, (2) re-verified the preprocessing against the
lab's own R/Julia code yourself, and (3) run a grouping-aware analysis end to
end. For the rationale behind every choice, see
[ASSUMPTIONS_AND_CHOICES.md](ASSUMPTIONS_AND_CHOICES.md).

All commands assume MSYS2 bash and the repo's virtualenv. Adjust `./venv/...`
to your environment.

---

## 1. Install

```bash
cd ml_multiomics
python -m venv .venv --upgrade-deps      # if you don't already have one
./venv/Scripts/pip install -e .          # editable install
```

Core dependencies are Python 3.12 + numpy / pandas / scipy / scikit-learn.
Optional: `shap` (SHAP feature importance), `mord` (ordinal regression). R 4.4
(`missMDA`) and Julia 1.11 are needed *only* to re-run the numerical
cross-checks (step 2) — not to use the library.

---

## 2. Verify it yourself

The preprocessing primitives are checked for **numerical parity against the
lab's actual R and Julia code**. Re-run the whole suite:

```bash
./venv/Scripts/python tests/crosscheck/run_all.py
```

Expected: `4/4 suites passed`, and a fresh `tests/crosscheck/RESULTS.md`. What it
verifies (see `tests/crosscheck/README.md` for the per-primitive tables):

- **Structural** (`test_scaffold.py`): block alignment by ID, the missingness
  gate, zero-leakage grouped CV on the real banana + psilocybin data.
- **vs IdeaBio.R / IdeaBio.jl**: z-score, log, MetaboAnalyst imputation match at
  machine precision; documented deliberate divergences are flagged, not silent.
- **vs missMDA::imputePCA**: the pure-Python imputePCA matches to ~1e-14.

If you don't have R/Julia, open the committed `RESULTS.md` to read the numbers
from the last run.

Run the method smoke tests too:

```bash
./venv/Scripts/python tests/test_scaffold.py
./venv/Scripts/python tests/test_methods.py
```

---

## 3. Quickstart — a grouping-aware analysis

This is a complete, runnable example (it is the verified snippet from the method
smoke test). It loads an omics block, attaches metadata and a target,
preprocesses (missing-aware), and runs Random Forest with **leave-one-group-out**
cross-validation.

```python
import pandas as pd, numpy as np
from ml_multiomics import OmicsDataset, Preprocessor, RandomForest
from ml_multiomics.core import parse_delimited

# 1. load an omics block into the container
df = pd.read_csv("data/badata-proteomics-imputed.csv").set_index("Sample").drop(columns=["Groups"])
ds = OmicsDataset(name="banana")
ds.add_block("proteomics", df, omics_type="proteomics")

# 2. attach sample metadata (pluggable parser) + a target
ds.set_sample_metadata(parse_delimited(df.index, sep="-", names=("stage", "replicate")))

# 3. preprocess (missing-aware: log2 + z-score, NaN preserved, scaled ONCE)
Preprocessor().run(ds)

# 4. fit + grouping-aware CV. groups = the independent unit; here each banana
#    replicate is independent, so each sample is its own group (leave-one-out).
X, y = ds.get("proteomics"), ds.sample_meta["stage"].to_numpy()
groups = np.arange(len(y))
rf = RandomForest().fit(X, y, target_type="nominal")
cv = rf.cross_validate(X, y, groups=groups, target_type="nominal")

print("task:", rf.task_)                              # classification
print("grouped-CV accuracy:", round(cv["accuracy"], 3))   # ~0.889
print(rf.importances(top_n=3).to_string(index=False))
```

### Bioreactor (psilocybin) data: grouping matters

For data with repeated measures (timepoints within a bioreactor), the grouping
vector is what prevents leakage. Parse the IDs and group by **bioreactor**:

```python
from ml_multiomics.core import parse_bioreactor_ids

meta = parse_bioreactor_ids(block.index)   # F503_C1_R1_T1 -> condition/replicate/timepoint
ds.set_sample_metadata(meta)

# group by bioreactor so no reactor's timepoints split across train/test
groups = ds.groups("bioreactor").to_numpy()
cv = rf.cross_validate(X, y, groups=groups, target_type="continuous")  # e.g. yield
```

### Read a p-value honestly

```python
from ml_multiomics.validation import permutation_resolution
res = permutation_resolution(groups, y)
print(res["finest_two_sided_p"])   # the smallest p the design can even produce
```

---

## 4. Where things live

```
src/ml_multiomics/
  core/          OmicsDataset, Block, TargetSpec, metadata parsers
  preprocessing/ missing-aware primitives, imputation, Preprocessor
  methods/       supervised/ (RandomForest, ...), base.py (BaseMethod)
  validation/    grouping-aware CV / permutation / bootstrap
tests/
  test_scaffold.py, test_methods.py
  crosscheck/    R & Julia parity harnesses + RESULTS.md
docs/
  ASSUMPTIONS_AND_CHOICES.md   (the "why" — for review/publication)
  TUTORIAL.md                  (this file)
```

## 5. Status

Implemented & verified: data model, missing-aware preprocessing (lab-parity),
grouping-aware resampling, Random Forest (classification + regression), sparse
PLS-DA (with stability selection), DIABLO (multi-block integration), WGCNA
(modules + dimensionality reduction), LASSO/ElasticNet, NMF (parts-based
reduction). Queued: ordinal regression, MOFA (lifted from `ml_psi_mofa`), and an
end-to-end Quarto report. See [../CONSOLIDATION_PLAN.md](../CONSOLIDATION_PLAN.md).
