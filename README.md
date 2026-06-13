# ml_multiomics

A reusable, **pure-Python** library for multi-omics machine learning, built for
the **low-sample, high-feature (low-n, high-p)** regime typical of omics studies.
Preprocessing matches the lab's standard conventions (verified numerically against
the IdeaBio R/Julia code), and every method validates with **leakage-free,
grouping-aware** cross-validation.

> **📖 Start here**
> - **Not a coder?** Open this repo in Claude Code and ask it to analyze your data — the [multiomics-analysis skill](.claude/skills/multiomics-analysis/SKILL.md) walks it through the whole workflow. See [examples/skill_walkthrough.py](examples/skill_walkthrough.py) for what that produces.
> - [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — **run your own dataset**: a plain-language decision guide (grouping, NaNs, transforms, method choice) with the code for each option. Reproducible without AI.
> - [docs/TUTORIAL.md](docs/TUTORIAL.md) — install, **verify it yourself**, run an analysis end to end.
> - [docs/ASSUMPTIONS_AND_CHOICES.md](docs/ASSUMPTIONS_AND_CHOICES.md) — every methodological choice + rationale, for review and publication.
> - [tests/crosscheck/README.md](tests/crosscheck/README.md) — numerical parity vs the lab's R/Julia code ([latest results](tests/crosscheck/RESULTS.md)).
> - [examples/psilocybin_report/](examples/psilocybin_report/) — a Quarto "Part 3: Omics & ML" report (predict a metabolite yield from proteomics).

## What it does

- **One container, any omics.** `OmicsDataset` holds N blocks (single-omics = 1),
  aligns by sample ID, tracks provenance. Loaders handle raw Excel/CSV.
- **Missing-aware preprocessing** (log + z-score), scaled **once**, matching the
  lab's conventions (z-score, log, MetaboAnalyst / imputePCA imputation — all
  verified to machine precision against the actual R/Julia code).
- **Methods on one interface** (`BaseMethod`), each with grouping-aware CV +
  group-level permutation + a `handles_missing` gate:
  - *supervised:* RandomForest (classify/regress), SparsePLSDA, DIABLO
    (multi-block), Lasso / ElasticNet, Ordinal regression
  - *reducers:* PCA, NMF, WGCNA — `.reduce()` → a samples×factor matrix for the
    **reduce → predict** pattern (essential for p≫n)
- **Standard analyses** (`analysis`): `compute_volcano` (Welch t-test + fold
  change + FDR), `anova_tukey`, `ora` (hypergeometric) — pure-Python ports of the
  lab's DE/enrichment, cross-checked against the R.

## Install

```bash
python -m venv venv --upgrade-deps
venv/Scripts/pip install -e .                 # core
venv/Scripts/pip install -e .[ordinal,shap,report]   # optional: mord / shap / Quarto rendering
```
Core deps: numpy, pandas, scipy, scikit-learn, statsmodels. Python ≥ 3.9.

## Test / verify

One command runs the structural tests, method/analysis smoke tests, and the
numerical cross-checks against the lab's actual R/Julia code, and writes a
captured report:

```bash
venv/Scripts/python tests/crosscheck/run_all.py     # -> 7/7 suites, writes RESULTS.md
```

## Package layout

```
src/ml_multiomics/
  core/          OmicsDataset, Block, TargetSpec, metadata parsers
  preprocessing/ missing-aware primitives, imputation, Preprocessor
  methods/       base.py (BaseMethod) + supervised/ + unsupervised/ (reducers)
  validation/    grouping-aware CV / permutation / bootstrap
  analysis/      differential expression + enrichment (lab-matched)
tests/           smoke tests + crosscheck/ (R & Julia parity, RESULTS.md)
docs/            USER_GUIDE, TUTORIAL, ASSUMPTIONS_AND_CHOICES
examples/        skill_walkthrough.py, psilocybin_report/ (Quarto)
```

## Status

Consolidated and verified: data model, preprocessing (lab-parity), grouping-aware
validation, 9 methods, standard DE/enrichment, a skill, and a rendering Quarto
report. MOFA (single-group) is the remaining method to fold in; see
[CONSOLIDATION_PLAN.md](CONSOLIDATION_PLAN.md).

## License

Apache-2.0. Author: Subaru Muroi.
