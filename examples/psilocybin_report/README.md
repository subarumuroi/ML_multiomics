# Psilocybin multi-omics report (standard analysis -> integrative ML)

A parameterized, **declared, self-documenting** Quarto report over the full
multi-block psilocybin data (proteomics + intracellular metabolomics CCM/PSI +
bioreactor) using `ml_multiomics`. It first **replicates the standard analysis**
(QC -> preprocessing -> differential expression -> enrichment), then crosses a
marked **ML divergence point** into two integrative assessments: continuous
**yield** (regression) and **construct C1 vs C2** (nominal).

Framing: at ~28 bioreactors this is **hypothesis-generation, not a predictor
leaderboard**. Each result is judged by **permutation signal** (vs the design's
resolution floor) + **bootstrap stability** + **biological annotation**; CV is a
binary overfitting sanity flag only. Proteomics (~4,375) dwarfs the metabolite
blocks, so it is **auto-reduced** (WGCNA/PCA/NMF) before integration — and the
report shows the naive-vs-reduced *stability* contrast that justifies it.
`met_ext_pb` is **excluded as a predictor** (unreliable as features) and only
sources the yield targets. The analysis is authored as an `AnalysisSpec` (the
user's declared decisions) and executed by the library; every step is
provenance-tracked.

## Files
- `analysis.py` — the engine (`run_psilocybin_report(compound, ...)`); declares
  the `AnalysisSpec`s and calls the library. Runnable on its own: `python analysis.py`
  (a reduced-compute smoke). Compute knobs: `n_permutations`, `stability_bootstrap`,
  `reducers`, `run_integration`, `run_construct`.
- `psilocybin_omics_ml.qmd` — the report; its cells render the engine output.

## Render

Quarto needs a Python with the jupyter stack. Install the extra and point Quarto
at the venv:

```bash
pip install -e .[report]          # ipykernel, nbclient, nbformat, jupyter_client, pyyaml
export QUARTO_PYTHON="$(pwd)/venv/Scripts/python.exe"   # the venv with ml_multiomics + [report]
quarto render examples/psilocybin_report/psilocybin_omics_ml.qmd --to html
```

## Choose the target metabolite

Default is `psilocybin_ext`. Change the `compound` parameter at the top of the
`.qmd`, or pass it at render time:

```bash
quarto render psilocybin_omics_ml.qmd -P compound:tryptamine_ext --output tryptamine_omics_ml.html
```

Available targets (from the yields file) include `psilocybin_ext`,
`tryptamine_ext`, `psilocin_ext`, `Baeocystine_ext`, `serotonin_ext`,
`norbaeocystin_ext`, `tryptophan_ext`. Internal metabolites / growth rate can be
targeted the same way once present in the yields table.

## Data

The engine reads the psilocybin project data by default from
`ml_psi_mofa/data/` (master_multiomics.csv + pseudobatched-external-yields-rates.csv).
Override the paths via `run_psilocybin_report(master_path=..., yields_path=...)`.
In the consolidated architecture this report lives in the *project repo*
(`ml_psi_mofa`), which depends on the `ml_multiomics` library.

Rendered `.html` outputs are git-ignored (regenerate with the command above).
