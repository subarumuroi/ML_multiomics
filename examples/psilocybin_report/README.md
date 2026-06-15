# Psilocybin "Part 3 — Omics & ML" report

A parameterized, **interpretation-framed** Quarto report that relates phase-3
**proteomics** to a metabolite **yield** using `ml_multiomics`. It extends the
IDEA Bio fermentation report (Parts 1–2) with the omics/ML analysis their
template stubs out.

Framing: at ~28 bioreactors this is **hypothesis-generation, not a predictor
leaderboard**. Cross-validation (leave-one-bioreactor-out) is a *guardrail*
(beats the predict-the-mean baseline? overfitting?), the descriptive model panel
is not a ranking, and reducing the ~4,000-protein block to a few PCA/NMF factors
is principled, interpretable preprocessing.

## Files
- `analysis.py` — the validated analysis engine (`run_yield_analysis(compound)`).
  Runnable on its own: `python analysis.py`.
- `psilocybin_omics_ml.qmd` — the report; its code cells call the engine.

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
Override the paths via `run_yield_analysis(master_path=..., yields_path=...)`.
In the consolidated architecture this report lives in the *project repo*
(`ml_psi_mofa`), which depends on the `ml_multiomics` library.

Rendered `.html` outputs are git-ignored (regenerate with the command above).
