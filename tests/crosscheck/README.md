# Preprocessing cross-checks — numerical parity vs the lab's R & Julia code

**Purpose.** Before any ML method is built on top of the preprocessing layer, we
verify that the pure-Python primitives reproduce the lab's *standard analyses*
numerically. Each cross-check runs the lab's **actual** reference code
(`StandardOmicAnalyses` / IdeaBio.R, and `IdeaBio.jl`) on a fixed input and diffs
it against the Python implementation. "Trust" here means *machine-precision
agreement*, not "looks similar."

This directory is the audit trail. To re-verify everything in one command:

```bash
./venv/Scripts/python tests/crosscheck/run_all.py      # writes RESULTS.md
```

Individual harnesses:

```bash
./venv/Scripts/python tests/crosscheck/run_crosscheck_r.py         # vs IdeaBio.R
./venv/Scripts/python tests/crosscheck/run_crosscheck_jl.py        # vs IdeaBio.jl
./venv/Scripts/python tests/crosscheck/run_crosscheck_imputepca.py # vs missMDA::imputePCA
```

`RESULTS.md` (committed) holds the captured output of the most recent `run_all`,
so the numbers can be reviewed without an R/Julia toolchain.

---

## Environment used

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.12 (repo venv) | numpy / pandas / scipy / sklearn |
| R | 4.4.0 | `missMDA` (imputePCA), `assertthat` |
| Julia | 1.11.3 | stdlib only (`Statistics`, `DelimitedFiles`) |
| shell | MSYS2 bash | |

---

## What each harness checks

### `run_crosscheck_r.py` → sources IdeaBio.R (`normalisations.R`, `imputations.R`)
Runs the lab's *actual* R functions via `ref_r.R` and diffs vs Python.

| primitive | reference R fn | result | note |
|-----------|----------------|--------|------|
| z-score (complete) | `zscore(x, 2)` | **MATCH** ~3e-15 | identical |
| z-score (missing) | `zscore(x, 2)` | **DIVERGE** (by design) | R has no `na.rm` → one NaN nukes the whole feature; Python skips NaN (matches Julia) |
| log10 (complete) | base `log10` | **MATCH** ~5e-15 | lab DE transform |
| MetaboAnalyst impute | `impute_missing_metaboanalyst` | **MATCH** 0.0 | `0.2 × min(positive)` per feature |
| imputePCA-by-group | `impute_matrix_by_group` | see imputePCA harness | machine-precision parity (below) |

### `run_crosscheck_jl.py` → reproduces IdeaBio.jl `src/omics` function bodies
The full IdeaBio.jl module is not loaded (its Makie/RCall/CairoMakie dependency
tree is too heavy for a quick check); instead `ref_jl.jl` reproduces the exact
numeric bodies of `normalise_zscore`, `transform_log2/10`,
`impute_default_metaboanalyst` using only Julia stdlib (whose `std` is the same
corrected n-1 as `StatsBase.zscore`).

| primitive | result | note |
|-----------|--------|------|
| z-score (complete) | **MATCH** 0.0 | Julia == Python == R |
| z-score (missing) | **MATCH** 0.0 | confirms Python matches Julia's skip-NaN (where R diverged) |
| log2: Python `log2(x+1)` vs Julia `log2(x)` | **DIVERGE** 0.585 | pure `+1` pseudocount — a documented default choice, not a bug |
| log2: plain vs Julia | **MATCH** 0.0 | proves the divergence is *only* the pseudocount |
| log10 (complete) | **MATCH** 0.0 | identical |
| MetaboAnalyst impute | **MATCH** 0.0 | identical on positive data |

### `run_crosscheck_imputepca.py` → `missMDA::imputePCA` + `impute_matrix_by_group`
The Python `imputepca` / `imputepca_by_group` are a faithful port of missMDA's
regularized iterative PCA (algorithm dumped from missMDA source): row-weighted
`svd.triplet`, `sigma2` residual-variance shrinkage `λ=(vs²−σ²)/vs` capped at
`vs[ncp+1]²`, deterministic zero-init EM loop, `scale=TRUE`, `ncp=2`.

| check | result | max abs diff |
|-------|--------|--------------|
| imputePCA whole matrix | **MATCH** | 5e-14 |
| imputePCA imputed cells only | **MATCH** | 3e-14 |
| `impute_matrix_by_group` (per-group) | **MATCH** | 6e-14 |

Tolerance for "MATCH" = `1e-6`; observed diffs are ~1e-14 (machine precision).

---

## Decisions that came out of these cross-checks

1. **z-score on missing data → skip-NaN is canonical.** R's `zscore` propagates
   NaN over a whole feature; Julia and Python skip NaN. We follow Julia because
   skip-NaN is *required* for MOFA (which receives z-scored-but-unimputed data) —
   R's behavior would turn any partially-missing feature into all-NaN. The lab's
   own R and Julia tools disagree here.
2. **log2 default = `log2(x+1)`** (mofa_prep convention; zero-safe; matches the
   validated psilocybin pipeline). IdeaBio.jl uses plain `log2(x)` (zeros →
   missing). Both available as profile options; the divergence is *only* the
   pseudocount.
3. **MetaboAnalyst**: Python/R use `0.2 × min(positive)`, Julia uses
   `0.2 × min(non-missing)` — identical on positive intensity data.
4. **imputePCA** is ported to pure Python at machine-precision parity, so the
   lab's default imputation needs no R dependency.

**Net:** every preprocessing primitive is either numerically identical to the
lab's actual code, or a documented, deliberate choice.

---

## Notes for reviewers

- Inputs are fixed and deterministic (hard-coded matrices / seeded RNG), so
  results are reproducible. Generated input/output CSVs are git-ignored; only the
  harnesses, this README, and `RESULTS.md` are committed.
- A "DIVERGE (by design / decision)" row is expected and does **not** fail the
  harness; only an unexpected divergence on a should-MATCH primitive fails.
- The structural (non-numeric) foundation test lives one level up:
  `tests/test_scaffold.py` (container alignment, missingness gate, zero-leakage
  grouped CV on real banana + psilocybin data).
