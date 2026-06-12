"""
run_crosscheck_r.py
===================
Numerical equivalence check: my Python preprocessing primitives vs the lab's
ACTUAL R functions (sourced from StandardOmicAnalyses/IdeaBio.R).

Writes a fixed input, runs ref_r.R (which sources the real lab code), then runs
the Python primitives on the same input and diffs. Reports MATCH / DIVERGE per
primitive with the max absolute difference.

Run:  ./venv/Scripts/python tests/crosscheck/run_crosscheck_r.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_multiomics.preprocessing import (
    zscore, log10_transform, metaboanalyst_impute,
)

HERE = Path(__file__).resolve().parent
LAB_R = Path("C:/Users/uqkmuroi/gitcode/ideabio/StandardOmicAnalyses/R")
RSCRIPT = "Rscript"

# Fixed, deterministic input: 6 samples x 4 features, all positive.
SAMPLES = ["A1", "A2", "A3", "B1", "B2", "B3"]
FEATURES = ["f1", "f2", "f3", "f4"]
X = np.array([
    [10.0,  2.0, 100.0,  5.0],
    [12.0,  3.0, 110.0,  6.0],
    [ 9.0,  2.5,  90.0,  4.0],
    [50.0, 20.0, 200.0, 50.0],
    [55.0, 22.0, 210.0, 55.0],
    [48.0, 18.0, 190.0, 45.0],
])
GROUPS = ["A", "A", "A", "B", "B", "B"]
# missing cells: (row1,col f2) and (row4,col f4)
MISSING_CELLS = [(1, "f2"), (4, "f4")]


def _fmt(diff):
    return "n/a" if diff is None or np.isnan(diff) else f"{diff:.2e}"


def main():
    xc = pd.DataFrame(X, index=SAMPLES, columns=FEATURES)
    xm = xc.copy()
    for r, c in MISSING_CELLS:
        xm.iloc[r, xm.columns.get_loc(c)] = np.nan

    xc.to_csv(HERE / "x_complete.csv")
    xm.to_csv(HERE / "x_missing.csv", na_rep="NA")
    pd.DataFrame({"group": GROUPS}, index=SAMPLES).to_csv(HERE / "groups.csv")

    # --- run the lab's actual R functions ---
    print("Running lab R reference (sourcing IdeaBio.R functions)...")
    res = subprocess.run(
        [RSCRIPT, str(HERE / "ref_r.R"), str(HERE), str(LAB_R)],
        capture_output=True, text=True,
    )
    print(res.stdout.strip())
    if res.returncode != 0:
        print("R STDERR:\n", res.stderr)
        sys.exit(2)

    def load_r(name):
        return pd.read_csv(HERE / name, index_col=0)

    rows = []

    # 1. z-score, complete data -> expect MATCH
    py = zscore(xc)
    r = load_r("r_zscore_complete.csv")
    d = float(np.nanmax(np.abs(py.to_numpy() - r.to_numpy())))
    rows.append(("zscore (complete)", "MATCH" if d < 1e-9 else "DIVERGE", _fmt(d),
                 "expect match"))

    # 2. z-score, missing data -> expect DIVERGE (R propagates NaN per column;
    #    Python/Julia skip NaN). Report which columns differ.
    py = zscore(xm)
    r = load_r("r_zscore_missing.csv")
    pa, ra = py.to_numpy(), r.to_numpy()
    both_nan = np.isnan(pa) & np.isnan(ra)
    diff_mask = ~both_nan & ~np.isclose(pa, ra, equal_nan=False)
    n_diff = int(np.nansum(diff_mask))
    rows.append(("zscore (missing)", "DIVERGE (expected)" if n_diff else "MATCH",
                 f"{n_diff} cells differ",
                 "R propagates NaN over whole column; Python skips NaN (matches Julia)"))

    # 3. log10, complete -> expect MATCH
    py = log10_transform(xc)
    r = load_r("r_log10_complete.csv")
    d = float(np.nanmax(np.abs(py.to_numpy() - r.to_numpy())))
    rows.append(("log10 (complete)", "MATCH" if d < 1e-9 else "DIVERGE", _fmt(d),
                 "lab DE transform"))

    # 4. MetaboAnalyst imputation -> expect MATCH (both 0.2 x min positive)
    py = metaboanalyst_impute(xm)
    r = load_r("r_metaboanalyst.csv")
    d = float(np.nanmax(np.abs(py.to_numpy() - r.to_numpy())))
    rows.append(("metaboanalyst impute", "MATCH" if d < 1e-9 else "DIVERGE", _fmt(d),
                 "0.2 x min(positive) per feature"))

    # 5. imputePCA per group -> Python has NO equivalent yet (GAP)
    ip = HERE / "r_imputepca.csv"
    rows.append(("imputePCA-by-group", "GAP", "R-only",
                 "R default imputation; not yet ported to Python"
                 + ("" if ip.exists() else " (R run also failed)")))

    # --- report ---
    print("\n" + "=" * 92)
    print(f"{'primitive':22s} {'result':20s} {'detail':16s} note")
    print("-" * 92)
    for name, status, detail, note in rows:
        print(f"{name:22s} {status:20s} {detail:16s} {note}")
    print("=" * 92)

    # fail only if something that should MATCH diverged
    hard_fail = any(
        s.startswith("DIVERGE") and "expected" not in s
        for _, s, _, _ in rows
    )
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
