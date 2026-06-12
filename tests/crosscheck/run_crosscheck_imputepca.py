"""
run_crosscheck_imputepca.py
==========================
Numerical parity check: my pure-Python regularized iterative PCA imputation vs
the lab's actual missMDA::imputePCA (whole matrix) and IdeaBio.R
impute_matrix_by_group (per-group).

Run:  ./venv/Scripts/python tests/crosscheck/run_crosscheck_imputepca.py
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

from ml_multiomics.preprocessing import imputepca, imputepca_by_group

HERE = Path(__file__).resolve().parent
LAB_R = Path("C:/Users/uqkmuroi/gitcode/ideabio/StandardOmicAnalyses/R")
RSCRIPT = "Rscript"

# 12 samples (2 groups of 6) x 6 features. Deterministic; a handful of NaN,
# none all-missing within a group/feature.
N_PER_GROUP, N_FEAT = 6, 6
rng = np.random.default_rng(7)
base = rng.normal(loc=50, scale=10, size=(2 * N_PER_GROUP, N_FEAT))
base[6:, :] += 15.0  # group B offset so groups differ
SAMPLES = [f"A{i}" for i in range(1, 7)] + [f"B{i}" for i in range(1, 7)]
FEATURES = [f"f{j}" for j in range(1, N_FEAT + 1)]
GROUPS = ["A"] * 6 + ["B"] * 6
# missing cells (row, col), spread across both groups
for r, c in [(0, 1), (2, 3), (4, 0), (7, 2), (9, 5), (11, 1)]:
    base[r, c] = np.nan
XM = pd.DataFrame(base, index=SAMPLES, columns=FEATURES)


def maxdiff(a, b):
    return float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def main():
    XM.to_csv(HERE / "ipca_x.csv", na_rep="NA")
    pd.DataFrame({"group": GROUPS}, index=SAMPLES).to_csv(HERE / "ipca_groups.csv")

    print("Running lab R imputePCA reference...")
    res = subprocess.run([RSCRIPT, str(HERE / "ref_imputepca.R"), str(HERE), str(LAB_R)],
                         capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        print("R STDERR:\n", res.stderr)
        sys.exit(2)

    r_whole = pd.read_csv(HERE / "r_ipca_whole.csv", index_col=0)
    r_bg = pd.read_csv(HERE / "r_ipca_bygroup.csv", index_col=0)

    py_whole = imputepca(XM)                         # whole-matrix
    py_bg = imputepca_by_group(XM, GROUPS)           # per-group

    # align column/row order to R output
    py_whole = py_whole.loc[r_whole.index, r_whole.columns]
    py_bg = py_bg.loc[r_bg.index, r_bg.columns]

    d_whole = maxdiff(py_whole.to_numpy(), r_whole.to_numpy())
    d_bg = maxdiff(py_bg.to_numpy(), r_bg.to_numpy())

    # only the imputed cells matter most; observed cells should be identical too
    miss = XM.isna().to_numpy()
    d_whole_missing = maxdiff(py_whole.to_numpy()[miss], r_whole.to_numpy()[miss])

    TOL = 1e-6
    rows = [
        ("imputePCA whole (all cells)", d_whole),
        ("imputePCA whole (imputed cells)", d_whole_missing),
        ("impute_matrix_by_group", d_bg),
    ]
    print("\n" + "=" * 70)
    print(f"{'check':38s} {'max abs diff':14s} result")
    print("-" * 70)
    ok = True
    for name, d in rows:
        res_s = "MATCH" if d < TOL else "DIVERGE"
        if d >= TOL:
            ok = False
        print(f"{name:38s} {d:.3e}      {res_s}")
    print("=" * 70)
    print(f"tolerance = {TOL:g}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
