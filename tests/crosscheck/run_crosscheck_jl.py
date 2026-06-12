"""
run_crosscheck_jl.py
====================
Numerical equivalence check: my Python preprocessing primitives vs IdeaBio.jl's
omics operations (reproduced verbatim in ref_jl.jl; the full module is not
loadable quickly due to its Makie/RCall dependency tree).

Key questions this answers:
  * Does my z-score match Julia on MISSING data? (it diverged from R there)
  * The log2 default: my log2(x+1) (mofa_prep) vs IdeaBio.jl plain log2(x)
  * MetaboAnalyst: my 0.2 x min(positive) vs Julia 0.2 x min(non-missing)

Run:  ./venv/Scripts/python tests/crosscheck/run_crosscheck_jl.py
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
    zscore, log2_transform, log10_transform, metaboanalyst_impute,
)

HERE = Path(__file__).resolve().parent
JULIA = "julia"

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
MISSING_CELLS = [(1, 1), (4, 3)]  # (row, col index)


def _max_abs_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(a - b)
    d[both_nan] = 0.0
    return float(np.nanmax(d))


def main():
    xc = pd.DataFrame(X, index=SAMPLES, columns=FEATURES)
    xm = xc.copy()
    for r, c in MISSING_CELLS:
        xm.iloc[r, c] = np.nan

    # plain numeric CSVs (no header/index, NaN marker) for Julia readdlm
    np.savetxt(HERE / "x_complete_plain.csv", xc.to_numpy(), delimiter=",")
    np.savetxt(HERE / "x_missing_plain.csv", xm.to_numpy(), delimiter=",")

    print("Running Julia reference (reproducing IdeaBio.jl ops)...")
    res = subprocess.run([JULIA, str(HERE / "ref_jl.jl"), str(HERE)],
                         capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        print("JULIA STDERR:\n", res.stderr)
        sys.exit(2)

    def load_jl(name):
        return np.genfromtxt(HERE / name, delimiter=",")

    rows = []

    # z-score complete -> expect MATCH
    d = _max_abs_diff(zscore(xc).to_numpy(), load_jl("jl_zscore_complete.csv"))
    rows.append(("zscore (complete)", "MATCH" if d < 1e-9 else "DIVERGE", f"{d:.2e}",
                 "Julia == Python == R on complete data"))

    # z-score MISSING -> expect MATCH (this is where Python diverged from R)
    d = _max_abs_diff(zscore(xm).to_numpy(), load_jl("jl_zscore_missing.csv"))
    rows.append(("zscore (missing)", "MATCH" if d < 1e-9 else "DIVERGE", f"{d:.2e}",
                 "confirms Python matches Julia's skip-NaN (R diverged here)"))

    # log2: mine log2(x+1) vs Julia plain log2(x) -> expect DIVERGE
    jl_log2 = load_jl("jl_log2_complete.csv")
    d_pseudo = _max_abs_diff(log2_transform(xc).to_numpy(), jl_log2)
    rows.append(("log2: mine(x+1) vs jl(x)", "DIVERGE (decision)" if d_pseudo > 1e-9 else "MATCH",
                 f"{d_pseudo:.2e}", "pseudocount: mofa_prep log2(x+1) vs IdeaBio.jl log2(x)"))
    # prove it's ONLY the pseudocount: plain np.log2 vs Julia -> MATCH
    d_plain = _max_abs_diff(np.log2(xc.to_numpy()), jl_log2)
    rows.append(("log2: plain vs jl", "MATCH" if d_plain < 1e-9 else "DIVERGE",
                 f"{d_plain:.2e}", "confirms divergence is purely the +1 pseudocount"))

    # log10 complete -> expect MATCH
    d = _max_abs_diff(log10_transform(xc).to_numpy(), load_jl("jl_log10_complete.csv"))
    rows.append(("log10 (complete)", "MATCH" if d < 1e-9 else "DIVERGE", f"{d:.2e}",
                 "Julia == Python"))

    # metaboanalyst on positive data -> expect MATCH (min positive == min nonmissing)
    d = _max_abs_diff(metaboanalyst_impute(xm).to_numpy(), load_jl("jl_metaboanalyst.csv"))
    rows.append(("metaboanalyst impute", "MATCH" if d < 1e-9 else "DIVERGE", f"{d:.2e}",
                 "0.2x min; mine=positive, jl=non-missing -- agree on positive data"))

    print("\n" + "=" * 96)
    print(f"{'primitive':28s} {'result':20s} {'maxdiff':12s} note")
    print("-" * 96)
    for name, status, detail, note in rows:
        print(f"{name:28s} {status:20s} {detail:12s} {note}")
    print("=" * 96)

    hard_fail = any(s.startswith("DIVERGE") and "decision" not in s
                    for _, s, _, _ in rows)
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
