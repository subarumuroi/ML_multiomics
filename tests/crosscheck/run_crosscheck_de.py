"""
run_crosscheck_de.py
===================
Numerical parity: ml_multiomics.analysis vs the lab's actual R DE code
(IdeaBio.R foldchange.R::compute_volcano) and base R phyper (the ORA statistic).

Run:  ./venv/Scripts/python tests/crosscheck/run_crosscheck_de.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_multiomics.analysis import compute_volcano

HERE = Path(__file__).resolve().parent
LAB_R = Path("C:/Users/uqkmuroi/gitcode/ideabio/StandardOmicAnalyses/R")
RSCRIPT = "Rscript"

# Parity is checked on a SINGLE contrast (2 groups). See the note below: the
# lab's compute_volcano mislabels features when there are >1 contrasts (an R bug
# we deliberately do NOT replicate), so multi-contrast outputs cannot be compared
# row-for-row. With one contrast the bug does not trigger and the statistic is
# directly comparable.
SAMPLES = ["c1_1", "c1_2", "c1_3", "c1_4", "c1_5", "c2_1", "c2_2", "c2_3", "c2_4"]
GROUPS = ["c1", "c1", "c1", "c1", "c1", "c2", "c2", "c2", "c2"]
rng = np.random.default_rng(11)
X = pd.DataFrame(np.abs(rng.normal(loc=50, scale=12, size=(9, 5))) + 1.0,
                 index=SAMPLES, columns=[f"f{i}" for i in range(5)])


def main():
    X.to_csv(HERE / "de_x.csv")
    pd.DataFrame({"group": GROUPS}, index=SAMPLES).to_csv(HERE / "de_groups.csv")

    print("Running lab R DE reference (compute_volcano + phyper)...")
    res = subprocess.run([RSCRIPT, str(HERE / "ref_de.R"), str(HERE), str(LAB_R)],
                         capture_output=True, text=True)
    print(res.stdout.strip())
    if res.returncode != 0:
        print("R STDERR:\n", res.stderr)
        sys.exit(2)

    # --- compute_volcano parity ---
    r_v = pd.read_csv(HERE / "r_volcano.csv")
    py_v = compute_volcano(X, GROUPS, logx=True, fdr_method="bonferroni")
    # align on (contrast, feature): R uses "c1-vs-c2" group labels too
    r_v = r_v.rename(columns={"group": "contrast"})
    key = ["contrast", "feature"]
    merged = py_v.merge(r_v, on=key, suffixes=("_py", "_r"))
    # R may order contrasts differently; require all rows matched
    matched_ok = len(merged) == len(py_v) == len(r_v)

    d_fc = float(np.nanmax(np.abs(merged["foldchange_py"] - merged["foldchange_r"])))
    d_l2 = float(np.nanmax(np.abs(merged["log2fc_py"] - merged["log2fc_r"])))
    d_p = float(np.nanmax(np.abs(merged["pvalue_py"] - merged["pvalue_r"])))
    d_q = float(np.nanmax(np.abs(merged["qvalue_py"] - merged["qvalue_r"])))

    # --- phyper (ORA hypergeometric statistic) parity ---
    r_h = pd.read_csv(HERE / "r_phyper.csv")
    py_h = hypergeom.sf(r_h["k"].to_numpy() - 1, r_h["N"].to_numpy(),
                        r_h["K"].to_numpy(), r_h["n"].to_numpy())
    d_h = float(np.nanmax(np.abs(py_h - r_h["p"].to_numpy())))

    TOL = 1e-9
    rows = [
        ("volcano rows aligned", 0.0 if matched_ok else 1.0),
        ("volcano foldchange", d_fc),
        ("volcano log2fc", d_l2),
        ("volcano Welch p-value", d_p),
        ("volcano qvalue (bonferroni)", d_q),
        ("ORA hypergeometric (phyper)", d_h),
    ]
    print("\n" + "=" * 64)
    print(f"{'check':32s} {'max abs diff':14s} result")
    print("-" * 64)
    ok = True
    for name, d in rows:
        res_s = "MATCH" if d < TOL else "DIVERGE"
        if d >= TOL:
            ok = False
        print(f"{name:32s} {d:.2e}      {res_s}")
    print("=" * 64)
    print("NOTE: parity verified on a single contrast. The lab's compute_volcano")
    print("      mislabels features for >1 contrast (uses rep(colnames, times=)")
    print("      instead of each=, mis-pairing labels with column-major values).")
    print("      Our Python labels correctly and is NOT bug-for-bug compatible.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
