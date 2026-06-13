"""
test_analysis.py
================
Smoke test for the standard-analysis module (differential expression +
enrichment). Numerical parity vs the lab's R lives in
tests/crosscheck/run_crosscheck_de.py; this checks the API shapes and behaviour.

    ./venv/Scripts/python tests/test_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_multiomics.analysis import compute_volcano, anova_tukey, ora

PASS, FAIL = "[PASS]", "[FAIL]"
_failures = []


def check(cond, msg):
    print(f"  {PASS if cond else FAIL} {msg}")
    if not cond:
        _failures.append(msg)


def test_volcano_and_anova():
    print("\n=== compute_volcano / anova_tukey ===")
    rng = np.random.default_rng(0)
    n = 12
    groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    X = pd.DataFrame(np.abs(rng.normal(50, 10, size=(n, 6))) + 1,
                     index=[f"s{i}" for i in range(n)],
                     columns=[f"f{i}" for i in range(6)])
    # make f0 differ strongly in group c
    X.iloc[8:, 0] *= 3.0

    v = compute_volcano(X, groups, logx=True, fdr_method="BH")
    check(set(["contrast", "feature", "foldchange", "log2fc", "pvalue", "qvalue"]).issubset(v.columns),
          "volcano returns expected columns")
    check(v["contrast"].nunique() == 3, "3 pairwise contrasts for 3 groups")
    # f0 should be among the smaller p-values in an a-vs-c or b-vs-c contrast
    ac = v[v["contrast"].str.contains("c")]
    check((ac.loc[ac["feature"] == "f0", "pvalue"] < 0.5).any(), "elevated f0 flagged (low p in a c-contrast)")

    a = anova_tukey(X, groups, logx=True, fdr_method="BH")
    check("pvalue" in a.columns and len(a) == 6, "anova_tukey returns one row per feature")
    check(any(col.startswith("tukey_p_") for col in a.columns), "anova_tukey returns Tukey pairwise p-values")


def test_ora():
    print("\n=== ora (hypergeometric over-representation) ===")
    universe = [f"g{i}" for i in range(200)]
    hits = [f"g{i}" for i in range(20)]              # 20 hits
    gene_sets = {
        "enriched_set": [f"g{i}" for i in range(15)] + ["g100", "g101", "g102", "g103", "g104"],  # 15/20 hits
        "random_set": [f"g{i}" for i in range(100, 130)],   # ~no hits
    }
    res = ora(hits, universe, gene_sets, min_gs_size=5, max_gs_size=100)
    check(len(res) >= 1, "ora returns results")
    top = res.iloc[0]
    check(top["term"] == "enriched_set", "the enriched set ranks first")
    check(top["pvalue"] < 0.05, f"enriched set is significant (p={top['pvalue']:.1e})")
    check(set(["term", "n_set", "n_hit_in_set", "gene_ratio", "bg_ratio", "pvalue", "padj", "sig"]).issubset(res.columns),
          "ora returns expected columns")


def main():
    test_volcano_and_anova()
    test_ora()
    print("\n" + "=" * 60)
    if _failures:
        print(f"{FAIL} {len(_failures)} check(s) failed")
        for f in _failures:
            print("   -", f)
        sys.exit(1)
    print(f"{PASS} all analysis smoke checks passed")


if __name__ == "__main__":
    main()
