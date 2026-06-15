"""
run_crosscheck_splsda.py
=======================
Parity PROBE: our native SparsePLSDA vs mixOmics::splsda on the real banana
proteomics (z-scored once, scale=FALSE in mixOmics so neither double-scales).
Reports per-component sample-variate correlation (algorithmic agreement) and
selected-feature overlap (practical agreement). This is a probe to decide
keep-Python-vs-use-R, not a pass/fail gate.

    ./venv/Scripts/python tests/crosscheck/run_crosscheck_splsda.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ml_multiomics import OmicsDataset, Preprocessor
from ml_multiomics.core import parse_delimited
from ml_multiomics.methods import SparsePLSDA

HERE = Path(__file__).resolve().parent
DATA = ROOT / "data"
KEEPX = 50


def main():
    df = pd.read_csv(DATA / "badata-proteomics-imputed.csv").set_index("Sample").drop(columns=["Groups"])
    ds = OmicsDataset("banana"); ds.add_block("proteomics", df, omics_type="proteomics")
    ds.set_sample_metadata(parse_delimited(df.index, sep="-", names=("stage", "replicate")))
    Preprocessor().run(ds)
    X = ds.get("proteomics"); y = ds.sample_meta["stage"].to_numpy()

    X.to_csv(HERE / "sp_X.csv")
    pd.DataFrame({"y": y}).to_csv(HERE / "sp_y.csv", index=False)

    print(f"Running mixOmics::splsda (n={X.shape[0]}, p={X.shape[1]}, keepX={KEEPX})...")
    r = subprocess.run(["Rscript", str(HERE / "ref_splsda.R"), str(HERE), str(KEEPX)],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("R STDERR:\n", r.stderr); sys.exit(2)

    rv = pd.read_csv(HERE / "r_splsda_variates.csv").to_numpy()
    ours = SparsePLSDA(n_components=2, keepX=KEEPX).fit(X, y)
    ov = ours.fit_["T"]  # sample variates (n x 2)

    print("\n" + "=" * 60)
    print(f"{'component':12s} {'|variate corr|':16s} {'feature overlap':16s}")
    print("-" * 60)
    for k in range(2):
        corr = abs(np.corrcoef(ov[:, k], rv[:, k])[0, 1])
        r_sel = set(int(i) - 1 for i in (HERE / f"r_splsda_sel{k+1}.txt").read_text().split())
        o_sel = set(np.where(ours.fit_["W"][:, k] != 0)[0].tolist())
        jac = len(r_sel & o_sel) / len(r_sel | o_sel) if (r_sel | o_sel) else float("nan")
        print(f"comp {k+1:<7d} {corr:<16.3f} {len(r_sel & o_sel)}/{KEEPX} (Jaccard {jac:.2f})")
    print("=" * 60)
    print("Interpretation: |corr|~1 + high overlap => our NIPALS matches mixOmics;")
    print("low values => the native port is NOT equivalent -> prefer the R original.")


if __name__ == "__main__":
    main()
