"""
skill_walkthrough.py
====================
End-to-end example following the multiomics-analysis SKILL playbook on the banana
data, with the plain-language interpretation a non-computational user would get.
Demonstrates the low-n high-p "reduce -> predict" recipe.

    ./venv/Scripts/python examples/skill_walkthrough.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_multiomics import OmicsDataset, Preprocessor, WGCNA, RandomForest
from ml_multiomics.core import parse_delimited
from ml_multiomics.validation import permutation_resolution

DATA = ROOT / "data"


def main():
    # Step 1-2: understand + load. Banana ripening, 1 layer (volatiles), each
    # replicate is an independent fruit.
    df = pd.read_csv(DATA / "badata-aromatics.csv").set_index("Sample").drop(columns=["Groups"])
    ds = OmicsDataset(name="banana_ripening")
    ds.add_block("aromatics", df, omics_type="volatiles")
    ds.set_sample_metadata(parse_delimited(df.index, sep="-", names=("stage", "replicate")))

    # Step 3: preprocess (defaults: missing-aware log + z-score, scaled once)
    Preprocessor().run(ds)

    # Step 4: grouping — each fruit is independent, so each sample is its own group
    X = ds.get("aromatics")
    y = ds.sample_meta["stage"].to_numpy()
    groups = np.arange(len(y))

    # Step 5-6: low-n high-p -> reduce (WGCNA) then predict (RandomForest)
    wg = WGCNA(corr_method="spearman").fit(X, y, target_type="ordinal")
    reduced = wg.reduce(strategy="eigengenes_and_hubs")
    rf = RandomForest().fit(reduced, y, target_type="nominal")
    cv = rf.cross_validate(reduced, y, groups=groups, target_type="nominal")
    res = permutation_resolution(groups, y)
    top = rf.importances(top_n=5)

    # Step 7: plain-language interpretation
    print("=" * 64)
    print("PLAIN-LANGUAGE SUMMARY (what a non-expert would be told)")
    print("=" * 64)
    print(f"Data:        {X.shape[1]} volatile features across {X.shape[0]} banana samples")
    print(f"Reduction:   collapsed to {reduced.shape[1]} co-abundance factors (WGCNA)")
    print(f"Question:    can the factors tell ripening stage apart?")
    print(f"Answer:      {cv['accuracy']:.0%} accuracy under leakage-free cross-validation")
    print(f"             (balanced accuracy {cv['balanced_accuracy']:.0%})")
    print(f"Caveat:      only {res['n_groups']} independent samples -- the smallest")
    print(f"             p-value this design can produce is {res['finest_two_sided_p']:.2g};")
    print(f"             treat as exploratory and cross-check with another method.")
    print(f"Top factors driving the call:")
    for _, r in top.iterrows():
        print(f"             - {r['feature']}  (importance {r['importance']:.3f})")
    print("=" * 64)

    assert reduced.shape[0] == X.shape[0]
    assert 0.0 <= cv["accuracy"] <= 1.0
    print("[PASS] skill walkthrough completed end-to-end")


if __name__ == "__main__":
    main()
