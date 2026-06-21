"""
metadata.py
===========
Pluggable sample-metadata parsers. These are DATASET-SPECIFIC conventions, not
universal — a loader picks the parser that matches the incoming IDs, or supplies
a metadata table directly.

Bioreactor convention (psilocybin / fermentation data ONLY):
    F#C#R#T#  e.g.  27-PSI_F503_C1_R1_T1  or  F503_C1_R1
      * F# and C# are both experimental CONDITION factors (F = fermentation batch,
        C = the C-factor condition, e.g. C1/C2). NEITHER is a genetic "construct";
        the helper imposes no biological meaning -- it just parses the structure.
      * R# = replicate number
      * T# = timepoint number
      * usual independent unit for grouping = bioreactor = F#C#R#

The helper stays deliberately FLEXIBLE/OPEN: it returns the raw factors (F, C, R, T)
plus two common composites (condition = F_C, bioreactor = F_C_R), and the USER decides
via the AnalysisSpec which factor is the grouping unit / a target / a covariate /
excluded. It never assigns roles itself.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

# F503  C1   R1   optional T1 ; tolerant of a leading project prefix (27-PSI_)
_BIOREACTOR_RE = re.compile(
    r"(?P<F>F\d+)[_-]?(?P<C>C\d+)[_-]?(?P<R>R\d+)(?:[_-]?(?P<T>T\d+))?",
    re.IGNORECASE,
)


def parse_bioreactor_ids(sample_ids: Iterable[str]) -> pd.DataFrame:
    """Parse F#C#R#T# bioreactor sample IDs into a flexible factor table.

    Returns a DataFrame indexed by the original sample_id with columns:
        F           (F#)   -- fermentation batch     (a condition factor)
        C           (C#)   -- C-factor condition     (a condition factor; NOT a construct)
        R           (R#)   -- replicate
        T           (T#, may be NaN) -- timepoint
        condition   (F#_C#)   -- convenience composite of the two condition factors
        bioreactor  (F#_C#_R#) -- convenience composite; the usual independent unit

    The four raw factors are exposed so the caller can compose ANY grouping/role
    (e.g. group by F#C#R#, contrast C, treat F as a covariate, aggregate over T) via
    the AnalysisSpec -- the helper assigns no roles and imposes no biology. Rows whose
    ID does not match the pattern get NaN fields (logged by caller).
    """
    records = {}
    for sid in sample_ids:
        s = str(sid)
        m = _BIOREACTOR_RE.search(s)
        if not m:
            records[s] = {"F": np.nan, "C": np.nan, "R": np.nan, "T": np.nan,
                          "condition": np.nan, "bioreactor": np.nan}
            continue
        f = m.group("F").upper()
        c = m.group("C").upper()
        r = m.group("R").upper()
        t = m.group("T")
        t = t.upper() if t else np.nan
        records[s] = {
            "F": f, "C": c, "R": r, "T": t,
            "condition": f"{f}_{c}",
            "bioreactor": f"{f}_{c}_{r}",
        }
    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "sample_id"
    return df


def parse_delimited(
    sample_ids: Iterable[str],
    sep: str = "-",
    names: tuple[str, str] = ("group", "replicate"),
) -> pd.DataFrame:
    """Generic parser for '<group><sep><replicate>' IDs, e.g. 'Green-1'.

    The independent unit for grouping is the full sample (each row independent),
    but ``group`` is also returned for stratification / labelling.
    """
    group_name, rep_name = names
    records = {}
    for sid in sample_ids:
        s = str(sid)
        parts = s.rsplit(sep, 1)
        if len(parts) == 2:
            records[s] = {group_name: parts[0], rep_name: parts[1]}
        else:
            records[s] = {group_name: s, rep_name: np.nan}
    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "sample_id"
    return df
