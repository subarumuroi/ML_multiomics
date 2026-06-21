"""
native_plots.py
===============
Thin, timeout-guarded wrappers that drive the iconic NATIVE R figures
(mixOmics DIABLO plotIndiv/circosPlot/plotLoadings/plotDiablo; WGCNA
soft-threshold/dendrogram/module-trait) during a report's cache build, and
return the produced PNG paths so the report can embed them next to the tables.

These are deliberately separate from the fitting bridges: producing the native
plots re-fits in an isolated R process, so a finicky plot (circosPlot is the
usual offender) cannot disturb the reported results, and a missing/absent R or
a hang degrades to "no native figure" rather than failing the build.

The caller passes blocks/matrices ALREADY on the scale the model sees
(imputed + z-scored); we only defensively median-fill any residual NaN, since R
cannot read NaN. Returns [] (never raises) when Rscript is absent, times out, or
the script errors -- the report notes the native figures were skipped.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RDIR = Path(__file__).resolve().parents[1] / "rscripts"


def _rscript_available(rscript: str) -> bool:
    return shutil.which(rscript) is not None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Median-fill residual NaN (R can't read NaN); drop all-NaN columns."""
    df = df.dropna(axis=1, how="all")
    if bool(df.isna().any().any()):
        df = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    return df


def _run(script: str, work: Path, prefix: str, timeout: int, rscript: str) -> list[Path]:
    try:
        res = subprocess.run([rscript, str(_RDIR / script), str(work)],
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("%s timed out after %ss -- native figures skipped", script, timeout)
        return []
    if res.returncode != 0:
        logger.warning("%s failed -- native figures skipped:\n%s", script, res.stderr[-800:])
        return []
    manifest = work / f"{prefix}_manifest.json"
    plotdir = work  # config sets plotdir = work
    if not manifest.exists():
        return []
    names = json.loads(manifest.read_text()).get("plots", [])
    return [plotdir / n for n in names if (plotdir / n).exists()]


def diablo_plots(blocks: dict, y, *, target_type: str, plotdir: str | Path,
                 prefix: str = "diablo", keepX: Optional[dict] = None,
                 design: float = 0.1, ncomp: int = 2,
                 rscript: str = "Rscript", timeout: int = 300) -> list[Path]:
    """Native mixOmics DIABLO figures -> list of PNG paths (empty if R unavailable)."""
    if not _rscript_available(rscript):
        return []
    plotdir = Path(plotdir); plotdir.mkdir(parents=True, exist_ok=True)
    bd = {nm: _clean(df if isinstance(df, pd.DataFrame) else pd.DataFrame(df))
          for nm, df in blocks.items()}
    common = None
    for df in bd.values():
        common = df.index if common is None else common.intersection(df.index)
    bd = {nm: df.loc[list(common)] for nm, df in bd.items()}
    yv = (pd.Series(y).reindex(list(common)) if hasattr(y, "reindex")
          else pd.Series(np.asarray(y), index=list(common)))
    work = Path(tempfile.mkdtemp(prefix="diablo_plots_"))
    try:
        for nm, df in bd.items():
            df.to_csv(work / f"block_{nm}.csv")
        pd.DataFrame({"y": yv.to_numpy()}).to_csv(work / "y.csv", index=False)
        if isinstance(keepX, dict):
            keepX = {k: (list(v) if isinstance(v, (list, tuple)) else int(v)) for k, v in keepX.items()}
        cfg = {"blocks": list(bd), "ncomp": int(ncomp), "design": float(design),
               "keepX": keepX, "target_type": target_type, "keepY": 1,
               "plotdir": str(work), "prefix": prefix}
        (work / "config.json").write_text(json.dumps(cfg))
        pngs = _run("diablo_plots.R", work, prefix, timeout, rscript)
        return _persist(pngs, plotdir)
    finally:
        _safe_rmtree(work, keep=plotdir)


def wgcna_plots(matrix: pd.DataFrame, *, plotdir: str | Path, prefix: str = "wgcna",
                y=None, network_type: str = "unsigned", min_module_size: int = 20,
                merge_cut_height: float = 0.25, power: Optional[int] = None,
                rscript: str = "Rscript", timeout: int = 300) -> list[Path]:
    """Native WGCNA figures -> list of PNG paths (empty if R unavailable / n too small)."""
    if not _rscript_available(rscript):
        return []
    plotdir = Path(plotdir); plotdir.mkdir(parents=True, exist_ok=True)
    X = _clean(matrix if isinstance(matrix, pd.DataFrame) else pd.DataFrame(matrix))
    work = Path(tempfile.mkdtemp(prefix="wgcna_plots_"))
    try:
        X.to_csv(work / "block.csv")
        if y is not None:
            yv = (pd.Series(y).reindex(list(X.index)) if hasattr(y, "reindex")
                  else pd.Series(np.asarray(y), index=list(X.index)))
            pd.DataFrame({"y": yv.to_numpy()}).to_csv(work / "y.csv", index=False)
        cfg = {"network_type": network_type, "min_module_size": int(min_module_size),
               "merge_cut_height": float(merge_cut_height),
               "power": None if power is None else int(power),
               "plotdir": str(work), "prefix": prefix}
        (work / "config.json").write_text(json.dumps(cfg))
        pngs = _run("wgcna_plots.R", work, prefix, timeout, rscript)
        return _persist(pngs, plotdir)
    finally:
        _safe_rmtree(work, keep=plotdir)


def _persist(pngs: Sequence[Path], plotdir: Path) -> list[Path]:
    """Copy PNGs out of the temp work dir into the persistent figure dir."""
    out = []
    for p in pngs:
        dest = plotdir / p.name
        try:
            dest.write_bytes(Path(p).read_bytes())
            out.append(dest)
        except OSError:
            pass
    return out


def _safe_rmtree(work: Path, keep: Path) -> None:
    if work.resolve() != keep.resolve():
        shutil.rmtree(work, ignore_errors=True)
