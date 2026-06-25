"""
dataset.py
==========
The canonical multi-omics container.

One ``OmicsDataset`` holds N omics blocks (single-omics = 1 block), a shared
sample-metadata table, an optional target spec, and a per-block provenance log.

Design rules:
  * Blocks may have DIFFERENT sample sets. Alignment is ALWAYS by sample ID with
    intersection, never by position.
  * Provenance is tracked per block (transformed / normalized / imputed + a step
    log) so methods can check data state and never silently re-transform.
  * Missingness is preserved through preprocessing; imputation is a separate,
    method-gated step (see methods.base.BaseMethod.handles_missing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_TYPES = ("nominal", "ordinal", "continuous", "none")


@dataclass
class TargetSpec:
    """Describes the prediction target.

    type ∈ {nominal, ordinal, continuous, none}. ``values`` is a Series indexed
    by sample_id. For ``ordinal`` supply ``ordinal_order`` (ordered categories).
    """
    name: str
    type: str
    values: Optional[pd.Series] = None
    ordinal_order: Optional[list] = None

    def __post_init__(self):
        if self.type not in TARGET_TYPES:
            raise ValueError(
                f"target type must be one of {TARGET_TYPES}; got {self.type!r}"
            )
        if self.values is not None:
            self.values = self.values.copy()
            self.values.index = self.values.index.astype(str)
        if self.type == "ordinal" and self.ordinal_order is None and self.values is not None:
            logger.warning(
                "ordinal target '%s' has no ordinal_order; category order is undefined.",
                self.name,
            )

    def encoded(self) -> pd.Series:
        """Return ordinal targets mapped to 0..k-1 in ``ordinal_order``."""
        if self.type != "ordinal":
            return self.values
        order = self.ordinal_order or sorted(self.values.dropna().unique())
        mapping = {cat: i for i, cat in enumerate(order)}
        return self.values.map(mapping)


class Block:
    """One omics layer: a samples x features DataFrame plus provenance."""

    def __init__(self, name: str, data: pd.DataFrame, omics_type: Optional[str] = None):
        data = data.copy()
        data.index = data.index.astype(str)
        self.name = name
        self.data = data
        self.omics_type = omics_type
        self.provenance: list[str] = []
        # state flags enforced by the preprocessing contract
        self.transformed = False
        self.normalized = False
        self.imputed = False

    # -- views -------------------------------------------------------------
    @property
    def samples(self) -> list[str]:
        return list(self.data.index)

    @property
    def features(self) -> list[str]:
        return list(self.data.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def has_missing(self) -> bool:
        return bool(self.data.isna().any().any())

    def missing_fraction(self) -> float:
        if self.data.size == 0:
            return 0.0
        return float(self.data.isna().to_numpy().mean())

    # -- provenance --------------------------------------------------------
    def log(self, step: str) -> None:
        self.provenance.append(step)
        logger.info("[%s] %s", self.name, step)

    def __repr__(self) -> str:
        return (
            f"Block(name={self.name!r}, shape={self.shape}, "
            f"omics_type={self.omics_type!r}, missing={self.missing_fraction():.1%}, "
            f"transformed={self.transformed}, normalized={self.normalized}, "
            f"imputed={self.imputed})"
        )


class OmicsDataset:
    """Container for N omics blocks + shared sample metadata + a target."""

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.blocks: dict[str, Block] = {}
        self.sample_meta: pd.DataFrame = pd.DataFrame()
        self.target: Optional[TargetSpec] = None

    # -- blocks ------------------------------------------------------------
    def add_block(self, name: str, data: pd.DataFrame, omics_type: Optional[str] = None) -> Block:
        if name in self.blocks:
            raise ValueError(f"block {name!r} already exists")
        block = Block(name, data, omics_type)
        self.blocks[name] = block
        return block

    @property
    def block_names(self) -> list[str]:
        return list(self.blocks)

    def get(self, name: str) -> pd.DataFrame:
        return self.blocks[name].data

    # -- metadata ----------------------------------------------------------
    def set_sample_metadata(self, meta: pd.DataFrame) -> "OmicsDataset":
        meta = meta.copy()
        meta.index = meta.index.astype(str)
        self.sample_meta = meta
        return self

    def groups(self, column: str) -> pd.Series:
        """Return the grouping/independent-unit vector from sample metadata."""
        if column not in self.sample_meta.columns:
            raise KeyError(
                f"metadata column {column!r} not found; have "
                f"{list(self.sample_meta.columns)}"
            )
        return self.sample_meta[column]

    # -- alignment ---------------------------------------------------------
    def common_samples(self, blocks: Optional[list[str]] = None) -> list[str]:
        """Intersection of sample IDs across the given blocks (order from first)."""
        names = blocks or self.block_names
        if not names:
            return []
        sets = [set(self.blocks[n].samples) for n in names]
        common = set.intersection(*sets)
        first = self.blocks[names[0]].samples
        return [s for s in first if s in common]

    def align(self, blocks: Optional[list[str]] = None) -> "OmicsDataset":
        """Subset all (or given) blocks + metadata to common samples, by ID."""
        names = blocks or self.block_names
        common = self.common_samples(names)
        for n in names:
            b = self.blocks[n]
            before = b.shape[0]
            b.data = b.data.loc[common]
            b.log(f"align: {before} -> {len(common)} common samples")
        if not self.sample_meta.empty:
            keep = [s for s in common if s in self.sample_meta.index]
            self.sample_meta = self.sample_meta.loc[keep]
        return self

    # -- target ------------------------------------------------------------
    def set_target(
        self,
        name: str,
        type: str,
        values: Optional[pd.Series] = None,
        column: Optional[str] = None,
        ordinal_order: Optional[list] = None,
    ) -> TargetSpec:
        """Set the prediction target from an explicit Series or a metadata column."""
        if values is None and column is not None:
            if column not in self.sample_meta.columns:
                raise KeyError(f"metadata column {column!r} not found")
            values = self.sample_meta[column]
        self.target = TargetSpec(name=name, type=type, values=values, ordinal_order=ordinal_order)
        return self.target

    # -- summary -----------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        rows = []
        for n, b in self.blocks.items():
            rows.append({
                "block": n,
                "omics_type": b.omics_type,
                "n_samples": b.shape[0],
                "n_features": b.shape[1],
                "pct_missing": round(100 * b.missing_fraction(), 2),
                "transformed": b.transformed,
                "normalized": b.normalized,
                "imputed": b.imputed,
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"OmicsDataset(name={self.name!r}, blocks={self.block_names}, "
            f"n_meta_cols={self.sample_meta.shape[1]}, "
            f"target={self.target.name if self.target else None})"
        )
