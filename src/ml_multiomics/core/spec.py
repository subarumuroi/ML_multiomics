"""
spec.py
=======
The declarative analysis contract: ``AnalysisSpec``.

This package is built for REUSE on arbitrary multi-omics data, so it must never
guess what the user wants. The user *declares* the analysis; the package
*executes, validates, flags, and records* it. Specifically the spec captures the
decisions a human must own:

  * which **grouping / independent-unit column** defines leave-one-out folds
    (the engine NEVER parses sample IDs — `parse_bioreactor_ids` / `parse_delimited`
    are optional helpers that *populate* such a column);
  * the **role of every layer** — predictor / target / covariate / exclude
    (every block must be assigned a role; nothing is assumed important);
  * optional per-layer **transform** overrides;
  * which predictor layers (if any) **integrate together** — cross-layer
    combination is opt-in and explicit; the package never auto-combines layers;
  * the **target** (a metadata column or an external Series) and its type.

`validate()` rejects ambiguous or silent setups so a misconfiguration fails loudly
rather than producing a confident-but-wrong result. `describe()` renders the
declared decisions for the provenance trail and the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .dataset import OmicsDataset, TargetSpec, TARGET_TYPES

#: Roles a layer can play in an analysis.
ROLES = ("predictor", "target", "covariate", "exclude")

#: Transform names understood by the Preprocessor (others are passed through
#: and validated downstream). Kept here for documentation / spec rendering.
KNOWN_TRANSFORMS = ("log2", "log10", "log2p1", "none")


@dataclass
class AnalysisSpec:
    """A user-authored declaration of one analysis over an ``OmicsDataset``.

    Required
    --------
    grouping_column : str
        Name of the metadata column holding the independent unit (e.g. a
        bioreactor or a piece of fruit). Defines leave-one-group-out folds,
        permutation, and bootstrap. The engine reads this column; it never
        infers structure from sample IDs.
    roles : dict[str, str]
        Maps EVERY block name to one of :data:`ROLES`. There is no default role
        — the user must decide what each layer is for.
    target_type : str
        One of ``nominal`` / ``ordinal`` / ``continuous``.

    Target source (exactly one)
    ---------------------------
    target_column : str
        A column in ``sample_meta`` to use as the target, OR
    target_values : pd.Series
        An external target indexed by sample id (e.g. a yield not stored in the
        omics matrix).

    Optional
    --------
    transforms : dict[str, str]
        Per-layer transform override; if absent the Preprocessor uses the
        omics-type default.
    integration_groups : list[list[str]] | None
        Lists of predictor layers to integrate together (DIABLO/multi-block).
        ``None`` = NO cross-layer integration (each predictor analysed single-
        block). Integration is always opt-in — the package never combines layers
        on its own.
    grouping_parser : str
        Provenance note for how ``grouping_column`` was produced
        (``"user-supplied"``, ``"parse_bioreactor_ids"``, ``"parse_delimited"``…).
    min_obs_frac : float | None
        Default detection-filter threshold (applied method-aware downstream).
    """

    grouping_column: str
    roles: dict[str, str]
    target_type: str
    target_column: Optional[str] = None
    target_values: Optional[pd.Series] = None
    target_name: Optional[str] = None
    ordinal_order: Optional[list] = None
    transforms: dict[str, str] = field(default_factory=dict)
    integration_groups: Optional[list[list[str]]] = None
    grouping_parser: str = "user-supplied"
    min_obs_frac: Optional[float] = None
    name: Optional[str] = None
    #: per-layer declaration of what upstream ALREADY did, so ML preprocessing can
    #: skip/adapt rather than double-apply. e.g.
    #: {"proteomics": {"transform": "log2", "normalized": False, "imputed": True}}.
    #: Undeclared layers are assumed raw (nothing applied).
    input_states: dict = field(default_factory=dict)
    #: per-layer raw/least-processed DataFrame to use INSTEAD of the dataset block
    #: when ML needs the un-processed values (the legitimate "revert"; e.g. banana's
    #: unimputed proteomics). Recorded in provenance.
    raw_sources: dict = field(default_factory=dict)

    # -- role views --------------------------------------------------------
    def _layers_with_role(self, role: str) -> list[str]:
        return [layer for layer, r in self.roles.items() if r == role]

    def predictor_layers(self) -> list[str]:
        return self._layers_with_role("predictor")

    def target_layers(self) -> list[str]:
        return self._layers_with_role("target")

    def covariate_layers(self) -> list[str]:
        return self._layers_with_role("covariate")

    def excluded_layers(self) -> list[str]:
        return self._layers_with_role("exclude")

    def transform_for(self, layer: str) -> Optional[str]:
        """The declared transform override for ``layer`` (None = use default)."""
        return self.transforms.get(layer)

    def input_state_for(self, layer: str) -> dict:
        """Declared upstream state for ``layer`` (empty = assumed raw / nothing applied)."""
        return dict(self.input_states.get(layer, {}))

    # -- validation --------------------------------------------------------
    def validate(self, ds: OmicsDataset) -> "AnalysisSpec":
        """Reject ambiguous / silent setups against a concrete dataset.

        Raises ValueError/KeyError with an actionable message. Returns self so
        callers can ``spec.validate(ds)`` inline.
        """
        errors: list[str] = []

        # target_type
        if self.target_type not in ("nominal", "ordinal", "continuous"):
            errors.append(
                f"target_type must be nominal/ordinal/continuous; got {self.target_type!r}"
            )

        # roles: every block assigned exactly one valid role; no unknown layers
        block_names = set(ds.block_names)
        for layer, role in self.roles.items():
            if role not in ROLES:
                errors.append(f"layer {layer!r} has invalid role {role!r}; choose from {ROLES}")
            if layer not in block_names:
                errors.append(f"role declared for unknown layer {layer!r}; dataset has {sorted(block_names)}")
        undeclared = sorted(block_names - set(self.roles))
        if undeclared:
            errors.append(
                "every layer must be assigned a role (the user decides what each is for); "
                f"undeclared: {undeclared}"
            )
        if not self.predictor_layers():
            errors.append("at least one layer must have role 'predictor'")

        # grouping column must exist and be populated
        if ds.sample_meta is None or ds.sample_meta.empty:
            errors.append("dataset has no sample_meta; cannot resolve a grouping column")
        elif self.grouping_column not in ds.sample_meta.columns:
            errors.append(
                f"grouping_column {self.grouping_column!r} not in sample_meta "
                f"(have {list(ds.sample_meta.columns)}); the engine does NOT parse sample IDs -- "
                "populate this column first (e.g. with parse_bioreactor_ids/parse_delimited)"
            )
        elif self.grouping_column in ds.sample_meta and ds.sample_meta[self.grouping_column].isna().any():
            errors.append(f"grouping_column {self.grouping_column!r} contains missing values")

        # target source: exactly one of column/values
        has_col = self.target_column is not None
        has_vals = self.target_values is not None
        if has_col == has_vals:
            errors.append("specify exactly one of target_column or target_values")
        if has_col and not ds.sample_meta.empty and self.target_column not in ds.sample_meta.columns:
            errors.append(f"target_column {self.target_column!r} not in sample_meta")
        if has_col and self.target_column in self.roles:
            errors.append(
                f"target_column {self.target_column!r} also names a layer role; "
                "the target must not double as a predictor layer"
            )
        if self.target_type == "ordinal" and not self.ordinal_order:
            errors.append("ordinal target requires ordinal_order (the category order)")

        # upstream-state / raw-source declarations must name real layers
        for layer in self.input_states:
            if layer not in block_names:
                errors.append(f"input_states declared for unknown layer {layer!r}")
        for layer in self.raw_sources:
            if layer not in block_names:
                errors.append(f"raw_sources declared for unknown layer {layer!r}")

        # integration groups: members must be predictors; combination is explicit
        if self.integration_groups is not None:
            preds = set(self.predictor_layers())
            for grp in self.integration_groups:
                bad = [m for m in grp if m not in preds]
                if bad:
                    errors.append(
                        f"integration group {grp} contains non-predictor layers {bad}; "
                        "only predictor layers may be integrated"
                    )

        if errors:
            raise ValueError("AnalysisSpec invalid:\n  - " + "\n  - ".join(errors))
        return self

    # -- resolution --------------------------------------------------------
    def resolve_target(self, ds: OmicsDataset) -> TargetSpec:
        """Build a :class:`TargetSpec` from the declared column or external Series."""
        if self.target_values is not None:
            values = self.target_values
        else:
            values = ds.sample_meta[self.target_column]
        return TargetSpec(
            name=self.target_name or self.target_column or "target",
            type=self.target_type,
            values=values,
            ordinal_order=self.ordinal_order,
        )

    def integration_sets(self) -> list[list[str]]:
        """Predictor-layer groupings to integrate (empty if none declared)."""
        if not self.integration_groups:
            return []
        return [list(g) for g in self.integration_groups]

    # -- rendering (provenance / report) -----------------------------------
    def describe(self) -> str:
        """Human-readable rendering of the declared decisions."""
        lines = [f"AnalysisSpec{f' ({self.name})' if self.name else ''}"]
        lines.append(
            f"  grouping unit : {self.grouping_column} (source: {self.grouping_parser})"
        )
        tgt_src = (
            f"column '{self.target_column}'" if self.target_column else "external values"
        )
        lines.append(
            f"  target        : {self.target_name or self.target_column or 'target'} "
            f"[{self.target_type}] from {tgt_src}"
        )
        lines.append("  layer roles   :")
        for layer in sorted(self.roles):
            tf = self.transforms.get(layer)
            tf_txt = f", transform={tf}" if tf else ""
            lines.append(f"      - {layer}: {self.roles[layer]}{tf_txt}")
        groups = self.integration_sets()
        if groups:
            lines.append("  integration   :")
            for g in groups:
                lines.append(f"      - {' + '.join(g)}")
        else:
            lines.append("  integration   : none declared (each predictor analysed single-block)")
        if self.input_states or self.raw_sources:
            lines.append("  upstream state:")
            for layer in sorted(set(self.input_states) | set(self.raw_sources)):
                st = self.input_states.get(layer, {})
                bits = [f"{k}={v}" for k, v in st.items()]
                if layer in self.raw_sources:
                    bits.append("raw_source=provided")
                lines.append(f"      - {layer}: {', '.join(bits) or 'declared'}")
        return "\n".join(lines)

    def to_record(self) -> dict:
        """Structured, JSON-serialisable snapshot for the provenance trail."""
        return {
            "name": self.name,
            "grouping_column": self.grouping_column,
            "grouping_parser": self.grouping_parser,
            "roles": dict(self.roles),
            "transforms": dict(self.transforms),
            "integration_groups": self.integration_sets(),
            "target": {
                "name": self.target_name or self.target_column or "target",
                "type": self.target_type,
                "source": ("column:" + self.target_column) if self.target_column else "external_values",
                "ordinal_order": self.ordinal_order,
            },
            "min_obs_frac": self.min_obs_frac,
            "input_states": dict(self.input_states),
            "raw_sources": sorted(self.raw_sources),  # layer names only (data not serialised)
        }
