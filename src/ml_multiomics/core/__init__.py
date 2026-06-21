"""
ml_multiomics.core
==================
The canonical data model shared by every method in the library.

Exports
-------
OmicsDataset          container: N omics blocks + sample metadata + target + provenance
Block                 one omics layer (samples x features) with a provenance log
TargetSpec            describes the prediction target (type + values)
AnalysisSpec          user-authored declaration of an analysis (roles/grouping/target)
ROLES                 the layer roles an AnalysisSpec can assign
parse_bioreactor_ids  optional helper to populate a grouping column from F#C#R#T# IDs
parse_delimited       optional helper to populate a grouping column from '<group>-<rep>' IDs
"""

from .dataset import OmicsDataset, Block, TargetSpec, TARGET_TYPES
from .metadata import parse_bioreactor_ids, parse_delimited
from .spec import AnalysisSpec, ROLES, KNOWN_TRANSFORMS

__all__ = [
    "OmicsDataset",
    "Block",
    "TargetSpec",
    "TARGET_TYPES",
    "AnalysisSpec",
    "ROLES",
    "KNOWN_TRANSFORMS",
    "parse_bioreactor_ids",
    "parse_delimited",
]
