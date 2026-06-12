"""
ml_multiomics.core
==================
The canonical data model shared by every method in the library.

Exports
-------
OmicsDataset          container: N omics blocks + sample metadata + target + provenance
Block                 one omics layer (samples x features) with a provenance log
TargetSpec            describes the prediction target (type + values)
parse_bioreactor_ids  pluggable metadata parser for F#C#R#T# bioreactor IDs
parse_delimited       generic metadata parser for '<group>-<replicate>' style IDs
"""

from .dataset import OmicsDataset, Block, TargetSpec, TARGET_TYPES
from .metadata import parse_bioreactor_ids, parse_delimited

__all__ = [
    "OmicsDataset",
    "Block",
    "TargetSpec",
    "TARGET_TYPES",
    "parse_bioreactor_ids",
    "parse_delimited",
]
