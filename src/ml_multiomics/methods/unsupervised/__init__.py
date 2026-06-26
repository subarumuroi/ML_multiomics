"""
Unsupervised methods — including the "reducers".

Reducers (WGCNA here; NMF, PCA, MOFA to follow) collapse many features into a
small samples x factor/module representation via ``.reduce()`` (WGCNA) or
``.transform()`` (factor models). That reduced matrix can be fed directly into a
supervised method (RandomForest, SparsePLSDA, ...) or added back to an
OmicsDataset as a new block — the "reduce -> predict" pattern for p >> n data.
"""

from .wgcna_r import WGCNA            # WGCNA is the R WGCNA package only (validated)
from .nmf import NMF
from .pca import PCA

__all__ = ["WGCNA", "NMF", "PCA"]
