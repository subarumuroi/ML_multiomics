# ============================================================================
# src/ml_multiomics/workflows/__init__.py
# ============================================================================
"""Complete analysis workflows."""

from .single_omics_workflow import SingleOmicsWorkflow
from .multi_omics_workflow import MultiOmicsWorkflow

__all__ = ['SingleOmicsWorkflow', 'MultiOmicsWorkflow']
