"""
ml_multiomics.validation
========================
Grouping-aware resampling. Every routine REQUIRES an explicit ``groups`` vector
= the independent unit (psilocybin: bioreactor F#C#R#; banana: replicate). This
prevents pseudoreplication leakage (e.g. two timepoints of one bioreactor
splitting across train/test).

Also provides honest-resolution reporting: ``permutation_resolution`` reports the
finest achievable group-level permutation p-value given the design, so a single
p-value is never read without its discrete floor.
"""

from .resampling import (
    leave_one_group_out,
    group_kfold,
    permutation_resolution,
    grouped_permutation_test,
    grouped_bootstrap_indices,
)

__all__ = [
    "leave_one_group_out",
    "group_kfold",
    "permutation_resolution",
    "grouped_permutation_test",
    "grouped_bootstrap_indices",
]
