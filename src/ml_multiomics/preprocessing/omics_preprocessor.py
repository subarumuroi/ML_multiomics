"""
Omics-specific preprocessors for metabolomics, volatiles, and proteomics data.

Each preprocessor handles the unique characteristics of its omics type.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_preprocessor import BasePreprocessor


class MetabolomicsPreprocessor(BasePreprocessor):
    """
    Preprocessor for metabolomics concentration data.
    
    Features:
    - Group-wise median imputation (assumes MAR - Missing At Random)
    - Log transformation (metabolite concentrations are log-normal)
    - Pareto or standard scaling
    - Conservative feature filtering
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(omics_type='metabolomics', config=config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for metabolomics."""
        return {
            'drop_threshold': 0.5,      # Drop features with >50% missing
            'fill_value': 0,             # Fill remaining NaNs with 0 (undetected)
            'transform': 'log',          # Log transform concentrations
            'scaling': 'pareto',         # Pareto scaling preserves structure
            'handle_negatives': True,    # Shift negative values if present
        }
    
    def handle_missing(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """
        Handle missing values for metabolomics data.
        
        Strategy:
        1. Drop columns that are entirely NaN (no information)
        2. Fill remaining NaN with fill_value (default: 0)
        
        ASSUMPTION: Missing values represent undetected/absent metabolites.
        This assumes Missing Not At Random (MNAR) - common for metabolomics
        where values below detection limit are recorded as missing.
        
        WARNING: If your data has a different missing mechanism (e.g., random
        technical failures), consider using group-wise median imputation instead.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with potential missing values
        group_col : str
            Column name for group labels (excluded from imputation)
            
        Returns
        -------
        pd.DataFrame
            Data with missing values handled
        """
        df = df.copy()
        feature_cols = [c for c in df.columns if c != group_col]
        # Drop columns that are all-NaN
        all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
        if all_nan_cols:
            self._log(f"Dropping {len(all_nan_cols)} all-NaN columns")
            df = df.drop(columns=all_nan_cols)
        # Fill remaining missing values in numeric columns
        feature_cols = [c for c in df.columns if c != group_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        fill_value = self.config.get('fill_value', 0)
        n_missing = df[numeric_cols].isna().sum().sum()
        if n_missing > 0:
            self._log(f"Filling {n_missing} missing values with {fill_value} (assumes undetected)")
        df[numeric_cols] = df[numeric_cols].fillna(fill_value)
        return df
    
    def apply_transformation(self, X: np.ndarray) -> np.ndarray:
        """
        Apply transformation with special handling for negative values.
        
        Metabolomics data should be non-negative, but sometimes baseline
        correction or normalization can produce negative values.
        """
        if self.config.get('handle_negatives', True):
            if (X < 0).any():
                # Per-feature shift to positive range
                n_negatives = (X < 0).sum()
                self._log(f"Warning: Found {n_negatives} negative values")
                
                # Shift each feature independently
                for j in range(X.shape[1]):
                    if (X[:, j] < 0).any():
                        shift = abs(X[:, j].min()) + 1
                        X[:, j] = X[:, j] + shift
                
                self._log("Shifted negative features to positive range")
        
        # Apply log transformation
        return super().apply_transformation(X)


class VolatilesPreprocessor(BasePreprocessor):
    """
    Preprocessor for volatile compounds (GC-MS area counts).
    
    Different from metabolomics because:
    - Area counts (not concentrations)
    - Often higher sparsity
    - May not need log transform depending on data
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(omics_type='volatiles', config=config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for volatiles."""
        return {
            'drop_threshold': 0.6,       # More lenient for sparse volatile data
            'fill_value': 0,              # Undetected = absent
            'transform': 'log',           # Log transform helps with skewness
            'scaling': 'pareto',        # changed from standard to pareto for DIABLO consistency
            'handle_negatives': False,    # Area counts shouldn't be negative
        }
    
    def handle_missing(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """
        Drop columns (except group_col) that are all-NaN, then fill remaining missing numeric values with fill_value (default 0).
        No group-wise imputation is performed. This treats missing as undetected/absent.
        """
        df = df.copy()
        feature_cols = [c for c in df.columns if c != group_col]
        all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)
        feature_cols = [c for c in df.columns if c != group_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        fill_value = self.config.get('fill_value', 0)
        df[numeric_cols] = df[numeric_cols].fillna(fill_value)
        return df


class ProteomicsPreprocessor(BasePreprocessor):
    """
    Preprocessor for proteomics data.
    
    Special considerations:
    - Often pre-imputed by acquisition software
    - Different missing data mechanisms (MNAR common)
    - Wide dynamic range
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(omics_type='proteomics', config=config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for proteomics."""
        return {
            'drop_threshold': 0.3,        # Stricter for proteomics
            'fill_value': 0,            # Should be imputed, but 0 for missing values
            'transform': 'log2',           # Log2 is standard for proteomics
            'scaling': 'pareto',         # changed from standard to pareto for DIABLO consistency
            'handle_negatives': False,     # Shouldn't have negatives
        }
    
    def handle_missing(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """
        Drop columns (except group_col) that are all-NaN, then impute remaining missing numeric values with group-wise median.
        """
        df = df.copy()
        feature_cols = [c for c in df.columns if c != group_col]
        # Drop columns that are all-NaN
        all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)
        feature_cols = [c for c in df.columns if c != group_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        # Group-wise median imputation for numeric columns with missing values
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df.groupby(group_col)[col].transform(lambda x: x.fillna(x.median()))
        return df