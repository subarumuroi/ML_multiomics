"""
Preprocessor for metabolomics data (amino acids, central carbon metabolism).

Handles concentration data with appropriate imputation and transformation.
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
        Handle missing values using group-wise median imputation.
        
        Strategy:
        1. Impute with group median (within Green/Ripe/Overripe)
        2. Fill remaining NaNs with fill_value (assumed absent)
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with missing values
        group_col : str
            Column containing group labels
            
        Returns
        -------
        pd.DataFrame
            Imputed dataframe
        """
        df = df.copy()
        feature_cols = [c for c in df.columns if c != group_col]
        
        # Group-wise median imputation
        def impute_group(group):
            group = group.copy()
            for col in feature_cols:
                if group[col].isna().any():
                    median_val = group[col].median()
                    if not pd.isna(median_val):
                        group[col] = group[col].fillna(median_val)
            return group
        
        df = df.groupby(group_col, group_keys=False).apply(impute_group)
        
        # Fill remaining NaNs
        fill_value = self.config.get('fill_value', 0)
        df[feature_cols] = df[feature_cols].fillna(fill_value)
        
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
            'scaling': 'standard',        # Standard scaling for area counts
            'handle_negatives': False,    # Area counts shouldn't be negative
        }
    
    def handle_missing(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """
        Handle missing values in volatile data.
        
        Volatiles are often truly absent (below detection limit),
        so we're more conservative with imputation.
        """
        df = df.copy()
        feature_cols = [c for c in df.columns if c != group_col]
        
        # Group-wise median imputation (more conservative)
        def impute_group(group):
            group = group.copy()
            for col in feature_cols:
                if group[col].isna().any():
                    # Only impute if at least 50% of group has values
                    if group[col].notna().sum() >= len(group) * 0.5:
                        median_val = group[col].median()
                        group[col] = group[col].fillna(median_val)
            return group
        
        df = df.groupby(group_col, group_keys=False).apply(impute_group)
        
        # Fill remaining with 0 (absent)
        fill_value = self.config.get('fill_value', 0)
        df[feature_cols] = df[feature_cols].fillna(fill_value)
        
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
            'fill_value': None,            # Don't fill - data is pre-imputed
            'transform': 'log2',           # Log2 is standard for proteomics
            'scaling': 'standard',         # Z-score normalization
            'handle_negatives': False,     # Shouldn't have negatives
        }
    
    def handle_missing(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """
        Handle missing values in proteomics data.
        
        For imputed proteomics data, we assume missing values are minimal.
        If using unimputed data, would need more sophisticated methods.
        """
        df = df.copy()
        feature_cols = [c for c in df.columns if c != group_col]
        
        n_missing = df[feature_cols].isna().sum().sum()
        
        if n_missing > 0:
            self._log(f"Warning: Found {n_missing} missing values in imputed proteomics data")
            
            # Simple median imputation as fallback
            fill_value = self.config.get('fill_value')
            if fill_value is not None:
                df[feature_cols] = df[feature_cols].fillna(fill_value)
            else:
                # Use global median
                df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        return df