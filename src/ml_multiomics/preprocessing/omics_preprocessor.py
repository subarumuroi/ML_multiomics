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
            'imputation': 'half_min',    # Half-minimum imputation (better than 0 for log)
            'transform': 'log',          # Log transform concentrations
            'scaling': 'pareto',         # Pareto scaling preserves structure
            'handle_negatives': True,    # Shift negative values if present
        }
    
    def handle_missing(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """
        Handle missing values for metabolomics data.
        
        Strategy:
        1. Drop columns that are entirely NaN (no information)
        2. Fill remaining NaN with half-minimum of each feature
           (better than 0 for subsequent log transform)
        
        ASSUMPTION: Missing values represent undetected/absent metabolites.
        This assumes Missing Not At Random (MNAR) - common for metabolomics
        where values below detection limit are recorded as missing.
        
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
        
        imputation = self.config.get('imputation', 'half_min')
        n_missing = df[numeric_cols].isna().sum().sum()
        
        if n_missing > 0:
            if imputation == 'half_min':
                # Half-minimum imputation: fill with half of the minimum positive value
                for col in numeric_cols:
                    if df[col].isna().any():
                        positive_vals = df[col][df[col] > 0]
                        if len(positive_vals) > 0:
                            half_min = positive_vals.min() / 2
                        else:
                            half_min = 1e-10  # Fallback for all-zero columns
                        df[col] = df[col].fillna(half_min)
                self._log(f"Filled {n_missing} missing values with half-minimum (MNAR assumption)")
            else:
                # Fallback to constant fill
                fill_value = self.config.get('fill_value', 0)
                df[numeric_cols] = df[numeric_cols].fillna(fill_value)
                self._log(f"Filled {n_missing} missing values with {fill_value}")
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
            'imputation': 'half_min',     # Half-minimum imputation (better for log)
            'apply_tsn': True,            # Total Sum Normalization (high CV in volatiles)
            'transform': 'log',           # Log transform helps with skewness
            'scaling': 'pareto',          # Pareto scaling for DIABLO consistency
            'handle_negatives': False,    # Area counts shouldn't be negative
        }
    
    def handle_missing(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """
        Handle missing values for volatiles data:
        1. Drop columns that are all-NaN
        2. Apply Total Sum Normalization (TSN) if enabled (reduces sample-to-sample variability)
        3. Fill remaining missing with half-minimum of each feature
        """
        df = df.copy()
        feature_cols = [c for c in df.columns if c != group_col]
        all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
        if all_nan_cols:
            self._log(f"Dropping {len(all_nan_cols)} all-NaN columns")
            df = df.drop(columns=all_nan_cols)
        
        feature_cols = [c for c in df.columns if c != group_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        # Apply Total Sum Normalization (TSN) if enabled
        # This normalizes each sample to the same total, reducing injection variability
        if self.config.get('apply_tsn', True):
            row_sums = df[numeric_cols].sum(axis=1)
            median_total = row_sums.median()
            cv_before = row_sums.std() / row_sums.mean() * 100
            
            # Normalize each row to median total (preserves scale)
            for col in numeric_cols:
                df[col] = df[col] / row_sums * median_total
            
            cv_after = df[numeric_cols].sum(axis=1).std() / df[numeric_cols].sum(axis=1).mean() * 100
            self._log(f"Applied TSN: CV reduced from {cv_before:.1f}% to {cv_after:.1f}%")
        
        # Half-minimum imputation
        n_missing = df[numeric_cols].isna().sum().sum()
        if n_missing > 0:
            imputation = self.config.get('imputation', 'half_min')
            if imputation == 'half_min':
                for col in numeric_cols:
                    if df[col].isna().any():
                        positive_vals = df[col][df[col] > 0]
                        if len(positive_vals) > 0:
                            half_min = positive_vals.min() / 2
                        else:
                            half_min = 1e-10
                        df[col] = df[col].fillna(half_min)
                self._log(f"Filled {n_missing} missing values with half-minimum")
            else:
                fill_value = self.config.get('fill_value', 0)
                df[numeric_cols] = df[numeric_cols].fillna(fill_value)
                self._log(f"Filled {n_missing} missing values with {fill_value}")
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
            'imputation': 'group_median',  # Group-wise median, then half-min fallback
            'transform': 'log2',           # Log2 is standard for proteomics
            'scaling': 'pareto',           # Pareto scaling for DIABLO consistency
            'handle_negatives': False,     # Shouldn't have negatives
        }
    
    def handle_missing(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """
        Handle missing values for proteomics data:
        1. Drop columns that are all-NaN
        2. Group-wise median imputation (assumes MAR within groups)
        3. Half-minimum fallback for remaining NaNs (if protein missing in entire group)
        """
        df = df.copy()
        feature_cols = [c for c in df.columns if c != group_col]
        # Drop columns that are all-NaN
        all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
        if all_nan_cols:
            self._log(f"Dropping {len(all_nan_cols)} all-NaN columns")
            df = df.drop(columns=all_nan_cols)
        
        feature_cols = [c for c in df.columns if c != group_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        
        n_missing_before = df[numeric_cols].isna().sum().sum()
        
        # Step 1: Group-wise median imputation
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df.groupby(group_col)[col].transform(lambda x: x.fillna(x.median()))
        
        n_after_group = df[numeric_cols].isna().sum().sum()
        n_filled_group = n_missing_before - n_after_group
        if n_filled_group > 0:
            self._log(f"Group-wise median imputed {n_filled_group} values")
        
        # Step 2: Half-minimum fallback for remaining NaNs
        # (occurs when protein is missing in ALL samples of a group)
        if n_after_group > 0:
            for col in numeric_cols:
                if df[col].isna().any():
                    positive_vals = df[col][df[col] > 0]
                    if len(positive_vals) > 0:
                        half_min = positive_vals.min() / 2
                    else:
                        half_min = 1e-10
                    df[col] = df[col].fillna(half_min)
            self._log(f"Half-minimum fallback filled {n_after_group} remaining values")
        
        return df