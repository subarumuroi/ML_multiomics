"""
Base preprocessor class for omics data.

Provides common interface and utilities for all omics-specific preprocessors.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Dict, Any


class BasePreprocessor(ABC):
    """
    Abstract base class for omics preprocessing.
    
    Each omics type (metabolomics, proteomics, etc.) should inherit from this
    and implement omics-specific preprocessing logic.
    """
    
    def __init__(self, omics_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        omics_type : str
            Type of omics data ('metabolomics', 'proteomics', 'volatiles')
        config : dict, optional
            Configuration overrides for preprocessing parameters
        """
        self.omics_type = omics_type
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.scaler = None
        self.feature_names = None
        self.preprocessing_log = []
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration for this omics type."""
        pass
    
    @abstractmethod
    def handle_missing(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """
        Handle missing values in omics-specific way.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with potential missing values
        group_col : str
            Column containing group labels for group-wise imputation
            
        Returns
        -------
        pd.DataFrame
            Data with missing values handled
        """
        pass
    
    def filter_low_quality_features(self, df: pd.DataFrame, 
                                     group_col: str) -> pd.DataFrame:
        """
        Remove features with too many missing values.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data
        group_col : str
            Column name for grouping variable
            
        Returns
        -------
        pd.DataFrame
            Filtered dataframe
        """
        threshold = self.config.get('drop_threshold', 0.5)
        
        # Separate features from group column
        feature_cols = [c for c in df.columns if c != group_col]
        
        # Calculate missing proportion per feature
        missing_prop = df[feature_cols].isna().sum() / len(df)
        
        # Keep features below threshold
        features_to_keep = missing_prop[missing_prop <= threshold].index.tolist()
        
        n_dropped = len(feature_cols) - len(features_to_keep)
        if n_dropped > 0:
            self._log(f"Dropped {n_dropped} features with >{threshold*100}% missing values")
        
        return df[[group_col] + features_to_keep]
    
    def apply_transformation(self, X: np.ndarray) -> np.ndarray:
        """
        Apply mathematical transformation (log, etc.).
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Transformed features
        """
        transform_type = self.config.get('transform', None)
        
        if transform_type == 'log':
            # Add small constant to avoid log(0)
            epsilon = 1e-10
            X_transformed = np.log(X + epsilon)
            self._log(f"Applied log transformation with epsilon={epsilon}")
            return X_transformed
        
        elif transform_type == 'log2':
            epsilon = 1e-10
            X_transformed = np.log2(X + epsilon)
            self._log(f"Applied log2 transformation with epsilon={epsilon}")
            return X_transformed
        
        return X
    
    def apply_scaling(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Apply scaling to features.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        fit : bool
            Whether to fit the scaler (True) or use existing (False)
            
        Returns
        -------
        np.ndarray
            Scaled features
        """
        scaling_type = self.config.get('scaling', 'standard')
        
        if scaling_type == 'standard':
            if fit:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                self._log("Applied standard scaling (z-score)")
            else:
                X_scaled = self.scaler.transform(X)
        
        elif scaling_type == 'minmax':
            if fit:
                self.scaler = MinMaxScaler()
                X_scaled = self.scaler.fit_transform(X)
                self._log("Applied min-max scaling")
            else:
                X_scaled = self.scaler.transform(X)
        
        elif scaling_type == 'pareto':
            # Pareto scaling: mean-centered, divided by sqrt(std)
            if fit:
                self.mean_ = np.mean(X, axis=0)
                self.std_ = np.std(X, axis=0)
                X_scaled = (X - self.mean_) / np.sqrt(self.std_)
                self._log("Applied Pareto scaling")
            else:
                X_scaled = (X - self.mean_) / np.sqrt(self.std_)
        
        else:
            X_scaled = X
            self._log("No scaling applied")
        
        return X_scaled
    
    def preprocess(self, df: pd.DataFrame, 
                   group_col: str = 'Groups') -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Complete preprocessing pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data with group column and features
        group_col : str
            Name of column containing group labels
            
        Returns
        -------
        X : np.ndarray
            Preprocessed feature matrix
        y : np.ndarray
            Group labels
        feature_names : list
            Names of features in X
        """
        self._log(f"Starting preprocessing for {self.omics_type}")
        self._log(f"Initial shape: {df.shape}")
        
        # Step 1: Filter low-quality features
        df = self.filter_low_quality_features(df, group_col)
        self._log(f"After filtering: {df.shape}")
        
        # Step 2: Handle missing values (omics-specific)
        df = self.handle_missing(df, group_col)
        self._log(f"After imputation: {df[df.columns[df.columns != group_col]].isna().sum().sum()} missing values remain")
        
        # Step 3: Separate features and labels
        feature_cols = [c for c in df.columns if c != group_col]

        # keep only numeric cols (ignore 'sample name/ids' if present)
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].values.astype(float)
        y = df[group_col].astype(str).values
        self.feature_names = list(numeric_cols)

        # there was a bug where sample name strings were causeing errors downstream
        X = np.array(X, dtype=float)
        
        # Step 4: Apply transformation
        X = self.apply_transformation(X)
        
        # Step 5: Apply scaling
        X = self.apply_scaling(X, fit=True)
        
        self._log(f"Preprocessing complete. Final shape: {X.shape}")
        
        return X, y, self.feature_names
    
    def _log(self, message: str):
        """Add message to preprocessing log."""
        self.preprocessing_log.append(message)
    
    def get_log(self) -> list:
        """Return preprocessing log."""
        return self.preprocessing_log
    
    def print_log(self):
        """Print preprocessing log."""
        print("\n".join(self.preprocessing_log))