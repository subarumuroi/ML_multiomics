"""
Simple concatenation-based multi-omics integration.

Provides baseline for comparison with more sophisticated methods.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Tuple, Optional


class ConcatenationBaseline:
    """
    Simple concatenation approach for multi-omics integration.
    
    Strategy:
    1. Horizontally concatenate all omics layers
    2. Apply standard ML classifier
    
    Serves as baseline to compare against DIABLO and other integration methods.
    """
    
    def __init__(self, 
                 classifier: str = 'random_forest',
                 classifier_params: Optional[Dict] = None):
        """
        Initialize concatenation baseline.
        
        Parameters
        ----------
        classifier : str
            Type of classifier ('random_forest', 'svm', 'logistic')
        classifier_params : dict, optional
            Parameters for the classifier
        """
        self.classifier_type = classifier
        self.classifier_params = classifier_params or {}
        self.model = None
        self.feature_names = None
        
    def _get_classifier(self):
        """Create classifier instance."""
        if self.classifier_type == 'random_forest':
            default_params = {
                'n_estimators': 500,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(self.classifier_params)
            return RandomForestClassifier(**default_params)
        
        elif self.classifier_type == 'svm':
            default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'random_state': 42
            }
            default_params.update(self.classifier_params)
            return SVC(**default_params)
        
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[list] = None):
        """
        Fit classifier on concatenated data.
        
        Parameters
        ----------
        X : np.ndarray
            Concatenated feature matrix
        y : np.ndarray
            Group labels
        feature_names : list, optional
            Feature names
        """
        self.model = self._get_classifier()
        self.model.fit(X, y)
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X.shape[1])]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray
            Concatenated features
            
        Returns
        -------
        np.ndarray
            Predicted labels
        """
        return self.model.predict(X)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                      cv: int = None) -> Dict:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        cv : int, optional
            Number of folds (default: Leave-One-Out)
            
        Returns
        -------
        dict
            Cross-validation results
        """
        if cv is None:
            cv = LeaveOneOut()
            cv_name = "Leave-One-Out"
        else:
            cv_name = f"{cv}-Fold"
        
        model = self._get_classifier()
        
        # Get predictions
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        # Get CV scores
        scores = cross_val_score(model, X, y, cv=cv)
        
        results = {
            'cv_type': cv_name,
            'accuracy': scores.mean(),
            'std': scores.std(),
            'scores': scores,
            'y_true': y,
            'y_pred': y_pred,
            'classification_report': classification_report(y, y_pred)
        }
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance (only for tree-based models).
        
        Parameters
        ----------
        top_n : int
            Number of top features
            
        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df = importance_df.head(top_n).reset_index(drop=True)
        
        return importance_df


class WeightedConcatenation:
    """
    Weighted concatenation approach.
    
    Applies different weights to different omics blocks before concatenation.
    Can help balance blocks with very different numbers of features.
    """
    
    def __init__(self, 
                 block_weights: Optional[Dict[str, float]] = None,
                 classifier: str = 'random_forest',
                 classifier_params: Optional[Dict] = None):
        """
        Initialize weighted concatenation.
        
        Parameters
        ----------
        block_weights : dict, optional
            {block_name: weight} for each block
            If None, uses equal weights
        classifier : str
            Type of classifier
        classifier_params : dict, optional
            Classifier parameters
        """
        self.block_weights = block_weights
        self.classifier_type = classifier
        self.classifier_params = classifier_params or {}
        self.model = None
        self.block_indices = None
        
    def fit_from_blocks(self, 
                       blocks: Dict[str, np.ndarray],
                       y: np.ndarray,
                       feature_names_dict: Optional[Dict[str, list]] = None):
        """
        Fit from separate blocks.
        
        Parameters
        ----------
        blocks : dict
            {block_name: X_block} dictionary
        y : np.ndarray
            Labels
        feature_names_dict : dict, optional
            {block_name: feature_names} dictionary
        """
        block_names = list(blocks.keys())
        
        # Set default weights if not provided
        if self.block_weights is None:
            self.block_weights = {name: 1.0 for name in block_names}
        
        # Apply weights and concatenate
        weighted_blocks = []
        self.block_indices = {}
        start_idx = 0
        
        for name in block_names:
            X_block = blocks[name]
            weight = self.block_weights.get(name, 1.0)
            
            # Apply weight
            X_weighted = X_block * weight
            weighted_blocks.append(X_weighted)
            
            # Track indices
            end_idx = start_idx + X_block.shape[1]
            self.block_indices[name] = (start_idx, end_idx)
            start_idx = end_idx
        
        X_concat = np.hstack(weighted_blocks)
        
        # Create combined feature names
        if feature_names_dict:
            combined_features = []
            for name in block_names:
                features = feature_names_dict[name]
                combined_features.extend([f"{name}_{f}" for f in features])
            self.feature_names = combined_features
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X_concat.shape[1])]
        
        # Fit model
        baseline = ConcatenationBaseline(
            classifier=self.classifier_type,
            classifier_params=self.classifier_params
        )
        baseline.fit(X_concat, y, self.feature_names)
        self.model = baseline.model
        
        return self
    
    def predict(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict from blocks.
        
        Parameters
        ----------
        blocks : dict
            {block_name: X_block} dictionary
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        # Apply weights and concatenate
        weighted_blocks = []
        for name in sorted(blocks.keys()):
            X_block = blocks[name]
            weight = self.block_weights.get(name, 1.0)
            weighted_blocks.append(X_block * weight)
        
        X_concat = np.hstack(weighted_blocks)
        return self.model.predict(X_concat)


class EarlyIntegration:
    """
    Early integration approach with multiple strategies.
    
    Combines different preprocessing and feature selection strategies
    for concatenated multi-omics data.
    """
    
    def __init__(self, 
                 strategy: str = 'simple',
                 feature_selection: Optional[str] = None,
                 n_features: Optional[int] = None):
        """
        Initialize early integration.
        
        Parameters
        ----------
        strategy : str
            Integration strategy:
            - 'simple': direct concatenation
            - 'weighted': weight by block size
            - 'normalized': normalize block contributions
        feature_selection : str, optional
            Feature selection method ('variance', 'mutual_info')
        n_features : int, optional
            Number of features to select
        """
        self.strategy = strategy
        self.feature_selection = feature_selection
        self.n_features = n_features
        
    def integrate_blocks(self, 
                        blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Integrate multiple blocks into single matrix.
        
        Parameters
        ----------
        blocks : dict
            {block_name: X_block} dictionary
            
        Returns
        -------
        np.ndarray
            Integrated feature matrix
        """
        if self.strategy == 'simple':
            return np.hstack(list(blocks.values()))
        
        elif self.strategy == 'weighted':
            # Weight inversely by number of features
            weighted_blocks = []
            for name, X_block in blocks.items():
                weight = 1.0 / np.sqrt(X_block.shape[1])
                weighted_blocks.append(X_block * weight)
            return np.hstack(weighted_blocks)
        
        elif self.strategy == 'normalized':
            # Normalize each block's contribution
            normalized_blocks = []
            for name, X_block in blocks.items():
                # Scale so each block has similar total variance
                block_std = np.std(X_block)
                normalized_blocks.append(X_block / (block_std + 1e-10))
            return np.hstack(normalized_blocks)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")