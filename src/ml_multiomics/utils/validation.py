"""
Validation utilities for omics analysis.

Provides cross-validation, permutation tests, and statistical validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils import resample
from typing import Dict, List, Tuple, Optional, Callable
import warnings


class CrossValidator:
    """
    Cross-validation framework for omics data.
    
    Handles small sample sizes common in omics studies.
    """
    
    def __init__(self, strategy: str = 'loo'):
        """
        Initialize cross-validator.
        
        Parameters
        ----------
        strategy : str
            CV strategy: 'loo' (Leave-One-Out), 'kfold', 'stratified'
        """
        self.strategy = strategy
        
    def get_cv_splitter(self, n_samples: int, y: Optional[np.ndarray] = None, k: int = 5):
        """
        Get appropriate CV splitter.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        y : np.ndarray, optional
            Labels for stratification
        k : int
            Number of folds (for k-fold)
            
        Returns
        -------
        CV splitter object
        """
        if self.strategy == 'loo':
            return LeaveOneOut()
        
        elif self.strategy == 'kfold':
            return StratifiedKFold(n_splits=min(k, n_samples), shuffle=True, random_state=42)
        
        elif self.strategy == 'stratified':
            return StratifiedKFold(n_splits=min(k, n_samples), shuffle=True, random_state=42)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def validate_model(self,
                      model,
                      X: np.ndarray,
                      y: np.ndarray,
                      k: int = 5,
                      scoring: str = 'accuracy') -> Dict:
        """
        Perform cross-validation on a model.
        
        Parameters
        ----------
        model : sklearn model
            Model to validate
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        k : int
            Number of folds
        scoring : str
            Scoring metric
            
        Returns
        -------
        dict
            CV results
        """
        cv_splitter = self.get_cv_splitter(len(X), y, k)
        
        # Get CV scores
        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)
        
        # Get predictions
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(model, X, y, cv=cv_splitter)
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'y_pred': y_pred,
            'y_true': y,
            'strategy': self.strategy,
            'n_splits': cv_splitter.get_n_splits(X, y)
        }
        
        return results


class PermutationTest:
    """
    Permutation testing for statistical validation.
    
    Tests whether model performance is better than random.
    """
    
    def __init__(self, n_permutations: int = 1000):
        """
        Initialize permutation test.
        
        Parameters
        ----------
        n_permutations : int
            Number of permutations
        """
        self.n_permutations = n_permutations
        
    def test_model(self,
                   model,
                   X: np.ndarray,
                   y: np.ndarray,
                   cv_strategy: str = 'loo',
                   metric: Callable = accuracy_score) -> Dict:
        """
        Perform permutation test on model.
        
        Parameters
        ----------
        model : sklearn model
            Model to test
        X : np.ndarray
            Features
        y : np.ndarray
            True labels
        cv_strategy : str
            Cross-validation strategy
        metric : callable
            Scoring metric
            
        Returns
        -------
        dict
            Test results including p-value
        """
        # Get true performance
        cv = CrossValidator(strategy=cv_strategy)
        cv_results = cv.validate_model(model, X, y)
        true_score = metric(cv_results['y_true'], cv_results['y_pred'])
        
        # Permutation scores
        perm_scores = []
        
        for i in range(self.n_permutations):
            # Shuffle labels
            y_perm = resample(y, replace=False, random_state=i)
            
            # Get permuted performance
            perm_cv = cv.validate_model(model, X, y_perm)
            perm_score = metric(perm_cv['y_true'], perm_cv['y_pred'])
            perm_scores.append(perm_score)
        
        perm_scores = np.array(perm_scores)
        
        # Calculate p-value
        p_value = (np.sum(perm_scores >= true_score) + 1) / (self.n_permutations + 1)
        
        results = {
            'true_score': true_score,
            'perm_scores': perm_scores,
            'perm_mean': perm_scores.mean(),
            'perm_std': perm_scores.std(),
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_permutations': self.n_permutations
        }
        
        return results


class BootstrapValidator:
    """
    Bootstrap validation for uncertainty estimation.
    """
    
class ModelComparator:
    """
    Compare multiple models on same dataset.
    """
    
    def __init__(self, cv_strategy: str = 'loo'):
        """
        Initialize model comparator.
        
        Parameters
        ----------
        cv_strategy : str
            Cross-validation strategy
        """
        self.cv_strategy = cv_strategy
        self.validator = CrossValidator(strategy=cv_strategy)
    
    def compare_models(self,
                      models: Dict[str, any],
                      X: np.ndarray,
                      y: np.ndarray,
                      metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Parameters
        ----------
        models : dict
            {model_name: model_instance} dictionary
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        metrics : list, optional
            Metrics to compute
            
        Returns
        -------
        pd.DataFrame
            Comparison results
        """
        if metrics is None:
            metrics = ['accuracy']
        
        results = []
        
        for model_name, model in models.items():
            model_results = {'Model': model_name}
            
            for metric in metrics:
                cv_results = self.validator.validate_model(
                    model, X, y, scoring=metric
                )
                model_results[f'{metric}_mean'] = cv_results['mean']
                model_results[f'{metric}_std'] = cv_results['std']
            
            results.append(model_results)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values(f'{metrics[0]}_mean', ascending=False)
        
        return comparison_df


