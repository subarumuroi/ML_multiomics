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
    
    def __init__(self, n_bootstrap: int = 1000):
        """
        Initialize bootstrap validator.
        
        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap samples
        """
        self.n_bootstrap = n_bootstrap
    
    def estimate_uncertainty(self,
                            model,
                            X: np.ndarray,
                            y: np.ndarray,
                            metric: Callable = accuracy_score) -> Dict:
        """
        Estimate model uncertainty via bootstrap.
        
        Parameters
        ----------
        model : sklearn model
            Model to evaluate
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        metric : callable
            Scoring metric
            
        Returns
        -------
        dict
            Bootstrap results with confidence intervals
        """
        n_samples = len(X)
        scores = []
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = resample(range(n_samples), n_samples=n_samples, random_state=i)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Out-of-bag samples
            oob_indices = list(set(range(n_samples)) - set(indices))
            if len(oob_indices) == 0:
                continue
            
            X_oob = X[oob_indices]
            y_oob = y[oob_indices]
            
            # Fit and evaluate
            model.fit(X_boot, y_boot)
            y_pred = model.predict(X_oob)
            score = metric(y_oob, y_pred)
            scores.append(score)
        
        scores = np.array(scores)
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'ci_lower': np.percentile(scores, 2.5),
            'ci_upper': np.percentile(scores, 97.5),
            'n_bootstrap': self.n_bootstrap
        }
        
        return results


class FeatureStabilityValidator:
    """
    Validate stability of feature selection across resampling.
    
    Important for omics where feature selection is critical.
    """
    
    def __init__(self, n_iterations: int = 100):
        """
        Initialize feature stability validator.
        
        Parameters
        ----------
        n_iterations : int
            Number of resampling iterations
        """
        self.n_iterations = n_iterations
    
    def assess_stability(self,
                        feature_selector: Callable,
                        X: np.ndarray,
                        y: np.ndarray,
                        feature_names: List[str],
                        top_k: int = 20) -> pd.DataFrame:
        """
        Assess stability of feature selection.
        
        Parameters
        ----------
        feature_selector : callable
            Function that takes (X, y) and returns selected feature indices
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        feature_names : list
            Feature names
        top_k : int
            Number of top features to track
            
        Returns
        -------
        pd.DataFrame
            Feature stability scores
        """
        n_samples, n_features = X.shape
        selection_counts = np.zeros(n_features)
        
        for i in range(self.n_iterations):
            # Resample
            indices = resample(range(n_samples), n_samples=n_samples, random_state=i)
            X_resample = X[indices]
            y_resample = y[indices]
            
            # Select features
            selected_indices = feature_selector(X_resample, y_resample)
            selection_counts[selected_indices] += 1
        
        # Calculate stability scores
        stability_scores = selection_counts / self.n_iterations
        
        # Create dataframe
        stability_df = pd.DataFrame({
            'Feature': feature_names,
            'Selection_Frequency': stability_scores,
            'Times_Selected': selection_counts.astype(int)
        })
        
        stability_df = stability_df.sort_values('Selection_Frequency', ascending=False)
        stability_df = stability_df.head(top_k).reset_index(drop=True)
        
        return stability_df


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


def validate_preprocessing_choices(preprocessor,
                                   df: pd.DataFrame,
                                   group_col: str,
                                   model,
                                   config_variations: List[Dict]) -> pd.DataFrame:
    """
    Validate different preprocessing configurations.
    
    Parameters
    ----------
    preprocessor : BasePreprocessor
        Preprocessor class
    df : pd.DataFrame
        Raw data
    group_col : str
        Group column
    model : sklearn model
        Model to evaluate
    config_variations : list
        List of config dictionaries to test
        
    Returns
    -------
    pd.DataFrame
        Results for each configuration
    """
    results = []
    
    for i, config in enumerate(config_variations):
        # Preprocess with this config
        prep = preprocessor(config=config)
        X, y, _ = prep.preprocess(df, group_col)
        
        # Validate
        validator = CrossValidator(strategy='loo')
        cv_results = validator.validate_model(model, X, y)
        
        results.append({
            'Config_ID': i,
            'Config': str(config),
            'Accuracy': cv_results['mean'],
            'Std': cv_results['std']
        })
    
    return pd.DataFrame(results).sort_values('Accuracy', ascending=False)