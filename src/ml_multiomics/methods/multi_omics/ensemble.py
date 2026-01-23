"""
Multi-omics ensemble integration methods.

Block-wise ensemble approaches that train separate models on each omics layer
and combine predictions via voting or averaging.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report


class BlockWiseEnsemble:
    """
    Block-wise ensemble for multi-omics integration.
    
    Trains a separate classifier on each omics block and combines predictions
    via voting (hard) or probability averaging (soft).
    
    Advantages:
    - Robust for small sample sizes
    - Each block contributes independently
    - Interpretable block-specific importance
    - Handles blocks with different feature dimensions
    """
    
    def __init__(self,
                 classifier: str = 'random_forest',
                 voting: str = 'soft',
                 classifier_params: Optional[Dict] = None):
        """
        Initialize block-wise ensemble.
        
        Parameters
        ----------
        classifier : str
            Classifier type ('random_forest', 'svm', 'logistic')
        voting : str
            'hard' (majority vote) or 'soft' (probability averaging)
        classifier_params : dict, optional
            Parameters for the classifier
        """
        self.classifier_type = classifier
        self.voting = voting
        self.classifier_params = classifier_params or {}
        self.block_models = {}
        self.block_names = []
        self.feature_names = {}
        self.classes_ = None
        
    def _get_classifier(self):
        """Get classifier instance."""
        if self.classifier_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            params = {**default_params, **self.classifier_params}
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unsupported classifier: {self.classifier_type}")
    
    def fit(self,
            blocks: Dict[str, np.ndarray],
            y: np.ndarray,
            feature_names: Optional[Dict[str, List[str]]] = None):
        """
        Fit ensemble models on each block.
        
        Parameters
        ----------
        blocks : dict
            {block_name: X_block} where X_block is (n_samples, n_features)
        y : np.ndarray
            Labels (n_samples,)
        feature_names : dict, optional
            {block_name: [feature_names]} for each block
        """
        self.block_names = list(blocks.keys())
        self.classes_ = np.unique(y)
        self.feature_names = feature_names or {}
        
        print(f"\n{'='*70}")
        print("BLOCK-WISE ENSEMBLE TRAINING")
        print(f"{'='*70}")
        print(f"Training {len(blocks)} block-specific models...")
        print(f"Classifier: {self.classifier_type}")
        print(f"Voting strategy: {self.voting}")
        
        # Train one model per block
        for block_name, X_block in blocks.items():
            print(f"\nTraining model for block: {block_name}")
            print(f"  Shape: {X_block.shape}")
            
            model = self._get_classifier()
            model.fit(X_block, y)
            self.block_models[block_name] = model
            
            # Store feature names - use provided names or generate generic ones
            if feature_names and block_name in feature_names:
                self.feature_names[block_name] = feature_names[block_name]
            else:
                self.feature_names[block_name] = [
                    f"{block_name}_F{i}" for i in range(X_block.shape[1])
                ]
            
            print(f"  ✓ Model trained successfully")
        
        print(f"\n{'='*70}")
        print(f"Ensemble training complete!")
        print(f"{'='*70}\n")
        
        return self
    
    def predict(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict using ensemble voting/averaging.
        
        Parameters
        ----------
        blocks : dict
            {block_name: X_block} for prediction
            
        Returns
        -------
        np.ndarray
            Predicted labels
        """
        if self.voting == 'hard':
            return self._predict_hard(blocks)
        else:
            return self._predict_soft(blocks)
    
    def _predict_hard(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """Hard voting: majority vote."""
        n_samples = list(blocks.values())[0].shape[0]
        votes = np.zeros((n_samples, len(self.classes_)))
        
        for block_name, X_block in blocks.items():
            if block_name in self.block_models:
                preds = self.block_models[block_name].predict(X_block)
                # Convert to vote matrix
                for i, pred in enumerate(preds):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    votes[i, class_idx] += 1
        
        # Return class with most votes
        return self.classes_[np.argmax(votes, axis=1)]
    
    def _predict_soft(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """Soft voting: average probabilities."""
        n_samples = list(blocks.values())[0].shape[0]
        proba_sum = np.zeros((n_samples, len(self.classes_)))
        n_models = 0
        
        for block_name, X_block in blocks.items():
            if block_name in self.block_models:
                proba = self.block_models[block_name].predict_proba(X_block)
                proba_sum += proba
                n_models += 1
        
        # Average probabilities
        proba_avg = proba_sum / n_models
        return self.classes_[np.argmax(proba_avg, axis=1)]
    
    def predict_proba(self, blocks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get ensemble probability predictions.
        
        Parameters
        ----------
        blocks : dict
            {block_name: X_block}
            
        Returns
        -------
        np.ndarray
            Average probabilities (n_samples, n_classes)
        """
        n_samples = list(blocks.values())[0].shape[0]
        proba_sum = np.zeros((n_samples, len(self.classes_)))
        n_models = 0
        
        for block_name, X_block in blocks.items():
            if block_name in self.block_models:
                proba = self.block_models[block_name].predict_proba(X_block)
                proba_sum += proba
                n_models += 1
        
        return proba_sum / n_models
    
    def cross_validate(self,
                      blocks: Dict[str, np.ndarray],
                      y: np.ndarray,
                      feature_names: Optional[Dict[str, List[str]]] = None,
                      cv_strategy: str = 'loo') -> Dict:
        """
        Perform cross-validation on ensemble.
        
        Parameters
        ----------
        blocks : dict
            {block_name: X_block}
        y : np.ndarray
            Labels
        feature_names : dict, optional
            {block_name: [feature_names]} for each block
        cv_strategy : str
            'loo' for leave-one-out
            
        Returns
        -------
        dict
            CV results with predictions and metrics
        """
        if cv_strategy != 'loo':
            raise NotImplementedError("Only LOO CV currently supported")
        
        print(f"\n{'='*70}")
        print("ENSEMBLE CROSS-VALIDATION")
        print(f"{'='*70}")
        print(f"Strategy: Leave-One-Out")
        print(f"Total samples: {len(y)}")
        
        loo = LeaveOneOut()
        y_true = []
        y_pred = []
        
        for fold, (train_idx, test_idx) in enumerate(loo.split(list(blocks.values())[0]), 1):
            # Split each block
            train_blocks = {
                name: X[train_idx] for name, X in blocks.items()
            }
            test_blocks = {
                name: X[test_idx] for name, X in blocks.items()
            }
            
            # Train ensemble on fold
            self.fit(train_blocks, y[train_idx], feature_names=feature_names)
            
            # Predict
            pred = self.predict(test_blocks)
            
            y_true.append(y[test_idx][0])
            y_pred.append(pred[0])
            
            if fold % 3 == 0 or fold == len(y):
                print(f"  Fold {fold}/{len(y)} complete")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        print(f"\n{'='*70}")
        print(f"Leave-One-Out Accuracy: {accuracy*100:.2f}%")
        print(f"Balanced Accuracy: {balanced_acc*100:.2f}%")
        print(f"{'='*70}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'mean': accuracy,
            'std': 0.0  # LOO doesn't have std across folds
        }
        
        return results
    
    def get_block_importance(self, top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance from each block's model.
        
        Parameters
        ----------
        top_n : int
            Number of top features per block
            
        Returns
        -------
        dict
            {block_name: DataFrame} with feature importance
        """
        importance_dict = {}
        
        for block_name, model in self.block_models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = self.feature_names.get(
                    block_name,
                    [f"{block_name}_F{i}" for i in range(len(importance))]
                )
                
                # Create DataFrame
                df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                })
                df = df.sort_values('Importance', ascending=False)
                df = df.head(top_n).reset_index(drop=True)
                
                importance_dict[block_name] = df
        
        return importance_dict
    
    def get_ensemble_summary(self) -> pd.DataFrame:
        """
        Get summary of ensemble models.
        
        Returns
        -------
        pd.DataFrame
            Summary of each block's model
        """
        summary = []
        
        for block_name, model in self.block_models.items():
            n_features = len(self.feature_names.get(block_name, []))
            
            summary.append({
                'Block': block_name,
                'N_Features': n_features,
                'Classifier': self.classifier_type,
                'Model': str(type(model).__name__)
            })
        
        return pd.DataFrame(summary)
