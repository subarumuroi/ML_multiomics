"""
Partial Least Squares Discriminant Analysis (PLS-DA).

Supervised dimensionality reduction and classification for omics data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from typing import Dict, List, Optional, Tuple


class PLSDAAnalysis:
    """
    PLS-DA for omics classification.
    
    Features:
    - Multi-class classification
    - Cross-validation
    - VIP scores for feature importance
    - Scores and loadings plots
    - Performance metrics
    """
    
    def __init__(self, n_components: int = 2):
        """
        Initialize PLS-DA.
        
        Parameters
        ----------
        n_components : int
            Number of latent variables
        """
        self.n_components = n_components
        self.pls = None
        self.label_binarizer = None
        self.classes = None
        self.scores = None
        self.loadings = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None):
        """
        Fit PLS-DA model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples × n_features)
        y : np.ndarray
            Group labels
        feature_names : list, optional
            Feature names
        """
        # Encode labels as binary matrix
        self.label_binarizer = LabelBinarizer()
        Y = self.label_binarizer.fit_transform(y)
        
        # Handle binary case (LabelBinarizer returns 1D for binary)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        self.classes = self.label_binarizer.classes_
        
        # Fit PLS
        self.pls = PLSRegression(n_components=self.n_components, scale=False)
        self.pls.fit(X, Y)
        
        # Store results
        self.scores = self.pls.x_scores_  # Scores (T)
        self.loadings = self.pls.x_loadings_  # Loadings (P)
        # same fix as for pca
        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predicted class labels
        """
        Y_pred = self.pls.predict(X)
        
        # Convert continuous predictions to class labels
        # For multi-class: argmax across columns
        if Y_pred.shape[1] > 1:
            class_indices = np.argmax(Y_pred, axis=1)
        else:
            # Binary case
            class_indices = (Y_pred.ravel() > 0.5).astype(int)
        
        return self.classes[class_indices]
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       cv: int = None) -> Dict[str, any]:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
        cv : int, optional
            Number of folds (default: Leave-One-Out)
            
        Returns
        -------
        dict
            Cross-validation results
        """
        if cv is None:
            # Use Leave-One-Out for small datasets
            cv = LeaveOneOut()
            cv_name = "Leave-One-Out"
        else:
            cv_name = f"{cv}-Fold"
        
        # Encode labels
        Y = self.label_binarizer.transform(y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        # Get CV predictions
        Y_pred_cv = cross_val_predict(self.pls, X, Y, cv=cv)
        
        # Convert to class labels
        if Y_pred_cv.shape[1] > 1:
            y_pred = self.classes[np.argmax(Y_pred_cv, axis=1)]
        else:
            y_pred = self.classes[(Y_pred_cv.ravel() > 0.5).astype(int)]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        
        results = {
            'cv_type': cv_name,
            'accuracy': accuracy,
            'y_true': y,
            'y_pred': y_pred,
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }
        
        return results
    
    def get_vip_scores(self) -> pd.DataFrame:
        """
        Calculate Variable Importance in Projection (VIP) scores.
        
        VIP scores measure the importance of each feature in the PLS model.
        VIP > 1 is commonly used as a threshold for important features.
        
        Returns
        -------
        pd.DataFrame
            Features ranked by VIP score
        """
        # Get model parameters
        W = self.pls.x_weights_  # Weights
        T = self.pls.x_scores_   # Scores
        Q = self.pls.y_loadings_ # Y loadings
        
        # Calculate VIP scores
        n_features = W.shape[0]
        n_components = W.shape[1]
        
        # Sum of squares of Y explained by each component
        ss = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
        
        # VIP calculation
        vip_scores = np.zeros(n_features)
        for i in range(n_features):
            weight_sq = (W[i, :] ** 2).reshape(1, -1)
            vip_scores[i] = np.sqrt(n_features * np.sum(ss * weight_sq) / np.sum(ss))
        
        # Create dataframe
        vip_df = pd.DataFrame({
            'Feature': self.feature_names,
            'VIP': vip_scores
        })
        
        vip_df = vip_df.sort_values('VIP', ascending=False).reset_index(drop=True)
        vip_df['Important'] = vip_df['VIP'] > 1.0
        
        return vip_df
    
    def plot_scores(self, y: np.ndarray,
                    lv_x: int = 1, lv_y: int = 2,
                    labels: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (10, 8)):
        """
        Plot PLS-DA scores.
        
        Parameters
        ----------
        y : np.ndarray
            Group labels
        lv_x, lv_y : int
            Which latent variables to plot (1-indexed)
        labels : list, optional
            Custom labels for groups
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        lv_x_idx = lv_x - 1
        lv_y_idx = lv_y - 1
        
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_groups = np.unique(y)
        colors = sns.color_palette('husl', n_colors=len(unique_groups))
        
        for i, group in enumerate(unique_groups):
            mask = y == group
            label = labels[i] if labels else str(group)
            
            ax.scatter(
                self.scores[mask, lv_x_idx],
                self.scores[mask, lv_y_idx],
                c=[colors[i]], s=150, alpha=0.7,
                label=label, edgecolors='black', linewidth=1.5
            )
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel(f'LV{lv_x}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'LV{lv_y}', fontsize=12, fontweight='bold')
        ax.set_title('PLS-DA Scores Plot', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fontsize=11)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_loadings(self, lv: int = 1, top_n: int = 15,
                     figsize: Tuple[int, int] = (10, 8)):
        """
        Plot loadings for a latent variable.
        
        Parameters
        ----------
        lv : int
            Which LV to plot (1-indexed)
        top_n : int
            Number of top features to show
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        lv_idx = lv - 1
        loadings = self.loadings[:, lv_idx]
        
        # Get top features by absolute loading
        top_indices = np.argsort(np.abs(loadings))[-top_n:][::-1]
        
        top_loadings = loadings[top_indices]
        top_features = [self.feature_names[i] for i in top_indices]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#e74c3c' if x < 0 else '#3498db' for x in top_loadings]
        
        bars = ax.barh(range(len(top_loadings)), top_loadings, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_loadings)))
        ax.set_yticklabels(top_features)
        ax.axvline(x=0, color='black', linewidth=1)
        
        ax.set_xlabel('Loading Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'LV{lv} Loadings (Top {top_n})', fontsize=14, fontweight='bold')
        
        for i, (bar, val) in enumerate(zip(bars, top_loadings)):
            x_pos = val + (0.01 if val > 0 else -0.01)
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, fontsize=9)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_vip(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot VIP scores.
        
        Parameters
        ----------
        top_n : int
            Number of top features to show
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        vip_df = self.get_vip_scores().head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#2ecc71' if x else '#95a5a6' for x in vip_df['Important']]
        
        bars = ax.barh(range(len(vip_df)), vip_df['VIP'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(vip_df)))
        ax.set_yticklabels(vip_df['Feature'])
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='VIP = 1.0')
        
        ax.set_xlabel('VIP Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'Variable Importance in Projection (Top {top_n})', 
                    fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fontsize=10)
        
        for i, (bar, val) in enumerate(zip(bars, vip_df['VIP'])):
            ax.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_confusion_matrix(self, cv_results: Dict,
                             figsize: Tuple[int, int] = (8, 6)):
        """
        Plot confusion matrix from cross-validation.
        
        Parameters
        ----------
        cv_results : dict
            Results from cross_validate()
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        cm = cv_results['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(f"Confusion Matrix ({cv_results['cv_type']} CV)\n"
                    f"Accuracy: {cv_results['accuracy']:.2%}", 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, ax