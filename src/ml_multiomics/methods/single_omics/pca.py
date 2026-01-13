"""
Principal Component Analysis (PCA) for omics data.

Provides exploratory visualization and variance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple


class PCAAnalysis:
    """
    PCA analysis for single omics datasets.
    
    Provides:
    - Dimensionality reduction
    - Variance explained analysis
    - Scores and loadings plots
    - Biplot visualization
    """
    
    def __init__(self, n_components: int = 5):
        """
        Initialize PCA analysis.
        
        Parameters
        ----------
        n_components : int
            Number of principal components to compute
        """
        self.n_components = n_components
        self.pca = None
        self.scores = None
        self.loadings = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit PCA model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples × n_features)
        feature_names : list, optional
            Names of features
        """
        self.pca = PCA(n_components=self.n_components)
        self.scores = self.pca.fit_transform(X)
        self.loadings = self.pca.components_.T  # n_features × n_components

        # fix to ensure feature names are Python list
        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        #create dedicated feature_list attribute for backward compatibility
        self.feature_list = self.feature_names

        return self
    
    def get_variance_explained(self) -> pd.DataFrame:
        """
        Get variance explained by each component.
        
        Returns
        -------
        pd.DataFrame
            Variance explained and cumulative variance
        """
        var_exp = self.pca.explained_variance_ratio_ * 100
        cum_var = np.cumsum(var_exp)
        
        df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(len(var_exp))],
            'Variance_Explained': var_exp,
            'Cumulative_Variance': cum_var
        })
        
        return df
    
    def get_loadings_df(self, pc: int = 1, top_n: int = 10) -> pd.DataFrame:
        """
        Get top features contributing to a principal component.
        
        Parameters
        ----------
        pc : int
            Principal component number (1-indexed)
        top_n : int
            Number of top features to return
            
        Returns
        -------
        pd.DataFrame
            Top features sorted by absolute loading value
        """
        pc_idx = pc - 1
        loadings = self.loadings[:, pc_idx]
        
        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Loading': loadings,
            'Abs_Loading': np.abs(loadings)
        })
        
        df = df.sort_values('Abs_Loading', ascending=False).head(top_n)
        return df.reset_index(drop=True)
    
    def plot_scree(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot scree plot showing variance explained.
        
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        var_df = self.get_variance_explained()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Bar plot for individual variance
        ax.bar(range(1, len(var_df) + 1), var_df['Variance_Explained'], 
               alpha=0.7, label='Individual')
        
        # Line plot for cumulative variance
        ax2 = ax.twinx()
        ax2.plot(range(1, len(var_df) + 1), var_df['Cumulative_Variance'], 
                'ro-', linewidth=2, markersize=8, label='Cumulative')
        ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
        
        ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold')
        ax.set_title('PCA Scree Plot', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        return fig, (ax, ax2)
    
    def plot_scores(self, y: np.ndarray, 
                    pc_x: int = 1, pc_y: int = 2,
                    labels: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (10, 8)):
        """
        Plot PCA scores (samples in PC space).
        
        Parameters
        ----------
        y : np.ndarray
            Group labels for coloring points
        pc_x, pc_y : int
            Which PCs to plot (1-indexed)
        labels : list, optional
            Custom labels for legend
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        pc_x_idx = pc_x - 1
        pc_y_idx = pc_y - 1
        
        var_df = self.get_variance_explained()
        var_x = var_df.iloc[pc_x_idx]['Variance_Explained']
        var_y = var_df.iloc[pc_y_idx]['Variance_Explained']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique groups
        unique_groups = np.unique(y)
        colors = sns.color_palette('husl', n_colors=len(unique_groups))
        
        for i, group in enumerate(unique_groups):
            mask = y == group
            label = labels[i] if labels else str(group)
            
            ax.scatter(
                self.scores[mask, pc_x_idx],
                self.scores[mask, pc_y_idx],
                c=[colors[i]], s=150, alpha=0.7,
                label=label, edgecolors='black', linewidth=1.5
            )
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel(f'PC{pc_x} ({var_x:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC{pc_y} ({var_y:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_title('PCA Scores Plot', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fontsize=11)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_loadings(self, pc: int = 1, top_n: int = 10,
                     figsize: Tuple[int, int] = (10, 8)):
        """
        Plot loadings for a principal component.
        
        Parameters
        ----------
        pc : int
            Which PC to plot (1-indexed)
        top_n : int
            Number of top features to show
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        loadings_df = self.get_loadings_df(pc, top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color bars by sign
        colors = ['#e74c3c' if x < 0 else '#3498db' for x in loadings_df['Loading']]
        
        bars = ax.barh(range(len(loadings_df)), loadings_df['Loading'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(loadings_df)))
        ax.set_yticklabels(loadings_df['Feature'])
        ax.axvline(x=0, color='black', linewidth=1)
        
        ax.set_xlabel('Loading Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        
        var_exp = self.get_variance_explained().iloc[pc-1]['Variance_Explained']
        ax.set_title(f'PC{pc} Loadings (Top {top_n}) - {var_exp:.1f}% variance', 
                    fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, loadings_df['Loading'])):
            x_pos = val + (0.01 if val > 0 else -0.01)
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, fontsize=9)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_biplot(self, y: np.ndarray,
                   pc_x: int = 1, pc_y: int = 2,
                   n_loadings: int = 5,
                   figsize: Tuple[int, int] = (12, 10)):
        """
        Create a biplot (scores + loadings overlay).
        
        Parameters
        ----------
        y : np.ndarray
            Group labels
        pc_x, pc_y : int
            Which PCs to plot (1-indexed)
        n_loadings : int
            Number of top loadings to show
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        pc_x_idx = pc_x - 1
        pc_y_idx = pc_y - 1
        
        # Get top loadings for both PCs
        loadings_x = self.get_loadings_df(pc_x, n_loadings)
        loadings_y = self.get_loadings_df(pc_y, n_loadings)
        top_features = list(set(loadings_x['Feature'].tolist() + loadings_y['Feature'].tolist()))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot scores
        unique_groups = np.unique(y)
        colors = sns.color_palette('husl', n_colors=len(unique_groups))
        
        for i, group in enumerate(unique_groups):
            mask = y == group
            ax.scatter(
                self.scores[mask, pc_x_idx],
                self.scores[mask, pc_y_idx],
                c=[colors[i]], s=120, alpha=0.6,
                label=str(group), edgecolors='black', linewidth=1
            )
        
        # Plot loadings as arrows
        scale = 0.7 * np.abs(self.scores[:, [pc_x_idx, pc_y_idx]]).max()
        
        for feature in top_features:
            idx = self.feature_list.index(feature)
            x = self.loadings[idx, pc_x_idx] * scale
            y = self.loadings[idx, pc_y_idx] * scale
            
            ax.arrow(0, 0, x, y, head_width=0.1*scale, head_length=0.1*scale,
                    fc='red', ec='red', alpha=0.6, linewidth=2)
            ax.text(x*1.1, y*1.1, feature, fontsize=9, color='darkred',
                   fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        var_df = self.get_variance_explained()
        var_x = var_df.iloc[pc_x_idx]['Variance_Explained']
        var_y = var_df.iloc[pc_y_idx]['Variance_Explained']
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_xlabel(f'PC{pc_x} ({var_x:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC{pc_y} ({var_y:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_title('PCA Biplot', fontsize=14, fontweight='bold')
        ax.legend(title='Group', frameon=True, fontsize=10)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig, ax