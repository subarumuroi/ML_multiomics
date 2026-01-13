"""
Advanced visualization utilities for multi-omics analysis.

Provides publication-quality plots and interactive visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Tuple, Optional
import warnings


class OmicsPlotter:
    """
    Publication-quality plotting for omics data.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize plotter.
        
        Parameters
        ----------
        style : str
            Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            pass  # Use default if style not available
        
        # Set default colors
        self.colors = sns.color_palette('husl', n_colors=10)
    
    def plot_sample_overview(self,
                            data_dict: Dict[str, pd.DataFrame],
                            group_col: str = 'Groups',
                            figsize: Tuple[int, int] = (14, 8)) -> Tuple:
        """
        Create overview plot of sample distribution across omics layers.
        
        Parameters
        ----------
        data_dict : dict
            {layer_name: dataframe} dictionary
        group_col : str
            Group column name
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, axes : matplotlib figure and axes
        """
        n_layers = len(data_dict)
        fig, axes = plt.subplots(1, n_layers, figsize=figsize)
        
        if n_layers == 1:
            axes = [axes]
        
        for ax, (layer_name, df) in zip(axes, data_dict.items()):
            # Count samples per group
            group_counts = df[group_col].value_counts()
            
            # Plot
            bars = ax.bar(range(len(group_counts)), group_counts.values,
                         color=self.colors[:len(group_counts)], alpha=0.7,
                         edgecolor='black', linewidth=1.5)
            
            ax.set_xticks(range(len(group_counts)))
            ax.set_xticklabels(group_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
            ax.set_title(f'{layer_name}\n({df.shape[1]-1} features)',
                        fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, group_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(val)}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_missing_data_heatmap(self,
                                 df: pd.DataFrame,
                                 group_col: str = 'Groups',
                                 figsize: Tuple[int, int] = (12, 8)) -> Tuple:
        """
        Visualize missing data patterns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with potential missing values
        group_col : str
            Group column name
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        feature_cols = [c for c in df.columns if c != group_col]
        
        # Create binary missing indicator
        missing_matrix = df[feature_cols].isna().astype(int)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(missing_matrix.T, cmap='RdYlGn_r', cbar_kws={'label': 'Missing'},
                   yticklabels=feature_cols if len(feature_cols) < 50 else False,
                   xticklabels=False, ax=ax)
        
        ax.set_xlabel('Samples', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title('Missing Data Pattern', fontsize=14, fontweight='bold')
        
        # Add summary statistics
        missing_pct = (missing_matrix.sum().sum() / missing_matrix.size) * 100
        ax.text(0.02, 0.98, f'Overall Missing: {missing_pct:.1f}%',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig, ax
    
    def plot_feature_distributions(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   feature_names: List[str],
                                   top_n: int = 9,
                                   figsize: Tuple[int, int] = (15, 10)) -> Tuple:
        """
        Plot distributions of top features across groups.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Group labels
        feature_names : list
            Feature names
        top_n : int
            Number of features to plot
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, axes : matplotlib figure and axes
        """
        # Select top features by variance
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-top_n:][::-1]
        
        # Create subplots
        n_rows = int(np.ceil(top_n / 3))
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        axes = axes.flatten()
        
        unique_groups = np.unique(y)
        
        for idx, feat_idx in enumerate(top_indices):
            ax = axes[idx]
            
            # Plot distributions for each group
            for i, group in enumerate(unique_groups):
                mask = y == group
                data = X[mask, feat_idx]
                
                ax.hist(data, bins=10, alpha=0.5, label=str(group),
                       color=self.colors[i], edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(feature_names[feat_idx], fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        
        # Remove extra subplots
        for idx in range(top_n, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        return fig, axes
    
    def plot_correlation_network(self,
                                 X: np.ndarray,
                                 feature_names: List[str],
                                 threshold: float = 0.7,
                                 figsize: Tuple[int, int] = (12, 10)) -> Tuple:
        """
        Plot feature correlation network.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        feature_names : list
            Feature names
        threshold : float
            Correlation threshold for showing edges
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        # Calculate correlations
        corr_matrix = np.corrcoef(X.T)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        mask = np.abs(corr_matrix) < threshold
        
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, square=True,
                   xticklabels=feature_names if len(feature_names) < 50 else False,
                   yticklabels=feature_names if len(feature_names) < 50 else False,
                   cbar_kws={'label': 'Correlation'}, ax=ax)
        
        ax.set_title(f'Feature Correlation Network (|r| > {threshold})',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_volcano(self,
                    fold_changes: np.ndarray,
                    p_values: np.ndarray,
                    feature_names: List[str],
                    fc_threshold: float = 1.5,
                    p_threshold: float = 0.05,
                    top_n_labels: int = 10,
                    figsize: Tuple[int, int] = (10, 8)) -> Tuple:
        """
        Create volcano plot for differential analysis.
        
        Parameters
        ----------
        fold_changes : np.ndarray
            Fold changes (linear scale)
        p_values : np.ndarray
            P-values
        feature_names : list
            Feature names
        fc_threshold : float
            Fold change threshold
        p_threshold : float
            P-value threshold
        top_n_labels : int
            Number of top features to label
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        # Calculate log values
        log2_fc = np.log2(fold_changes + 1e-10)
        neg_log10_p = -np.log10(p_values + 1e-10)
        
        # Determine significance
        significant = (np.abs(log2_fc) > np.log2(fc_threshold)) & (p_values < p_threshold)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot non-significant
        ax.scatter(log2_fc[~significant], neg_log10_p[~significant],
                  c='gray', alpha=0.5, s=30, label='Not Significant')
        
        # Plot significant
        ax.scatter(log2_fc[significant], neg_log10_p[significant],
                  c='red', alpha=0.7, s=50, label='Significant')
        
        # Add threshold lines
        ax.axhline(y=-np.log10(p_threshold), color='blue', linestyle='--',
                  alpha=0.5, label=f'p = {p_threshold}')
        ax.axvline(x=np.log2(fc_threshold), color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=-np.log2(fc_threshold), color='green', linestyle='--',
                  alpha=0.5, label=f'FC = ±{fc_threshold}')
        
        # Label top features
        top_indices = np.argsort(neg_log10_p * np.abs(log2_fc))[-top_n_labels:]
        for idx in top_indices:
            ax.annotate(feature_names[idx],
                       xy=(log2_fc[idx], neg_log10_p[idx]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('log2(Fold Change)', fontsize=12, fontweight='bold')
        ax.set_ylabel('-log10(p-value)', fontsize=12, fontweight='bold')
        ax.set_title('Volcano Plot', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_confidence_ellipses(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                comp_x: int = 0,
                                comp_y: int = 1,
                                confidence: float = 0.95,
                                figsize: Tuple[int, int] = (10, 8)) -> Tuple:
        """
        Plot scores with confidence ellipses.
        
        Parameters
        ----------
        X : np.ndarray
            Scores matrix
        y : np.ndarray
            Group labels
        comp_x, comp_y : int
            Which components to plot
        confidence : float
            Confidence level for ellipses
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        from scipy.stats import chi2
        
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_groups = np.unique(y)
        
        for i, group in enumerate(unique_groups):
            mask = y == group
            points = X[mask][:, [comp_x, comp_y]]
            
            # Plot points
            ax.scatter(points[:, 0], points[:, 1],
                      c=[self.colors[i]], s=150, alpha=0.7,
                      label=str(group), edgecolors='black', linewidth=1.5)
            
            # Calculate and plot ellipse
            if len(points) > 2:
                mean = points.mean(axis=0)
                cov = np.cov(points.T)
                
                # Chi-square value for confidence level
                chi2_val = chi2.ppf(confidence, df=2)
                
                # Eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                
                # Ellipse parameters
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width, height = 2 * np.sqrt(chi2_val * eigenvalues)
                
                # Plot ellipse
                ellipse = Ellipse(mean, width, height, angle=angle,
                                facecolor='none', edgecolor=self.colors[i],
                                linewidth=2, linestyle='--', alpha=0.7)
                ax.add_patch(ellipse)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_xlabel(f'Component {comp_x + 1}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Component {comp_y + 1}', fontsize=12, fontweight='bold')
        ax.set_title(f'Scores with {confidence*100:.0f}% Confidence Ellipses',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.2)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_performance_comparison(self,
                                   results_dict: Dict[str, Dict],
                                   metric: str = 'accuracy',
                                   figsize: Tuple[int, int] = (10, 6)) -> Tuple:
        """
        Compare model performance across methods.
        
        Parameters
        ----------
        results_dict : dict
            {method_name: results_dict} with 'mean' and 'std' keys
        metric : str
            Metric name
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        methods = list(results_dict.keys())
        means = [results_dict[m]['mean'] for m in methods]
        stds = [results_dict[m]['std'] for m in methods]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=self.colors[:len(methods)], alpha=0.7,
                     edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_title(f'Model Performance Comparison ({metric})',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig, ax


def save_publication_figure(fig, filename: str, dpi: int = 300):
    """
    Save figure in publication quality.
    
    Parameters
    ----------
    fig : matplotlib figure
        Figure to save
    filename : str
        Output filename
    dpi : int
        Resolution
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"Figure saved: {filename}")