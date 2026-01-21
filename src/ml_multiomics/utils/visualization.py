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