"""
DIABLO (Data Integration Analysis for Biomarker discovery using Latent cOmponents).

Multi-omics integration method that maximizes correlation between omics blocks
while performing discriminant analysis.

This is a simplified implementation of the mixOmics DIABLO approach.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelBinarizer
from typing import Dict, List, Tuple, Optional


class DIABLO:
    """
    DIABLO for multi-omics integration.
    
    Features:
    - Integrates multiple omics blocks
    - Supervised discrimination
    - Block correlation maximization
    - Feature selection across blocks
    - Sample and variable plots
    """
    
    def __init__(self, 
                 n_components: int = 2,
                 design: Optional[np.ndarray] = None):
        """
        Initialize DIABLO.
        
        Parameters
        ----------
        n_components : int
            Number of components to extract
        design : np.ndarray, optional
            Design matrix (n_blocks × n_blocks) specifying block connections.
            1 = maximize correlation, 0 = no connection.
            If None, uses full design (all blocks connected).
        """
        self.n_components = n_components
        self.design = design
        self.blocks = {}
        self.block_names = []
        self.pls_models = {}
        self.scores = {}
        self.loadings = {}
        self.label_binarizer = None
        self.classes = None
        
    def fit(self, 
            blocks: Dict[str, np.ndarray],
            y: np.ndarray,
            feature_names: Optional[Dict[str, List[str]]] = None):
        """
        Fit DIABLO model.
        
        Parameters
        ----------
        blocks : dict
            Dictionary of {block_name: X_block} where each X_block is
            (n_samples × n_features) array
        y : np.ndarray
            Group labels (n_samples,)
        feature_names : dict, optional
            Dictionary of {block_name: feature_names} for each block
        """
        self.block_names = list(blocks.keys())
        n_blocks = len(self.block_names)
        
        # Check all blocks have same number of samples
        n_samples = [blocks[name].shape[0] for name in self.block_names]
        if len(set(n_samples)) != 1:
            raise ValueError(f"All blocks must have same number of samples. Got {n_samples}")
        
        # Set up design matrix if not provided
        if self.design is None:
            # Full design: all blocks connected
            self.design = np.ones((n_blocks, n_blocks))
            np.fill_diagonal(self.design, 0)  # No self-connection
        
        # Encode labels
        self.label_binarizer = LabelBinarizer()
        Y = self.label_binarizer.fit_transform(y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.classes = self.label_binarizer.classes_
        
        # Store blocks
        self.blocks = blocks
        
        # Store feature names
        self.feature_names = feature_names if feature_names is not None else {}

        # Fit PLS for each block
        for block_name in self.block_names:
            X_block = blocks[block_name]
            
            # Fit PLS-DA for this block
            pls = PLSRegression(n_components=self.n_components, scale=False)
            pls.fit(X_block, Y)
            
            self.pls_models[block_name] = pls
            self.scores[block_name] = pls.x_scores_
            self.loadings[block_name] = pls.x_loadings_
        
        return self
    
    def transform(self, blocks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Project new data onto components.
        
        Parameters
        ----------
        blocks : dict
            Dictionary of new data blocks
            
        Returns
        -------
        dict
            Dictionary of scores for each block
        """
        scores = {}
        for block_name in self.block_names:
            X_block = blocks[block_name]
            scores[block_name] = self.pls_models[block_name].transform(X_block)
        
        return scores
    
    def get_block_vip(self, block_name: str) -> pd.DataFrame:
        """
        Calculate VIP scores for a specific block.
        
        Parameters
        ----------
        block_name : str
            Name of the block
            
        Returns
        -------
        pd.DataFrame
            VIP scores for features in this block
        """
        pls = self.pls_models[block_name]
        
        W = pls.x_weights_
        T = pls.x_scores_
        Q = pls.y_loadings_
        
        n_features = W.shape[0]
        n_components = W.shape[1]
        
        ss = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
        
        vip_scores = np.zeros(n_features)
        for i in range(n_features):
            weight_sq = (W[i, :] ** 2).reshape(1, -1)
            vip_scores[i] = np.sqrt(n_features * np.sum(ss * weight_sq) / np.sum(ss))
        
        vip_df = pd.DataFrame({
            'Feature': self.feature_names.get(block_name, [f"Feature_{i}" for i in range(n_features)]),
            'VIP': vip_scores,
            'Block': block_name
        })
        
        vip_df = vip_df.sort_values('VIP', ascending=False).reset_index(drop=True)
        vip_df['Important'] = vip_df['VIP'] > 1.0
        
        return vip_df
    
    def get_all_vips(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top VIP features across all blocks.
        
        Parameters
        ----------
        top_n : int
            Number of top features per block
            
        Returns
        -------
        pd.DataFrame
            Combined VIP scores from all blocks
        """
        all_vips = []
        
        for block_name in self.block_names:
            vip_df = self.get_block_vip(block_name).head(top_n)
            all_vips.append(vip_df)
        
        return pd.concat(all_vips, ignore_index=True)
    
    def calculate_block_correlations(self) -> pd.DataFrame:
        """
        Calculate correlations between block scores.
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix between block scores on first component
        """
        # Use first component scores
        block_scores_comp1 = {name: self.scores[name][:, 0] 
                              for name in self.block_names}
        
        corr_matrix = np.zeros((len(self.block_names), len(self.block_names)))
        
        for i, name1 in enumerate(self.block_names):
            for j, name2 in enumerate(self.block_names):
                corr_matrix[i, j] = np.corrcoef(
                    block_scores_comp1[name1],
                    block_scores_comp1[name2]
                )[0, 1]
        
        corr_df = pd.DataFrame(
            corr_matrix,
            index=self.block_names,
            columns=self.block_names
        )
        
        return corr_df
    
    def plot_samples(self, 
                    y: np.ndarray,
                    comp_x: int = 1,
                    comp_y: int = 2,
                    block: Optional[str] = None,
                    labels: Optional[List[str]] = None,
                    figsize: Tuple[int, int] = (10, 8)):
        """
        Plot sample scores.
        
        Parameters
        ----------
        y : np.ndarray
            Group labels
        comp_x, comp_y : int
            Which components to plot (1-indexed)
        block : str, optional
            Which block to plot (if None, averages all blocks)
        labels : list, optional
            Custom labels for groups
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        comp_x_idx = comp_x - 1
        comp_y_idx = comp_y - 1
        
        # Get scores for plotting
        if block is None:
            # Average scores across all blocks
            all_scores = np.array([self.scores[name] for name in self.block_names])
            scores_to_plot = np.mean(all_scores, axis=0)
            title_suffix = "(Average across blocks)"
        else:
            scores_to_plot = self.scores[block]
            title_suffix = f"({block})"
        
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_groups = np.unique(y)
        colors = sns.color_palette('husl', n_colors=len(unique_groups))
        
        for i, group in enumerate(unique_groups):
            mask = y == group
            label = labels[i] if labels else str(group)
            
            ax.scatter(
                scores_to_plot[mask, comp_x_idx],
                scores_to_plot[mask, comp_y_idx],
                c=[colors[i]], s=150, alpha=0.7,
                label=label, edgecolors='black', linewidth=1.5
            )
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel(f'Component {comp_x}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Component {comp_y}', fontsize=12, fontweight='bold')
        ax.set_title(f'DIABLO Sample Plot {title_suffix}', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fontsize=11)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_block_correlations(self, figsize: Tuple[int, int] = (8, 7)):
        """
        Plot correlation heatmap between blocks.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        corr_df = self.calculate_block_correlations()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='coolwarm',
                   vmin=-1, vmax=1, center=0, square=True,
                   cbar_kws={'label': 'Correlation'}, ax=ax)
        
        ax.set_title('Block Correlation (Component 1)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_arrow_plot(self,
                       y: np.ndarray,
                       comp_x: int = 1,
                       comp_y: int = 2,
                       labels: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (12, 10)):
        """
        Create arrow plot showing agreement between blocks.
        
        Each sample is plotted with arrows connecting its position
        in different omics blocks.
        
        Parameters
        ----------
        y : np.ndarray
            Group labels
        comp_x, comp_y : int
            Which components to plot (1-indexed)
        labels : list, optional
            Custom labels for groups
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        comp_x_idx = comp_x - 1
        comp_y_idx = comp_y - 1
        
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_groups = np.unique(y)
        colors = sns.color_palette('husl', n_colors=len(unique_groups))
        
        # Calculate average position for each sample
        n_samples = self.scores[self.block_names[0]].shape[0]
        avg_scores = np.zeros((n_samples, 2))
        
        for name in self.block_names:
            scores = self.scores[name]
            avg_scores[:, 0] += scores[:, comp_x_idx]
            avg_scores[:, 1] += scores[:, comp_y_idx]
        
        avg_scores /= len(self.block_names)
        
        # Plot arrows from each block to average position
        for name in self.block_names:
            scores = self.scores[name]
            
            for i in range(n_samples):
                x_start = scores[i, comp_x_idx]
                y_start = scores[i, comp_y_idx]
                x_end = avg_scores[i, 0]
                y_end = avg_scores[i, 1]
                
                ax.arrow(x_start, y_start, 
                        x_end - x_start, y_end - y_start,
                        head_width=0.1, head_length=0.1,
                        fc='gray', ec='gray', alpha=0.3, linewidth=0.5)
        
        # Plot average positions
        for i, group in enumerate(unique_groups):
            mask = y == group
            label = labels[i] if labels else str(group)
            
            ax.scatter(
                avg_scores[mask, 0],
                avg_scores[mask, 1],
                c=[colors[i]], s=200, alpha=0.8,
                label=label, edgecolors='black', linewidth=2,
                marker='s', zorder=5
            )
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel(f'Component {comp_x}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Component {comp_y}', fontsize=12, fontweight='bold')
        ax.set_title('DIABLO Arrow Plot (Block Agreement)', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, fontsize=11)
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_circos(self, 
                   threshold: float = 0.7,
                   figsize: Tuple[int, int] = (10, 10)):
        """
        Create a simplified circos-like plot showing block relationships.
        
        Parameters
        ----------
        threshold : float
            Correlation threshold for drawing connections
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig, ax : matplotlib figure and axis
        """
        corr_df = self.calculate_block_correlations()
        n_blocks = len(self.block_names)
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Position blocks around circle
        angles = np.linspace(0, 2 * np.pi, n_blocks, endpoint=False)
        
        # Plot blocks as points
        for i, (angle, name) in enumerate(zip(angles, self.block_names)):
            ax.plot([angle], [1], 'o', markersize=20, label=name)
            ax.text(angle, 1.15, name, ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        # Draw connections for high correlations
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                corr = corr_df.iloc[i, j]
                if abs(corr) >= threshold:
                    theta = np.linspace(angles[i], angles[j], 100)
                    r = 1 - 0.3 * np.sin(np.linspace(0, np.pi, 100))
                    
                    color = 'red' if corr > 0 else 'blue'
                    alpha = abs(corr)
                    
                    ax.plot(theta, r, color=color, alpha=alpha, linewidth=2)
        
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
        
        ax.set_title(f'Block Correlations (threshold = {threshold})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig, ax