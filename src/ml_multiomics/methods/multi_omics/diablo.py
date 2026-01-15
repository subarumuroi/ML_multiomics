"""
DIABLO (Data Integration Analysis for Biomarker discovery using Latent cOmponents).

Multi-omics integration using R's mixOmics package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from ml_multiomics.utils.r_interface import run_diablo_r


class DIABLO:
    """
    DIABLO for multi-omics integration via R mixOmics.
    
    Features:
    - Integrates multiple omics blocks
    - Supervised discrimination
    - Block correlation maximization
    - Feature selection across blocks
    - Sample and variable plots
    """
    
    def __init__(self, n_components: int = 2):
        """
        Initialize DIABLO.
        
        Parameters
        ----------
        n_components : int
            Number of components to extract (used for tuning in R)
        """
        self.n_components = n_components
        self.block_names = []
        self.scores = {}
        self.loadings = {}
        self.selected_features = {}
        self.feature_names = {}
        self.r_results = None
        self.classes = None
        
    def fit(self, 
            blocks: Dict[str, np.ndarray],
            y: np.ndarray,
            feature_names: Optional[Dict[str, List[str]]] = None):
        """
        Fit DIABLO model using R mixOmics.
        
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
        self.classes = np.unique(y)
        
        # Check all blocks have same number of samples
        n_samples = [blocks[name].shape[0] for name in self.block_names]
        if len(set(n_samples)) != 1:
            raise ValueError(f"All blocks must have same number of samples. Got {n_samples}")
        
        # Store feature names
        self.feature_names = feature_names if feature_names is not None else {}
        
        # Convert blocks to DataFrames with feature names
        data_blocks = {}
        for block_name, X_block in blocks.items():
            features = self.feature_names.get(
                block_name, 
                [f"{block_name}_feature_{i}" for i in range(X_block.shape[1])]
            )
            data_blocks[block_name] = pd.DataFrame(X_block, columns=features)
        
        # Create sample IDs
        sample_ids = [f"sample_{i}" for i in range(n_samples[0])]
        
        # Run DIABLO in R
        print("Running DIABLO via R mixOmics...")
        
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'diablo_results'
            
            self.r_results = run_diablo_r(
                data_blocks=data_blocks,
                y=y,
                sample_ids=sample_ids,
                output_dir=str(output_dir),
                timeout=300
            )
        
        # Extract results
        self._extract_results()
        
        return self
    
    def _extract_results(self):
        """Extract and store results from R output."""
        # Variates (sample scores) - exclude Y block
        if 'variates' in self.r_results:
            self.scores = {
                name: df.values 
                for name, df in self.r_results['variates'].items()
                if name != 'Y'  # Filter out response matrix
            }
        
        # Loadings (feature contributions) - exclude Y block
        if 'loadings' in self.r_results:
            self.loadings = {
                name: df.values 
                for name, df in self.r_results['loadings'].items()
                if name != 'Y'  # Filter out response matrix
            }
        
        # Selected features
        if 'selected_features' in self.r_results:
            self.selected_features = self.r_results['selected_features']
    
    def get_block_vip(self, block_name: str) -> pd.DataFrame:
        """
        Get important features for a specific block.
        
        Parameters
        ----------
        block_name : str
            Name of the block
            
        Returns
        -------
        pd.DataFrame
            Important features for this block
        """
        if block_name not in self.selected_features:
            return pd.DataFrame(columns=['Feature', 'VIP', 'Block', 'Important'])
        
        selected_df = self.selected_features[block_name].copy()
        
        # Add block name
        selected_df['Block'] = block_name
        selected_df['Important'] = True
        
        # Ensure we have 'Feature' column
        if 'feature' in selected_df.columns:
            selected_df = selected_df.rename(columns={'feature': 'Feature'})
        elif 'Feature' not in selected_df.columns and selected_df.index.name is None:
            selected_df['Feature'] = selected_df.index
        
        # Use loading value as VIP (absolute value)
        if 'value.var' in selected_df.columns:
            selected_df['VIP'] = selected_df['value.var'].abs()
        elif 'comp1' in selected_df.columns:
            selected_df['VIP'] = selected_df['comp1'].abs()
        else:
            # Rank by appearance order
            selected_df['VIP'] = np.arange(len(selected_df), 0, -1)
        
        # Sort and return key columns
        selected_df = selected_df.sort_values('VIP', ascending=False)
        return selected_df[['Feature', 'VIP', 'Block', 'Important']].reset_index(drop=True)
        
    def get_all_vips(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top features across all blocks.
        
        Parameters
        ----------
        top_n : int
            Number of top features per block
            
        Returns
        -------
        pd.DataFrame
            Combined important features from all blocks
        """
        all_vips = []
        
        for block_name in self.block_names:
            vip_df = self.get_block_vip(block_name).head(top_n)
            all_vips.append(vip_df)
        
        if all_vips:
            return pd.concat(all_vips, ignore_index=True)
        else:
            return pd.DataFrame(columns=['Feature', 'VIP', 'Block', 'Important'])
    
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