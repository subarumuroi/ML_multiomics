"""
Multi-omics data integrator.

Handles combining preprocessed omics layers with different sample sizes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings


class OmicsIntegrator:
    """
    Integrates multiple preprocessed omics datasets.
    
    Handles:
    - Different sample sizes across layers
    - Sample ID matching
    - Data alignment
    - Concatenation strategies
    """
    
    def __init__(self):
        """Initialize integrator."""
        self.layers = {}
        self.common_samples = None
        self.layer_info = {}
        
    def add_layer(self, 
                  name: str,
                  X: np.ndarray,
                  y: np.ndarray,
                  feature_names: List[str],
                  sample_ids: Optional[List[str]] = None):
        """
        Add an omics layer.
        
        Parameters
        ----------
        name : str
            Name of omics layer (e.g., 'metabolomics', 'proteomics')
        X : np.ndarray
            Preprocessed feature matrix (n_samples × n_features)
        y : np.ndarray
            Group labels
        feature_names : list
            Feature names
        sample_ids : list, optional
            Sample identifiers for matching across layers
        """
        if sample_ids is None:
            sample_ids = [f"{name}_sample_{i}" for i in range(len(X))]
        
        self.layers[name] = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'sample_ids': sample_ids,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
        
        print(f"Added layer '{name}': {X.shape[0]} samples × {X.shape[1]} features")
    
    def find_common_samples(self) -> List[str]:
        """
        Find samples present in all layers.
        
        Returns
        -------
        list
            Common sample IDs across all layers
        """
        if len(self.layers) == 0:
            raise ValueError("No layers added yet")
        
        # Get intersection of all sample IDs
        all_sample_sets = [set(layer['sample_ids']) for layer in self.layers.values()]
        common = set.intersection(*all_sample_sets)
        
        self.common_samples = sorted(list(common))
        
        print(f"\nFound {len(self.common_samples)} common samples across {len(self.layers)} layers")
        
        for layer_name, layer_data in self.layers.items():
            n_unique = len(set(layer_data['sample_ids']) - common)
            if n_unique > 0:
                print(f"  - {layer_name}: {n_unique} unique samples will be excluded")
        
        return self.common_samples
    
    def align_layers(self) -> Dict[str, Dict]:
        """
        Align all layers to common samples.
        
        Returns
        -------
        dict
            Aligned layers with consistent sample ordering
        """
        if self.common_samples is None:
            self.find_common_samples()
        
        aligned_layers = {}
        
        for name, layer in self.layers.items():
            # Find indices of common samples in this layer
            sample_to_idx = {sid: i for i, sid in enumerate(layer['sample_ids'])}
            common_indices = [sample_to_idx[sid] for sid in self.common_samples]
            
            # Subset to common samples
            aligned_layers[name] = {
                'X': layer['X'][common_indices, :],
                'y': layer['y'][common_indices],
                'feature_names': layer['feature_names'],
                'sample_ids': self.common_samples,
                'n_samples': len(common_indices),
                'n_features': layer['n_features']
            }
        
        # Verify consistent labels across layers
        y_arrays = [layer['y'] for layer in aligned_layers.values()]
        for i in range(1, len(y_arrays)):
            if not np.array_equal(y_arrays[0], y_arrays[i]):
                warnings.warn("Group labels differ across aligned layers!")
        
        print(f"\nAligned all layers to {len(self.common_samples)} common samples")
        
        return aligned_layers
    
    def concatenate(self, 
                    layer_names: Optional[List[str]] = None,
                    align: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Concatenate multiple omics layers horizontally.
        
        Parameters
        ----------
        layer_names : list, optional
            Which layers to concatenate (default: all)
        align : bool
            Whether to align to common samples first
            
        Returns
        -------
        X_concat : np.ndarray
            Concatenated feature matrix
        y : np.ndarray
            Group labels
        feature_names : list
            Combined feature names with layer prefixes
        """
        if layer_names is None:
            layer_names = list(self.layers.keys())
        
        # Align if requested
        if align:
            layers_to_use = self.align_layers()
        else:
            layers_to_use = self.layers
        
        # Check sample sizes match
        sample_sizes = [layers_to_use[name]['n_samples'] for name in layer_names]
        if len(set(sample_sizes)) > 1:
            raise ValueError(f"Cannot concatenate layers with different sample sizes: {sample_sizes}")
        
        # Concatenate features
        X_blocks = []
        feature_names_combined = []
        
        for name in layer_names:
            layer = layers_to_use[name]
            X_blocks.append(layer['X'])
            
            # Prefix feature names with layer name
            prefixed_names = [f"{name}_{feat}" for feat in layer['feature_names']]
            feature_names_combined.extend(prefixed_names)
        
        X_concat = np.hstack(X_blocks)
        
        # Use labels from first layer (they should all match)
        y = layers_to_use[layer_names[0]]['y']
        
        print(f"\nConcatenated {len(layer_names)} layers:")
        print(f"  Final shape: {X_concat.shape}")
        print(f"  Total features: {X_concat.shape[1]}")
        
        return X_concat, y, feature_names_combined
    
    def get_layer_blocks(self, 
                        layer_names: Optional[List[str]] = None) -> Dict[str, Tuple[int, int]]:
        """
        Get feature index ranges for each layer in concatenated data.
        
        Useful for methods that need to know which features belong to which layer.
        
        Parameters
        ----------
        layer_names : list, optional
            Which layers (default: all)
            
        Returns
        -------
        dict
            {layer_name: (start_idx, end_idx)} for each layer
        """
        if layer_names is None:
            layer_names = list(self.layers.keys())
        
        aligned_layers = self.align_layers()
        
        blocks = {}
        start_idx = 0
        
        for name in layer_names:
            n_features = aligned_layers[name]['n_features']
            end_idx = start_idx + n_features
            blocks[name] = (start_idx, end_idx)
            start_idx = end_idx
        
        return blocks
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all layers.
        
        Returns
        -------
        pd.DataFrame
            Summary of each layer
        """
        summary_data = []
        
        for name, layer in self.layers.items():
            summary_data.append({
                'Layer': name,
                'Samples': layer['n_samples'],
                'Features': layer['n_features'],
                'Groups': len(np.unique(layer['y'])),
                'Missing_Values': np.isnan(layer['X']).sum()
            })
        
        df = pd.DataFrame(summary_data)
        return df


class MultiBlockData:
    """
    Container for multi-block omics data.
    
    Maintains separate blocks while ensuring sample alignment.
    Used for methods like DIABLO that need block structure preserved.
    """
    
    def __init__(self):
        """Initialize multi-block container."""
        self.blocks = {}
        self.y = None
        self.sample_ids = None
        
    def add_block(self, 
                  name: str,
                  X: np.ndarray,
                  feature_names: List[str]):
        """
        Add a data block.
        
        Parameters
        ----------
        name : str
            Block name
        X : np.ndarray
            Feature matrix (must have same n_samples as other blocks)
        feature_names : list
            Feature names
        """
        if len(self.blocks) > 0:
            # Check sample size matches existing blocks
            existing_n = list(self.blocks.values())[0]['X'].shape[0]
            if X.shape[0] != existing_n:
                raise ValueError(f"Block '{name}' has {X.shape[0]} samples, expected {existing_n}")
        
        self.blocks[name] = {
            'X': X,
            'feature_names': feature_names,
            'n_features': X.shape[1]
        }
        
    def set_labels(self, y: np.ndarray, sample_ids: Optional[List[str]] = None):
        """
        Set group labels.
        
        Parameters
        ----------
        y : np.ndarray
            Group labels
        sample_ids : list, optional
            Sample identifiers
        """
        self.y = y
        self.sample_ids = sample_ids
        
    def get_block_names(self) -> List[str]:
        """Get names of all blocks."""
        return list(self.blocks.keys())
    
    def get_block(self, name: str) -> np.ndarray:
        """Get feature matrix for a block."""
        return self.blocks[name]['X']
    
    def get_n_blocks(self) -> int:
        """Get number of blocks."""
        return len(self.blocks)
    
    def get_n_samples(self) -> int:
        """Get number of samples."""
        if len(self.blocks) == 0:
            return 0
        return list(self.blocks.values())[0]['X'].shape[0]
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all blocks."""
        summary_data = []
        
        for name, block in self.blocks.items():
            summary_data.append({
                'Block': name,
                'Features': block['n_features'],
                'Total_Values': block['X'].size,
                'Mean': np.mean(block['X']),
                'Std': np.std(block['X'])
            })
        
        return pd.DataFrame(summary_data)