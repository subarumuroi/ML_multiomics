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
        self.alignment_method = None  # 'name_based' or 'position_based'
        self.layer_info = {}
        
    def add_layer(self, 
                  name: str,
                  X: np.ndarray,
                  y: np.ndarray,
                  feature_names: List[str],
                  sample_ids: Optional[List[str]] = None):
        """
        Add an omics layer. Explicit sample_ids are required for robust alignment.
        Raises ValueError if sample_ids is not provided.
        """
        if sample_ids is None:
            raise ValueError(f"Explicit sample_ids must be provided for layer '{name}'. Auto-generated IDs are not allowed.")
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
        
        Uses two strategies:
        1. Name-based: Match by sample IDs (requires consistent naming)
        2. Position-based: Match by position within each group (fallback for inconsistent naming)
        
        Returns
        -------
        list
            Common sample IDs across all layers
        """
        if len(self.layers) == 0:
            raise ValueError("No layers added yet")
        
        # Strategy 1: Try name-based alignment
        all_sample_sets = [set(layer['sample_ids']) for layer in self.layers.values()]
        common = set.intersection(*all_sample_sets)
        
        if len(common) > 0:
            # Name-based alignment successful
            self.common_samples = sorted(list(common))
            self.alignment_method = 'name_based'
            
            print(f"\n[Name-based alignment] Found {len(self.common_samples)} common samples across {len(self.layers)} layers")
            
            for layer_name, layer_data in self.layers.items():
                n_unique = len(set(layer_data['sample_ids']) - common)
                if n_unique > 0:
                    print(f"  - {layer_name}: {n_unique} unique samples will be excluded")
        else:
            # Strategy 2: Position-based alignment within groups
            print("\n[Name-based alignment failed - no common sample IDs]")
            print("[Falling back to position-based alignment within groups]")
            
            self.alignment_method = 'position_based'
            self.common_samples = self._align_by_position()
            
            print(f"\nAligned {len(self.common_samples)} samples by position within groups")
        
        return self.common_samples
    
    def _align_by_position(self) -> List[str]:
        """
        Align samples by their position within each group.
        
        Assumes samples at the same position within the same group across
        different omics layers are the same biological sample.
        
        Returns
        -------
        list
            Synthetic sample IDs (e.g., 'Green_1', 'Green_2', ...)
        """
        # Get groups from first layer (all layers should have same group structure)
        first_layer = list(self.layers.values())[0]
        unique_groups = np.unique(first_layer['y'])
        
        # For each group, find minimum sample count across layers
        group_sample_counts = {group: [] for group in unique_groups}
        
        for layer in self.layers.values():
            for group in unique_groups:
                group_mask = layer['y'] == group
                n_samples_in_group = group_mask.sum()
                group_sample_counts[group].append(n_samples_in_group)
        
        # Generate synthetic aligned sample IDs
        aligned_sample_ids = []
        for group in unique_groups:
            min_count = min(group_sample_counts[group])
            max_count = max(group_sample_counts[group])
            
            if min_count != max_count:
                print(f"  Warning: Group '{group}' has varying sample counts across layers ({min_count}-{max_count})")
                print(f"           Using first {min_count} samples from each layer")
            
            for i in range(min_count):
                aligned_sample_ids.append(f"{group}_{i+1}")
        
        return aligned_sample_ids
    
    def align_layers(self) -> Dict[str, Dict]:
        """
        Align all layers to common samples.
        
        Uses either name-based or position-based alignment depending on
        what find_common_samples() determined.
        
        Returns
        -------
        dict
            Aligned layers with consistent sample ordering
        """
        if self.common_samples is None:
            self.find_common_samples()
        
        aligned_layers = {}
        
        if self.alignment_method == 'name_based':
            # Name-based alignment: match by sample IDs
            for name, layer in self.layers.items():
                sample_to_idx = {sid: i for i, sid in enumerate(layer['sample_ids'])}
                common_indices = [sample_to_idx[sid] for sid in self.common_samples]
                
                aligned_layers[name] = {
                    'X': layer['X'][common_indices, :],
                    'y': layer['y'][common_indices],
                    'feature_names': layer['feature_names'],
                    'sample_ids': self.common_samples,
                    'n_samples': len(common_indices),
                    'n_features': layer['n_features']
                }
        
        else:  # position_based
            # Position-based alignment: match by position within groups
            for name, layer in self.layers.items():
                aligned_indices = []
                synthetic_ids = []
                y_aligned = []
                
                for sample_id in self.common_samples:
                    # Parse synthetic ID: "GroupName_N"
                    group_name, position = sample_id.rsplit('_', 1)
                    position = int(position) - 1  # Convert to 0-indexed
                    
                    # Find samples in this group
                    group_mask = layer['y'] == group_name
                    group_indices = np.where(group_mask)[0]
                    
                    if position < len(group_indices):
                        idx = group_indices[position]
                        aligned_indices.append(idx)
                        synthetic_ids.append(sample_id)
                        y_aligned.append(layer['y'][idx])
                
                aligned_layers[name] = {
                    'X': layer['X'][aligned_indices, :],
                    'y': np.array(y_aligned),
                    'feature_names': layer['feature_names'],
                    'sample_ids': synthetic_ids,
                    'n_samples': len(aligned_indices),
                    'n_features': layer['n_features']
                }
        
        # Verify consistent labels across layers
        y_arrays = [layer['y'] for layer in aligned_layers.values()]
        for i in range(1, len(y_arrays)):
            if not np.array_equal(y_arrays[0], y_arrays[i]):
                warnings.warn("Group labels differ across aligned layers!")
        
        print(f"\nAligned all layers to {len(self.common_samples)} common samples using {self.alignment_method} method")
        
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