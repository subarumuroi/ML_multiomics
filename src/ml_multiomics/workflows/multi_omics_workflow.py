"""
Multi-omics integration workflow.

Complete pipeline for integrating and analyzing multiple omics datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
from ml_multiomics.preprocessing import MetabolomicsPreprocessor, VolatilesPreprocessor, ProteomicsPreprocessor, OmicsIntegrator, MultiBlockData

# Import integration methods
from ml_multiomics.methods.multi_omics import DIABLO, ConcatenationBaseline, WeightedConcatenation


class MultiOmicsWorkflow:
    """
    Complete workflow for multi-omics integration.
    
    Pipeline:
    1. Preprocess each omics layer separately
    2. Integrate layers (align samples)
    3. Apply integration methods (DIABLO, concatenation)
    4. Compare performance
    5. Identify key features across omics
    """
    
    def __init__(self):
        """Initialize multi-omics workflow."""
        self.preprocessors = {}
        self.integrator = OmicsIntegrator()
        self.preprocessed_data = {}
        self.integration_methods = {}
        self.results = {}
        
    def add_omics_layer(self,
                       name: str,
                       df: pd.DataFrame,
                       omics_type: str,
                       group_col: str = 'Groups',
                       sample_id_col: Optional[str] = None,
                       config: Optional[Dict] = None):
        """
        Add and preprocess an omics layer.
        
        Parameters
        ----------
        name : str
            Name for this layer (e.g., 'amino_acids', 'proteomics')
        df : pd.DataFrame
            Raw data
        omics_type : str
            Type ('metabolomics', 'volatiles', 'proteomics')
        group_col : str
            Group column name
        sample_id_col : str, optional
            Sample ID column (if None, uses index)
        config : dict, optional
            Custom preprocessing config
        """
        print(f"\n{'='*60}")
        print(f"PREPROCESSING: {name.upper()}")
        print(f"{'='*60}")
        
        # Initialize preprocessor
        if omics_type == 'metabolomics':
            preprocessor = MetabolomicsPreprocessor(config)
        elif omics_type == 'volatiles':
            preprocessor = VolatilesPreprocessor(config)
        elif omics_type == 'proteomics':
            preprocessor = ProteomicsPreprocessor(config)
        else:
            raise ValueError(f"Unknown omics type: {omics_type}")
        
        # Preprocess
        X, y, feature_names = preprocessor.preprocess(df, group_col)
        
        # Get sample IDs
        if sample_id_col:
            sample_ids = df[sample_id_col].tolist()
        else:
            sample_ids = df.index.tolist()
        
        # Store preprocessed data
        self.preprocessed_data[name] = {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'sample_ids': sample_ids
        }
        
        # Add to integrator
        self.integrator.add_layer(
            name=name,
            X=X,
            y=y,
            feature_names=feature_names,
            sample_ids=sample_ids
        )
        
        print("\nPreprocessing Log:")
        preprocessor.print_log()
        
        return X, y, feature_names
    
    def prepare_integration(self) -> MultiBlockData:
        """
        Prepare data for multi-omics integration.
        
        Aligns all layers to common samples.
        
        Returns
        -------
        MultiBlockData
            Aligned multi-block data container
        """
        print(f"\n{'='*60}")
        print(f"PREPARING INTEGRATION")
        print(f"{'='*60}")
        
        # Align layers
        aligned_layers = self.integrator.align_layers()
        
        # Create MultiBlockData container
        multi_block = MultiBlockData()
        
        for name, layer in aligned_layers.items():
            multi_block.add_block(
                name=name,
                X=layer['X'],
                feature_names=layer['feature_names']
            )
        
        # Set labels (should be same for all aligned layers)
        multi_block.set_labels(
            y=list(aligned_layers.values())[0]['y'],
            sample_ids=list(aligned_layers.values())[0]['sample_ids']
        )
        
        print("\nIntegration Summary:")
        print(multi_block.get_summary().to_string(index=False))
        
        return multi_block
    
    def run_diablo(self,
                   multi_block: MultiBlockData,
                   n_components: int = 2,
                   design: Optional[np.ndarray] = None,
                   plot: bool = True) -> DIABLO:
        """
        Run DIABLO integration.
        
        Parameters
        ----------
        multi_block : MultiBlockData
            Aligned multi-block data
        n_components : int
            Number of components
        design : np.ndarray, optional
            Design matrix
        plot : bool
            Whether to generate plots
            
        Returns
        -------
        DIABLO
            Fitted DIABLO object
        """
        print(f"\n{'='*60}")
        print(f"DIABLO INTEGRATION")
        print(f"{'='*60}")
        
        # Prepare blocks dictionary
        blocks = {name: multi_block.get_block(name) 
                 for name in multi_block.get_block_names()}

        feature_names = {name: multi_block.blocks[name]['feature_names']
                         for name in multi_block.get_block_names()}
        
        # Fit DIABLO
        diablo = DIABLO(n_components=n_components)
        diablo.fit(blocks, multi_block.y, feature_names=feature_names)
        
        # Get block correlations
        corr_df = diablo.calculate_block_correlations()
        print("\nBlock Correlations (Component 1):")
        print(corr_df.to_string())
        
        self.integration_methods['diablo'] = diablo
        self.results['diablo_correlations'] = corr_df
        
        # Get VIP scores
        all_vips = diablo.get_all_vips(top_n=10)
        print(f"\nTop Features per Block:")
        for block_name in multi_block.get_block_names():
            block_vips = all_vips[all_vips['Block'] == block_name]
            print(f"\n{block_name}:")
            print(block_vips[['Feature', 'VIP']].head(5).to_string(index=False))
        
        self.results['diablo_vips'] = all_vips
        
        # Generate plots
        if plot:
            print("\nGenerating DIABLO plots...")
            
            # Sample plot
            fig_samples, _ = diablo.plot_samples(multi_block.y, comp_x=1, comp_y=2)
            self.results['fig_diablo_samples'] = fig_samples
            
            # Block correlations
            fig_corr, _ = diablo.plot_block_correlations()
            self.results['fig_diablo_correlations'] = fig_corr
            
            # Arrow plot
            fig_arrow, _ = diablo.plot_arrow_plot(multi_block.y, comp_x=1, comp_y=2)
            self.results['fig_diablo_arrow'] = fig_arrow
            
            # Circos plot
            fig_circos, _ = diablo.plot_circos(threshold=0.5)
            self.results['fig_diablo_circos'] = fig_circos
        
        return diablo

    def run_concatenation_baseline(self,
                                   multi_block: Optional[MultiBlockData] = None,
                                   classifier: str = 'random_forest',
                                   cv: bool = True) -> ConcatenationBaseline:
        """
        Run simple concatenation baseline.
        
        Parameters
        ----------
        multi_block : MultiBlockData, optional
            If None, uses integrator to concatenate
        classifier : str
            Classifier type
        cv : bool
            Whether to perform cross-validation
            
        Returns
        -------
        ConcatenationBaseline
            Fitted baseline model
        """
        print(f"\n{'='*60}")
        print(f"CONCATENATION BASELINE")
        print(f"{'='*60}")
        
        # Get concatenated data
        if multi_block is None:
            X_concat, y, feature_names = self.integrator.concatenate(align=True)
        else:
            blocks = {name: multi_block.get_block(name) 
                     for name in multi_block.get_block_names()}
            X_concat = np.hstack(list(blocks.values()))
            y = multi_block.y
            feature_names = []
            for name in multi_block.get_block_names():
                features = multi_block.blocks[name]['feature_names']
                feature_names.extend([f"{name}_{f}" for f in features])
        
        # Fit baseline
        baseline = ConcatenationBaseline(classifier=classifier)
        baseline.fit(X_concat, y, feature_names)
        
        self.integration_methods['concatenation'] = baseline
        
        # Cross-validation
        if cv:
            print("\nPerforming Leave-One-Out Cross-Validation...")
            cv_results = baseline.cross_validate(X_concat, y)
            
            print(f"\n{cv_results['cv_type']} Accuracy: {cv_results['accuracy']:.2%} ± {cv_results['std']:.2%}")
            print("\nClassification Report:")
            print(cv_results['classification_report'])
            
            self.results['concatenation_cv'] = cv_results
        
        # Feature importance (for tree-based models)
        if classifier == 'random_forest':
            importance_df = baseline.get_feature_importance(top_n=20)
            print(f"\nTop 10 Important Features:")
            print(importance_df.head(10).to_string(index=False))
            
            self.results['concatenation_importance'] = importance_df
        
        return baseline
    
    def compare_methods(self) -> pd.DataFrame:
        """
        Compare performance of different integration methods.
        
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        print(f"\n{'='*60}")
        print(f"METHOD COMPARISON")
        print(f"{'='*60}")
        
        comparison_data = []
        
        # Get results for each method
        if 'concatenation_cv' in self.results:
            comparison_data.append({
                'Method': 'Concatenation',
                'Accuracy': self.results['concatenation_cv']['accuracy'],
                'Std': self.results['concatenation_cv']['std']
            })
        
        # Add more methods as they're implemented
        # DIABLO doesn't have built-in CV in this implementation,
        # but could be added
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print("\nPerformance Comparison:")
            print(comparison_df.to_string(index=False))
            
            self.results['method_comparison'] = comparison_df
            return comparison_df
        
        return None

    def generate_statistical_report(self):
        """Generate report on statistical power and limitations."""
        # Get number of samples
        n_samples = len(self.integrator.common_samples)
        
        # Count total features from preprocessed data
        total_features = 0
        for layer_name, layer_data in self.preprocessed_data.items():
            total_features += len(layer_data['feature_names'])
        
        # Get number of classes
        first_layer = list(self.preprocessed_data.values())[0]
        n_classes = len(np.unique(first_layer['y']))
        
        feature_to_sample_ratio = total_features / n_samples
        
        print("\n" + "="*60)
        print("STATISTICAL LIMITATIONS REPORT")
        print("="*60)
        print(f"Sample Size: {n_samples}")
        print(f"Total Features: {total_features}")
        print(f"Feature:Sample Ratio: {feature_to_sample_ratio:.0f}:1")
        print(f"\n⚠️  HIGH OVERFITTING RISK")
        print("   Results are hypothesis-generating only")
        print("   Recommend validation with n>30")
        print("="*60)

    def run_full_integration(self,
                           data_dict: Dict[str, pd.DataFrame],
                           omics_types: Dict[str, str],
                           group_col: str = 'Groups',
                           sample_id_col: Optional[str] = None,
                           n_components: int = 2) -> Dict:
        """
        Run complete multi-omics integration pipeline.
        
        Parameters
        ----------
        data_dict : dict
            {layer_name: dataframe} dictionary
        omics_types : dict
            {layer_name: omics_type} dictionary
        group_col : str
            Group column name
        sample_id_col : str, optional
            Sample ID column
        n_components : int
            Number of components for integration
            
        Returns
        -------
        dict
            Dictionary of all results
        """
        # Step 1: Preprocess all layers
        for name, df in data_dict.items():
            self.add_omics_layer(
                name=name,
                df=df,
                omics_type=omics_types[name],
                group_col=group_col,
                sample_id_col=sample_id_col
            )
        
        # Step 2: Prepare integration
        multi_block = self.prepare_integration()
        self.generate_statistical_report()
        
        # Step 3: Run DIABLO
        self.run_diablo(multi_block, n_components=n_components, plot=True)
        
        # Step 4: Run MOFA+
        #self.run_mofa_integration(n_factors=10)

        # Step 5: Run concatenation baseline
        self.run_concatenation_baseline(multi_block, cv=True)
    
        # Step 6: Compare methods
        self.compare_methods()
        
        print(f"\n{'='*60}")
        print(f"INTEGRATION COMPLETE")
        print(f"{'='*60}")
        
        return self.results
    
    def save_results(self, output_dir: str):
        """
        Save all results.
        
        Parameters
        ----------
        output_dir : str
            Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tables
        for key in ['diablo_correlations', 'diablo_vips', 
                   'concatenation_importance', 'method_comparison']:
            if key in self.results:
                self.results[key].to_csv(
                    f"{output_dir}/{key}.csv", index=False)
        
        # Save MOFA results 
        if hasattr(self, 'mofa_model'):
            mofa_dir = f"{output_dir}/mofa"
            os.makedirs(mofa_dir, exist_ok=True)
            self.mofa_model.save_results(mofa_dir)
            
            # Save MOFA plots
            fig_var = self.mofa_model.plot_variance_explained()
            fig_var.savefig(f"{mofa_dir}/variance_explained.png", 
                        dpi=300, bbox_inches='tight')
            plt.close(fig_var)
            
            fig_scores = self.mofa_model.plot_factor_scores(1, 2)
            fig_scores.savefig(f"{mofa_dir}/factor_scores.png", 
                            dpi=300, bbox_inches='tight')
            plt.close(fig_scores)

        # Save figures
        for key, value in self.results.items():
            if key.startswith('fig_'):
                fig_name = key.replace('fig_', '')
                value.savefig(f"{output_dir}/{fig_name}.png", dpi=300, bbox_inches='tight')
        
        print(f"\nResults saved to: {output_dir}")
    
    def display_results(self):
        """Display all plots."""
        plt.show()


# Example usage
if __name__ == "__main__":
    # Load all omics datasets
    data_dict = {
        'amino_acids': pd.read_csv("data/badata-amino-acids.csv"),
        'central_carbon': pd.read_csv("data/badata-metabolomics.csv"),
        'aromatics': pd.read_csv("data/badata-aromatics.csv"),
        'proteomics': pd.read_csv("data/badata-proteomics-imputed.csv")
    }
    
    omics_types = {
        'amino_acids': 'metabolomics',
        'central_carbon': 'metabolomics',
        'aromatics': 'volatiles',
        'proteomics': 'proteomics'
    }
    
    # Initialize workflow
    workflow = MultiOmicsWorkflow()
    
    # Run full integration
    results = workflow.run_full_integration(
        data_dict=data_dict,
        omics_types=omics_types,
        group_col='Groups',
        n_components=2
    )
    
    # Save results
    workflow.save_results("results/multi_omics_integration")
    
    # Display plots
    workflow.display_results()