"""
Single omics analysis workflow.

Complete pipeline for analyzing individual omics datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
from ml_multiomics.preprocessing import MetabolomicsPreprocessor, VolatilesPreprocessor, ProteomicsPreprocessor

# Import analysis methods
from ml_multiomics.methods.single_omics import PCAAnalysis, PLSDAAnalysis


class SingleOmicsWorkflow:
    """
    Complete workflow for single omics analysis.
    
    Pipeline:
    1. Load data
    2. Preprocess (filter, impute, transform, scale)
    3. PCA (exploratory)
    4. PLS-DA (supervised classification)
    5. Generate reports and visualizations
    """
    
    def __init__(self, omics_type: str, config: Optional[Dict] = None):
        """
        Initialize workflow.
        
        Parameters
        ----------
        omics_type : str
            Type of omics ('metabolomics', 'volatiles', 'proteomics')
        config : dict, optional
            Custom preprocessing configuration
        """
        self.omics_type = omics_type
        self.config = config
        
        # Initialize preprocessor
        if omics_type == 'metabolomics':
            self.preprocessor = MetabolomicsPreprocessor(config)
        elif omics_type == 'volatiles':
            self.preprocessor = VolatilesPreprocessor(config)
        elif omics_type == 'proteomics':
            self.preprocessor = ProteomicsPreprocessor(config)
        else:
            raise ValueError(f"Unknown omics type: {omics_type}")
        
        # Initialize analysis objects
        self.pca = None
        self.plsda = None
        
        # Store results
        self.X = None
        self.y = None
        self.feature_names = None
        self.results = {}
        
    def run_preprocessing(self, 
                         df: pd.DataFrame,
                         group_col: str = 'Groups') -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Run preprocessing pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data with group column
        group_col : str
            Name of group column
            
        Returns
        -------
        X : np.ndarray
            Preprocessed features
        y : np.ndarray
            Group labels
        feature_names : list
            Feature names
        """
        print(f"\n{'='*60}")
        print(f"PREPROCESSING: {self.omics_type.upper()}")
        print(f"{'='*60}")
        
        X, y, feature_names = self.preprocessor.preprocess(df, group_col)
        
        # Print log
        print("\nPreprocessing Log:")
        self.preprocessor.print_log()
        
        # Store results
        self.X = X
        self.y = y
        self.feature_names = feature_names
        
        return X, y, feature_names
    
    def run_pca(self, 
                n_components: int = 5,
                plot: bool = True) -> PCAAnalysis:
        """
        Run PCA analysis.
        
        Parameters
        ----------
        n_components : int
            Number of components
        plot : bool
            Whether to generate plots
            
        Returns
        -------
        PCAAnalysis
            Fitted PCA object
        """
        print(f"\n{'='*60}")
        print(f"PCA ANALYSIS")
        print(f"{'='*60}")
        
        self.pca = PCAAnalysis(n_components=n_components)
        self.pca.fit(self.X, self.feature_names)
        
        # Get variance explained
        var_df = self.pca.get_variance_explained()
        print("\nVariance Explained:")
        print(var_df.to_string(index=False))
        
        # Store results
        self.results['pca_variance'] = var_df
        
        # Generate plots
        if plot:
            print("\nGenerating PCA plots...")
            
            # Scree plot
            fig_scree, _ = self.pca.plot_scree()
            self.results['fig_pca_scree'] = fig_scree
            
            # Scores plot
            fig_scores, _ = self.pca.plot_scores(self.y, pc_x=1, pc_y=2)
            self.results['fig_pca_scores'] = fig_scores
            
            # Loadings plot
            fig_loadings, _ = self.pca.plot_loadings(pc=1, top_n=15)
            self.results['fig_pca_loadings'] = fig_loadings
            
            # Biplot
            fig_biplot, _ = self.pca.plot_biplot(self.y, pc_x=1, pc_y=2, n_loadings=5)
            self.results['fig_pca_biplot'] = fig_biplot
        
        return self.pca
    
    def run_plsda(self,
                  n_components: int = 2,
                  cv: bool = True,
                  plot: bool = True) -> PLSDAAnalysis:
        """
        Run PLS-DA analysis.
        
        Parameters
        ----------
        n_components : int
            Number of latent variables
        cv : bool
            Whether to perform cross-validation
        plot : bool
            Whether to generate plots
            
        Returns
        -------
        PLSDAAnalysis
            Fitted PLS-DA object
        """
        print(f"\n{'='*60}")
        print(f"PLS-DA ANALYSIS")
        print(f"{'='*60}")
        
        self.plsda = PLSDAAnalysis(n_components=n_components)
        self.plsda.fit(self.X, self.y, self.feature_names)
        
        # Cross-validation
        if cv:
            print("\nPerforming Leave-One-Out Cross-Validation...")
            cv_results = self.plsda.cross_validate(self.X, self.y)
            
            print(f"\n{cv_results['cv_type']} Accuracy: {cv_results['accuracy']:.2%}")
            print("\nClassification Report:")
            print(cv_results['classification_report'])
            
            self.results['plsda_cv'] = cv_results
        
        # VIP scores
        vip_df = self.plsda.get_vip_scores()
        print(f"\nTop 10 Important Features (VIP > 1):")
        print(vip_df.head(10).to_string(index=False))
        
        self.results['plsda_vip'] = vip_df
        
        # Generate plots
        if plot:
            print("\nGenerating PLS-DA plots...")
            
            # Scores plot
            fig_scores, _ = self.plsda.plot_scores(self.y, lv_x=1, lv_y=2)
            self.results['fig_plsda_scores'] = fig_scores
            
            # VIP plot
            fig_vip, _ = self.plsda.plot_vip(top_n=20)
            self.results['fig_plsda_vip'] = fig_vip
            
            # Loadings plot
            fig_loadings, _ = self.plsda.plot_loadings(lv=1, top_n=15)
            self.results['fig_plsda_loadings'] = fig_loadings
            
            # Confusion matrix
            if cv:
                fig_cm, _ = self.plsda.plot_confusion_matrix(cv_results)
                self.results['fig_plsda_cm'] = fig_cm
        
        return self.plsda
    
    def run_full_analysis(self,
                         df: pd.DataFrame,
                         group_col: str = 'Groups',
                         n_pca_components: int = 5,
                         n_plsda_components: int = 2) -> Dict:
        """
        Run complete single omics analysis pipeline.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data
        group_col : str
            Group column name
        n_pca_components : int
            Number of PCA components
        n_plsda_components : int
            Number of PLS-DA components
            
        Returns
        -------
        dict
            Dictionary of all results
        """
        # Step 1: Preprocessing
        self.run_preprocessing(df, group_col)
        
        # Step 2: PCA
        self.run_pca(n_components=n_pca_components, plot=True)
        
        # Step 3: PLS-DA
        self.run_plsda(n_components=n_plsda_components, cv=True, plot=True)
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
        return self.results
    
    def save_results(self, output_dir: str):
        """
        Save all results to directory.
        
        Parameters
        ----------
        output_dir : str
            Output directory path
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tables
        if 'pca_variance' in self.results:
            self.results['pca_variance'].to_csv(
                f"{output_dir}/pca_variance_explained.csv", index=False)
        
        if 'plsda_vip' in self.results:
            self.results['plsda_vip'].to_csv(
                f"{output_dir}/plsda_vip_scores.csv", index=False)
        
        # Save figures
        for key, value in self.results.items():
            if key.startswith('fig_'):
                fig_name = key.replace('fig_', '')
                value.savefig(f"{output_dir}/{fig_name}.png", dpi=300, bbox_inches='tight')
        
        print(f"\nResults saved to: {output_dir}")
    
    def display_results(self):
        """Display all generated plots."""
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example: Analyze amino acids data
    
    # Load data
    aa_data = pd.read_csv("data/badata-amino-acids.csv")
    
    # Initialize workflow
    workflow = SingleOmicsWorkflow(omics_type='metabolomics')
    
    # Run full analysis
    results = workflow.run_full_analysis(
        df=aa_data,
        group_col='Groups',
        n_pca_components=5,
        n_plsda_components=2
    )
    
    # Save results
    workflow.save_results("results/amino_acids")
    
    # Display plots
    workflow.display_results()