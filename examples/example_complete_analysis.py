"""
Complete example analysis using the multi-omics framework.

This script demonstrates:
1. Single omics analysis for each layer
2. Multi-omics integration with DIABLO
3. Baseline concatenation comparison
4. Feature extraction and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import workflows
from workflows.single_omics_workflow import SingleOmicsWorkflow
from workflows.multi_omics_workflow import MultiOmicsWorkflow

# Import utilities
from utils.validation import CrossValidator, PermutationTest, ModelComparator
from utils.visualization import OmicsPlotter, save_publication_figure


def run_single_omics_analyses(data_paths: dict, output_dir: str = "results/single_omics"):
    """
    Run individual analysis for each omics layer.
    
    Parameters
    ----------
    data_paths : dict
        {layer_name: (file_path, omics_type)} dictionary
    output_dir : str
        Output directory
    """
    print("\n" + "="*80)
    print("SINGLE OMICS ANALYSES")
    print("="*80)
    
    results = {}
    
    for layer_name, (file_path, omics_type) in data_paths.items():
        print(f"\n\nAnalyzing: {layer_name}")
        print("-" * 80)
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Create workflow
        workflow = SingleOmicsWorkflow(
            omics_type=omics_type,
            config=None  # Use defaults
        )
        
        # Run full analysis
        layer_results = workflow.run_full_analysis(
            df=df,
            group_col='Groups',
            n_pca_components=5,
            n_plsda_components=2
        )
        
        # Save results
        layer_output_dir = f"{output_dir}/{layer_name}"
        workflow.save_results(layer_output_dir)
        
        results[layer_name] = {
            'workflow': workflow,
            'results': layer_results
        }
    
    return results


def run_multi_omics_integration(data_paths: dict, output_dir: str = "results/multi_omics"):
    """
    Run multi-omics integration analysis.
    
    Parameters
    ----------
    data_paths : dict
        {layer_name: (file_path, omics_type)} dictionary
    output_dir : str
        Output directory
    """
    print("\n" + "="*80)
    print("MULTI-OMICS INTEGRATION")
    print("="*80)
    
    # Load all data
    data_dict = {}
    omics_types = {}
    
    for layer_name, (file_path, omics_type) in data_paths.items():
        data_dict[layer_name] = pd.read_csv(file_path)
        omics_types[layer_name] = omics_type
    
    # Create workflow
    workflow = MultiOmicsWorkflow()
    
    # Run integration
    results = workflow.run_full_integration(
        data_dict=data_dict,
        omics_types=omics_types,
        group_col='Groups',
        n_components=2
    )
    
    # Save results
    workflow.save_results(output_dir)
    
    return workflow, results


def generate_summary_report(single_results: dict, 
                           multi_results: dict,
                           output_file: str = "results/analysis_summary.txt"):
    """
    Generate text summary of all analyses.
    
    Parameters
    ----------
    single_results : dict
        Results from single omics analyses
    multi_results : dict
        Results from multi-omics integration
    output_file : str
        Output file path
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-OMICS ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Single omics summary
        f.write("SINGLE OMICS RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        for layer_name, data in single_results.items():
            results = data['results']
            
            f.write(f"{layer_name.upper()}\n")
            
            # PCA variance
            if 'pca_variance' in results:
                var_df = results['pca_variance']
                total_var = var_df['Cumulative_Variance'].iloc[1]  # First 2 PCs
                f.write(f"  PCA: First 2 components explain {total_var:.1f}% variance\n")
            
            # PLS-DA accuracy
            if 'plsda_cv' in results:
                cv_res = results['plsda_cv']
                f.write(f"  PLS-DA ({cv_res['cv_type']}): {cv_res['accuracy']:.2%} accuracy\n")
            
            # Top features
            if 'plsda_vip' in results:
                vip_df = results['plsda_vip']
                n_important = (vip_df['VIP'] > 1.0).sum()
                f.write(f"  Important features (VIP > 1): {n_important}\n")
                f.write(f"  Top feature: {vip_df.iloc[0]['Feature']} (VIP = {vip_df.iloc[0]['VIP']:.2f})\n")
            
            f.write("\n")
        
        # Multi-omics summary
        f.write("\nMULTI-OMICS INTEGRATION RESULTS\n")
        f.write("-"*80 + "\n\n")
        
        # DIABLO correlations
        if 'diablo_correlations' in multi_results:
            f.write("DIABLO Block Correlations:\n")
            corr_df = multi_results['diablo_correlations']
            f.write(corr_df.to_string() + "\n\n")
        
        # Concatenation performance
        if 'concatenation_cv' in multi_results:
            cv_res = multi_results['concatenation_cv']
            f.write(f"Concatenation Baseline ({cv_res['cv_type']}):\n")
            f.write(f"  Accuracy: {cv_res['accuracy']:.2%} ± {cv_res['std']:.2%}\n\n")
        
        # Top multi-omics features
        if 'diablo_vips' in multi_results:
            vip_df = multi_results['diablo_vips']
            f.write("Top Features Across All Omics:\n")
            for block in vip_df['Block'].unique():
                block_vips = vip_df[vip_df['Block'] == block].head(3)
                f.write(f"\n  {block}:\n")
                for _, row in block_vips.iterrows():
                    f.write(f"    - {row['Feature']} (VIP = {row['VIP']:.2f})\n")
    
    print(f"\nSummary report saved to: {output_file}")


def create_overview_visualizations(data_paths: dict, 
                                   single_results: dict,
                                   output_dir: str = "results/overview"):
    """
    Create overview visualizations.
    
    Parameters
    ----------
    data_paths : dict
        Data paths dictionary
    single_results : dict
        Single omics results
    output_dir : str
        Output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plotter = OmicsPlotter()
    
    # 1. Sample distribution overview
    data_dict = {name: pd.read_csv(path) for name, (path, _) in data_paths.items()}
    fig, _ = plotter.plot_sample_overview(data_dict, group_col='Groups')
    save_publication_figure(fig, f"{output_dir}/sample_distribution.png")
    plt.close()
    
    # 2. Performance comparison
    performance_dict = {}
    for layer_name, data in single_results.items():
        if 'plsda_cv' in data['results']:
            cv_res = data['results']['plsda_cv']
            performance_dict[layer_name] = {
                'mean': cv_res['accuracy'],
                'std': 0.0  # LOO doesn't have std in same way
            }
    
    if performance_dict:
        fig, _ = plotter.plot_performance_comparison(
            performance_dict, metric='accuracy'
        )
        save_publication_figure(fig, f"{output_dir}/performance_comparison.png")
        plt.close()
    
    print(f"\nOverview visualizations saved to: {output_dir}")


def main():
    """
    Run complete analysis pipeline.
    """
    # Define data paths
    # Modify these paths to match your data location
    data_paths = {
        'amino_acids': ('data/badata-amino-acids.csv', 'metabolomics'),
        'central_carbon': ('data/badata-metabolomics.csv', 'metabolomics'),
        'aromatics': ('data/badata-aromatics.csv', 'volatiles'),
        'proteomics': ('data/badata-proteomics-imputed.csv', 'proteomics')
    }
    
    # Create output directories
    Path("results/single_omics").mkdir(parents=True, exist_ok=True)
    Path("results/multi_omics").mkdir(parents=True, exist_ok=True)
    Path("results/overview").mkdir(parents=True, exist_ok=True)
    
    # Step 1: Single omics analyses
    print("\n" + "#"*80)
    print("STEP 1: INDIVIDUAL OMICS ANALYSES")
    print("#"*80)
    
    single_results = run_single_omics_analyses(
        data_paths=data_paths,
        output_dir="results/single_omics"
    )
    
    # Step 2: Multi-omics integration
    print("\n" + "#"*80)
    print("STEP 2: MULTI-OMICS INTEGRATION")
    print("#"*80)
    
    multi_workflow, multi_results = run_multi_omics_integration(
        data_paths=data_paths,
        output_dir="results/multi_omics"
    )
    
    # Step 3: Generate summary report
    print("\n" + "#"*80)
    print("STEP 3: GENERATING SUMMARY REPORT")
    print("#"*80)
    
    generate_summary_report(
        single_results=single_results,
        multi_results=multi_results,
        output_file="results/analysis_summary.txt"
    )
    
    # Step 4: Create overview visualizations
    print("\n" + "#"*80)
    print("STEP 4: CREATING OVERVIEW VISUALIZATIONS")
    print("#"*80)
    
    create_overview_visualizations(
        data_paths=data_paths,
        single_results=single_results,
        output_dir="results/overview"
    )
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nResults saved in:")
    print("  - results/single_omics/     (individual omics analyses)")
    print("  - results/multi_omics/      (integration results)")
    print("  - results/overview/         (summary visualizations)")
    print("  - results/analysis_summary.txt (text summary)")
    print("\nYou can now explore the generated figures and tables.")


if __name__ == "__main__":
    main()