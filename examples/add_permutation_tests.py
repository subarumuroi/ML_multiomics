"""
Quick script to add permutation testing to existing analysis results.

Run this after completing your multi-omics integration to add
statistical validation via permutation tests.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_multiomics.workflows.multi_omics_workflow import MultiOmicsWorkflow


def add_permutation_tests_to_existing_analysis(n_permutations=1000):
    """
    Add permutation testing to completed multi-omics analysis.
    
    Parameters
    ----------
    n_permutations : int
        Number of permutations (1000 for quick POC, 5000-10000 for publication)
    """
    print("\n" + "="*80)
    print("ADDING PERMUTATION TESTS TO EXISTING ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
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
    
    # Preprocess layers (needed to get multi_block)
    print("\nPreprocessing omics layers...")
    for name, df in data_dict.items():
        workflow.add_omics_layer(
            name=name,
            df=df,
            omics_type=omics_types[name],
            group_col='Groups'
        )
    
    # Prepare integration
    multi_block = workflow.prepare_integration()
    
    # Load existing CV results to populate workflow.results
    print("\nLoading existing CV results...")
    existing_results = pd.read_csv("results/multi_omics/method_comparison.csv")
    
    # Populate results dictionary with existing CV results
    for _, row in existing_results.iterrows():
        if row['Method'] == 'Concatenation':
            workflow.results['concatenation_cv'] = {
                'accuracy': row['Accuracy'],
                'std': row['Std']
            }
        elif row['Method'] == 'Block-wise Ensemble':
            workflow.results['ensemble_cv'] = {
                'accuracy': row['Accuracy'],
                'std': row['Std']
            }
        elif row['Method'] == 'DIABLO':
            workflow.results['diablo_cv'] = {
                'accuracy': row['Accuracy'],
                'std': row['Std']
            }
    
    # Re-fit models for permutation testing
    print("\nRe-fitting models for permutation testing...")
    
    # Fit concatenation
    from ml_multiomics.methods.multi_omics import ConcatenationBaseline
    blocks = {name: multi_block.get_block(name) 
             for name in multi_block.get_block_names()}
    X_concat = np.hstack(list(blocks.values()))
    baseline = ConcatenationBaseline(classifier='random_forest')
    baseline.fit(X_concat, multi_block.y)
    workflow.integration_methods['concatenation'] = baseline
    
    # Fit ensemble
    from ml_multiomics.methods.multi_omics import BlockWiseEnsemble
    feature_names = {name: multi_block.blocks[name]['feature_names']
                    for name in multi_block.get_block_names()}
    ensemble = BlockWiseEnsemble(classifier='random_forest', voting='soft')
    ensemble.fit(blocks, multi_block.y, feature_names=feature_names)
    workflow.integration_methods['ensemble'] = ensemble
    
    # Fit DIABLO
    from ml_multiomics.methods.multi_omics import DIABLO
    diablo = DIABLO(n_components=2)
    diablo.fit(blocks, multi_block.y, 
               feature_names=feature_names,
               sample_ids=multi_block.sample_ids)
    workflow.integration_methods['diablo'] = diablo
    
    # Run permutation tests
    print("\n" + "="*80)
    print(f"RUNNING PERMUTATION TESTS (n={n_permutations})")
    print("="*80)
    print("\nThis will take a few minutes...")
    print(f"- Testing if performance > random chance")
    print(f"- {n_permutations} permutations per method")
    
    perm_results = workflow.run_permutation_tests(
        multi_block=multi_block,
        n_permutations=n_permutations,
        random_state=42
    )
    
    # Save permutation results
    output_file = "results/multi_omics/permutation_tests.csv"
    perm_results.to_csv(output_file, index=False)
    
    print(f"\n✓ Permutation test results saved to: {output_file}")
    
    # Create summary report
    create_permutation_summary_report(perm_results, len(multi_block.y))
    
    return perm_results


def create_permutation_summary_report(perm_df: pd.DataFrame, n_samples: int):
    """Create a summary report of permutation test results."""
    
    report_file = "results/multi_omics/permutation_test_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PERMUTATION TEST REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: Multi-omics integration (n={n_samples} samples)\n")
        f.write(f"Test: Null hypothesis = Model performance is no better than chance\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(perm_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n\n")
        
        for _, row in perm_df.iterrows():
            f.write(f"{row['Method']}:\n")
            f.write(f"  True Accuracy: {row['True_Accuracy']:.3f}\n")
            f.write(f"  Permutation Null: {row['Perm_Mean']:.3f} ± {row['Perm_Std']:.3f}\n")
            f.write(f"  P-value: {row['P_Value']:.4f}\n")
            
            if row['Significant']:
                f.write(f"  ✓ SIGNIFICANT: Performance better than chance (p<0.05)\n")
            else:
                f.write(f"  ✗ NOT SIGNIFICANT: Cannot reject null hypothesis (p>=0.05)\n")
            
            if row['True_Accuracy'] == 1.0 and n_samples < 15:
                f.write(f"  ⚠ WARNING: Perfect accuracy with n={n_samples} suggests overfitting\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("STATISTICAL CONSIDERATIONS\n")
        f.write("="*80 + "\n\n")
        
        if n_samples < 15:
            f.write(f"⚠ VERY SMALL SAMPLE SIZE (n={n_samples}):\n")
            f.write("  - Results are exploratory/hypothesis-generating only\n")
            f.write("  - High risk of overfitting to training data\n")
            f.write("  - Validation with n>30 strongly recommended\n")
            f.write("  - Focus on methodology demonstration for POC\n\n")
        elif n_samples < 30:
            f.write(f"⚠ SMALL SAMPLE SIZE (n={n_samples}):\n")
            f.write("  - Limited statistical power\n")
            f.write("  - Results should be validated independently\n")
            f.write("  - Recommend n>50 for robust conclusions\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("  1. Acquire more samples when possible (target n>30)\n")
        f.write("  2. Validate findings on independent dataset\n")
        f.write("  3. Focus on reproducible biomarker selection\n")
        f.write("  4. Report limitations clearly in presentations/publications\n")
        f.write("  5. Use this POC to demonstrate methodology to partners\n\n")
        
        f.write("="*80 + "\n")
        f.write("PROOF-OF-CONCEPT VALUE\n")
        f.write("="*80 + "\n\n")
        
        f.write("This analysis demonstrates:\n")
        f.write("  ✓ Complete statistical validation framework\n")
        f.write("  ✓ Permutation testing methodology for omics integration\n")
        f.write("  ✓ Handling of multi-block data with R/Python integration\n")
        f.write("  ✓ Ready-to-scale pipeline for larger datasets\n")
        f.write("  ✓ Professional statistical rigor in analysis workflow\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n✓ Summary report saved to: {report_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Add permutation testing to multi-omics analysis'
    )
    parser.add_argument(
        '--n-permutations', 
        type=int, 
        default=1000,
        help='Number of permutations (default: 1000)'
    )
    
    args = parser.parse_args()
    
    print(f"\nRunning with {args.n_permutations} permutations")
    print("(Use --n-permutations 5000 for publication-quality results)")
    
    perm_results = add_permutation_tests_to_existing_analysis(
        n_permutations=args.n_permutations
    )
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - results/multi_omics/permutation_tests.csv")
    print("  - results/multi_omics/permutation_test_report.txt")
    print("\n")
