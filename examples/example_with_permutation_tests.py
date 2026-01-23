"""
Example: Multi-omics integration with permutation testing.

This example demonstrates how to add permutation testing to validate
that model performance is better than random chance - particularly
important for small sample sizes.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from ml_multiomics.workflows.multi_omics_workflow import MultiOmicsWorkflow


def run_multi_omics_with_permutation_tests():
    """
    Run complete multi-omics analysis with permutation testing.
    """
    print("\n" + "="*80)
    print("MULTI-OMICS INTEGRATION WITH PERMUTATION TESTING")
    print("="*80)
    
    # Define data paths
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
    
    # Run standard integration pipeline
    print("\n" + "="*80)
    print("STEP 1: STANDARD INTEGRATION PIPELINE")
    print("="*80)
    
    results = workflow.run_full_integration(
        data_dict=data_dict,
        omics_types=omics_types,
        group_col='Groups',
        n_components=2
    )
    
    # Get the multi-block data for permutation testing
    multi_block = workflow.integrator.get_multi_block()
    
    # Run permutation tests
    print("\n" + "="*80)
    print("STEP 2: PERMUTATION TESTING")
    print("="*80)
    print("\nRunning permutation tests to validate model significance...")
    print("This tests: H0 = Model performance is no better than random chance")
    print("\nNote: With n=9 samples, this is a proof-of-concept demonstration")
    print("of the statistical framework for larger datasets.")
    
    # Run permutation tests (use fewer permutations for POC to save time)
    perm_results = workflow.run_permutation_tests(
        multi_block=multi_block,
        n_permutations=1000,  # Use 1000 for quick POC; 10000+ for production
        random_state=42
    )
    
    # Save all results including permutation tests
    output_dir = "results/multi_omics"
    workflow.save_results(output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - method_comparison.csv: Performance comparison")
    print("  - permutation_tests.csv: Statistical validation")
    print("  - diablo_*.csv: DIABLO-specific results")
    print("  - ensemble_importance_*.csv: Block-specific feature importance")
    print("  - Various plots (.png)")
    
    return workflow, perm_results


def interpret_permutation_results(perm_df: pd.DataFrame, n_samples: int):
    """
    Provide interpretation guidance for permutation test results.
    
    Parameters
    ----------
    perm_df : pd.DataFrame
        Permutation test results
    n_samples : int
        Number of samples in dataset
    """
    print("\n" + "="*80)
    print("PERMUTATION TEST INTERPRETATION GUIDE")
    print("="*80)
    
    print(f"\nDataset size: n={n_samples} samples")
    print("\n1. P-VALUE INTERPRETATION:")
    print("   - p < 0.05: Model performance significantly better than random")
    print("   - p >= 0.05: Cannot reject null hypothesis (performance = random)")
    
    print("\n2. SAMPLE SIZE CONSIDERATIONS (n={n_samples}):")
    if n_samples < 15:
        print("   ⚠ VERY SMALL: Results are hypothesis-generating only")
        print("   ⚠ High risk of overfitting and spurious findings")
        print("   ⚠ Recommend n>30 for reliable conclusions")
    elif n_samples < 30:
        print("   ⚠ SMALL: Limited statistical power")
        print("   ⚠ Results should be validated on independent data")
        print("   ⚠ Recommend n>50 for robust inference")
    else:
        print("   ✓ ADEQUATE: Sufficient for preliminary conclusions")
        print("   ✓ Still recommend independent validation")
    
    print("\n3. PERFECT ACCURACY CONCERNS:")
    print("   - Perfect accuracy (1.0) with small n suggests:")
    print("     • Dataset may be too easy (strong class separation)")
    print("     • High risk of overfitting to training data")
    print("     • Limited generalizability to new samples")
    print("   - Even if p<0.05, treat perfect scores with caution")
    
    print("\n4. METHOD COMPARISON:")
    print("   - When all methods achieve same accuracy:")
    print("     • Statistical tests can't distinguish methods")
    print("     • Focus on biological interpretation (feature selection)")
    print("     • Consider method complexity vs. performance trade-off")
    
    print("\n5. RECOMMENDED ACTIONS:")
    print("   ✓ Acquire more samples when possible")
    print("   ✓ Validate on independent/external dataset")
    print("   ✓ Focus on reproducible biomarker selection")
    print("   ✓ Use results to guide experimental design")
    print("   ✓ Report limitations clearly in publications")
    
    print("\n6. REPORTING FOR POC/PILOT STUDIES:")
    print("   - Clearly state exploratory/hypothesis-generating nature")
    print("   - Present permutation p-values with sample size caveat")
    print("   - Emphasize framework validation over specific results")
    print("   - Highlight methodology readiness for scaled studies")
    
    print("\n" + "="*80)
    
    # Method-specific interpretation
    print("\nMETHOD-SPECIFIC RESULTS:")
    for _, row in perm_df.iterrows():
        print(f"\n{row['Method']}:")
        print(f"  Accuracy: {row['True_Accuracy']:.3f}")
        print(f"  P-value: {row['P_Value']:.4f}")
        
        if row['Significant']:
            print(f"  ✓ Significantly better than chance (p<0.05)")
        else:
            print(f"  ✗ Not significantly different from chance (p>=0.05)")
        
        if row['True_Accuracy'] == 1.0:
            print(f"  ⚠ Perfect accuracy - high overfitting risk with n={n_samples}")


if __name__ == "__main__":
    # Run analysis with permutation testing
    workflow, perm_results = run_multi_omics_with_permutation_tests()
    
    # Interpret results
    n_samples = len(workflow.integrator.common_samples)
    interpret_permutation_results(perm_results, n_samples)
    
    print("\n" + "="*80)
    print("Done! Check results/multi_omics/ for all outputs.")
    print("="*80)
