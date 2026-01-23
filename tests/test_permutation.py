"""
Quick test of permutation testing functionality.

Tests that the permutation testing framework works without
running the full expensive computation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_multiomics.workflows.multi_omics_workflow import MultiOmicsWorkflow


def test_permutation_framework():
    """
    Test that permutation testing can be called and returns expected format.
    """
    print("\n" + "="*80)
    print("TESTING PERMUTATION FRAMEWORK")
    print("="*80)
    
    try:
        # Load data
        print("\n1. Loading data...")
        data_dict = {
            'amino_acids': pd.read_csv("data/badata-amino-acids.csv"),
            'central_carbon': pd.read_csv("data/badata-metabolomics.csv"),
            'aromatics': pd.read_csv("data/badata-aromatics.csv"),
            'proteomics': pd.read_csv("data/badata-proteomics-imputed.csv")
        }
        print("   ✓ Data loaded")
        
        # Initialize and preprocess
        print("\n2. Preprocessing...")
        workflow = MultiOmicsWorkflow()
        
        omics_types = {
            'amino_acids': 'metabolomics',
            'central_carbon': 'metabolomics',
            'aromatics': 'volatiles',
            'proteomics': 'proteomics'
        }
        
        for name, df in data_dict.items():
            workflow.add_omics_layer(
                name=name,
                df=df,
                omics_type=omics_types[name],
                group_col='Groups'
            )
        print("   ✓ Preprocessing complete")
        
        # Prepare integration
        print("\n3. Preparing integration...")
        multi_block = workflow.prepare_integration()
        print(f"   ✓ Multi-block prepared (n={len(multi_block.y)} samples)")
        
        # Fit models quickly (no CV)
        print("\n4. Fitting models (quick mode)...")
        
        # Concatenation
        from ml_multiomics.methods.multi_omics import ConcatenationBaseline
        blocks = {name: multi_block.get_block(name) 
                 for name in multi_block.get_block_names()}
        X_concat = np.hstack(list(blocks.values()))
        baseline = ConcatenationBaseline(classifier='random_forest')
        baseline.fit(X_concat, multi_block.y)
        workflow.integration_methods['concatenation'] = baseline
        workflow.results['concatenation_cv'] = {'accuracy': 1.0, 'std': 0.0}
        
        # Ensemble
        from ml_multiomics.methods.multi_omics import BlockWiseEnsemble
        feature_names = {name: multi_block.blocks[name]['feature_names']
                        for name in multi_block.get_block_names()}
        ensemble = BlockWiseEnsemble(classifier='random_forest', voting='soft')
        ensemble.fit(blocks, multi_block.y, feature_names=feature_names)
        workflow.integration_methods['ensemble'] = ensemble
        workflow.results['ensemble_cv'] = {'accuracy': 1.0, 'std': 0.0}
        
        # DIABLO
        from ml_multiomics.methods.multi_omics import DIABLO
        diablo = DIABLO(n_components=2)
        diablo.fit(blocks, multi_block.y, 
                   feature_names=feature_names,
                   sample_ids=multi_block.sample_ids)
        workflow.integration_methods['diablo'] = diablo
        workflow.results['diablo_cv'] = {'accuracy': 1.0, 'std': 0.0}
        
        print("   ✓ All models fitted")
        
        # Run minimal permutation test (just 10 permutations for speed)
        print("\n5. Running permutation test (n=10, quick test)...")
        perm_results = workflow.run_permutation_tests(
            multi_block=multi_block,
            n_permutations=10,  # Minimal for testing
            random_state=42
        )
        print("   ✓ Permutation test completed")
        
        # Validate output format
        print("\n6. Validating output format...")
        required_cols = ['Method', 'True_Accuracy', 'Perm_Mean', 'Perm_Std', 'P_Value', 'Significant']
        assert all(col in perm_results.columns for col in required_cols), \
            f"Missing columns. Expected {required_cols}, got {perm_results.columns.tolist()}"
        
        assert len(perm_results) == 3, f"Expected 3 methods, got {len(perm_results)}"
        
        expected_methods = ['Concatenation', 'Block-wise Ensemble', 'DIABLO']
        assert set(perm_results['Method'].tolist()) == set(expected_methods), \
            f"Method names don't match. Expected {expected_methods}"
        
        print("   ✓ Output format correct")
        print("   ✓ All required columns present")
        print(f"   ✓ All 3 methods tested")
        
        # Display results
        print("\n7. Results preview:")
        print(perm_results.to_string(index=False))
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        print("\nPermutation testing framework is working correctly!")
        print("\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_permutation_framework()
    sys.exit(0 if success else 1)
