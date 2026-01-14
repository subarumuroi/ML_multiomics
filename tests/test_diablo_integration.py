"""Test DIABLO with real multi-omics data."""

import numpy as np
import pandas as pd
from ml_multiomics.preprocessing import (
    MetabolomicsPreprocessor, 
    ProteomicsPreprocessor
)
from ml_multiomics.methods.multi_omics import DIABLO

# Load different omics
df_amino = pd.read_csv('data/badata-amino-acids.csv')
df_protein = pd.read_csv('data/badata-proteomics-imputed.csv')

# Preprocess each
print("Preprocessing amino acids...")
prep_amino = MetabolomicsPreprocessor()
X_amino, y_amino, feat_amino = prep_amino.preprocess(df_amino, 'Groups')

print("Preprocessing proteomics...")
prep_protein = ProteomicsPreprocessor()
X_protein, y_protein, feat_protein = prep_protein.preprocess(df_protein, 'Groups')

print(f"\nAmino acids: {X_amino.shape}")
print(f"Proteomics: {X_protein.shape}")

# Create multi-block
blocks = {
    'amino_acids': X_amino,
    'proteomics': X_protein[:, :50]  # Just first 50 proteins for speed
}

feature_names = {
    'amino_acids': feat_amino,
    'proteomics': feat_protein[:50]
}

print(f"\nRunning DIABLO with real multi-omics...")

# Fit DIABLO
diablo = DIABLO(n_components=2)
diablo.fit(blocks, y_amino, feature_names)

print("\n✅ DIABLO fit successful!")

# Check results
print(f"\nBlock correlations:\n{diablo.calculate_block_correlations()}")

# Get VIPs
vips = diablo.get_all_vips(top_n=5)
print(f"\nTop features per block:\n{vips}")

print("\n✅ Real multi-omics test passed!")