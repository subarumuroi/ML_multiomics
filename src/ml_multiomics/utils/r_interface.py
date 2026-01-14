"""
Simple interface for calling R's DIABLO implementation.
"""

import os
import subprocess
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional


def get_r_script_path() -> str:
    """Get path to run_diablo.R script."""
    package_root = Path(__file__).parent.parent.parent.parent
    script_path = package_root / 'scripts' / 'run_diablo.R'
    
    if not script_path.exists():
        raise FileNotFoundError(f"R script not found: {script_path}")
    
    return str(script_path)


def run_diablo_r(data_blocks: Dict[str, pd.DataFrame],
                 y: np.ndarray,
                 sample_ids: list,
                 output_dir: str,
                 timeout: int = 300) -> Dict:
    """
    Run DIABLO using R's mixOmics package.
    
    Parameters
    ----------
    data_blocks : dict
        {block_name: DataFrame} - preprocessed data blocks
    y : np.ndarray
        Group labels
    sample_ids : list
        Sample identifiers
    output_dir : str
        Where to save results
    timeout : int
        Max execution time in seconds
        
    Returns
    -------
    dict
        DIABLO results (variates, loadings, performance, etc.)
    """
    # Create temp directory for input data
    input_dir = os.path.join(output_dir, 'diablo_input')
    os.makedirs(input_dir, exist_ok=True)
    
    # Save blocks as CSV
    for block_name, df in data_blocks.items():
        df_copy = df.copy()
        df_copy.index = sample_ids
        df_copy.to_csv(os.path.join(input_dir, f"block_{block_name}.csv"))
    
    # Save labels
    y_df = pd.DataFrame({'label': y}, index=sample_ids)
    y_df.to_csv(os.path.join(input_dir, 'labels.csv'))
    
    # Run R script
    r_output_dir = os.path.join(output_dir, 'diablo_output')
    os.makedirs(r_output_dir, exist_ok=True)
    
    cmd = ['Rscript', get_r_script_path(), input_dir, r_output_dir]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    
    if result.returncode != 0:
        raise RuntimeError(
            f"DIABLO R script failed:\n{result.stderr}"
        )
    
    # Load results
    return _load_results(r_output_dir)


def _load_results(results_dir: str) -> Dict:
    """Load DIABLO results from R output directory."""
    results = {}
    
    # Load summary
    summary_file = os.path.join(results_dir, 'model_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            results['summary'] = json.load(f)
    
    # Load variates, loadings, selected features
    for prefix in ['variates', 'loadings', 'selected_features']:
        results[prefix] = {}
        for f in os.listdir(results_dir):
            if f.startswith(f'{prefix}_') and f.endswith('.csv'):
                block_name = f.replace(f'{prefix}_', '').replace('.csv', '')
                results[prefix][block_name] = pd.read_csv(
                    os.path.join(results_dir, f), index_col=0
                )
    
    # Load performance
    perf_file = os.path.join(results_dir, 'performance_metrics.csv')
    if os.path.exists(perf_file):
        results['performance'] = pd.read_csv(perf_file)
    
    return results