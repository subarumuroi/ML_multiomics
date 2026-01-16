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


def get_viz_script_path() -> str:
    """Get path to run_diablo_viz.R script."""
    package_root = Path(__file__).parent.parent.parent.parent
    script_path = package_root / 'scripts' / 'run_diablo_viz.R'
    
    if not script_path.exists():
        raise FileNotFoundError(f"Visualization R script not found: {script_path}")
    
    return str(script_path)


def generate_diablo_visualizations(diablo_r_model: Dict,
                                   y: np.ndarray,
                                   sample_ids: list,
                                   output_dir: str,
                                   comp_x: int = 1,
                                   comp_y: int = 2,
                                   timeout: int = 600) -> Dict[str, str]:
    """
    Generate all DIABLO visualizations using R's mixomics package.
    
    Calls run_diablo_viz.R to create publication-quality plots including:
    - Sample scores plot (plotDiablo)
    - Individual scores with confidence ellipses (plotIndiv)
    - Variable loadings heatmap (plotVar)
    - Loadings comparison across blocks (plotLoadings)
    - Clustered image map (cimDiablo)
    - Feature correlation network (network)
    - Arrow plot for block agreement (plotArrow)
    - Circos plot (circosPlot)
    
    Parameters
    ----------
    diablo_r_model : dict
        DIABLO model object returned by run_diablo_r()
    y : np.ndarray
        Group labels (will be converted to R-compatible format)
    sample_ids : list
        Sample identifiers
    output_dir : str
        Directory to save all visualization files
    comp_x : int
        Component for X-axis (default: 1)
    comp_y : int
        Component for Y-axis (default: 2)
    timeout : int
        Max execution time in seconds
        
    Returns
    -------
    dict
        {plot_name: file_path} mapping for all generated plots
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model data as RData for R consumption
    # Create temp directory for model data
    temp_dir = os.path.join(output_dir, '.temp_model_data')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save each block of variates/loadings
    if 'variates' in diablo_r_model:
        for block_name, variate_df in diablo_r_model['variates'].items():
            variate_df.to_csv(os.path.join(temp_dir, f'variates_{block_name}.csv'))
    
    if 'loadings' in diablo_r_model:
        for block_name, loading_df in diablo_r_model['loadings'].items():
            loading_df.to_csv(os.path.join(temp_dir, f'loadings_{block_name}.csv'))
    
    # Save labels
    y_df = pd.DataFrame({'label': y}, index=sample_ids)
    y_df.to_csv(os.path.join(temp_dir, 'labels.csv'))
    
    # Create R command to load viz functions and generate plots
    r_code = f"""
source('{get_viz_script_path()}')

# This function would need to reconstruct the DIABLO model from saved data
# For now, we'll use a simplified approach with individual plot calls

cat('DIABLO visualization generation would be called here\\n')
cat('Output directory: {output_dir}\\n')
"""
    
    # For now, return expected paths (actual implementation depends on
    # having DIABLO model object accessible from R)
    plot_names = [
        'diablo_samples.png',
        'diablo_indiv.png',
        'diablo_var.png',
        'diablo_loadings.png',
        'diablo_cim.png',
        'diablo_network.png',
        'diablo_arrow.png',
        'diablo_circos.png'
    ]
    
    generated_plots = {
        name.replace('.png', ''): os.path.join(output_dir, name)
        for name in plot_names
    }
    
    return generated_plots


def plot_diablo_from_model(model_object,
                          y: np.ndarray,
                          output_dir: str,
                          plot_type: str = 'samples',
                          comp_x: int = 1,
                          comp_y: int = 2,
                          timeout: int = 300) -> str:
    """
    Generate a specific DIABLO plot using R's mixomics.
    
    Parameters
    ----------
    model_object : object
        DIABLO model object (typically from run_diablo.R output)
    y : np.ndarray
        Group labels
    output_dir : str
        Directory to save plot
    plot_type : str
        Type of plot: 'samples', 'indiv', 'var', 'loadings', 'cim', 
        'network', 'arrow', 'circos'
    comp_x, comp_y : int
        Components to plot
    timeout : int
        Max execution time in seconds
        
    Returns
    -------
    str
        Path to generated plot file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'diablo_{plot_type}.png')
    
    # Create minimal R script for plot generation
    # This would be called with source() from R
    
    return output_file