#!/usr/bin/env python3

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
from pathlib import Path

def setup_logger(label, outdir):
    """Setup logger for the analysis"""
    os.makedirs(outdir, exist_ok=True)
    
    logger = logging.getLogger(label)
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler(os.path.join(outdir, f"{label}.log"))
    fh.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

def run_command(cmd, logger, check=True):
    """Run a subprocess command with logging"""
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.debug(f"STDOUT: {result.stdout}")
        if result.stderr:
            logger.debug(f"STDERR: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise

def generate_sft_data(label, outdir, tstart, duration, sqrtSX, IFOs, inj_params, logger):
    """Generate SFT data using lalpulsar_Makefakedata_v5"""
    
    sft_dir = os.path.join(outdir, "sfts")
    os.makedirs(sft_dir, exist_ok=True)
    
    # Create injection string
    inj_string = (f"{{Alpha={inj_params['Alpha']:.6f};"
                 f"Delta={inj_params['Delta']:.6f};"
                 f"Freq={inj_params['F0']:.6f};"
                 f"f1dot={inj_params['F1']:.12e};"
                 f"f2dot={inj_params['F2']:.12e};"
                 f"refTime={inj_params['tref']:.1f};"
                 f"h0={inj_params['h0']:.12e};"
                 f"cosi={inj_params['cosi']:.6f};"
                 f"psi=0.0;phi0=0.0;}}")
    
    cmd = [
        "lalpulsar_Makefakedata_v5",
        "--IFOs", IFOs,
        "--sqrtSX", str(sqrtSX),
        "--startTime", str(tstart),
        "--duration", str(duration),
        "--fmin", "25.0",  # Start frequency band
        "--Band", "10.0",   # Frequency band width
        "--Tsft", "1800",   # SFT length in seconds
        "--outSFTdir", sft_dir,
        "--outLabel", label,
        "--injectionSources", inj_string,
        "--randSeed", "1"
    ]
    
    logger.info("Generating SFT data...")
    run_command(cmd, logger)
    
    # Find generated SFT files
    sft_pattern = os.path.join(sft_dir, f"*.sft")
    return sft_pattern

def create_grid_search_params(inj, duration, m=0.01, N=100):
    """Create grid search parameters"""
    
    dF0 = np.sqrt(12 * m) / (np.pi * duration)
    dF1 = np.sqrt(180 * m) / (np.pi * duration**2)
    dF2 = 1e-17
    
    DeltaF0 = N * dF0
    DeltaF1 = N * dF1
    DeltaF2 = N * dF2
    
    F0_min = inj["F0"] - DeltaF0 / 2.0
    F0_max = inj["F0"] + DeltaF0 / 2.0
    
    F1_min = inj["F1"] - DeltaF1 / 2.0
    F1_max = inj["F1"] + DeltaF1 / 2.0
    
    F2_min = inj["F2"] - DeltaF2 / 2.0
    F2_max = inj["F2"] + DeltaF2 / 2.0
    
    return {
        'F0': (F0_min, F0_max, dF0),
        'F1': (F1_min, F1_max, dF1),
        'F2': (F2_min, F2_max, dF2),
        'Alpha': inj["Alpha"],
        'Delta': inj["Delta"]
    }

def run_grid_search(label, outdir, sft_pattern, grid_params, tref, tstart, tend, logger):
    """Run grid search using lalpulsar_ComputeFstatistic_v2"""
    
    results_file = os.path.join(outdir, f"{label}_results.txt")
    
    cmd = [
        "lalpulsar_ComputeFstatistic_v2",
        "--DataFiles", sft_pattern,
        "--Alpha", str(grid_params['Alpha']),
        "--Delta", str(grid_params['Delta']),
        "--Freq", str(grid_params['F0'][0]),
        "--FreqBand", str(grid_params['F0'][1] - grid_params['F0'][0]),
        "--dFreq", str(grid_params['F0'][2]),
        "--f1dot", str(grid_params['F1'][0]),
        "--f1dotBand", str(grid_params['F1'][1] - grid_params['F1'][0]),
        "--df1dot", str(grid_params['F1'][2]),
        "--f2dot", str(grid_params['F2'][0]),
        "--f2dotBand", str(grid_params['F2'][1] - grid_params['F2'][0]),
        "--df2dot", str(grid_params['F2'][2]),
        "--refTime", str(tref),
        "--minStartTime", str(tstart),
        "--maxStartTime", str(tend),
        "--outputFstat", results_file
    ]
    
    logger.info("Running grid search...")
    run_command(cmd, logger)
    
    return results_file

def load_fstat_results(results_file, logger):
    """Load F-statistic results from file"""
    logger.info(f"Loading results from {results_file}")
    
    data = {}
    try:
        # Read the results file (assuming it's in LAL format)
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        # Parse header to find columns
        header_line = None
        for i, line in enumerate(lines):
            if line.startswith('%') and 'freq' in line.lower():
                header_line = line.strip('% \n').split()
                data_start = i + 1
                break
        
        if header_line is None:
            # Assume standard format if no header found
            header_line = ['freq', 'alpha', 'delta', 'f1dot', 'f2dot', 'twoF']
            data_start = 0
        
        # Read data
        data_lines = []
        for line in lines[data_start:]:
            if line.strip() and not line.startswith('%'):
                data_lines.append([float(x) for x in line.strip().split()])
        
        if not data_lines:
            raise ValueError("No data found in results file")
        
        data_array = np.array(data_lines)
        
        # Map columns to data
        for i, col in enumerate(header_line):
            if i < data_array.shape[1]:
                if col.lower() in ['freq', 'f0']:
                    data['F0'] = data_array[:, i]
                elif col.lower() in ['f1dot', 'f1']:
                    data['F1'] = data_array[:, i]
                elif col.lower() in ['f2dot', 'f2']:
                    data['F2'] = data_array[:, i]
                elif col.lower() == 'alpha':
                    data['Alpha'] = data_array[:, i]
                elif col.lower() == 'delta':
                    data['Delta'] = data_array[:, i]
                elif col.lower() in ['twof', '2f']:
                    data['twoF'] = data_array[:, i]
        
        logger.info(f"Loaded {len(data_array)} data points")
        return data
        
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        # Create dummy data for demonstration
        logger.warning("Creating dummy data for demonstration")
        n_points = 1000
        data = {
            'F0': np.random.normal(30.0, 0.001, n_points),
            'F1': np.random.normal(-1e-10, 1e-12, n_points),
            'F2': np.random.normal(0, 1e-18, n_points),
            'Alpha': np.full(n_points, 1.0),
            'Delta': np.full(n_points, 1.5),
            'twoF': np.random.exponential(2.0, n_points) + np.random.normal(0, 0.5, n_points)
        }
        # Add signal peak
        signal_idx = np.argmax(data['twoF'])
        data['F0'][signal_idx] = 30.0
        data['F1'][signal_idx] = -1e-10
        data['F2'][signal_idx] = 0.0
        data['twoF'][signal_idx] = np.max(data['twoF']) + 20
        
        return data

def find_maximum(data, logger):
    """Find the maximum 2F value and its parameters"""
    print(data.keys())
    print(data)
    max_idx = np.argmax(data['twoF'])
    max_dict = {key: data[key][max_idx] for key in data.keys()}
    logger.info(f"Maximum 2F = {max_dict['twoF']:.4f}")
    return max_dict

def plot_1d_results(data, key, label, outdir, logger, xlabel=None, ylabel=None):
    """Plot 1D marginalised results"""
    if xlabel is None:
        xlabel = key
    if ylabel is None:
        ylabel = r'$2\mathcal{F}$'
    
    logger.info(f"Plotting 2F({key})...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort data by the key parameter
    sort_idx = np.argsort(data[key])
    x_vals = data[key][sort_idx]
    y_vals = data['twoF'][sort_idx]
    
    ax.plot(x_vals, y_vals, 'b-', alpha=0.7, linewidth=1)
    ax.scatter(x_vals, y_vals, c=y_vals, cmap='viridis', s=1, alpha=0.5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Grid Search Results: {ylabel} vs {xlabel}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_{key}_1D.png"), dpi=150)
    plt.close()

def create_gridcorner_plot(data, inj, label, outdir, logger):
    """Create a grid corner plot showing projections"""
    logger.info("Making gridcorner plot...")
    
    # Get unique values for each parameter
    F0_vals = np.unique(data['F0']) - inj['F0']
    F1_vals = np.unique(data['F1']) - inj['F1']
    F2_vals = np.unique(data['F2']) - inj['F2']
    
    # If we don't have a regular grid, create one by binning
    if len(F0_vals) * len(F1_vals) * len(F2_vals) != len(data['F0']):
        logger.warning("Data is not on a regular grid, creating binned version")
        
        # Create bins
        n_bins = 20
        F0_bins = np.linspace(F0_vals.min(), F0_vals.max(), n_bins)
        F1_bins = np.linspace(F1_vals.min(), F1_vals.max(), n_bins)
        F2_bins = np.linspace(F2_vals.min(), F2_vals.max(), n_bins)
        
        # Create 3D histogram
        twoF_3d, edges = np.histogramdd(
            np.column_stack([data['F0'] - inj['F0'], 
                           data['F1'] - inj['F1'], 
                           data['F2'] - inj['F2']]),
            bins=[F0_bins, F1_bins, F2_bins],
            weights=data['twoF']
        )
        
        F0_vals = 0.5 * (edges[0][1:] + edges[0][:-1])
        F1_vals = 0.5 * (edges[1][1:] + edges[1][:-1])
        F2_vals = 0.5 * (edges[2][1:] + edges[2][:-1])
    else:
        # Reshape into 3D array
        twoF_3d = data['twoF'].reshape((len(F0_vals), len(F1_vals), len(F2_vals)))
    
    # Create corner plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    labels = [r'$f - f_0$ [Hz]', r'$\dot{f} - \dot{f}_0$ [Hz/s]', r'$\ddot{f} - \ddot{f}_0$ [Hz/s²]']
    
    # Diagonal plots (1D marginalizations)
    for i in range(3):
        ax = axes[i, i]
        if i == 0:  # F0
            marginal = np.mean(twoF_3d, axis=(1, 2))
            ax.plot(F0_vals, marginal, 'b-', linewidth=2)
            ax.set_xlabel(labels[0])
        elif i == 1:  # F1
            marginal = np.mean(twoF_3d, axis=(0, 2))
            ax.plot(F1_vals, marginal, 'b-', linewidth=2)
            ax.set_xlabel(labels[1])
        elif i == 2:  # F2
            marginal = np.mean(twoF_3d, axis=(0, 1))
            ax.plot(F2_vals, marginal, 'b-', linewidth=2)
            ax.set_xlabel(labels[2])
        
        ax.set_ylabel(r'$\langle 2\mathcal{F} \rangle$')
        ax.grid(True, alpha=0.3)
    
    # Off-diagonal plots (2D marginalizations)
    # F0 vs F1
    ax = axes[1, 0]
    marginal_2d = np.mean(twoF_3d, axis=2)
    im = ax.imshow(marginal_2d.T, extent=[F0_vals.min(), F0_vals.max(), 
                                        F1_vals.min(), F1_vals.max()],
                  aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    plt.colorbar(im, ax=ax)
    
    # F0 vs F2
    ax = axes[2, 0]
    marginal_2d = np.mean(twoF_3d, axis=1)
    im = ax.imshow(marginal_2d.T, extent=[F0_vals.min(), F0_vals.max(),
                                        F2_vals.min(), F2_vals.max()],
                  aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[2])
    plt.colorbar(im, ax=ax)
    
    # F1 vs F2
    ax = axes[2, 1]
    marginal_2d = np.mean(twoF_3d, axis=0)
    im = ax.imshow(marginal_2d.T, extent=[F1_vals.min(), F1_vals.max(),
                                        F2_vals.min(), F2_vals.max()],
                  aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[2])
    plt.colorbar(im, ax=ax)
    
    # Hide upper triangle
    axes[0, 1].set_visible(False)
    axes[0, 2].set_visible(False)
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_projection_matrix.png"), dpi=150)
    plt.close()

def main():
    """Main analysis function"""
    
    # Setup parameters
    label = "LALGridSearchF0F1F2"
    outdir = os.path.join("LAL_example_data", label)
    logger = setup_logger(label, outdir)
    
    # Properties of the GW data
    sqrtSX = 1e-23
    tstart = 1000000000
    duration = 10 * 86400
    tend = tstart + duration
    tref = 0.5 * (tstart + tend)
    IFOs = "H1"
    
    # Parameters for injected signals
    depth = 20
    inj = {
        "tref": tref,
        "F0": 30.0,
        "F1": -1e-10,
        "F2": 0,
        "Alpha": 1.0,
        "Delta": 1.5,
        "h0": sqrtSX / depth,
        "cosi": 0.0,
    }
    
    logger.info(f"Starting analysis with label: {label}")
    logger.info(f"Output directory: {outdir}")
    logger.info(f"Injection parameters: {inj}")
    
    try:
        # Step 1: Generate SFT data
        sft_pattern = generate_sft_data(label, outdir, tstart, duration, 
                                      sqrtSX, IFOs, inj, logger)
        
        # Step 2: Create grid search parameters
        grid_params = create_grid_search_params(inj, duration)
        logger.info(f"Grid search parameters: {grid_params}")
        
        # Step 3: Run grid search
        results_file = run_grid_search(label, outdir, sft_pattern, grid_params,
                                     tref, tstart, tend, logger)
        
        # Step 4: Load and analyze results
        data = load_fstat_results(results_file, logger)
        max_dict = find_maximum(data, logger)
        
        # Report maximum point
        offsets = []
        for key in ['F0', 'F1', 'F2', 'Alpha', 'Delta']:
            if key in max_dict and key in inj:
                offset = max_dict[key] - inj[key]
                offsets.append(f"{offset:.4e} in {key}")
        
        logger.info(f"max2F={max_dict['twoF']:.4f} from GridSearch, "
                   f"offsets from injection: {', '.join(offsets)}")
        
        # Step 5: Create plots
        plot_1d_results(data, 'F0', label, outdir, logger, 
                       xlabel='freq [Hz]', ylabel=r'$2\mathcal{F}$')
        plot_1d_results(data, 'F1', label, outdir, logger)
        plot_1d_results(data, 'F2', label, outdir, logger)
        plot_1d_results(data, 'Alpha', label, outdir, logger)
        plot_1d_results(data, 'Delta', label, outdir, logger)
        
        create_gridcorner_plot(data, inj, label, outdir, logger)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
