import os
import numpy as np
import subprocess
import shutil
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from typing import Dict, Any
import matplotlib as mpl

# --- one-off style tweaks ---------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": "serif",  # Times/Computer Modern-style text
        "mathtext.fontset": "cm",
        "axes.spines.top": False,  # hide unnecessary spines
        "axes.spines.right": False,
        "axes.linewidth": 1.2,  # make the remaining spines a little bolder
    }
)

# Convenience: absolute paths for the lalapps binaries
MFD_BIN = shutil.which("lalapps_Makefakedata_v5")
FS_BIN = shutil.which("lalapps_ComputeFstatistic_v2")
if MFD_BIN is None or FS_BIN is None:
    raise RuntimeError("Could not find lalapps binaries in $PATH$. Make sure "
                      "LALSuite is installed and its 'bin' directory is on "
                      "your shell PATH before running this script.")

# Helper regex to capture the "2F" result line of lalapps_ComputeFstatistic_v2
re_twoF = re.compile(r"^twoF\s*=\s*([0-9eE+\-.]+)")

class CalculationParams:
    def __init__(self, inj_params: Dict[str, float], DeltaF0: float, DeltaF1: float, 
                 DeltaF2: float, dF0: float, dF1_refined: float, dF2_refined: float,
                 sky: bool, outdir: str, sftfilepath: str, tref: int, nsegs: int,
                 plot: bool, labels: Dict[str, str], tstart: int, duration: int):
        self.inj_params = inj_params
        self.DeltaF0 = DeltaF0
        self.DeltaF1 = DeltaF1
        self.DeltaF2 = DeltaF2
        self.dF0 = dF0
        self.dF1_refined = dF1_refined
        self.dF2_refined = dF2_refined
        self.sky = sky
        self.outdir = outdir
        self.sftfilepath = sftfilepath
        self.tref = tref
        self.nsegs = nsegs
        self.plot = plot
        self.labels = labels
        self.tstart = tstart
        self.duration = duration

def run_makefakedata(sftdir: str, params: Dict[str, Any]) -> str:
    """Generate SFTs containing the injected signal and return a glob pattern."""
    os.makedirs(sftdir, exist_ok=True)

    cmd = [
        MFD_BIN,
        "--outSingleSFT=1",
        f"--outputSFTbase={sftdir}/SFT",
        f"--fmin={params['F0'] - 0.5}",      # narrow band suffices
        "--Band=1",
        f"--Tsft={params['Tsft']}",
        f"--IFOs={params['detectors']}",
        f"--sqrtSX={params['sqrtSX']}",
        f"--tstart={params['tstart']}",
        f"--duration={params['duration']}",
        # injection parameters
        f"--F0={params['F0']}",
        f"--F1={params['F1']}",
        f"--F2={params['F2']}",
        f"--Alpha={params['Alpha']}",
        f"--Delta={params['Delta']}",
        f"--h0={params['h0']}",
        f"--cosi={params['cosi']}",
        # misc
        "--EphemEarth=DE421",
        "--EphemSun=DE421",
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return os.path.join(sftdir, "*.sft")

def compute_twoF(sft_glob: str, f0: float, f1: float, f2: float,
                alpha: float, delta: float, tref: int) -> float:
    """Run lalapps_ComputeFstatistic_v2 for a single template and return 2F."""
    cmd = [
        FS_BIN,
        f"--Alpha={alpha}",
        f"--Delta={delta}",
        f"--F0={f0}",
        f"--f1dot={f1}",
        f"--f2dot={f2}",
        f"--refTime={tref}",
        "--singleFstat=1",          # ask for one template only
        f"--SFTFiles={sft_glob}",
    ]

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        m = re_twoF.match(line.strip())
        if m:
            return float(m.group(1))
    raise RuntimeError("Could not parse twoF from lalapps_ComputeFstatistic_v2 output.")

def calculate_mismatch(i: int, params: CalculationParams, random_offsets: Dict[str, float]) -> float:
    """Calculate mismatch for a single grid search with random offsets."""
    # Create temporary directory for this iteration's SFTs
    sftdir = os.path.join(params.outdir, f"SFTs_{i:03d}")
    sft_glob = run_makefakedata(sftdir, params.inj_params)

    # Set up grid parameters with random offsets
    F0s = [
        params.inj_params["F0"] - params.DeltaF0 / 2.0 + random_offsets["F0"],
        params.inj_params["F0"] + params.DeltaF0 / 2.0 + random_offsets["F0"],
        params.dF0
    ]
    F1s = [
        params.inj_params["F1"] - params.DeltaF1 / 2.0 + random_offsets["F1"],
        params.inj_params["F1"] + params.DeltaF1 / 2.0 + random_offsets["F1"],
        params.dF1_refined
    ]
    F2s = [
        params.inj_params["F2"] - params.DeltaF2 / 2.0 + random_offsets["F2"],
        params.inj_params["F2"] + params.DeltaF2 / 2.0 + random_offsets["F2"],
        params.dF2_refined
    ]

    # Generate grid points
    f0_vals = np.arange(F0s[0], F0s[1] + F0s[2]/2, F0s[2])
    f1_vals = np.arange(F1s[0], F1s[1] + F1s[2]/2, F1s[2])
    f2_vals = np.arange(F2s[0], F2s[1] + F2s[2]/2, F2s[2])

    # Initialize variables for tracking loudest template
    loudest_twoF = -np.inf
    loudest_params = None

    # Scan the grid
    for f0 in f0_vals:
        for f1 in f1_vals:
            for f2 in f2_vals:
                twoF = compute_twoF(
                    sft_glob, f0, f1, f2,
                    params.inj_params["Alpha"],
                    params.inj_params["Delta"],
                    params.tref
                )
                if twoF > loudest_twoF:
                    loudest_twoF = twoF
                    loudest_params = (f0, f1, f2)

    # Compute perfect-match 2F
    twoF_inj = compute_twoF(
        sft_glob,
        params.inj_params["F0"],
        params.inj_params["F1"],
        params.inj_params["F2"],
        params.inj_params["Alpha"],
        params.inj_params["Delta"],
        params.tref
    )

    # Calculate mismatch
    rho2_no = twoF_inj - 4.0
    rho2_mis = loudest_twoF - 4.0
    mu_empirical = (rho2_no - rho2_mis) / rho2_no

    # Clean up
    shutil.rmtree(sftdir)
    return mu_empirical

def single_grid(pool, config):
    """Run a single grid search with multiple random offsets."""
    all_random_offsets = []
    for i in range(config.numbers):
        random_offsets = {
            "F0": np.random.uniform(-config.dF0, config.dF0),
            "F1": np.random.uniform(-config.dF1_refined, config.dF1_refined),
            "F2": np.random.uniform(-config.dF2_refined, config.dF2_refined)
        }
        all_random_offsets.append(random_offsets)

    params = CalculationParams(
        inj_params=config.inj,
        DeltaF0=config.DeltaF0,
        DeltaF1=config.DeltaF1,
        DeltaF2=config.DeltaF2,
        dF0=config.dF0,
        dF1_refined=config.dF1_refined,
        dF2_refined=config.dF2_refined,
        sky=config.sky,
        outdir=config.outdir,
        sftfilepath=config.sftfilepath,
        tref=config.inj["tref"],
        nsegs=config.nsegs,
        plot=config.plot,
        labels=config.labels,
        tstart=config.tstart,
        duration=config.duration
    )

    futures = [
        pool.submit(calculate_mismatch, i, params, all_random_offsets[i])
        for i in range(config.numbers)
    ]
    mismatches = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Create DataFrame with results
    num_entries = len(mismatches)
    data = {
        'mismatch': mismatches,
        'mf': [getattr(config, 'mf', None)] * num_entries,
        'mf1': [getattr(config, 'mf1', None)] * num_entries,
        'mf2': [getattr(config, 'mf2', None)] * num_entries,
        'df': [config.dF0] * num_entries,
        'df1': [config.dF1_refined] * num_entries,
        'df2': [config.dF2_refined] * num_entries,
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Create parameter grids
    mf_values = np.linspace(0.1, 0.5, 10)  # 10 points from 0.1 to 0.5
    mf1_values = np.linspace(0.1, 0.5, 10)  # 10 points from 0.1 to 0.5  
    mf2_values = np.linspace(0.001, 0.3, 5)  # 5 points from 0.001 to 0.3

    # Create tmp directory for saving individual dataframes
    tmp_dir = os.path.join("LAL_example_data", "tmp_dataframes")
    os.makedirs(tmp_dir, exist_ok=True)

    # Store original values to restore later
    original_outdir = "LAL_example_data"

    print(f"Starting parameter sweep:")
    print(f"mf: {len(mf_values)} values from {mf_values[0]:.3f} to {mf_values[-1]:.3f}")
    print(f"mf1: {len(mf1_values)} values from {mf1_values[0]:.3f} to {mf1_values[-1]:.3f}")
    print(f"mf2: {len(mf2_values)} values from {mf2_values[0]:.6f} to {mf2_values[-1]:.6f}")
    print(f"Total combinations: {len(mf_values) * len(mf1_values) * len(mf2_values)}")

    all_dataframes = []
    run_counter = 0
    total_runs = len(mf_values) * len(mf1_values) * len(mf2_values)

    with concurrent.futures.ProcessPoolExecutor() as pool:
        # Loop through all parameter combinations
        for i, mf_val in enumerate(mf_values):
            for j, mf1_val in enumerate(mf1_values):
                for k, mf2_val in enumerate(mf2_values):
                    run_counter += 1
                    tmp_filename = f"run_{run_counter:03d}_df.csv"
                    if os.path.exists(os.path.join(tmp_dir, tmp_filename)):
                        print(f"Skipping run {run_counter} as {tmp_filename} already exists.")
                        continue

                    print(f"\n{'='*60}")
                    print(f"Run {run_counter}/{total_runs}")
                    print(f"mf={mf_val:.3f}, mf1={mf1_val:.3f}, mf2={mf2_val:.6f}")
                    print(f"{'='*60}")

                    # Create config object for this run
                    class Config:
                        def __init__(self):
                            self.numbers = 100  # Number of random offsets
                            self.sky = False
                            self.plot = False
                            self.nsegs = 1
                            self.tstart = 1000000000
                            self.duration = 30 * 86400
                            self.Tsft = 1800
                            self.detectors = "H1,L1"
                            self.sqrtSX = 1e-22
                            self.inj = {
                                "tref": self.tstart,
                                "F0": 30.0,
                                "F1": -1e-10,
                                "F2": 0,
                                "Alpha": 0.5,
                                "Delta": 1,
                                "h0": 0.5 * self.sqrtSX,
                                "cosi": 1.0,
                            }
                            self.labels = {
                                "F0": "$f$ [Hz]",
                                "F1": "$\\dot{f}$ [Hz/s]",
                                "F2": "$\\ddot{f}$ [Hz/s^2]",
                                "2F": "$2\\mathcal{F}$",
                                "Alpha": "$\\alpha$",
                                "Delta": "$\\delta$",
                            }
                            self.mf = mf_val
                            self.mf1 = mf1_val
                            self.mf2 = mf2_val
                            self.dF0 = np.sqrt(12 * self.mf) / (np.pi * self.duration)
                            self.dF1 = np.sqrt(180 * self.mf1) / (np.pi * self.duration**2)
                            self.dF2 = np.sqrt(25200 * self.mf2) / (np.pi * self.duration**3)
                            self.dF1_refined = self.dF1 / 2  # gamma1 = 2
                            self.dF2_refined = self.dF2 / 2  # gamma2 = 2
                            self.DeltaF0 = 8 * self.dF0
                            self.DeltaF1 = 8 * self.dF1_refined
                            self.DeltaF2 = 8 * self.dF2_refined
                            self.outdir = os.path.join(original_outdir, f"run_{run_counter:03d}_mf{mf_val:.3f}_mf1{mf1_val:.3f}_mf2{mf2_val:.6f}")
                            os.makedirs(self.outdir, exist_ok=True)
                            self.sftfilepath = os.path.join(self.outdir, "*.sft")

                    config = Config()

                    try:
                        # Run the single grid calculation
                        df_result = single_grid(pool, config)

                        # Add parameter information to the dataframe
                        df_result['run_id'] = run_counter
                        df_result['mf_input'] = mf_val
                        df_result['mf1_input'] = mf1_val
                        df_result['mf2_input'] = mf2_val
                        df_result['dF0_calculated'] = config.dF0
                        df_result['dF1_refined_calculated'] = config.dF1_refined
                        df_result['dF2_refined_calculated'] = config.dF2_refined

                        # Save individual dataframe to tmp folder
                        tmp_filepath = os.path.join(tmp_dir, tmp_filename)
                        df_result.to_csv(tmp_filepath, index=False)

                        # Store in memory for final merge
                        all_dataframes.append(df_result)

                        print(f"✓ Run {run_counter} completed successfully")
                        print(f"  Mean mismatch: {df_result['mismatch'].mean():.6f}")
                        print(f"  Std mismatch: {df_result['mismatch'].std():.6f}")
                        print(f"  Saved to: {tmp_filepath}")

                    except Exception as e:
                        print(f"✗ Run {run_counter} failed with error: {e}")
                        continue

    print(f"\n{'='*60}")
    print("MERGING ALL RESULTS")
    print(f"{'='*60}")

    # Merge all dataframes
    if all_dataframes:
        print(f"Merging {len(all_dataframes)} successful runs...")

        # Concatenate all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Save the combined dataframe
        combined_filepath = os.path.join(original_outdir, "combined_mismatch_results.csv")
        combined_df.to_csv(combined_filepath, index=False)

        # Create summary statistics by parameter combination
        summary_stats = combined_df.groupby(['mf_input', 'mf1_input', 'mf2_input']).agg({
            'mismatch': ['count', 'mean', 'std', 'min', 'max', 'median'],
            'run_id': 'first'
        }).round(6)

        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        summary_stats = summary_stats.reset_index()

        # Save summary statistics
        summary_filepath = os.path.join(original_outdir, "parameter_sweep_summary.csv")
        summary_stats.to_csv(summary_filepath, index=False)

        print(f"✓ Combined dataframe saved to: {combined_filepath}")
        print(f"✓ Summary statistics saved to: {summary_filepath}")
        print(f"✓ Individual run dataframes saved in: {tmp_dir}")

        print(f"\nFinal Dataset Info:")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Total parameter combinations: {len(summary_stats)}")
        print(f"  Columns: {list(combined_df.columns)}")

        print(f"\nOverall Statistics:")
        print(f"  Mean mismatch across all runs: {combined_df['mismatch'].mean():.6f}")
        print(f"  Std mismatch across all runs: {combined_df['mismatch'].std():.6f}")
        print(f"  Min mismatch: {combined_df['mismatch'].min():.6f}")
        print(f"  Max mismatch: {combined_df['mismatch'].max():.6f}")

    else:
        print("✗ No successful runs to merge!")

    print(f"\n{'='*60}")
    print("PARAMETER SWEEP COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved in: {original_outdir}") 