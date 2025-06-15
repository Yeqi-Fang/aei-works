import os
import numpy as np
import pyfstat
import config
from typing import Dict, Any, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import concurrent.futures
import gc
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import glob

from utils import plot_grid_vs_samples, plot_2F_scatter, CalculationParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ParameterGrid:
    mf: float
    mf1: float
    mf2: float
    dF0: float
    dF1_refined: float
    dF2_refined: float
    DeltaF0: float
    DeltaF1: float
    DeltaF2: float

class ConfigContext:
    """Context manager for safely modifying config values"""
    def __init__(self, config, **kwargs):
        self.config = config
        self.original_values = {}
        self.new_values = kwargs

    def __enter__(self):
        for key, value in self.new_values.items():
            self.original_values[key] = getattr(self.config, key)
            setattr(self.config, key, value)
        return self.config

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.original_values.items():
            setattr(self.config, key, value)

def check_disk_space(path: str, required_space: int) -> bool:
    """Check if there's enough disk space available"""
    total, used, free = shutil.disk_usage(path)
    return free > required_space

def check_sft_files(sftfilepath: str) -> bool:
    """Check if SFT files exist and are accessible"""
    # Expand the pattern to find matching files
    sft_files = glob.glob(sftfilepath)
    
    if not sft_files:
        logger.error(f"No SFT files found matching pattern: {sftfilepath}")
        return False
    
    logger.info(f"Found {len(sft_files)} SFT files matching pattern: {sftfilepath}")
    return True

def calculate_parameters(mf: float, mf1: float, mf2: float, T_coh: float, gamma1: float, gamma2: float) -> ParameterGrid:
    """Calculate all derived parameters for a given set of mismatch values"""
    dF0 = np.sqrt(12 * mf) / (np.pi * T_coh)
    dF1 = np.sqrt(180 * mf1) / (np.pi * T_coh**2)
    dF2 = np.sqrt(25200 * mf2) / (np.pi * T_coh**3)
    
    dF1_refined = dF1 / gamma1
    dF2_refined = dF2 / gamma2
    
    DeltaF0 = 8 * dF0
    DeltaF1 = 8 * dF1_refined
    DeltaF2 = 8 * dF2_refined
    
    return ParameterGrid(
        mf=mf, mf1=mf1, mf2=mf2,
        dF0=dF0, dF1_refined=dF1_refined, dF2_refined=dF2_refined,
        DeltaF0=DeltaF0, DeltaF1=DeltaF1, DeltaF2=DeltaF2
    )

def precalculate_parameter_grids(mf_values: np.ndarray, mf1_values: np.ndarray, mf2_values: np.ndarray,
                               T_coh: float, gamma1: float, gamma2: float) -> Dict[Tuple[float, float, float], ParameterGrid]:
    """Pre-calculate all parameter combinations"""
    grids = {}
    for mf in mf_values:
        for mf1 in mf1_values:
            for mf2 in mf2_values:
                key = (mf, mf1, mf2)
                grids[key] = calculate_parameters(mf, mf1, mf2, T_coh, gamma1, gamma2)
    return grids

def cleanup_temp_files(tmp_dir: str, run_id: int):
    """Clean up temporary files for a failed run"""
    tmp_file = os.path.join(tmp_dir, f"run_{run_id:03d}_df.csv")
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

def calculate_mismatch(i: int, params_dict: dict, random_offsets: Dict[str, float]) -> float:
    """Calculate mismatch for a single point"""
    try:
        # Import modules inside the function to avoid pickling issues
        import pyfstat
        import numpy as np
        import os
        import gc

        # Convert params_dict back to CalculationParams
        from dataclasses import dataclass
        from typing import Dict, Any

        @dataclass
        class LocalCalculationParams:
            inj_params: Dict[str, Any]
            DeltaF0: float
            DeltaF1: float
            DeltaF2: float
            dF0: float
            dF1_refined: float
            dF2_refined: float
            sky: bool
            outdir: str
            sftfilepath: str
            tref: int
            nsegs: int
            plot: bool
            labels: Dict[str, str]
            tstart: int
            duration: int

        params = LocalCalculationParams(**params_dict)

        # Calculate search ranges
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

        search_keys = ["F0", "F1", "F2"]

        if params.sky:
            dSky = 0.01
            DeltaSky = 10 * dSky
            Alphas = [
                params.inj_params["Alpha"] - DeltaSky / 2.0,
                params.inj_params["Alpha"] + DeltaSky / 2.0,
                dSky
            ]
            Deltas = [
                params.inj_params["Delta"] - DeltaSky / 2.0,
                params.inj_params["Delta"] + DeltaSky / 2.0,
                dSky
            ]
            search_keys += ["Alpha", "Delta"]
        else:
            Alphas = [params.inj_params["Alpha"]]
            Deltas = [params.inj_params["Delta"]]

        search_keys_label = "".join(search_keys)

        # Create output directory if it doesn't exist
        os.makedirs(params.outdir, exist_ok=True)

        # Run grid search with proper cleanup
        try:
            gridsearch = pyfstat.GridSearch(
                label=f"GridSearch_iter_{i}" + search_keys_label,
                outdir=params.outdir,
                sftfilepattern=params.sftfilepath,
                F0s=F0s,
                F1s=F1s,
                F2s=F2s,
                Alphas=Alphas,
                Deltas=Deltas,
                tref=params.tref,
                nsegs=params.nsegs,
            )
            gridsearch.run()
            gridsearch.print_max_twoF()
            gridsearch.generate_loudest()

            search_ranges = {
                "F0": [params.inj_params["F0"]],
                "Alpha": [params.inj_params["Alpha"]],
                "Delta": [params.inj_params["Delta"]],
            }

            fs = pyfstat.SemiCoherentSearch(
                label=f"MismatchTest_{i}",
                outdir=params.outdir,
                tref=params.tref,
                nsegs=params.nsegs,
                sftfilepattern=params.sftfilepath,
                minStartTime=params.tstart,
                maxStartTime=params.tstart + params.duration,
                search_ranges=search_ranges,
            )

            grid_res = gridsearch.data
            inj_pars = {k: params.inj_params[k] for k in ("F0", "F1", "F2", "Alpha", "Delta")}
            twoF_inj = fs.get_semicoherent_det_stat(params=inj_pars)
            rho2_no = twoF_inj - 4.0

            grid_maxidx = np.argmax(grid_res["twoF"])
            twoF_mis = grid_res["twoF"][grid_maxidx]
            rho2_mis = twoF_mis - 4.0

            mu_empirical = (rho2_no - rho2_mis) / rho2_no

            return mu_empirical

        finally:
            # Cleanup resources
            if 'gridsearch' in locals():
                del gridsearch
            if 'fs' in locals():
                del fs
            gc.collect()

    except Exception as e:
        logger.error(f"Error in calculate_mismatch for iteration {i}: {str(e)}")
        raise

def single_grid(pool: concurrent.futures.ProcessPoolExecutor, 
                params: CalculationParams,
                run_id: int,
                tmp_dir: str) -> pd.DataFrame:
    """Run a single grid calculation with improved error handling"""
    try:
        # Convert CalculationParams to dict for pickling
        params_dict = {
            'inj_params': params.inj_params,
            'DeltaF0': params.DeltaF0,
            'DeltaF1': params.DeltaF1,
            'DeltaF2': params.DeltaF2,
            'dF0': params.dF0,
            'dF1_refined': params.dF1_refined,
            'dF2_refined': params.dF2_refined,
            'sky': params.sky,
            'outdir': params.outdir,
            'sftfilepath': params.sftfilepath,
            'tref': params.tref,
            'nsegs': params.nsegs,
            'plot': params.plot,
            'labels': params.labels,
            'tstart': params.tstart,
            'duration': params.duration
        }

        # Generate random offsets
        random_offsets = [{
            "F0": np.random.uniform(-params.dF0, params.dF0),
            "F1": np.random.uniform(-params.dF1_refined, params.dF1_refined),
            "F2": np.random.uniform(-params.dF2_refined, params.dF2_refined)
        } for _ in range(config.numbers)]

        # Run calculations in parallel with timeout and proper error handling
        futures = []
        mismatches = []
        completed = 0
        failed = 0

        try:
            # Submit all tasks
            futures = [
                pool.submit(calculate_mismatch, i, params_dict, random_offsets[i])
                for i in range(config.numbers)
            ]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures, timeout=3600):  # 1-hour timeout
                try:
                    result = future.result()
                    mismatches.append(result)
                    completed += 1
                    if completed % 10 == 0:  # Log progress every 10 completions
                        logger.info(f"Run {run_id}: Completed {completed}/{config.numbers} calculations")
                except Exception as e:
                    failed += 1
                    logger.error(f"Run {run_id}: Calculation failed: {str(e)}")
                    continue

        except concurrent.futures.TimeoutError:
            logger.error(f"Run {run_id}: Timeout after 1 hour")
            # Cancel remaining futures
            for future in futures:
                future.cancel()
            raise

        if not mismatches:
            raise RuntimeError(f"Run {run_id}: No successful calculations completed")

        if failed > 0:
            logger.warning(f"Run {run_id}: {failed} calculations failed out of {config.numbers}")

        # Create and save results
        df_result = pd.DataFrame({
            'mismatch': mismatches,
            'mf': [params.mf] * len(mismatches),
            'mf1': [params.mf1] * len(mismatches),
            'mf2': [params.mf2] * len(mismatches),
            'df': [params.dF0] * len(mismatches),
            'df1': [params.dF1_refined] * len(mismatches),
            'df2': [params.dF2_refined] * len(mismatches),
            'run_id': run_id
        })

        # Save to temporary file with proper error handling
        tmp_filepath = os.path.join(tmp_dir, f"run_{run_id:03d}_df.csv")
        try:
            df_result.to_csv(tmp_filepath, index=False)
        except Exception as e:
            logger.error(f"Run {run_id}: Failed to save results: {str(e)}")
            raise

        return df_result

    except Exception as e:
        logger.error(f"Run {run_id}: Error in single_grid: {str(e)}")
        cleanup_temp_files(tmp_dir, run_id)
        raise

def generate_sft_files(config: Any, outdir: str) -> str:
    """Generate SFT files for testing if they don't exist"""
    import pyfstat
    import os

    # Create SFT directory
    sft_dir = os.path.join(outdir, "SFTs")
    os.makedirs(sft_dir, exist_ok=True)

    # Check if SFT files already exist
    sft_pattern = os.path.join(sft_dir, "*.sft")
    if glob.glob(sft_pattern):
        logger.info(f"Using existing SFT files in {sft_dir}")
        return sft_pattern

    logger.info("Generating new SFT files...")
    
    # Define default values
    default_tref = 1234567890  # Default reference time
    default_alpha = 0.0  # Default right ascension
    default_delta = 0.0  # Default declination
    default_tstart = 1234567890  # Default start time
    default_duration = 86400  # Default duration (1 day)
    default_tsft = 1800  # Default SFT duration (30 minutes)
    default_detectors = "H1"  # Default detector
    
    # Get values from config with defaults
    tref = getattr(config, 'tref', default_tref)
    alpha = getattr(config, 'inj_params', {}).get('Alpha', default_alpha)
    delta = getattr(config, 'inj_params', {}).get('Delta', default_delta)
    tstart = getattr(config, 'tstart', default_tstart)
    duration = getattr(config, 'duration', default_duration)
    tsft = getattr(config, 'tsft', default_tsft)
    detectors = getattr(config, 'detectors', default_detectors)
    
    # Generate SFT files with frequency parameters matching the search range
    writer = pyfstat.Writer(
        label="PyFstatExample",
        outdir=sft_dir,
        tref=tref,
        F0=30.0,  # Center frequency in the SFT range
        F1=0.0,
        F2=0.0,
        Alpha=alpha,
        Delta=delta,
        h0=1e-24,
        cosi=0.0,
        psi=0.0,
        phi=0.0,
        Tsft=tsft,
        Band=1.0,   # 1 Hz band to cover the search range
        detectors=detectors,
        sqrtSX=1e-23,  # Noise level
        randSeed=42,
        tstart=tstart,
        duration=duration,
        SFTWindowType="tukey",
        SFTWindowParam=0.001,  # Using SFTWindowParam instead of SFTWindowBeta
    )
    writer.make_data()

    logger.info(f"Generated SFT files in {sft_dir}")
    return sft_pattern

def process_parameter_sweep(mf_values: np.ndarray, 
                          mf1_values: np.ndarray, 
                          mf2_values: np.ndarray,
                          config: Any) -> None:
    """Main function to process the parameter sweep"""
    # Create output directories
    tmp_dir = os.path.join(config.outdir, "tmp_dataframes")
    os.makedirs(tmp_dir, exist_ok=True)

    # Define default injection parameters if not present in config
    default_inj_params = {
        "F0": 30.0,  # Default frequency matching SFT range
        "F1": 0.0,    # Default frequency derivative
        "F2": 0.0,    # Default second frequency derivative
        "Alpha": 0.0, # Default right ascension
        "Delta": 0.0  # Default declination
    }

    # Define default config values
    default_config = {
        'sftfilepath': None,  # Will be set after SFT generation
        'tref': 1234567890,      # Default reference time
        'nsegs': 1,              # Default number of segments
        'tstart': 1234567890,    # Default start time
        'duration': 86400,       # Default duration (1 day)
        'sky': False,            # Default sky search flag
        'plot': False,           # Default plotting flag
        'labels': {},            # Default labels
        'numbers': 10,           # Default number of calculations
        'num_workers': 4,        # Default number of workers
        'T_coh': 86400,         # Default coherent time
        'gamma1': 1.0,          # Default gamma1
        'gamma2': 1.0           # Default gamma2
    }

    # Use injection parameters from config if available, otherwise use defaults
    inj_params = getattr(config, 'inj_params', default_inj_params)

    # Get config values with defaults
    tref = getattr(config, 'tref', default_config['tref'])
    nsegs = getattr(config, 'nsegs', default_config['nsegs'])
    tstart = getattr(config, 'tstart', default_config['tstart'])
    duration = getattr(config, 'duration', default_config['duration'])
    sky = getattr(config, 'sky', default_config['sky'])
    plot = getattr(config, 'plot', default_config['plot'])
    labels = getattr(config, 'labels', default_config['labels'])
    numbers = getattr(config, 'numbers', default_config['numbers'])
    num_workers = getattr(config, 'num_workers', default_config['num_workers'])
    T_coh = getattr(config, 'T_coh', default_config['T_coh'])
    gamma1 = getattr(config, 'gamma1', default_config['gamma1'])
    gamma2 = getattr(config, 'gamma2', default_config['gamma2'])

    # Generate or find SFT files
    sftfilepath = generate_sft_files(config, config.outdir)

    # Pre-calculate parameter grids
    param_grids = precalculate_parameter_grids(
        mf_values, mf1_values, mf2_values,
        T_coh, gamma1, gamma2
    )

    # Calculate required disk space (rough estimate)
    estimated_space = len(param_grids) * numbers * 1000  # 1KB per result
    if not check_disk_space(config.outdir, estimated_space):
        raise RuntimeError(f"Insufficient disk space. Need at least {estimated_space/1e9:.2f}GB")

    # Determine optimal number of workers
    num_workers = min(num_workers, multiprocessing.cpu_count() - 1)
    logger.info(f"Using {num_workers} workers for parallel processing")

    all_dataframes = []
    run_counter = 0
    total_runs = len(param_grids)

    with concurrent.futures.ProcessPoolExecutor(num_workers) as pool:
        for (mf, mf1, mf2), param_grid in param_grids.items():
            run_counter += 1
            tmp_filename = f"run_{run_counter:03d}_df.csv"
            
            if os.path.exists(os.path.join(tmp_dir, tmp_filename)):
                logger.info(f"Skipping run {run_counter} as {tmp_filename} already exists")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Run {run_counter}/{total_runs}")
            logger.info(f"mf={mf:.3f}, mf1={mf1:.3f}, mf2={mf2:.6f}")
            logger.info(f"{'='*60}")

            try:
                # Create calculation parameters
                calc_params = CalculationParams(
                    inj_params=inj_params,
                    DeltaF0=param_grid.DeltaF0,
                    DeltaF1=param_grid.DeltaF1,
                    DeltaF2=param_grid.DeltaF2,
                    dF0=param_grid.dF0,
                    dF1_refined=param_grid.dF1_refined,
                    dF2_refined=param_grid.dF2_refined,
                    sky=sky,
                    outdir=os.path.join(config.outdir, f"r{run_counter:03d}"),
                    sftfilepath=sftfilepath,
                    tref=tref,
                    nsegs=nsegs,
                    plot=plot,
                    labels=labels,
                    tstart=tstart,
                    duration=duration
                )

                # Run calculation
                df_result = single_grid(pool, calc_params, run_counter, tmp_dir)
                all_dataframes.append(df_result)

                logger.info(f"✓ Run {run_counter} completed successfully")
                logger.info(f"  Mean mismatch: {df_result['mismatch'].mean():.6f}")
                logger.info(f"  Std mismatch: {df_result['mismatch'].std():.6f}")

            except Exception as e:
                logger.error(f"✗ Run {run_counter} failed with error: {str(e)}")
                continue

    # Process results
    if all_dataframes:
        logger.info(f"\n{'='*60}")
        logger.info("MERGING ALL RESULTS")
        logger.info(f"{'='*60}")

        # Combine results
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_filepath = os.path.join(config.outdir, "combined_mismatch_results.csv")
        combined_df.to_csv(combined_filepath, index=False)

        # Create summary statistics
        summary_stats = combined_df.groupby(['mf', 'mf1', 'mf2']).agg({
            'mismatch': ['count', 'mean', 'std', 'min', 'max', 'median'],
            'run_id': 'first'
        }).round(6)

        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        summary_stats = summary_stats.reset_index()

        summary_filepath = os.path.join(config.outdir, "parameter_sweep_summary.csv")
        summary_stats.to_csv(summary_filepath, index=False)

        logger.info(f"✓ Combined dataframe saved to: {combined_filepath}")
        logger.info(f"✓ Summary statistics saved to: {summary_filepath}")

if __name__ == "__main__":
    # Create parameter grids with corrected ranges
    mf_values = np.linspace(0.1, 0.5, 10)
    mf1_values = np.linspace(0.1, 0.5, 10)
    mf2_values = np.linspace(0.0005, 0.001, 5)  # Corrected range

    try:
        # Run parameter sweep
        process_parameter_sweep(mf_values, mf1_values, mf2_values, config)
    except Exception as e:
        logger.error(f"Fatal error in parameter sweep: {str(e)}")
        raise
    finally:
        # Cleanup temporary files
        tmp_dir = os.path.join(config.outdir, "tmp_dataframes")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir) 