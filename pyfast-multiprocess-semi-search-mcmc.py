import os
import numpy as np
import pyfstat
import config
from typing import Dict, Any
import pandas as pd
# make sure to put these after the pyfstat import, to not break notebook inline plots
import matplotlib.pyplot as plt
import multiprocessing

# %matplotlib inline
from utils import plot_grid_vs_samples, plot_2F_scatter, CalculationParams
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

import concurrent.futures

# flip this switch for a more expensive 4D (F0,F1,Alpha,Delta) run
# instead of just (F0,F1)
# (still only a few minutes on current laptops)

# log =


# general setup

logger = pyfstat.set_up_logger(
    label=config.label, outdir=config.outdir, log_level="WARNING"
)
if config.sky:
    config.outdir += "AlphaDelta"
printout = False
# parameters for the data set to generate


# parameters for injected signals

# create SFT files
logger.info("Generating SFTs with injected signal...")
writer = pyfstat.Writer(
    label=config.label + "SimulatedSignal",
    outdir=config.outdir,
    tstart=config.tstart,
    duration=config.duration,
    detectors=config.detectors,
    sqrtSX=config.sqrtSX,
    Tsft=config.Tsft,
    **config.inj,
    Band=1,  # default band estimation would be too narrow for a wide grid/prior
)
writer.make_data()

# set up square search grid with fixed (F0,F1) mismatch
# and (optionally) some ad-hoc sky coverage

print(config.DeltaF0, config.DeltaF1, config.DeltaF2)


mismatches = []


def calculate_mismatch(i: int, params: CalculationParams, random_offsets: Dict[str, float]) -> float:

    import pyfstat
    import numpy as np
    import os

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

    search_keys = ["F0", "F1", "F2"]  # only the ones that aren't 0-width

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

    # run the grid search
    # logger.info("Performing GridSearch...")
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

    # do some plots of the GridSearch results
    if not params.sky:  # this plotter can't currently deal with too large result arrays
        # logger.info("Plotting 1D 2F distributions...")
        if params.plot:
            for key in search_keys:
                gridsearch.plot_1D(
                    xkey=key, xlabel=params.labels[key], ylabel=params.labels["2F"]
                )

        # logger.info(
        #     "Making GridSearch {:s} corner plot...".format("-".join(search_keys))
        # )
        vals = [
            np.unique(gridsearch.data[key]) - params.inj_params[key] for key in search_keys
        ]
        twoF = gridsearch.data["twoF"].reshape([len(kval) for kval in vals])
        corner_labels = [
            "$f - f_0$ [Hz]",
            "$\\dot{f} - \\dot{f}_0$ [Hz/s]",
        ]
        if params.sky:
            corner_labels.append("$\\alpha - \\alpha_0$")
            corner_labels.append("$\\delta - \\delta_0$")
        corner_labels.append(params.labels["2F"])
        if params.plot:
            gridcorner_fig, gridcorner_axes = pyfstat.gridcorner(
                twoF,
                vals,
                projection="log_mean",
                labels=corner_labels,
                whspace=0.1,
                factor=1.8,
            )
            gridcorner_fig.savefig(
                os.path.join(params.outdir, gridsearch.label + "_corner.png")
            )
            # plt.show()

    # we'll use the two local plotting functions defined above
    # to avoid code duplication in the sky case
    if params.plot:
        plot_2F_scatter(gridsearch.data, "grid", "F0", "F1", params)
        if params.sky:
            plot_2F_scatter(gridsearch.data, "grid", "Alpha", "Delta", params)

    # -----------------------------------------------------------
    #  Mismatch diagnosis (API-safe version, PyFstat ≥ 2.x)
    # -----------------------------------------------------------

    search_ranges = {
        "F0": [params.inj_params["F0"]],  # a single value ⇒ zero width,
        "Alpha": [params.inj_params["Alpha"]],
        "Delta": [params.inj_params["Delta"]],
    }

    
    
    fs = pyfstat.SemiCoherentSearch(
        label=f"MismatchTest_{i}",  # SemiCoherentSearch需要label
        outdir=params.outdir,  # 需要outdir
        tref=params.tref,
        nsegs=params.nsegs,  # 添加分段数
        sftfilepattern=params.sftfilepath,
        minStartTime=params.tstart,
        maxStartTime=params.tstart + params.duration,
        search_ranges=search_ranges,
    )

    grid_res = gridsearch.data

    # template exactly at the injected parameters
    inj_pars = {k: params.inj_params[k] for k in ("F0", "F1", "F2", "Alpha", "Delta")}

    twoF_inj = fs.get_semicoherent_det_stat(params=inj_pars)

    rho2_no = twoF_inj - 4.0  # ρ²_no-mismatch

    # --- 2) loudest point from the grid you already ran ------------
    grid_maxidx = np.argmax(grid_res["twoF"])
    twoF_mis = grid_res["twoF"][grid_maxidx]
    rho2_mis = twoF_mis - 4.0  # ρ²_mismatch

    # --- 3) empirical mismatch -------------------------------------
    mu_empirical = (rho2_no - rho2_mis) / rho2_no

    if printout:
        print("\n--------- mismatch check (ρ-based) ---------")
        print(f"2F(injection)  = {twoF_inj:10.3f}")
        print(f"2F(loudest)    = {twoF_mis:10.3f}")
        print(f"ρ²_no-mismatch = {rho2_no:10.3f}")
        print(f"ρ²_mismatch    = {rho2_mis:10.3f}")
        print(f"μ  (empirical) = {mu_empirical:10.3e}")
        print("-------------------------------------------")

    # mismatches.append(mu_empirical)
    del gridsearch  # 1️⃣ free Python references
    del fs  # 2️⃣ free ComputeFstat object
    import gc

    gc.collect()  # 3️⃣ force GC inside the worker

    return mu_empirical


def single_grid(pool):
    
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
        sftfilepath=writer.sftfilepath,  # This needs to be available
        tref=config.inj["tref"],
        nsegs=config.nsegs,
        plot=config.plot,
        labels=config.labels,
        tstart=config.tstart,
        duration=config.duration
    )
    
    # run the mismatch calculation in parallel
    # with concurrent.futures.ProcessPoolExecutor(config.num_workers) as executor:
    #     futures = []
    #     for i in range(config.numbers):
    #         futures.append(executor.submit(calculate_mismatch, i, params, all_random_offsets[i]))
            
    #     mismatches = [future.result() for future in concurrent.futures.as_completed(futures)]

    
    futures = [
        pool.submit(calculate_mismatch, i, params, all_random_offsets[i])
        for i in range(config.numbers)
    ]
    mismatches = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    
    # save the mismatch results to a csv file
    mismatch_file = os.path.join(config.outdir, "mismatches.csv")
    np.savetxt(
        mismatch_file,
        mismatches,
        delimiter=",",
        header="Empirical Mismatch (μ)",
        comments="",
    )

    # plot the mismatch distribution

    # fig, ax = plt.subplots(figsize=(10, 6))
    # choose bin edges so the last bar ends at 1.0, like in the photo
    # bins = np.linspace(0, 1, 11)        # 10 equal-width bins → 11 edges
    # ax.hist(
    #     mismatches,
    #     bins=10,
    #     density=True,
    #     color="#5B9BD5",  # pleasant blue
    #     alpha=0.85,
    #     edgecolor="black",
    #     linewidth=1.0,
    # )

    # # axis labels & limits
    # ax.set_xlabel(r"mismatch $\mu$", fontsize=20)
    # ax.set_ylabel("normalized histogram", fontsize=20)
    # ax.set_xlim(0, 1)
    # # ax.set_ylim(0, 0.25)

    # # ticks & grid
    # ax.tick_params(axis="both", which="major", labelsize=14, length=6)
    # ax.grid(axis="y", linewidth=0.6, alpha=0.35)

    # fig.tight_layout()
    # fig.savefig(os.path.join(config.outdir, f"mismatch_distribution-max-mismatch:{config.mf}.pdf"))
    # plt.show()

    # rumtime
    print("runtime: ", config.tau_total)



    # -------------------------------------------------------------------------
    # NEW CODE: Create and save a pandas DataFrame with additional information
    # -------------------------------------------------------------------------
    
    # Assuming config contains mf, mf1, mf2. If not, replace with actual values.
    # We map df, df1, df2 to the refined grid spacings used.
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
    
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a new CSV file
    # df_output_path = os.path.join(config.outdir, "mismatch_data_frame.csv")
    # df.to_csv(df_output_path, index=False)
    
    # print("\nSuccessfully created pandas DataFrame:")
    # print(df.head())
    # print(f"\nDataFrame saved to: {df_output_path}")

    return df

if __name__ == "__main__":
    # MCMC/Random sampling parameters
    n_samples = 300  # Total number of random parameter combinations to sample
    
    # Define parameter ranges for sampling
    mf_range = (0.1, 0.5)      # continuous uniform distribution
    mf1_range = (0.1, 0.5)     # continuous uniform distribution  
    mf2_range = (0.001, 0.3)   # continuous log-uniform distribution (min must be > 0)
    
    # Function to generate random parameter combination
    def generate_random_parameters():
        mf = np.random.uniform(mf_range[0], mf_range[1])
        mf1 = np.random.uniform(mf1_range[0], mf1_range[1])
        
        # Log-uniform sampling for mf2
        # Sample uniformly in log space, then convert back to linear space
        log_mf2_min = np.log10(mf2_range[0])
        log_mf2_max = np.log10(mf2_range[1])
        log_mf2 = np.random.uniform(log_mf2_min, log_mf2_max)
        mf2 = 10**log_mf2
        
        return mf, mf1, mf2
    
    # Create tmp directory for saving individual dataframes
    tmp_dir = os.path.join(config.outdir, "tmp_dataframes")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Store original values to restore later
    original_mf = config.mf
    original_mf1 = config.mf1
    original_mf2 = config.mf2
    original_dF0 = config.dF0
    original_dF1_refined = config.dF1_refined
    original_dF2_refined = config.dF2_refined
    original_outdir = config.outdir

    print(f"Starting MCMC parameter sampling:")
    print(f"mf range: [{mf_range[0]:.3f}, {mf_range[1]:.3f}] (uniform)")
    print(f"mf1 range: [{mf1_range[0]:.3f}, {mf1_range[1]:.3f}] (uniform)")
    print(f"mf2 range: [{mf2_range[0]:.6f}, {mf2_range[1]:.3f}] (log-uniform)")
    print(f"Total samples: {n_samples}")

    all_dataframes = []
    run_counter = 0
    
    # Set random seed for reproducibility (optional)
    np.random.seed(42)
    
    with concurrent.futures.ProcessPoolExecutor(config.num_workers) as pool:
        # Generate and process random parameter samples
        for sample_idx in range(n_samples):
            run_counter += 1
            
            # Generate random parameter combination
            mf_val, mf1_val, mf2_val = generate_random_parameters()
            
            tmp_filename = f"mcmc_run_{run_counter:04d}_df.csv"
            if os.path.exists(os.path.join(tmp_dir, tmp_filename)):
                print(f"Skipping run {run_counter} as {tmp_filename} already exists.")
                continue
                
            print(f"\n{'='*60}")
            print(f"MCMC Sample {run_counter}/{n_samples}")
            print(f"mf={mf_val:.6f}, mf1={mf1_val:.6f}, mf2={mf2_val:.6f}")
            print(f"{'='*60}")
            
            # Update config parameters for this combination
            config.mf = mf_val
            config.mf1 = mf1_val
            config.mf2 = mf2_val
            
            # Recalculate derived parameters
            config.dF0 = np.sqrt(12 * config.mf) / (np.pi * config.T_coh)
            config.dF1 = np.sqrt(180 * config.mf1) / (np.pi * config.T_coh**2)
            config.dF2 = np.sqrt(25200 * config.mf2) / (np.pi * config.T_coh**3)
            config.dF1_refined = config.dF1 / config.gamma1
            config.dF2_refined = config.dF2 / config.gamma2
            
            config.DeltaF0 = 8 * config.dF0 
            config.DeltaF1 = 8 * config.dF1_refined
            config.DeltaF2 = 8 * config.dF2_refined
            
            # Update output directory for this run
            config.outdir = os.path.join(original_outdir, f"mcmc_run_{run_counter:04d}_mf{mf_val:.6f}_mf1{mf1_val:.6f}_mf2{mf2_val:.6f}")
            os.makedirs(config.outdir, exist_ok=True)
            
            print(f"Updated parameters:")
            print(f"  dF0 = {config.dF0:.6e}")
            print(f"  dF1_refined = {config.dF1_refined:.6e}")
            print(f"  dF2_refined = {config.dF2_refined:.6e}")
            print(f"  Output dir: {config.outdir}")
            
            try:
                # Run the single grid calculation
                df_result = single_grid(pool)   # pass the pool here
                
                # Add parameter information to the dataframe
                df_result['run_id'] = run_counter
                df_result['sample_idx'] = sample_idx
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
                
                print(f"✓ Sample {run_counter} completed successfully")
                print(f"  Mean mismatch: {df_result['mismatch'].mean():.6f}")
                print(f"  Std mismatch: {df_result['mismatch'].std():.6f}")
                print(f"  Saved to: {tmp_filepath}")
                
            except Exception as e:
                print(f"✗ Sample {run_counter} failed with error: {e}")
                # Continue with next sample even if this one fails
                continue
    
    # Restore original config values
    config.mf = original_mf
    config.mf1 = original_mf1
    config.mf2 = original_mf2
    config.dF0 = original_dF0
    config.dF1_refined = original_dF1_refined
    config.dF2_refined = original_dF2_refined
    config.outdir = original_outdir
    
    print(f"\n{'='*60}")
    print("MERGING ALL MCMC RESULTS")
    print(f"{'='*60}")
    
    # Merge all dataframes
    if all_dataframes:
        print(f"Merging {len(all_dataframes)} successful MCMC samples...")
        
        # Concatenate all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Save the combined dataframe
        combined_filepath = os.path.join(original_outdir, "mcmc_mismatch_results.csv")
        combined_df.to_csv(combined_filepath, index=False)
        
        # Create summary statistics
        summary_stats = combined_df.groupby(['mf_input', 'mf1_input', 'mf2_input']).agg({
            'mismatch': ['count', 'mean', 'std', 'min', 'max', 'median'],
            'run_id': 'first'
        }).round(6)
        
        # Also create overall statistics
        overall_stats = combined_df.agg({
            'mismatch': ['count', 'mean', 'std', 'min', 'max', 'median'],
            'mf_input': ['min', 'max', 'mean', 'std'],
            'mf1_input': ['min', 'max', 'mean', 'std'],
            'mf2_input': ['min', 'max', 'mean', 'std']
        }).round(6)
        
        # Flatten column names for summary stats
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
        summary_stats = summary_stats.reset_index()
        
        # Save summary statistics
        summary_filepath = os.path.join(original_outdir, "mcmc_sampling_summary.csv")
        summary_stats.to_csv(summary_filepath, index=False)
        
        # Save overall statistics
        overall_filepath = os.path.join(original_outdir, "mcmc_overall_stats.csv")
        overall_stats.to_csv(overall_filepath)
        
        print(f"✓ Combined dataframe saved to: {combined_filepath}")
        print(f"✓ Summary statistics saved to: {summary_filepath}")
        print(f"✓ Overall statistics saved to: {overall_filepath}")
        print(f"✓ Individual run dataframes saved in: {tmp_dir}")
        
        print(f"\nFinal MCMC Dataset Info:")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Total parameter samples: {len(combined_df.groupby(['mf_input', 'mf1_input', 'mf2_input']))}")
        print(f"  Columns: {list(combined_df.columns)}")
        
        print(f"\nOverall Statistics:")
        print(f"  Mean mismatch across all samples: {combined_df['mismatch'].mean():.6f}")
        print(f"  Std mismatch across all samples: {combined_df['mismatch'].std():.6f}")
        print(f"  Min mismatch: {combined_df['mismatch'].min():.6f}")
        print(f"  Max mismatch: {combined_df['mismatch'].max():.6f}")
        
        # Parameter space coverage
        print(f"\nParameter Space Coverage:")
        print(f"  mf range sampled: [{combined_df['mf_input'].min():.6f}, {combined_df['mf_input'].max():.6f}]")
        print(f"  mf1 range sampled: [{combined_df['mf1_input'].min():.6f}, {combined_df['mf1_input'].max():.6f}]")
        print(f"  mf2 range sampled: [{combined_df['mf2_input'].min():.6f}, {combined_df['mf2_input'].max():.6f}] (log-uniform)")
        
    else:
        print("✗ No successful MCMC samples to merge!")
    
    print(f"\n{'='*60}")
    print("MCMC PARAMETER SAMPLING COMPLETED")
    print(f"{'='*60}")
    # print(f"Total runtime: {config.tau_total}")
    print(f"Results saved in: {original_outdir}")