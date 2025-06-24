import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn
import matplotlib
matplotlib.use("Agg")
# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)


# Create output directory
N = 500  # Number of mismatch trials per parameter set
n_param_trials = 500  # Number of parameter combinations to try
print_output = False  # Set to False to suppress output
label = "LALSemiCoherentF0F1F2_search"
outdir = os.path.join("LAL_example_data", label)
os.makedirs(outdir, exist_ok=True)
os.makedirs(os.path.join("data"), exist_ok=True)  # Create data directory for CSV

# Properties of the GW data
sqrtSX = 1
tstart = 1000000000
duration = 120 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)
IFO = "H1, L1"  # Interferometers to use

# Parameters for injected signals
depth = 20
h0 = sqrtSX / depth
F0_inj = 151.5
F1_inj = -1e-10
F2_inj = -1e-20
Alpha_inj = 0.5
Delta_inj = 1
cosi_inj = 1
psi_inj = 0.0
phi0_inj = 0.0

# Semi-coherent search parameters
tStack = 15 * 86400  # 15 day coherent segments
nStacks = int(duration / tStack)  # Number of segments

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

# Step 1: Generate SFT data (only needs to be done once)
sft_dir = os.path.join(outdir, "sfts")
os.makedirs(sft_dir, exist_ok=True)
os.makedirs(os.path.join(outdir, "dats"), exist_ok=True)
os.makedirs(os.path.join(outdir, "commands"), exist_ok=True)
sft_pattern = os.path.join(sft_dir, "*.sft")

# Check if SFTs already exist
if not os.path.exists(os.path.join(sft_dir, "H1-1800_SemiCoh-1000000000-1800.sft")):
    print("Generating SFT data with injected signal...")
    
    injection_params = (
        f"{{Alpha={Alpha_inj:.15g}; Delta={Delta_inj:.15g}; Freq={F0_inj:.15g}; "
        f"f1dot={F1_inj:.15e}; f2dot={F2_inj:.15e}; refTime={tref:.15g}; "
        f"h0={h0:.15e}; cosi={cosi_inj:.15g}; psi={psi_inj:.15g}; phi0={phi0_inj:.15g};}}"
    )

    sft_label = "SemiCoh"

    makefakedata_cmd = [
        "lalpulsar_Makefakedata_v5",
        f"--IFOs={IFO}",
        f"--sqrtSX={sqrtSX:.15e}, {sqrtSX:.15e}",
        f"--startTime={int(tstart)}",
        f"--duration={int(duration)}",
        f"--fmin={F0_inj - 0.5:.15g}",
        f"--Band=1.0",
        "--Tsft=1800",
        f"--outSFTdir={sft_dir}",
        f"--outLabel={sft_label}",
        f"--injectionSources={injection_params}",
        "--randSeed=1234"
    ]

    result = subprocess.run(makefakedata_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error generating SFTs: {result.stderr}")
        raise RuntimeError("Failed to generate SFTs")
else:
    print("SFTs already exist, skipping generation...")

# Step 2: Create segment list file (CRITICAL!)
segFile = os.path.join(outdir, "segments.dat")
with open(segFile, 'w') as f:
    for i in range(nStacks):
        seg_start = tstart + i * tStack
        seg_end = seg_start + tStack
        nsft = int(tStack / 1800)  # Number of SFTs in segment
        f.write(f"{int(seg_start)} {int(seg_end)} {nsft}\n")

# Shared command parts
def get_shared_cmd():
    return [
        f"--DataFiles1={sft_pattern}",
        "--gridType1=3",  # IMPORTANT: 3=file mode for sky grid
        f"--skyGridFile={{{Alpha_inj} {Delta_inj}}}",
        f"--refTime={tref:.15f}",
        f"--tStack={tStack:.15g}",
        f"--nStacksMax={nStacks}",
        "--nCand1=1000",
        "--printCand1",
        "--semiCohToplist",
        f"--minStartTime1={int(tstart)}",
        f"--maxStartTime1={int(tend)}",
        "--recalcToplistStats=TRUE",
        "--FstatMethod=ResampBest",
        "--FstatMethodRecalc=DemodBest",
    ]

def single_run(param_idx, trial_idx, mf, mf1, mf2, F0_random, F1_random, F2_random):
    """Run a single search with given parameters"""
    
    # Calculate grid spacings based on current mf values
    dF0 = np.sqrt(12 * mf) / (np.pi * tStack)
    dF1 = np.sqrt(180 * mf1) / (np.pi * tStack**2)
    df2 = np.sqrt(25200 * mf2) / (np.pi * tStack**3)

    # Search bands
    N1 = 2
    N2 = 3
    N3 = 3
    gamma1 = 8
    gamma2 = 20

    DeltaF0 = N1 * dF0
    DeltaF1 = N2 * dF1
    DeltaF2 = N3 * df2

    F0_min = F0_inj - DeltaF0 / 2.0 + F0_random
    F0_max = F0_inj + DeltaF0 / 2.0 + F0_random
    F1_min = F1_inj - DeltaF1 / 2.0 + F1_random
    F1_max = F1_inj + DeltaF1 / 2.0 + F1_random
    F2_min = F2_inj - DeltaF2 / 2.0 + F2_random
    F2_max = F2_inj + DeltaF2 / 2.0 + F2_random

    output_file = os.path.join(outdir, f"dats/semicoh_results_p{param_idx}_t{trial_idx}.dat")
    
    # Build command with proper formatting
    hierarchsearch_cmd = [
        "lalpulsar_HierarchSearchGCT",
        f"--fnameout={output_file}",
        f"--Freq={F0_min:.15g}",
        f"--FreqBand={DeltaF0:.15g}",
        f"--dFreq={dF0:.15e}",
        f"--f1dot={F1_min:.15e}",
        f"--f1dotBand={DeltaF1:.15e}",
        f"--df1dot={dF1:.15e}",
        f"--f2dot={F2_min:.15e}",
        f"--f2dotBand={DeltaF2:.15e}",
        f"--df2dot={df2:.15e}",
        f"--gammaRefine={gamma1:.15g}",
        f"--gamma2Refine={gamma2:.15g}",
    ] + get_shared_cmd()

    # Save command for debugging
    cmd_file = os.path.join(outdir, f"commands/command_p{param_idx}_t{trial_idx}.sh")
    with open(cmd_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(" \\\n    ".join(hierarchsearch_cmd))
        f.write("\n")
    os.chmod(cmd_file, 0o755)

    # Run the command
    result = subprocess.run(hierarchsearch_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running HierarchSearchGCT for param {param_idx}, trial {trial_idx}:")
        print(f"stderr: {result.stderr}")
        raise RuntimeError("Failed to run semi-coherent search")
    
    # Read results and find maximum 2F
    if not os.path.exists(output_file):
        raise RuntimeError(f"Output file {output_file} does not exist")
        
    with open(output_file, 'r') as f:
        lines = f.readlines()
        
    data = []
    for line in lines:
        if line.strip() and not line.startswith('%'):
            parts = line.split()
            if len(parts) >= 8:
                try:
                    freq = float(parts[0])
                    alpha = float(parts[1])
                    delta = float(parts[2])
                    f1dot = float(parts[3])
                    f2dot = float(parts[4])
                    nc = float(parts[5])
                    twoF = float(parts[6])
                    twoFr = float(parts[7])
                    data.append([freq, f1dot, f2dot, twoFr])
                except ValueError:
                    continue

    if data:
        data = np.array(data)
        twoF_vals = data[:, 3]
        max_idx = np.argmax(twoF_vals)
        max_twoF = twoF_vals[max_idx]
        return max_twoF
    else:
        raise RuntimeError(f"No valid data found in {output_file}")

def compute_perfect_2F(param_idx, mf, mf1, mf2):
    """Compute the perfectly matched 2F value for given parameters"""
    
    # Calculate grid spacings
    dF0 = np.sqrt(12 * mf) / (np.pi * tStack)
    dF1 = np.sqrt(180 * mf1) / (np.pi * tStack**2)
    df2 = np.sqrt(25200 * mf2) / (np.pi * tStack**3)
    
    perfect_output_file = os.path.join(outdir, f"perfectly_matched_results_p{param_idx}.dat")

    perfect_search_cmd = [
        "lalpulsar_HierarchSearchGCT",
        f"--Freq={F0_inj:.15g}",
        "--FreqBand=0",
        f"--dFreq={dF0:.15e}",
        f"--f1dot={F1_inj:.15e}",
        "--f1dotBand=0",
        f"--df1dot={dF1:.15e}",
        f"--f2dot={F2_inj:.15e}",
        "--f2dotBand=0",
        f"--df2dot={df2:.15e}",
        f"--fnameout={perfect_output_file}",
        f"--gammaRefine=1",
        f"--gamma2Refine=1",
    ] + get_shared_cmd()

    result = subprocess.run(perfect_search_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error computing perfect 2F: {result.stderr}")
        raise RuntimeError("Failed to compute perfectly matched 2F value")

    with open(perfect_output_file, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        if line.strip() and not line.startswith('%'):
            parts = line.split()
            if len(parts) >= 8:
                try:
                    freq = float(parts[0])
                    alpha = float(parts[1])
                    delta = float(parts[2])
                    f1dot = float(parts[3])
                    f2dot = float(parts[4])
                    f3dot = float(parts[5])
                    twoF = float(parts[6])
                    twoFr = float(parts[7])
                    data.append([freq, f1dot, f2dot, twoFr])
                except ValueError:
                    continue

    if data:
        data = np.array(data)
        twoF_vals = data[:, 3]
        max_idx = np.argmax(twoF_vals)
        perfect_2F = twoF_vals[max_idx]
        return perfect_2F
    else:
        raise RuntimeError("No valid data found in perfect match file")


# Check for existing .dat files and load results
all_results = []
completed_trials = {}  # {param_idx: set of completed trial_idx}
perfect_2F_cache = {}  # {param_idx: perfect_2F_value}

# Define CSV filename for saving results
csv_filename = f"data/lal_parameter_search_results_{n_param_trials}params_{N}trials.csv"

print(f"Starting parameter search with {n_param_trials} parameter combinations, {N} trials each")

print("Scanning for existing .dat files...")
for param_idx in range(n_param_trials):
    completed_trials[param_idx] = set()
    
    # Check for perfect 2F file
    perfect_file = os.path.join(outdir, f"perfectly_matched_results_p{param_idx}.dat")
    if os.path.exists(perfect_file):
        # Read perfect 2F value
        try:
            with open(perfect_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.strip() and not line.startswith('%'):
                    parts = line.split()
                    if len(parts) >= 8:
                        perfect_2F_cache[param_idx] = float(parts[7])
                        break
        except:
            pass
    
    # Check for trial files
    for trial_idx in range(N):
        trial_file = os.path.join(outdir, f"dats/semicoh_results_p{param_idx}_t{trial_idx}.dat")
        if os.path.exists(trial_file):
            completed_trials[param_idx].add(trial_idx)

total_completed = sum(len(trials) for trials in completed_trials.values())
print(f"Found {total_completed} completed trials across {len([p for p in completed_trials if completed_trials[p]])} parameter sets")


print(f"Starting parameter search with {n_param_trials} parameter combinations, {N} trials each")

for param_idx in range(n_param_trials):
    
    # Skip if this parameter set is already completed
    if len(completed_trials[param_idx]) == N:
        print(f"Skipping parameter trial {param_idx + 1}/{n_param_trials} (already completed)")
        continue
    
    # Generate random parameters for this trial
    mf, mf1, mf2 = generate_random_parameters()
    
    print(f"\nParameter trial {param_idx + 1}/{n_param_trials}:")
    print(f"  mf = {mf:.4f}, mf1 = {mf1:.4f}, mf2 = {mf2:.6f}")
    
    # Calculate grid spacings for random offsets
    dF0 = np.sqrt(12 * mf) / (np.pi * tStack)
    dF1 = np.sqrt(180 * mf1) / (np.pi * tStack**2)
    df2 = np.sqrt(25200 * mf2) / (np.pi * tStack**3)
    
    # Generate random offsets for this parameter set
    F0_randoms = np.random.uniform(-dF0 / 2.0, dF0 / 2.0, size=N)
    F1_randoms = np.random.uniform(-dF1 / 2.0, dF1 / 2.0, size=N)
    F2_randoms = np.random.uniform(-df2 / 2.0, df2 / 2.0, size=N)
    
    # Compute perfect 2F for this parameter set (if not cached)
    if param_idx in perfect_2F_cache:
        perfect_2F = perfect_2F_cache[param_idx]
        print(f"  Using cached perfect 2F = {perfect_2F:.2f}")
    else:
        print(f"  Computing perfect 2F...")
        perfect_2F = compute_perfect_2F(param_idx, mf, mf1, mf2)
        perfect_2F_cache[param_idx] = perfect_2F
        print(f"  Perfect 2F = {perfect_2F:.2f}")
    
    # Run N trials with this parameter set
    max_twoFs = []
    
    with Progress(
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        "[progress.completed]{task.completed}/{task.total}",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        completed_count = len(completed_trials[param_idx])
        task = progress.add_task(f"Processing trials for param set {param_idx + 1}", total=N, completed=completed_count)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
            futures = []
            for trial_idx in range(N):
                if trial_idx in completed_trials[param_idx]:
                    # Load existing result
                    trial_file = os.path.join(outdir, f"dats/semicoh_results_p{param_idx}_t{trial_idx}.dat")
                    try:
                        with open(trial_file, 'r') as f:
                            lines = f.readlines()
                        for line in lines:
                            if line.strip() and not line.startswith('%'):
                                parts = line.split()
                                if len(parts) >= 8:
                                    max_twoF = float(parts[7])
                                    max_twoFs.append(max_twoF)
                                    mismatch = (perfect_2F - max_twoF) / (perfect_2F - 4)
                                    all_results.append({
                                        'param_idx': param_idx,
                                        'trial_idx': trial_idx,
                                        'mf': mf,
                                        'mf1': mf1,
                                        'mf2': mf2,
                                        'perfect_2F': perfect_2F,
                                        'max_2F': max_twoF,
                                        'mismatch': mismatch
                                    })
                                    break
                    except Exception as e:
                        print(f"Error reading existing trial {trial_idx}: {e}")
                else:
                    # Submit new job
                    future = executor.submit(
                        single_run, 
                        param_idx, 
                        trial_idx, 
                        mf, 
                        mf1, 
                        mf2,
                        F0_randoms[trial_idx],
                        F1_randoms[trial_idx],
                        F2_randoms[trial_idx]
                    )
                    futures.append((trial_idx, future))
            
            for trial_idx, future in futures:
                try:
                    max_twoF = future.result()
                    max_twoFs.append(max_twoF)
                    
                    # Calculate mismatch
                    mismatch = (perfect_2F - max_twoF) / (perfect_2F - 4)
                    
                    # Store result
                    all_results.append({
                        'param_idx': param_idx,
                        'trial_idx': trial_idx,
                        'mf': mf,
                        'mf1': mf1,
                        'mf2': mf2,
                        'perfect_2F': perfect_2F,
                        'max_2F': max_twoF,
                        'mismatch': mismatch
                    })
                    
                    progress.advance(task, 1)
                except Exception as e:
                    print(f"Error in trial {trial_idx}: {e}")
                    progress.advance(task, 1)
    
    # Calculate statistics for this parameter set
    mismatches = [(perfect_2F - twoF) / (perfect_2F - 4) for twoF in max_twoFs]
    mean_mismatch = np.mean(mismatches)
    std_mismatch = np.std(mismatches)
    
    print(f"  Mean mismatch: {mean_mismatch:.4f} ± {std_mismatch:.4f}")
    
    # Save intermediate results after each parameter set
    df_temp = pd.DataFrame(all_results)
    df_temp.to_csv(csv_filename, index=False)
    print(f"  Saved intermediate results to {csv_filename}")
        
    # Create histogram for this parameter set
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        mismatches,
        bins=10,
        density=True,
        color="#5B9BD5",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.0,
    )
    
    ax.set_xlabel(r"mismatch $\mu$", fontsize=20)
    ax.set_ylabel("normalized histogram", fontsize=20)
    ax.set_xlim(0, 1)
    ax.tick_params(axis="both", which="major", labelsize=14, length=6)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    
    # Add parameter info to plot
    ax.text(0.95, 0.95, f'mf={mf:.3f}\nmf1={mf1:.3f}\nmf2={mf2:.4f}\nmean={mean_mismatch:.3f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(f"images/mismatch_distribution_lal_p{param_idx}.pdf")
    plt.close(fig)

# Save all results to CSV
df = pd.DataFrame(all_results)
csv_filename = f"data/lal_parameter_search_results_{n_param_trials}params_{N}trials.csv"
df.to_csv(csv_filename, index=False)
print(f"\nResults saved to {csv_filename}")

# Create summary statistics
summary = df.groupby(['param_idx', 'mf', 'mf1', 'mf2']).agg({
    'mismatch': ['mean', 'std', 'min', 'max'],
    'perfect_2F': 'first'
}).round(4)

print("\nSummary by parameter set:")
print(summary)

# Save summary to CSV as well
summary.to_csv(f"data/lal_parameter_search_summary_{n_param_trials}params_{N}trials.csv")

print(f"\nTotal rows in dataset: {len(df)} (expected: {n_param_trials * N})")