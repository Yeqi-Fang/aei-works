import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn


# Create output directory
N = 500
print_output = False  # Set to False to suppress output
label = "LALSemiCoherentF0F1F2_corrected_fast"
outdir = os.path.join("LAL_example_data", label)
os.makedirs(outdir, exist_ok=True)

# Properties of the GW data
sqrtSX = 1e-22
tstart = 1126051217
duration = 120 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)
IFO = "H1"  # Interferometers to use

# Parameters for injected signals
depth = 0.02
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
tStack = 20 * 86400  # 15 day coherent segments
nStacks = int(duration / tStack)  # Number of segments

# Step 1: Generate SFT data
# print("Generating SFT data with injected signal...")

sft_dir = os.path.join(outdir, "sfts")
os.makedirs(sft_dir, exist_ok=True)
os.makedirs(os.path.join(outdir, "dats"), exist_ok=True)
os.makedirs(os.path.join(outdir, "commands"), exist_ok=True)
sft_pattern = os.path.join(sft_dir, "*.sft")

injection_params = (
    f"{{Alpha={Alpha_inj:.15g}; Delta={Delta_inj:.15g}; Freq={F0_inj:.15g}; "
    f"f1dot={F1_inj:.15e}; f2dot={F2_inj:.15e}; refTime={tref:.15g}; "
    f"h0={h0:.15e}; cosi={cosi_inj:.15g}; psi={psi_inj:.15g}; phi0={phi0_inj:.15g};}}"
)

sft_label = "SemiCoh"

makefakedata_cmd = [
    "lalpulsar_Makefakedata_v5",
    f"--IFOs={IFO}",
    f"--sqrtSX={sqrtSX:.15e}",
    f"--startTime={int(tstart)}",
    f"--duration={int(duration)}",
    f"--fmin={F0_inj - 0.1:.15g}",
    f"--Band=0.2",
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

# Step 2: Create segment list file (CRITICAL!)
# segFile = os.path.join(outdir, "segments.dat")
# with open(segFile, 'w') as f:
#     for i in range(nStacks):
#         seg_start = tstart + i * tStack
#         seg_end = seg_start + tStack
#         nsft = int(tStack / 1800)  # Number of SFTs in segment
#         f.write(f"{int(seg_start)} {int(seg_end)} {nsft}\n")

# print(f"Created segment file with {nStacks} segments")

# Step 3: Set up grid search parameters
mf = 0.01
mf1 = 0.001
mf2 = 0.08
dF0 = np.sqrt(12 * mf) / (np.pi * tStack)
dF1 = np.sqrt(180 * mf1) / (np.pi * tStack**2)
df2 = np.sqrt(25200 * mf2) / (np.pi * tStack**3)

# Search bands
N1 = 10
N2 = 5
N3 = 5
gamma1 = 11
gamma2 = 91


DeltaF0 = N1 * dF0
DeltaF1 = N2 * dF1
DeltaF2 = N3 * df2

F0_randoms = np.random.uniform(- dF0 / 2.0, dF0 / 2.0, size=N)
F1_randoms = np.random.uniform(- dF1 / 2.0, dF1 / 2.0, size=N)
F2_randoms = np.random.uniform(- df2 / 2.0, df2 / 2.0, size=N)



shared_cmd = [
    f"--DataFiles1={sft_pattern}",
    "--gridType1=3",  # IMPORTANT: 3=file mode for sky grid
    f"--skyGridFile={{{Alpha_inj} {Delta_inj}}}",
    f"--refTime={tref:.15f}",
    f"--tStack={tStack:.15g}",
    f"--nStacksMax={nStacks}",
    "--nCand1=10",
    "--printCand1",
    "--semiCohToplist",
    f"--minStartTime1={int(tstart)}",
    f"--maxStartTime1={int(tend)}",
    "--recalcToplistStats=TRUE",
    "--FstatMethod=DemodBest",
    "--FstatMethodRecalc=DemodOptC",
    "--Dterms=8",
    # "--peakThrF=2.6",
    # "--computeBSGL",
    # "--oLGX=0.5,0.5",
    # "--BSGLlogcorr=0",  # Disable BSG log correction    
    # "--Fstar0=24",
]

def single_run(i):


    F0_min = F0_inj - DeltaF0 / 2.0 + F0_randoms[i]
    F0_max = F0_inj + DeltaF0 / 2.0 + F0_randoms[i]
    F1_min = F1_inj - DeltaF1 / 2.0 + F1_randoms[i]
    F1_max = F1_inj + DeltaF1 / 2.0 + F1_randoms[i]
    F2_min = F2_inj - DeltaF2 / 2.0 + F2_randoms[i]
    F2_max = F2_inj + DeltaF2 / 2.0 + F2_randoms[i]

    output_file = os.path.join(outdir, f"dats/semicoh_results_{i}.dat")
    

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
    ] + shared_cmd

    # Save command for debugging
    # cmd_file = os.path.join(outdir, f"commands/command{i}.sh")
    # with open(cmd_file, 'w') as f:
    #     f.write("#!/bin/bash\n")
    #     f.write(" \\\n    ".join(hierarchsearch_cmd))
    #     f.write("\n")
    # os.chmod(cmd_file, 0o755)

    # print("Running command (saved to command.sh)...")
    result = subprocess.run(hierarchsearch_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running HierarchSearchGCT:")
        print(f"stderr: {result.stderr}")
        print(f"stdout: {result.stdout}")
        raise RuntimeError("Failed to run semi-coherent search")

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
    
    task = progress.add_task("Processing runs", total=N)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(single_run, i) for i in range(N)]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Wait for each future to complete
                progress.advance(task, 1)
            except Exception as e:
                print(f"Error in thread: {e}")
                progress.advance(task, 1)


results = []
max_twoFs = []

for i in range(N):
    output_file = os.path.join(outdir, f"dats/semicoh_results_{i}.dat")
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist, skipping.")
        continue
    with open(output_file, 'r') as f:
        lines = f.readlines()
    # Look for the data section
        data = []
        in_data = False
        for line in lines:
            if line.strip() and not line.startswith('%'):
                parts = line.split()
                if len(parts) >= 7:
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
            F0_vals = data[:, 0]
            F1_vals = data[:, 1]
            F2_vals = data[:, 2]
            twoF_vals = data[:, 3]
            
            # Find maximum
            max_idx = np.argmax(twoF_vals)
            max_twoF = twoF_vals[max_idx]
            max_F0 = F0_vals[max_idx]
            max_F1 = F1_vals[max_idx]
            max_F2 = F2_vals[max_idx]

    max_twoFs.append(max_twoF)

perfect_output_file = os.path.join(outdir, "perfectly_matched_results.dat")

F0_min = F0_inj - DeltaF0 / 2.0
F0_max = F0_inj + DeltaF0 / 2.0
F1_min = F1_inj - DeltaF1 / 2.0
F1_max = F1_inj + DeltaF1 / 2.0
F2_min = F2_inj - DeltaF2 / 2.0
F2_max = F2_inj + DeltaF2 / 2.0


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
    f"--gammaRefine=8",
    f"--gamma2Refine=20",
    
] + shared_cmd 



# 运行命令并捕获输出
result = subprocess.run(perfect_search_cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"Error running lalpulsar_HierarchSearchGCT: {result.stderr}")
    raise RuntimeError("Failed to compute perfectly matched 2F value")

with open(perfect_output_file, 'r') as f:
    lines = f.readlines()

# Look for the data section
data = []
in_data = False
for line in lines:
    if line.strip() and not line.startswith('%'):
        parts = line.split()
        if len(parts) >= 7:
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
    F0_vals = data[:, 0]
    F1_vals = data[:, 1]
    F2_vals = data[:, 2]
    twoF_vals = data[:, 3]
    
    # Find maximum
    max_idx = np.argmax(twoF_vals)
    perfect_2F = twoF_vals[max_idx]
    
    
mismatches = []
for i in range(N):
    max_twoF = max_twoFs[i]
    mismatches.append((perfect_2F - max_twoF) / (perfect_2F - 4))


fig, ax = plt.subplots(figsize=(10, 6))
# choose bin edges so the last bar ends at 1.0, like in the photo
# bins = np.linspace(0, 1, 11)        # 10 equal-width bins → 11 edges
ax.hist(
    mismatches,
    bins=10,
    density=True,
    color="#5B9BD5",  # pleasant blue
    alpha=0.85,
    edgecolor="black",
    linewidth=1.0,
)

# axis labels & limits
ax.set_xlabel(r"mismatch $\mu$", fontsize=20)
ax.set_ylabel("normalized histogram", fontsize=20)
ax.set_xlim(0, 1)
# ax.set_ylim(0, 0.25)

# ticks & grid
ax.tick_params(axis="both", which="major", labelsize=14, length=6)
ax.grid(axis="y", linewidth=0.6, alpha=0.35)

fig.tight_layout()
fig.savefig(f"images/mismatch_distribution_lal-{mf}-{mf1}-{mf2}-{N}-{depth}.pdf")
# plt.show()

# rumtime
# print("runtime: ", config.tau_total)

print("mean mismatch: ", np.mean(mismatches))
