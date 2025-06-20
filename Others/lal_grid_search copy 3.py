import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec

# Create output directory
label = "LALSemiCoherentF0F1F22"
outdir = os.path.join("LAL_example_data", label)
os.makedirs(outdir, exist_ok=True)

# Properties of the GW data
sqrtSX = 1e-23
tstart = 1000000000
duration = 10 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)
IFO = "H1"

# Parameters for injected signals
depth = 5
h0 = sqrtSX / depth
F0_inj = 30.0
F1_inj = -1e-10
F2_inj = 0
Alpha_inj = 1.0
Delta_inj = 1.5
cosi_inj = 0.0
psi_inj = 0.0
phi0_inj = 0.0

# Semi-coherent search parameters
tStack = 86400  # 1 day coherent segments
nStacks = int(duration / tStack)  # Number of segments

# Step 1: Generate SFT data with injected signal using lalpulsar_Makefakedata_v5
print("Generating SFT data with injected signal...")

sft_dir = os.path.join(outdir, "sfts")
os.makedirs(sft_dir, exist_ok=True)

injection_params = (
    f"{{Alpha={Alpha_inj}; Delta={Delta_inj}; Freq={F0_inj}; "
    f"f1dot={F1_inj}; f2dot={F2_inj}; refTime={tref}; "
    f"h0={h0}; cosi={cosi_inj}; psi={psi_inj}; phi0={phi0_inj};}}"
)

sft_label = "SemiCohF0F1F22"

makefakedata_cmd = [
    "lalpulsar_Makefakedata_v5",
    f"--IFOs={IFO}",
    f"--sqrtSX={sqrtSX}",
    f"--startTime={tstart}",
    f"--duration={duration}",
    f"--fmin={F0_inj - 1.0}",
    f"--Band=2.0",
    "--Tsft=1800",
    f"--outSFTdir={sft_dir}",
    f"--outLabel={sft_label}",
    f"--injectionSources={injection_params}",
    "--randSeed=42"
]

result = subprocess.run(makefakedata_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error generating SFTs: {result.stderr}")
    raise RuntimeError("Failed to generate SFTs")

print("SFTs generated successfully!")

# Check if SFTs were actually created
import glob
sft_files = glob.glob(os.path.join(sft_dir, "*.sft"))
print(f"Found {len(sft_files)} SFT files")

# Step 2: Set up grid search parameters for semi-coherent search
# For semi-coherent search, the mismatch is different due to shorter coherent time
mf = 0.1  # Coherent mismatch for F0
mf1 = 0.1  # Coherent mismatch for F1 and F2
mf2 = 0.003  # Coherent mismatch for F2
# m_semi = 0.2  # Semi-coherent mismatch
def format_scientific(x):
    """Format number to avoid scientific notation issues"""
    if abs(x) < 1e-4 and x != 0:
        # Use fixed notation with enough decimal places
        return f"{x:.15f}"
    else:
        # Use regular formatting
        return f"{x:.15g}"

# Calculate grid spacings based on coherent segment length (tStack)
dF0 = np.sqrt(12 * mf) / (np.pi * tStack)
dF1 = np.sqrt(180 * mf1) / (np.pi * tStack**2)
dF2 = np.sqrt(25200 * mf2) / (np.pi * tStack**3)

N1 = 10
N2 = 10
N3 = 4

DeltaF0 = N1 * dF0
DeltaF1 = N2 * dF1
DeltaF2 = N3 * dF2

F0_min = F0_inj - DeltaF0 / 2.0
F0_max = F0_inj + DeltaF0 / 2.0
F1_min = F1_inj - DeltaF1 / 2.0
F1_max = F1_inj + DeltaF1 / 2.0
F2_min = F2_inj - DeltaF2 / 2.0
F2_max = F2_inj + DeltaF2 / 2.0

# Step 3: Run HierarchSearchGCT for semi-coherent search
print("\nRunning semi-coherent F-statistic search...")

output_file = "semicoherent_results.dat"
sft_pattern = os.path.join(sft_dir, "*.sft")

# Create a simple sky grid file for single point search
skygrid_file = os.path.join(outdir, "skygrid.dat")
with open(skygrid_file, 'w') as f:
    f.write(f"{Alpha_inj} {Delta_inj}\n")

hierarchsearch_cmd = [
    "lalpulsar_HierarchSearchGCT",
    f"--DataFiles1={sft_pattern}",
    f"--skyGridFile={skygrid_file}",
    f"--refTime={format_scientific(tref)}",
    f"--Freq={format_scientific(F0_min)}",
    f"--FreqBand={format_scientific(DeltaF0)}",
    f"--dFreq={format_scientific(dF0)}",
    f"--f1dot={format_scientific(F1_min)}",
    f"--f1dotBand={format_scientific(DeltaF1)}",
    f"--df1dot={format_scientific(dF1)}",
    f"--f2dot={format_scientific(F2_min)}",
    f"--f2dotBand={format_scientific(DeltaF2)}",
    f"--df2dot={format_scientific(dF2)}",
    f"--nStacksMax={nStacks}",
    f"--tStack={tStack}",
    f"--mismatch1={mf}",
    f"--fnameout={os.path.join(outdir, output_file)}",
    f"--nCand1=1000",  # Keep top 1000 candidates
    "--printCand1=TRUE",
    "--semiCohToplist=TRUE",
    f"--minStartTime1={tstart}",
    f"--maxStartTime1={tend}",
    "--FstatMethod=ResampBest",
    "--computeBSGL=FALSE"
]

result = subprocess.run(hierarchsearch_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error running HierarchSearchGCT: {result.stderr}")
    print(f"stdout: {result.stdout}")
    raise RuntimeError("Failed to run semi-coherent search")

print("Semi-coherent search completed!")

# Step 4: Parse results
print("\nParsing results...")

# Read the toplist output
output_file_full = os.path.join(outdir, output_file)

# HierarchSearchGCT output format includes header lines
# Read the file and parse the toplist section
with open(output_file_full, 'r') as f:
    lines = f.readlines()

# Find the toplist section
toplist_start = None
for i, line in enumerate(lines):
    if 'freq' in line and 'Alpha' in line and '2F' in line:
        toplist_start = i
        break

if toplist_start is None:
    print("Could not find toplist in output file")
    # Try alternative parsing
    data = []
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
                    twoF = float(parts[6])
                    data.append([freq, f1dot, f2dot, twoF])
                except:
                    pass
    
    if not data:
        raise RuntimeError("Could not parse output file")
    
    data = np.array(data)
    F0_vals = data[:, 0]
    F1_vals = data[:, 1]
    F2_vals = data[:, 2]
    twoF_vals = data[:, 3]
else:
    # Parse from identified toplist
    data = []
    for line in lines[toplist_start+1:]:
        if line.strip() and not line.startswith('%'):
            parts = line.split()
            if len(parts) >= 7:
                try:
                    data.append([float(x) for x in parts[:7]])
                except:
                    break
    
    data = np.array(data)
    F0_vals = data[:, 0]
    F1_vals = data[:, 3]
    F2_vals = data[:, 4]
    twoF_vals = data[:, 6]

# Find maximum
if len(twoF_vals) > 0:
    max_idx = np.argmax(twoF_vals)
    max_twoF = twoF_vals[max_idx]
    max_F0 = F0_vals[max_idx]
    max_F1 = F1_vals[max_idx]
    max_F2 = F2_vals[max_idx]

    print(f"\nSemi-coherent search results:")
    print(f"Maximum 2F = {max_twoF:.4f}")
    print(f"Found at:")
    print(f"  F0 = {max_F0:.6f} Hz")
    print(f"  F1 = {max_F1:.4e} Hz/s")
    print(f"  F2 = {max_F2:.4e} Hz/s^2")
    print(f"\nOffsets from injection:")
    print(f"  F0: {max_F0 - F0_inj:.4e} Hz")
    print(f"  F1: {max_F1 - F1_inj:.4e} Hz/s")
    print(f"  F2: {max_F2 - F2_inj:.4e} Hz/s^2")
else:
    print("No candidates found in toplist")

# Step 5: Create comparison plots if we have enough data
if len(F0_vals) > 10:
    print("\nCreating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 2F vs F0
    axes[0, 0].scatter(F0_vals, twoF_vals, alpha=0.6)
    axes[0, 0].axvline(F0_inj, color='r', linestyle='--', label='Injection')
    axes[0, 0].set_xlabel('Frequency [Hz]')
    axes[0, 0].set_ylabel('$2\\mathcal{F}$')
    axes[0, 0].set_title('Semi-coherent 2F vs Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2F vs F1
    axes[0, 1].scatter(F1_vals, twoF_vals, alpha=0.6)
    axes[0, 1].axvline(F1_inj, color='r', linestyle='--', label='Injection')
    axes[0, 1].set_xlabel('$\\dot{f}$ [Hz/s]')
    axes[0, 1].set_ylabel('$2\\mathcal{F}$')
    axes[0, 1].set_title('Semi-coherent 2F vs Spindown')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 2F vs F2
    axes[1, 0].scatter(F2_vals, twoF_vals, alpha=0.6)
    axes[1, 0].axvline(F2_inj, color='r', linestyle='--', label='Injection')
    axes[1, 0].set_xlabel('$\\ddot{f}$ [Hz/s$^2$]')
    axes[1, 0].set_ylabel('$2\\mathcal{F}$')
    axes[1, 0].set_title('Semi-coherent 2F vs Second Spindown')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 2F distribution
    axes[1, 1].hist(twoF_vals, bins=30, alpha=0.7)
    axes[1, 1].set_xlabel('$2\\mathcal{F}$')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of 2F values')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "semicoherent_results.png"))
    plt.close()

print(f"\nAll results saved to {outdir}")
print(f"\nSemi-coherent search summary:")
print(f"- Used {nStacks} segments of {tStack/86400:.1f} days each")
print(f"- Total observation time: {duration/86400:.1f} days")
print(f"- Coherent mismatch per segment: {mf}")
print(f"- Grid spacings: dF0={dF0:.6f} Hz, dF1={dF1:.2e} Hz/s, dF2={dF2:.2e} Hz/s^2")