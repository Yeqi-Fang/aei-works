import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create output directory
label = "LALSemiCoherentF0F1F2_corrected0"
outdir = os.path.join("LAL_example_data", label)
os.makedirs(outdir, exist_ok=True)

# Properties of the GW data
sqrtSX = 1e-22
tstart = 1000000000
duration = 120 * 86400
tend = tstart + duration
tref = 0.5 * (tstart + tend)
IFO = "H1, L1"  # Interferometers to use

# Parameters for injected signals
depth = 0.2
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

# Step 1: Generate SFT data
print("Generating SFT data with injected signal...")

sft_dir = os.path.join(outdir, "sfts")
os.makedirs(sft_dir, exist_ok=True)

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
    f"--fmin={F0_inj - 1.0:.15g}",
    f"--Band=2.0",
    "--Tsft=1800",
    f"--outSFTdir={sft_dir}",
    f"--outLabel={sft_label}",
    f"--injectionSources={injection_params}",
]

result = subprocess.run(makefakedata_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error generating SFTs: {result.stderr}")
    raise RuntimeError("Failed to generate SFTs")

print("SFTs generated successfully!")

# Step 2: Create segment list file (CRITICAL!)
segFile = os.path.join(outdir, "segments.dat")
with open(segFile, 'w') as f:
    for i in range(nStacks):
        seg_start = tstart + i * tStack
        seg_end = seg_start + tStack
        nsft = int(tStack / 1800)  # Number of SFTs in segment
        f.write(f"{int(seg_start)} {int(seg_end)} {nsft}\n")

print(f"Created segment file with {nStacks} segments")

# Step 3: Set up grid search parameters
mf = 0.15
mf1 = 0.3
mf2 = 0.003
dF0 = np.sqrt(12 * mf) / (np.pi * tStack)
dF1 = np.sqrt(180 * mf1) / (np.pi * tStack**2)
df2 = np.sqrt(25200 * mf2) / (np.pi * tStack**3)

# Search bands
N1 = 10
N2 = 10
N3 = 20
gamma1 = 8
gamma2 = 20


DeltaF0 = N1 * dF0
DeltaF1 = N2 * dF1
DeltaF2 = N3 * df2

F0_random = np.random.uniform(- dF0 / 2.0, dF0 / 2.0)
F1_random = np.random.uniform(- dF1 / 2.0, dF1 / 2.0)
F2_random = np.random.uniform(- df2 / 2.0, df2 / 2.0)

F0_min = F0_inj - DeltaF0 / 2.0 + F0_random
F0_max = F0_inj + DeltaF0 / 2.0 + F0_random
F1_min = F1_inj - DeltaF1 / 2.0 + F1_random
F1_max = F1_inj + DeltaF1 / 2.0 + F1_random
F2_min = F2_inj - DeltaF2 / 2.0 + F2_random
F2_max = F2_inj + DeltaF2 / 2.0 + F2_random

print(f"\nGrid parameters:")
print(f"F0 range: [{F0_min:.6f}, {F0_max:.6f}] Hz")
print(f"F1 range: [{F1_min:.6e}, {F1_max:.6e}] Hz/s")
print(f"dF0 = {dF0:.6e} Hz")
print(f"dF1 = {dF1:.6e} Hz/s")

# Step 4: Create sky grid file
skygrid_file = os.path.join(outdir, "skygrid.dat")
with open(skygrid_file, 'w') as f:
    f.write(f"{Alpha_inj:.15g} {Delta_inj:.15g}\n")

# Step 5: Run HierarchSearchGCT
print("\nRunning semi-coherent F-statistic search...")

output_file = os.path.join(outdir, "semicoh_results.dat")
sft_pattern = os.path.join(sft_dir, "*.sft")

# Build command with proper formatting
hierarchsearch_cmd = [
    "lalpulsar_HierarchSearchGCT",
    f"--DataFiles1={sft_pattern}",
    "--gridType1=3",  # IMPORTANT: 3=file mode for sky grid
    f"--skyGridFile={{{Alpha_inj} {Delta_inj}}}",
    f"--refTime={tref:.15g}",
    f"--Freq={F0_min:.15g}",
    f"--FreqBand={DeltaF0:.15g}",
    f"--dFreq={dF0:.15e}",
    f"--f1dot={F1_min:.15e}",
    f"--f1dotBand={DeltaF1:.15e}",
    f"--df1dot={dF1:.15e}",
    f"--f2dot={F2_min:.15e}",
    f"--f2dotBand={DeltaF2:.15e}",
    f"--df2dot={df2:.15e}",
    f"--tStack={tStack:.15g}",
    f"--nStacksMax={nStacks}",
    f"--mismatch1={mf:.15g}",
    f"--fnameout={output_file}",
    "--nCand1=1000",
    "--printCand1",
    "--semiCohToplist",
    f"--minStartTime1={int(tstart)}",
    f"--maxStartTime1={int(tend)}",
    f"--gammaRefine={gamma1:.15g}",
    f"--gamma2Refine={gamma2:.15g}",
    "--recalcToplistStats=TRUE",
    "--FstatMethod=ResampBest",
    "--FstatMethodRecalc=DemodBest",
    # "--peakThrF=2.6",
    # "--SortToplist=0",
    # "--computeBSGL",
    # "--oLGX=0.5,0.5",
    # "--BSGLlogcorr=0",  
]

# Save command for debugging
cmd_file = os.path.join(outdir, "command.sh")
with open(cmd_file, 'w') as f:
    f.write("#!/bin/bash\n")
    f.write(" \\\n    ".join(hierarchsearch_cmd))
    f.write("\n")
os.chmod(cmd_file, 0o755)

print("Running command (saved to command.sh)...")
result = subprocess.run(hierarchsearch_cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"Error running HierarchSearchGCT:")
    print(f"stderr: {result.stderr}")
    print(f"stdout: {result.stdout}")
    raise RuntimeError("Failed to run semi-coherent search")

print("Semi-coherent search completed!")

# Step 6: Parse results
print("\nParsing results...")

# Read and parse the output file
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
    
    
    # save to CSV file
    results_csv = os.path.join(outdir, "semicoh_results.csv")
    df = pd.DataFrame(data, columns=['F0', 'F1', 'F2', '2F'])
    df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")
    
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
    
    print(f"\nSemi-coherent search results:")
    print(f"Maximum 2F = {max_twoF:.4f}")
    print(f"Found at:")
    print(f"  F0 = {max_F0:.6f} Hz (injection: {F0_inj} Hz)")
    print(f"  F1 = {max_F1:.4e} Hz/s (injection: {F1_inj:.4e} Hz/s)")
    print(f"  F2 = {max_F2:.4e} Hz/s^2 (injection: {F2_inj:.4e} Hz/s^2)")
    print(f"\nOffsets from injection:")
    print(f"  ΔF0: {max_F0 - F0_inj:.4e} Hz")
    print(f"  ΔF1: {max_F1 - F1_inj:.4e} Hz/s")
    print(f"  ΔF2: {max_F2 - F2_inj:.4e} Hz/s^2")
    
    # Create plots
    if len(F0_vals) > 10:
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
        
        # F0 vs F1
        # axes[1, 0].scatter(F0_vals, F1_vals, c=twoF_vals, cmap='viridis', alpha=0.6)
        # axes[1, 0].axvline(F0_inj, color='r', linestyle='--', alpha=0.5)
        # axes[1, 0].axhline(F1_inj, color='r', linestyle='--', alpha=0.5)
        # axes[1, 0].set_xlabel('Frequency [Hz]')
        # axes[1, 0].set_ylabel('$\\dot{f}$ [Hz/s]')
        # axes[1, 0].set_title('Parameter Space')
        axes[1, 0].scatter(F2_vals, twoF_vals, alpha=0.6)
        axes[1, 0].axvline(F2_inj, color='r', linestyle='--', label='Injection')
        axes[1, 0].set_xlabel('$\\ddot{f}$ [Hz/$s^2$]')
        axes[1, 0].set_ylabel('$2\\mathcal{F}$')
        axes[1, 0].set_title('Semi-coherent 2F vs Spindown')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 2F distribution
        axes[1, 1].hist(twoF_vals, bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('$2\\mathcal{F}$')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of 2F values')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "semicoh_results.png"))
        # plt.close()
else:
    print("No candidates found in output file")

print(f"\nAll results saved to {outdir}")


# 在现有代码的基础上添加以下部分

# Step 7: 计算perfectly matched点的2F值
print("\nStep 7: Computing 2F at perfectly matched (injected) point...")

# 准备ComputeFstatistic_v2命令
# computeF_cmd = [
#     "lalpulsar_ComputeFstatistic_v2",
#     f"--DataFiles={sft_pattern}",
#     f"--refTime={tref:.15g}",
#     f"--Alpha={Alpha_inj:.15g}",
#     f"--Delta={Delta_inj:.15g}",
#     f"--Freq={F0_inj:.15g}",
#     f"--f1dot={F1_inj:.15e}",
#     f"--f2dot={F2_inj:.15e}",
#     "--outputLoudest=loudest2.dat",
#     f"--minStartTime={tstart}",
#     f"--maxStartTime={tend}"
# ]
perfect_output_file = os.path.join(outdir, "perfectly_matched_results.dat")

F0_min = F0_inj - DeltaF0 / 2.0
F0_max = F0_inj + DeltaF0 / 2.0
F1_min = F1_inj - DeltaF1 / 2.0
F1_max = F1_inj + DeltaF1 / 2.0
F2_min = F2_inj - DeltaF2 / 2.0
F2_max = F2_inj + DeltaF2 / 2.0


perfect_search_cmd = [
    "lalpulsar_HierarchSearchGCT",
    f"--DataFiles1={sft_pattern}",
    "--gridType1=3",  # IMPORTANT: 3=file mode for sky grid
    f"--skyGridFile={{{Alpha_inj} {Delta_inj}}}",
    f"--refTime={tref:.15g}",
    f"--Freq={F0_inj:.15g}",
    "--FreqBand=0",
    f"--dFreq={dF0:.15e}",
    f"--f1dot={F1_inj:.15e}",
    "--f1dotBand=0",
    f"--df1dot={dF1:.15e}",
    f"--f2dot={F2_inj:.15e}",
    "--f2dotBand=0",
    f"--df2dot={df2:.15e}",
    f"--tStack={tStack:.15g}",
    f"--nStacksMax={nStacks}",
    "--mismatch1=0.2",
    f"--fnameout={perfect_output_file}",
    "--nCand1=1000",
    "--printCand1",
    "--semiCohToplist",
    f"--minStartTime1={int(tstart)}",
    f"--maxStartTime1={int(tend)}",
    "--Dterms=8",
    "--blocksRngMed=101",  # Running median window
    f"--gammaRefine={gamma1:.15g}",
    f"--gamma2Refine={gamma2:.15g}",
    "--recalcToplistStats=TRUE",
    "--FstatMethod=ResampBest",
    "--FstatMethodRecalc=DemodBest",
    # "--peakThrF=2.6",
    # "--SortToplist=0",
    # "--computeBSGL",
    # "--oLGX=0.5,0.5",
    # "--BSGLlogcorr=0",  # Disable BSG log correction    
]


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
    perfect_2F = twoF_vals[max_idx]
    


print("perfectly match: 2F = ", perfect_2F)
print("found maximum: 2F = ", max_twoF)
mismatch = (perfect_2F - max_twoF) / (perfect_2F - 4)

print(f"\nMismatch between perfectly matched point and found maximum: {mismatch:.6f}")