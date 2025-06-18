import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec

# Create output directory
label = "LALGridSearchF0F1F2"  # Changed: removed underscores
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
depth = 20
h0 = sqrtSX / depth
F0_inj = 30.0
F1_inj = -1e-10
F2_inj = 0
Alpha_inj = 1.0
Delta_inj = 1.5
cosi_inj = 0.0
psi_inj = 0.0
phi0_inj = 0.0

# Step 1: Generate SFT data with injected signal using lalpulsar_Makefakedata_v5
print("Generating SFT data with injected signal...")

sft_dir = os.path.join(outdir, "sfts")
os.makedirs(sft_dir, exist_ok=True)

injection_params = (
    f"{{Alpha={Alpha_inj}; Delta={Delta_inj}; Freq={F0_inj}; "
    f"f1dot={F1_inj}; f2dot={F2_inj}; refTime={tref}; "
    f"h0={h0}; cosi={cosi_inj}; psi={psi_inj}; phi0={phi0_inj};}}"
)

# Use a simple alphanumeric label for SFT files
sft_label = "GridF0F1F2"  # Changed: simple alphanumeric label for SFT files

makefakedata_cmd = [
    "lalpulsar_Makefakedata_v5",
    f"--IFOs={IFO}",
    f"--sqrtSX={sqrtSX}",
    f"--startTime={tstart}",
    f"--duration={duration}",
    f"--fmin={F0_inj - 1.0}",  # Set frequency band around signal
    f"--Band=2.0",
    "--Tsft=1800",
    f"--outSFTdir={sft_dir}",
    f"--outLabel={sft_label}",  # Changed: use the simple label
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

# Step 2: Set up grid search parameters
m = 0.01
dF0 = np.sqrt(12 * m) / (np.pi * duration)
dF1 = np.sqrt(180 * m) / (np.pi * duration**2)
dF2 = 1e-17
N = 100
DeltaF0 = N * dF0
DeltaF1 = N * dF1
DeltaF2 = N * dF2

F0_min = F0_inj - DeltaF0 / 2.0
F0_max = F0_inj + DeltaF0 / 2.0
F1_min = F1_inj - DeltaF1 / 2.0
F1_max = F1_inj + DeltaF1 / 2.0
F2_min = F2_inj - DeltaF2 / 2.0
F2_max = F2_inj + DeltaF2 / 2.0

# Step 3: Run ComputeFstatistic_v2 for grid search
print("\nRunning F-statistic grid search...")

output_fstat = "fstat_results.dat"  # Use relative path when running from outdir
sft_pattern = "sfts/*.sft"  # Use relative path when running from outdir

computefstat_cmd = [
    "lalpulsar_ComputeFstatistic_v2",
    f"--DataFiles={sft_pattern}",
    f"--refTime={tref}",
    f"--Alpha={Alpha_inj}",
    f"--Delta={Delta_inj}",
    f"--Freq={F0_min}",
    f"--FreqBand={DeltaF0}",
    f"--dFreq={dF0}",
    f"--f1dot={F1_min}",
    f"--f1dotBand={DeltaF1}",
    f"--df1dot={dF1}",
    f"--f2dot={F2_min}",
    f"--f2dotBand={DeltaF2}",
    f"--df2dot={dF2}",
    f"--outputFstat={output_fstat}",
    "--outputLoudest=loudest.dat",
    f"--minStartTime={tstart}",
    f"--maxStartTime={tend}"
]

# Run from within the output directory
result = subprocess.run(computefstat_cmd, capture_output=True, text=True, cwd=outdir)
if result.returncode != 0:
    print(f"Error running ComputeFstatistic_v2: {result.stderr}")
    raise RuntimeError("Failed to run F-statistic search")

print("F-statistic search completed!")

# Step 4: Parse results and find maximum
print("\nParsing results...")

# Read the F-statistic results
output_fstat_full = os.path.join(outdir, output_fstat)
# The output format of ComputeFstatistic_v2 is:
# columns: freq | alpha | delta | f1dot | f2dot | twoF
data = pd.read_csv(output_fstat_full, sep=r'\s+', comment='%', 
                   names=['freq', 'alpha', 'delta', 'f1dot', 'f2dot', 'twoF'])

# Extract values
F0_vals = data['freq'].values
F1_vals = data['f1dot'].values 
F2_vals = data['f2dot'].values
twoF_vals = data['twoF'].values

# Find maximum
max_idx = np.argmax(twoF_vals)
max_twoF = twoF_vals[max_idx]
max_F0 = F0_vals[max_idx]
max_F1 = F1_vals[max_idx]
max_F2 = F2_vals[max_idx]


print(f'max F0 = {max_F0:.4f} Hz')
print(f'max F1 = {max_F1:.4e} Hz/s')
print(f'max F2 = {max_F2:.4e} Hz/s^2')
print(f"\nMaximum 2F = {max_twoF:.4f}")
print(f"Offsets from injection:")
print(f"  F0: {max_F0 - F0_inj:.4e} Hz")
print(f"  F1: {max_F1 - F1_inj:.4e} Hz/s")
print(f"  F2: {max_F2 - F2_inj:.4e} Hz/s^2")

# Step 5: Create plots
print("\nCreating plots...")

# 1D plots
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

# Plot 2F vs F0 (at injection F1, F2)
mask_F1F2 = (np.abs(F1_vals - F1_inj) < dF1/2) & (np.abs(F2_vals - F2_inj) < dF2/2)
if np.any(mask_F1F2):
    F0_slice = F0_vals[mask_F1F2]
    twoF_slice = twoF_vals[mask_F1F2]
    sort_idx = np.argsort(F0_slice)
    axes[0].plot(F0_slice[sort_idx], twoF_slice[sort_idx], 'b-')
axes[0].axvline(F0_inj, color='r', linestyle='--', label='Injection')
axes[0].set_xlabel('Frequency [Hz]')
axes[0].set_ylabel('$2\\mathcal{F}$')
axes[0].legend()
axes[0].grid(True)

# Plot 2F vs F1 (at injection F0, F2)
mask_F0F2 = (np.abs(F0_vals - F0_inj) < dF0/2) & (np.abs(F2_vals - F2_inj) < dF2/2)
if np.any(mask_F0F2):
    F1_slice = F1_vals[mask_F0F2]
    twoF_slice = twoF_vals[mask_F0F2]
    sort_idx = np.argsort(F1_slice)
    axes[1].plot(F1_slice[sort_idx], twoF_slice[sort_idx], 'b-')
axes[1].axvline(F1_inj, color='r', linestyle='--', label='Injection')
axes[1].set_xlabel('$\\dot{f}$ [Hz/s]')
axes[1].set_ylabel('$2\\mathcal{F}$')
axes[1].legend()
axes[1].grid(True)

# Plot 2F vs F2 (at injection F0, F1)
mask_F0F1 = (np.abs(F0_vals - F0_inj) < dF0/2) & (np.abs(F1_vals - F1_inj) < dF1/2)
if np.any(mask_F0F1):
    F2_slice = F2_vals[mask_F0F1]
    twoF_slice = twoF_vals[mask_F0F1]
    sort_idx = np.argsort(F2_slice)
    axes[2].plot(F2_slice[sort_idx], twoF_slice[sort_idx], 'b-')
axes[2].axvline(F2_inj, color='r', linestyle='--', label='Injection')
axes[2].set_xlabel('$\\ddot{f}$ [Hz/s$^2$]')
axes[2].set_ylabel('$2\\mathcal{F}$')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "1D_projections.png"))
plt.close()

# Create grid corner plot
print("Creating grid corner plot...")

# Get unique values for each parameter
F0_unique = np.sort(np.unique(F0_vals))
F1_unique = np.sort(np.unique(F1_vals))
F2_unique = np.sort(np.unique(F2_vals))

# Shifted values for plotting
F0_shifted = F0_unique - F0_inj
F1_shifted = F1_unique - F1_inj
F2_shifted = F2_unique - F2_inj

# Reshape twoF values into 3D array
n_F0 = len(F0_unique)
n_F1 = len(F1_unique)
n_F2 = len(F2_unique)

# Create a mapping from parameter values to indices
tolerance = 1e-10  # Small tolerance for floating point comparison
def find_index(val, unique_vals):
    idx = np.argmin(np.abs(unique_vals - val))
    if np.abs(unique_vals[idx] - val) > tolerance:
        print(f"Warning: no exact match found for value {val}")
    return idx

# Initialize 3D array
twoF_3D = np.zeros((n_F0, n_F1, n_F2))

# Fill the 3D array
for i in range(len(F0_vals)):
    i_F0 = find_index(F0_vals[i], F0_unique)
    i_F1 = find_index(F1_vals[i], F1_unique)
    i_F2 = find_index(F2_vals[i], F2_unique)
    twoF_3D[i_F0, i_F1, i_F2] = twoF_vals[i]

# Create corner plot
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.1, wspace=0.1)

# 2D projections
# F0-F1
ax_F0F1 = fig.add_subplot(gs[1, 0])
Z_F0F1 = np.max(twoF_3D, axis=2)
im = ax_F0F1.imshow(Z_F0F1.T, origin='lower', aspect='auto',
                    extent=[F0_shifted[0], F0_shifted[-1], F1_shifted[0], F1_shifted[-1]],
                    cmap='viridis')
ax_F0F1.set_xlabel('$f - f_0$ [Hz]')
ax_F0F1.set_ylabel('$\\dot{f} - \\dot{f}_0$ [Hz/s]')

# F0-F2
ax_F0F2 = fig.add_subplot(gs[2, 0])
Z_F0F2 = np.max(twoF_3D, axis=1)
ax_F0F2.imshow(Z_F0F2.T, origin='lower', aspect='auto',
               extent=[F0_shifted[0], F0_shifted[-1], F2_shifted[0], F2_shifted[-1]],
               cmap='viridis')
ax_F0F2.set_xlabel('$f - f_0$ [Hz]')
ax_F0F2.set_ylabel('$\\ddot{f} - \\ddot{f}_0$ [Hz/s$^2$]')

# F1-F2
ax_F1F2 = fig.add_subplot(gs[2, 1])
Z_F1F2 = np.max(twoF_3D, axis=0)
ax_F1F2.imshow(Z_F1F2.T, origin='lower', aspect='auto',
               extent=[F1_shifted[0], F1_shifted[-1], F2_shifted[0], F2_shifted[-1]],
               cmap='viridis')
ax_F1F2.set_xlabel('$\\dot{f} - \\dot{f}_0$ [Hz/s]')
ax_F1F2.set_ylabel('$\\ddot{f} - \\ddot{f}_0$ [Hz/s$^2$]')

# 1D projections
# F0
ax_F0 = fig.add_subplot(gs[0, 0])
ax_F0.plot(F0_shifted, np.max(np.max(twoF_3D, axis=2), axis=1))
ax_F0.set_ylabel('$2\\mathcal{F}$')
ax_F0.set_xticklabels([])

# F1
ax_F1 = fig.add_subplot(gs[1, 1])
ax_F1.plot(F1_shifted, np.max(np.max(twoF_3D, axis=2), axis=0))
ax_F1.set_ylabel('$2\\mathcal{F}$')
ax_F1.set_xticklabels([])

# F2
ax_F2 = fig.add_subplot(gs[2, 2])
ax_F2.plot(F2_shifted, np.max(np.max(twoF_3D, axis=1), axis=0))
ax_F2.set_xlabel('$\\ddot{f} - \\ddot{f}_0$ [Hz/s$^2$]')
ax_F2.set_ylabel('$2\\mathcal{F}$')

plt.colorbar(im, ax=[ax_F0F1, ax_F0F2, ax_F1F2], label='$2\\mathcal{F}$')
plt.savefig(os.path.join(outdir, "grid_corner_plot.png"))
plt.close()

print(f"\nAll results saved to {outdir}")