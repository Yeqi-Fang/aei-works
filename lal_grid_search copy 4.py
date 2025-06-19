import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create output directory
label = "LALSemiCoherentF0F1F2_corrected"
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
depth = 0.45
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
    "--randSeed=42"
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
m_coh = 0.05
dF0 = np.sqrt(12 * m_coh) / (np.pi * tStack)
dF1 = np.sqrt(180 * m_coh) / (np.pi * tStack**2)
df2 = np.sqrt(25200 * m_coh) / (np.pi * tStack**3)

# Search bands
N1 = 10
N2 = 10
N3 = 10
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
    f"--skyGridFile={skygrid_file}",
    "--skyRegion=allsky",  # IMPORTANT: needed even with sky grid file
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
    "--mismatch1=0.2",
    f"--fnameout={output_file}",
    "--nCand1=1000",
    "--printCand1",
    "--semiCohToplist",
    f"--minStartTime1={int(tstart)}",
    f"--maxStartTime1={int(tend)}",
    "--FstatMethod=ResampBest",
    "--computeBSGL=FALSE",
    "--Dterms=8",
    "--blocksRngMed=101",  # Running median window
    f"--gammaRefine={gamma1:.15g}",
    f"--gamma2Refine={gamma2:.15g}",
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
                f3dot = float(parts[5])
                twoF = float(parts[6])
                data.append([freq, f1dot, f2dot, twoF])
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
        plt.close()
else:
    print("No candidates found in output file")

print(f"\nAll results saved to {outdir}")


# 在现有代码的基础上添加以下部分

# Step 7: 计算perfectly matched点的2F值
print("\nStep 7: Computing 2F at perfectly matched (injected) point...")

# 准备ComputeFstatistic_v2命令
computeF_cmd = [
    "lalpulsar_ComputeFstatistic_v2",
    f"--DataFiles={sft_pattern}",
    f"--refTime={tref:.15g}",
    f"--Alpha={Alpha_inj:.15g}",
    f"--Delta={Delta_inj:.15g}",
    f"--Freq={F0_inj:.15g}",
    f"--f1dot={F1_inj:.15e}",
    f"--f2dot={F2_inj:.15e}",
    "--outputLoudest=loudest2.dat",
    f"--minStartTime={tstart}",
    f"--maxStartTime={tend}"
]

# 运行命令并捕获输出
result = subprocess.run(computeF_cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"Error running ComputeFstatistic_v2: {result.stderr}")
else:
    # 从输出中提取2F值
    output_lines = result.stdout.strip().split('\n')
    print(result.stdout)  # 打印输出以便调试
    perfect_2F = None
    
    for line in output_lines:
        if "twoF" in line or "2F" in line:
            # 尝试从输出中提取2F值
            parts = line.split()
            for i, part in enumerate(parts):
                if "2F" in part and i+1 < len(parts):
                    try:
                        perfect_2F = float(parts[i+1])
                        break
                    except:
                        continue
            if perfect_2F is not None:
                break
    
    # 如果没有找到，尝试另一种方法：使用outputLoudest文件
    # if perfect_2F is None:
    #     loudest_file = os.path.join(outdir, "perfectly_matched_loudest.dat")
    #     computeF_cmd_with_output = computeF_cmd.copy()
    #     computeF_cmd_with_output[computeF_cmd_with_output.index("--outputLoudest=NONE")] = f"--outputLoudest={loudest_file}"
        
    #     result = subprocess.run(computeF_cmd_with_output, capture_output=True, text=True)
        
    #     if result.returncode == 0 and os.path.exists(loudest_file):
    #         with open(loudest_file, 'r') as f:
    #             lines = f.readlines()
    #             for line in lines:
    #                 if not line.startswith('%') and line.strip():
    #                     parts = line.split()
    #                     if len(parts) >= 7:
    #                         perfect_2F = float(parts[6])  # 2F通常在第7列
    #                         break
    
    # if perfect_2F is not None:
    #     print(f"\nPerfectly matched 2F = {perfect_2F:.4f}")
        
    #     # 与grid search结果比较
    #     if 'max_twoF' in locals():
    #         print(f"\nComparison:")
    #         print(f"  Perfectly matched 2F: {perfect_2F:.4f}")
    #         print(f"  Grid search max 2F:   {max_twoF:.4f}")
    #         print(f"  Difference:          {perfect_2F - max_twoF:.4f}")
    #         print(f"  Relative difference: {(perfect_2F - max_twoF)/perfect_2F * 100:.2f}%")
            
    #         # 检查grid search是否找到了注入信号
    #         if max_twoF < perfect_2F * 0.9:  # 如果grid search的2F小于完美匹配的90%
    #             print("\nWARNING: Grid search may have missed the injection!")
    #         else:
    #             print("\nGrid search successfully recovered the injection!")
                
        # 创建比较图
        if len(F0_vals) > 10 and perfect_2F is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制2F vs 频率
            ax.scatter(F0_vals, twoF_vals, alpha=0.6, label='Grid search')
            ax.axvline(F0_inj, color='r', linestyle='--', label='Injection frequency')
            ax.axhline(perfect_2F, color='g', linestyle='--', label=f'Perfect match 2F = {perfect_2F:.2f}')
            ax.scatter([max_F0], [max_twoF], color='orange', s=100, marker='*', 
                      label=f'Grid max 2F = {max_twoF:.2f}')
            
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('$2\\mathcal{F}$')
            ax.set_title('Comparison: Perfect Match vs Grid Search')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "perfect_match_comparison.png"))
            plt.close()
            
            print(f"\nComparison plot saved to {os.path.join(outdir, 'perfect_match_comparison.png')}")
    else:
        print("Could not extract 2F value from ComputeFstatistic_v2 output")

# 额外分析：计算注入点周围的网格密度
if len(F0_vals) > 0:
    # 找到最接近注入频率的网格点
    closest_idx = np.argmin(np.abs(F0_vals - F0_inj))
    closest_F0 = F0_vals[closest_idx]
    closest_F1 = F1_vals[closest_idx]
    closest_F2 = F2_vals[closest_idx]
    closest_2F = twoF_vals[closest_idx]
    
    print(f"\nClosest grid point to injection:")
    print(f"  F0: {closest_F0:.6f} Hz (offset: {closest_F0 - F0_inj:.4e} Hz)")
    print(f"  F1: {closest_F1:.4e} Hz/s (offset: {closest_F1 - F1_inj:.4e} Hz/s)")
    print(f"  F2: {closest_F2:.4e} Hz/s^2 (offset: {closest_F2 - F2_inj:.4e} Hz/s^2)")
    print(f"  2F: {closest_2F:.4f}")
    
    if 'perfect_2F' in locals() and perfect_2F is not None:
        mismatch = 1 - closest_2F / perfect_2F
        print(f"  Mismatch: {mismatch:.4f}")

print(f"\nAll analysis results saved to {outdir}")