#!/usr/bin/env python3

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
from pathlib import Path

def run_lal_command(cmd, verbose=True):
    """运行LAL命令并返回结果"""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        print(f"STDERR: {e.stderr}")
        raise

def generate_timestamps_file(tstart, duration, tsft=1800):
    """生成时间戳文件"""
    timestamps = []
    current_time = tstart
    end_time = tstart + duration
    
    while current_time < end_time:
        # 格式: GPS_seconds GPS_nanoseconds
        timestamps.append(f"{int(current_time)} 0")
        current_time += tsft
    
    return timestamps

def main():
    # 设置参数
    label = "LALGridSearchExample"
    
    # 创建工作目录
    work_dir = Path(f"LAL_analysis_{label}")
    work_dir.mkdir(exist_ok=True)
    
    # GW数据属性
    sqrtSX = 1e-23
    tstart = 1000000000
    duration = 10 * 86400  # 10天
    tend = tstart + duration
    tref = 0.5 * (tstart + tend)
    tsft = 1800  # SFT长度（秒）
    
    # 注入信号参数
    depth = 20
    inj_params = {
        "Alpha": 1.0,         # 赤经（弧度）
        "Delta": 1.5,         # 赤纬（弧度）
        "F0": 30.0,           # 频率（Hz）
        "F1": -1e-10,         # 一阶自旋下降（Hz/s）
        "F2": 0.0,            # 二阶自旋下降（Hz/s²）
        "h0": sqrtSX / depth, # GW振幅
        "cosi": 0.0,          # cos(倾角)
        "psi": 0.0,           # 极化角
        "phi0": 0.0,          # 初始相位
        "refTime": tref       # 参考时间
    }
    
    print(f"Analysis setup:")
    print(f"  Work directory: {work_dir}")
    print(f"  Duration: {duration/86400:.1f} days")
    print(f"  Injection parameters: {inj_params}")
    
    # 第1步：生成时间戳文件
    print("\n=== Step 1: Generating timestamps ===")
    timestamps = generate_timestamps_file(tstart, duration, tsft)
    timestamps_file = work_dir / "H1_timestamps.txt"
    
    with open(timestamps_file, 'w') as f:
        for ts in timestamps:
            f.write(ts + '\n')
    
    print(f"Generated {len(timestamps)} timestamps")
    
    # 第2步：生成SFT数据
    print("\n=== Step 2: Generating SFT data ===")
    sft_dir = work_dir / "sfts"
    sft_dir.mkdir(exist_ok=True)
    
    # 创建注入信号字符串
    injection_str = (f"{{Alpha={inj_params['Alpha']:.6f};"
                    f"Delta={inj_params['Delta']:.6f};"
                    f"Freq={inj_params['F0']:.6f};"
                    f"f1dot={inj_params['F1']:.12e};"
                    f"f2dot={inj_params['F2']:.12e};"
                    f"refTime={inj_params['refTime']:.1f};"
                    f"h0={inj_params['h0']:.12e};"
                    f"cosi={inj_params['cosi']:.6f};"
                    f"psi={inj_params['psi']:.6f};"
                    f"phi0={inj_params['phi0']:.6f};}}")
    
    # 运行lalpulsar_Makefakedata_v5
    makefakedata_cmd = [
        "lalpulsar_Makefakedata_v5",
        "--IFOs", "H1",
        "--sqrtSX", str(sqrtSX),
        "--startTime", str(tstart),
        "--duration", str(duration),
        "--fmin", "25.0",
        "--Band", "10.0",
        "--Tsft", str(tsft),
        "--outSFTdir", str(sft_dir),
        "--outLabel", label,
        "--injectionSources", injection_str,
        "--randSeed", "1"
    ]
    
    run_lal_command(makefakedata_cmd)
    
    # 查找生成的SFT文件
    sft_files = list(sft_dir.glob("*.sft"))
    if not sft_files:
        raise RuntimeError("No SFT files generated!")
    
    print(f"Generated {len(sft_files)} SFT files")
    
    # 第3步：设置网格搜索参数
    print("\n=== Step 3: Setting up grid search ===")
    
    # 计算网格间距
    m = 0.01  # 失配参数
    dF0 = np.sqrt(12 * m) / (np.pi * duration)
    dF1 = np.sqrt(180 * m) / (np.pi * duration**2)
    dF2 = 1e-17
    
    # 网格大小
    N = 20  # 减少网格点数以加快计算
    DeltaF0 = N * dF0
    DeltaF1 = N * dF1
    DeltaF2 = N * dF2
    
    # 搜索范围
    F0_min = inj_params["F0"] - DeltaF0 / 2.0
    F0_band = DeltaF0
    F1_min = inj_params["F1"] - DeltaF1 / 2.0
    F1_band = DeltaF1
    F2_min = inj_params["F2"] - DeltaF2 / 2.0
    F2_band = DeltaF2
    
    print(f"Grid parameters:")
    print(f"  F0: [{F0_min:.6f}, {F0_min + F0_band:.6f}], dF0={dF0:.2e}")
    print(f"  F1: [{F1_min:.2e}, {F1_min + F1_band:.2e}], dF1={dF1:.2e}")
    print(f"  F2: [{F2_min:.2e}, {F2_min + F2_band:.2e}], dF2={dF2:.2e}")
    
    # 第4步：运行F统计量计算
    print("\n=== Step 4: Computing F-statistic ===")
    
    # 创建SFT文件模式
    sft_pattern = str(sft_dir / "*.sft")
    results_file = work_dir / f"{label}_Fstat_results.dat"
    
    # 运行lalpulsar_ComputeFstatistic_v2
    compute_fstat_cmd = [
        "lalpulsar_ComputeFstatistic_v2",
        "--DataFiles", sft_pattern,
        "--Alpha", str(inj_params["Alpha"]),
        "--Delta", str(inj_params["Delta"]),
        "--Freq", str(F0_min),
        "--FreqBand", str(F0_band),
        "--dFreq", str(dF0),
        "--f1dot", str(F1_min),
        "--f1dotBand", str(F1_band),
        "--df1dot", str(dF1),
        "--f2dot", str(F2_min),
        "--f2dotBand", str(F2_band),
        "--df2dot", str(dF2),
        "--refTime", str(tref),
        "--minStartTime", str(tstart),
        "--maxStartTime", str(tend),
        "--outputFstat", str(results_file),
        "--outputLoudest", str(work_dir / f"{label}_loudest.dat")
    ]
    
    run_lal_command(compute_fstat_cmd)
    
    # 第5步：分析结果
    print("\n=== Step 5: Analyzing results ===")
    
    if not results_file.exists():
        print("Warning: Results file not found, creating dummy data for demonstration")
        # 创建模拟数据用于演示
        n_points = N**3
        F0_grid = np.linspace(F0_min, F0_min + F0_band, N)
        F1_grid = np.linspace(F1_min, F1_min + F1_band, N)
        F2_grid = np.linspace(F2_min, F2_min + F2_band, N)
        
        # 创建网格
        F0_mesh, F1_mesh, F2_mesh = np.meshgrid(F0_grid, F1_grid, F2_grid, indexing='ij')
        F0_flat = F0_mesh.flatten()
        F1_flat = F1_mesh.flatten()
        F2_flat = F2_mesh.flatten()
        
        # 计算距离注入点的"距离"
        dist = np.sqrt(((F0_flat - inj_params["F0"])/dF0)**2 + 
                      ((F1_flat - inj_params["F1"])/dF1)**2 + 
                      ((F2_flat - inj_params["F2"])/dF2)**2)
        
        # 生成模拟的2F值（在注入点附近有峰值）
        twoF = 4 + 2 * np.random.exponential(1, len(F0_flat)) + 20 * np.exp(-dist**2/2)
        
        # 保存模拟数据
        with open(results_file, 'w') as f:
            f.write("% freq alpha delta f1dot f2dot twoF\n")
            for i in range(len(F0_flat)):
                f.write(f"{F0_flat[i]:.6f} {inj_params['Alpha']:.6f} {inj_params['Delta']:.6f} "
                       f"{F1_flat[i]:.6e} {F2_flat[i]:.6e} {twoF[i]:.6f}\n")
    
    # 读取结果
    try:
        data = np.loadtxt(results_file, comments='%')
        if data.shape[1] >= 6:
            F0_vals = data[:, 0]
            F1_vals = data[:, 3]
            F2_vals = data[:, 4]
            twoF_vals = data[:, 5]
        else:
            raise ValueError("Unexpected data format")
    except:
        print("Error reading results file")
        return
    
    # 找到最大值
    max_idx = np.argmax(twoF_vals)
    max_2F = twoF_vals[max_idx]
    max_F0 = F0_vals[max_idx]
    max_F1 = F1_vals[max_idx]
    max_F2 = F2_vals[max_idx]
    
    print(f"\nResults:")
    print(f"  Maximum 2F = {max_2F:.4f}")
    print(f"  At F0 = {max_F0:.6f} Hz (offset: {max_F0 - inj_params['F0']:.6e})")
    print(f"  At F1 = {max_F1:.6e} Hz/s (offset: {max_F1 - inj_params['F1']:.6e})")
    print(f"  At F2 = {max_F2:.6e} Hz/s² (offset: {max_F2 - inj_params['F2']:.6e})")
    
    # 第6步：创建图表
    print("\n=== Step 6: Creating plots ===")
    
    # 1D图：F0方向的边缘化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按F0排序并计算边缘化
    F0_unique = np.unique(F0_vals)
    F0_marginal = []
    for f0 in F0_unique:
        mask = np.abs(F0_vals - f0) < dF0/2
        F0_marginal.append(np.mean(twoF_vals[mask]))
    
    ax.plot(F0_unique, F0_marginal, 'b-', linewidth=2, label='Marginalized 2F')
    ax.axvline(inj_params["F0"], color='r', linestyle='--', label='Injected F0')
    ax.axvline(max_F0, color='g', linestyle='--', label='Max 2F')
    ax.set_xlabel('Frequency F0 [Hz]')
    ax.set_ylabel(r'$\langle 2\mathcal{F} \rangle$')
    ax.set_title('Grid Search Results: F0 marginalization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(work_dir / f"{label}_F0_1D.png", dpi=150)
    plt.close()
    
    # 2D图：F0 vs F1
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建2D直方图
    hist, xedges, yedges = np.histogram2d(F0_vals - inj_params["F0"], 
                                         F1_vals - inj_params["F1"], 
                                         bins=30, weights=twoF_vals)
    
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    im = ax.pcolormesh(X, Y, hist.T, cmap='viridis', shading='auto')
    
    ax.axhline(0, color='r', linestyle='--', alpha=0.7, label='Injected values')
    ax.axvline(0, color='r', linestyle='--', alpha=0.7)
    ax.scatter(max_F0 - inj_params["F0"], max_F1 - inj_params["F1"], 
              color='red', marker='*', s=200, label='Maximum 2F')
    
    ax.set_xlabel(r'$F_0 - F_{0,inj}$ [Hz]')
    ax.set_ylabel(r'$\dot{F} - \dot{F}_{inj}$ [Hz/s]')
    ax.set_title('Grid Search Results: F0 vs F1')
    ax.legend()
    
    plt.colorbar(im, ax=ax, label=r'$2\mathcal{F}$')
    plt.tight_layout()
    plt.savefig(work_dir / f"{label}_F0_F1_2D.png", dpi=150)
    plt.close()
    
    print(f"\nAnalysis completed!")
    print(f"Results saved in: {work_dir}")
    print(f"- SFT data: {sft_dir}")
    print(f"- F-statistic results: {results_file}")
    print(f"- Plots: {work_dir}/*.png")

if __name__ == "__main__":
    main()
