import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
import tempfile
import shutil
import time
import psutil
import atexit
import json
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
perfect_2F_cache = {}


def get_last_completed_run(output_dir, label):
    """检查已完成的运行次数"""
    json_dir = os.path.join(output_dir, label)
    if not os.path.exists(json_dir):
        return 0
    
    completed_runs = []
    for filename in os.listdir(json_dir):
        if filename.startswith("config_mismatch_run_") and filename.endswith(".json"):
            try:
                run_number = int(filename.split("_")[-1].split(".")[0])
                completed_runs.append(run_number)
            except ValueError:
                continue
    
    return max(completed_runs) if completed_runs else 0


def sample_parameters(duration):
    """从参数空间随机采样"""
    
    # 参数范围定义
    mf_range = np.array([0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.7, 2.0])
    mf1_range = np.array([0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0])
    mf2_range = np.array([0.0001, 0.0002, 0.0005, 0.0008, 0.001, 0.002, 0.005, 0.008, 0.01, 0.03, 0.05, 0.07, 0.1])
    T_coh_range = np.array([6, 10, 15, 20, 30, 40, 60])
    
    # 随机采样基础参数
    mf = np.random.choice(mf_range)
    mf1 = np.random.choice(mf1_range)
    mf2 = np.random.choice(mf2_range)
    T_coh = np.random.choice(T_coh_range)
    
    # 计算衍生参数
    tStack = T_coh * 86400
    nStacks = int(duration / tStack)
    
    # 确保nStacks至少为1
    if nStacks < 1:
        nStacks = 1
        tStack = duration
    
    # 计算gamma的上限
    gamma1_upper = np.sqrt(max(1, 5 * nStacks ** 2 - 4))
    gamma2_upper = np.sqrt(max(5, 35 * nStacks ** 4 - 140 * nStacks ** 2 + 108)) / np.sqrt(3)
    
    gamma1_upper = min(gamma1_upper, 50)
    gamma2_upper = min(gamma2_upper, 100)
    
    # 修复：确保gamma范围有效
    gamma1_candidates = np.arange(1, int(gamma1_upper) + 1, 2)
    gamma2_candidates = np.arange(5, int(gamma2_upper) + 1, 2)
    
    # 如果候选数组为空，使用默认值
    if len(gamma1_candidates) == 0:
        gamma1 = 1
    else:
        gamma1 = np.random.choice(gamma1_candidates)
        
    if len(gamma2_candidates) == 0:
        gamma2 = 5
    else:
        gamma2 = np.random.choice(gamma2_candidates)
    
    return {
        'mf': mf,
        'mf1': mf1,
        'mf2': mf2,
        'T_coh': T_coh,
        'tStack': tStack,
        'nStacks': nStacks,
        'gamma1': gamma1,
        'gamma2': gamma2,
        'gamma1_upper': gamma1_upper,
        'gamma2_upper': gamma2_upper
    }


def setup_aggressive_memory_optimization():
    """设置激进的内存优化环境"""
    
    # 检查系统内存
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"System memory: {memory_gb:.1f} GB")
    
    # 创建专用内存工作空间
    if os.path.exists('/dev/shm'):
        memory_base = '/dev/shm'
        print("Using /dev/shm for aggressive memory optimization")
    else:
        memory_base = tempfile.gettempdir()
        print(f"Warning: /dev/shm not available, using {memory_base}")
    
    # 创建唯一的内存工作目录
    memory_workspace = os.path.join(memory_base, f"lal_aggressive_{os.getpid()}_{int(time.time())}")
    os.makedirs(memory_workspace, exist_ok=True)
    
    # 创建子目录结构
    memory_sft_dir = os.path.join(memory_workspace, "sfts")
    memory_dats_dir = os.path.join(memory_workspace, "dats")
    memory_temp_dir = os.path.join(memory_workspace, "temp")
    
    os.makedirs(memory_sft_dir, exist_ok=True)
    os.makedirs(memory_dats_dir, exist_ok=True)
    os.makedirs(memory_temp_dir, exist_ok=True)
    
    # 设置LAL环境变量进行激进优化
    os.environ['TMPDIR'] = memory_temp_dir
    os.environ['LAL_FSTAT_METHOD'] = 'DemodBest'
    os.environ['LAL_CACHE_SIZE'] = '0'  # 禁用LAL内部缓存
    os.environ['OMP_NUM_THREADS'] = '1'  # 禁用OpenMP以避免线程竞争
    
    print(f"Memory workspace created: {memory_workspace}")
    return memory_workspace, memory_sft_dir, memory_dats_dir


def cleanup_memory_workspace(workspace_path):
    """清理内存工作空间"""
    if os.path.exists(workspace_path):
        try:
            shutil.rmtree(workspace_path)
            print(f"Cleaned up memory workspace: {workspace_path}")
        except Exception as e:
            print(f"Warning: Could not clean up {workspace_path}: {e}")




def generate_sft_data_once(memory_workspace, memory_sft_dir, memory_dats_dir, 
                          sqrtSX, tstart, duration, F0_inj, F1_inj, F2_inj, 
                          Alpha_inj, Delta_inj, cosi_inj, psi_inj, phi0_inj, 
                          tref, h0, IFO):
    """生成SFT数据一次"""
    
    memory_sft_pattern = os.path.join(memory_sft_dir, "*.sft")
        
    print("Generating SFT data in memory...")
    start_sft_time = time.time()

    injection_params = (
        f"{{Alpha={Alpha_inj:.15g}; Delta={Delta_inj:.15g}; Freq={F0_inj:.15g}; "
        f"f1dot={F1_inj:.15e}; f2dot={F2_inj:.15e}; refTime={tref:.15g}; "
        f"h0={h0:.15e}; cosi={cosi_inj:.15g}; psi={psi_inj:.15g}; phi0={phi0_inj:.15g};}}"
    )

    makefakedata_cmd = [
        "lalpulsar_Makefakedata_v5",
        f"--IFOs={IFO}",
        f"--sqrtSX={sqrtSX:.15e}",
        f"--startTime={int(tstart)}",
        f"--duration={int(duration)}",
        f"--fmin={F0_inj - 0.2:.15g}",      # 最小化频带
        f"--Band=0.4",                      # 最小化频带  
        "--Tsft=1800",
        f"--outSFTdir={memory_sft_dir}",
        f"--outLabel=MemOpt",
        f"--injectionSources={injection_params}",
        "--randSeed=1234"
    ]

    result = subprocess.run(makefakedata_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error generating SFTs: {result.stderr}")
        cleanup_memory_workspace(memory_workspace)
        raise RuntimeError("Failed to generate SFTs")

    sft_time = time.time() - start_sft_time
    print(f"SFT generation completed in {sft_time:.2f} seconds")

    # 检查生成的SFT文件
    sft_files = [f for f in os.listdir(memory_sft_dir) if f.endswith('.sft')]
    print(f"Generated {len(sft_files)} SFT files in memory")
    
    return memory_sft_pattern, sft_time


def one_config(config, sqrtSX, tstart, duration, tend, tref, IFO, depth, h0, 
               F0_inj, F1_inj, F2_inj, Alpha_inj, Delta_inj, cosi_inj, 
               psi_inj, phi0_inj, memory_sft_pattern, memory_dats_dir, label, run_number=None):
    """运行一个配置的搜索"""
    
    # 参数设置
    N = 500

    mf = config['mf']
    mf1 = config['mf1']
    mf2 = config['mf2']
    T_coh = config['T_coh']
    tStack = config['tStack']
    nStacks = config['nStacks']  # 使用config中的值
    gamma1 = config['gamma1']
    gamma2 = config['gamma2']
    
    print(f"Starting aggressive memory-optimized search with N={N}")
    print(f"Signal depth: {depth}, h0: {h0:.2e}")
    print(f"Duration: {duration/86400:.1f} days, Segments: {nStacks}")
    
    # 设置搜索网格参数
    dF0 = np.sqrt(12 * mf) / (np.pi * tStack)
    dF1 = np.sqrt(180 * mf1) / (np.pi * tStack**2)
    dF2 = np.sqrt(25200 * mf2) / (np.pi * float(tStack)**3)
    
    N1 = 2
    N2 = 3
    N3 = 3
    
    DeltaF0 = N1 * dF0
    DeltaF1 = N2 * dF1
    DeltaF2 = N3 * dF2
    
    # 预生成随机数以避免重复计算
    print("Pre-generating random offsets...")
    F0_randoms = np.random.uniform(- dF0 / 2.0, dF0 / 2.0, size=N)
    F1_randoms = np.random.uniform(- dF1 / 2.0, dF1 / 2.0, size=N)
    F2_randoms = np.random.uniform(- dF2 / 2.0, dF2 / 2.0, size=N)
    
    # 激进优化的共享命令
    shared_cmd = [
        f"--DataFiles1={memory_sft_pattern}",
        f"--assumeSqrtSX={sqrtSX:.15e}",
        "--gridType1=3",
        f"--skyGridFile={{{Alpha_inj} {Delta_inj}}}",
        f"--refTime={tref:.15f}",
        f"--tStack={tStack:.15g}",
        f"--nStacksMax={nStacks}",
        "--nCand1=30",
        "--printCand1",
        "--semiCohToplist",
        f"--minStartTime1={int(tstart)}",
        f"--maxStartTime1={int(tend)}",
        "--recalcToplistStats=TRUE",
        "--FstatMethod=DemodBest",
        "--FstatMethodRecalc=DemodOptC",
    ]
    
    print(f"Grid spacing: dF0={dF0:.2e}, dF1={dF1:.2e}, dF2={dF2:.2e}")
    print(f"Search bands: ΔF0={DeltaF0:.2e}, ΔF1={DeltaF1:.2e}, ΔF2={DeltaF2:.2e}")
    
    def single_run_aggressive(i):
        """激进优化的单次搜索"""
        try:
            # 计算搜索参数
            F0_min = F0_inj - DeltaF0 / 2.0 + F0_randoms[i]
            F1_min = F1_inj - DeltaF1 / 2.0 + F1_randoms[i]
            F2_min = F2_inj - DeltaF2 / 2.0 + F2_randoms[i]
            
            # 内存中的输出文件
            memory_output_file = os.path.join(memory_dats_dir, f"results_{i}.dat")
            
            # 构建命令
            hierarchsearch_cmd = [
                "lalpulsar_HierarchSearchGCT",
                f"--fnameout={memory_output_file}",
                f"--Freq={F0_min:.15g}",
                f"--FreqBand={DeltaF0:.15g}",
                f"--dFreq={dF0:.15e}",
                f"--f1dot={F1_min:.15e}",
                f"--f1dotBand={DeltaF1:.15e}",
                f"--df1dot={dF1:.15e}",
                f"--f2dot={F2_min:.15e}",
                f"--f2dotBand={DeltaF2:.15e}",
                f"--df2dot={dF2:.15e}",
                f"--gammaRefine={gamma1:.15g}",
                f"--gamma2Refine={gamma2:.15g}",
            ] + shared_cmd
            
            # 执行搜索
            result = subprocess.run(
                hierarchsearch_cmd, 
                capture_output=True, 
                text=True,
                timeout=180,
                env=os.environ.copy()
            )
            
            if result.returncode != 0:
                print(f"Error in run {i}: {result.stderr[:200]}...")
                return None
            
            # 立即解析结果以减少内存占用
            if os.path.exists(memory_output_file):
                with open(memory_output_file, 'r') as f:
                    lines = f.readlines()
                
                max_twoF = 0.0
                for line in lines:
                    if line.strip() and not line.startswith('%'):
                        parts = line.split()
                        if len(parts) >= 8:
                            try:
                                twoFr = float(parts[7])
                                if twoFr > max_twoF:
                                    max_twoF = twoFr
                            except (ValueError, IndexError):
                                continue
                
                # 删除临时文件以节省内存
                os.remove(memory_output_file)
                return max_twoF
            
            return None
            
        except Exception as e:
            print(f"Exception in run {i}: {e}")
            return None
    
    # 执行并行搜索
    print("Starting parallel search...")
    search_start_time = time.time()
    
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
        refresh_per_second=2,
    ) as progress:
        
        task = progress.add_task("Processing runs", total=N)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(single_run_aggressive, i) for i in range(N)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        max_twoFs.append(result)
                    progress.advance(task, 1)
                except Exception as e:
                    print(f"Error in future: {e}")
                    progress.advance(task, 1)
    
    search_time = time.time() - search_start_time
    print(f"Parallel search completed in {search_time:.2f} seconds")
    print(f"Successfully processed {len(max_twoFs)}/{N} runs")
    
    # 计算完美匹配的2F值
    # Check cache first
    T_coh = config['T_coh']
    if T_coh in perfect_2F_cache:
        perfect_2F = perfect_2F_cache[T_coh]
        print(f"Using cached perfect match 2F: {perfect_2F:.6f} for T_coh={T_coh}")
    else:
        print("Computing perfect match 2F value...")
        perfect_output_file = os.path.join(memory_dats_dir, "perfect_match.dat")
        
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
            f"--df2dot={dF2:.15e}",
            f"--fnameout={perfect_output_file}",
            f"--gammaRefine={gamma1}",
            f"--gamma2Refine={gamma2}",
        ] + shared_cmd
        
        result = subprocess.run(perfect_search_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Error computing perfect match: {result.stderr}")
            raise RuntimeError("Failed to compute perfect match 2F value")
        
        # 解析完美匹配结果
        perfect_2F = 0.0
        if os.path.exists(perfect_output_file):
            with open(perfect_output_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.strip() and not line.startswith('%'):
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            twoFr = float(parts[7])
                            if twoFr > perfect_2F:
                                perfect_2F = twoFr
                        except (ValueError, IndexError):
                            continue
        
        print(f"Perfect match 2F: {perfect_2F:.6f}")
        
        perfect_2F_cache[T_coh] = perfect_2F
        print(f"Cached perfect match 2F: {perfect_2F:.6f} for T_coh={T_coh}")
    
    # 计算失配分布
    successful_runs = 0
    if len(max_twoFs) > 0 and perfect_2F > 4:
        mismatches = []
        for max_twoF in max_twoFs:
            if max_twoF > 0:
                mismatch = (perfect_2F - max_twoF) / (perfect_2F - 4)
                mismatches.append(max(0, min(1, mismatch)))
        
        if len(mismatches) > 0:
            successful_runs = len(mismatches)
            mean_mismatch = np.mean(mismatches)
            std_mismatch = np.std(mismatches)
            
            print(f"Mismatch statistics:")
            print(f"  Mean: {mean_mismatch:.6f}")
            print(f"  Std:  {std_mismatch:.6f}")
            print(f"  Min:  {np.min(mismatches):.6f}")
            print(f"  Max:  {np.max(mismatches):.6f}")
            print(f"  Valid samples: {len(mismatches)}/{N}")
            
            # # 生成并保存直方图
            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.hist(
            #     mismatches,
            #     bins=20,
            #     density=True,
            #     color="#5B9BD5",
            #     alpha=0.85,
            #     edgecolor="black",
            #     linewidth=1.0,
            # )
            
            # ax.set_xlabel(r"mismatch $\mu$", fontsize=20)
            # ax.set_ylabel("normalized histogram", fontsize=20)
            # ax.set_xlim(0, 1)
            # ax.tick_params(axis="both", which="major", labelsize=14, length=6)
            # ax.grid(axis="y", linewidth=0.6, alpha=0.35)
            # ax.set_title(f"Mismatch Distribution (N={len(mismatches)}, μ={mean_mismatch:.4f})", fontsize=16)
            
            # fig.tight_layout()
            
            # # 保存图片
            # os.makedirs("images", exist_ok=True)
            # fig.savefig(f"images/MC/mismatch_distribution_aggressive_{mf}-{mf1}-{mf2}-{T_coh}-{gamma1}-{gamma2}-{N}-{depth}.pdf")
            # fig.savefig(f"images/MC/mismatch_distribution_aggressive_{mf}-{mf1}-{mf2}-{T_coh}-{gamma1}-{gamma2}-{N}-{depth}.png")
            # print(f"Histogram saved to images/MC/mismatch_distribution_aggressive_{mf}-{mf1}-{mf2}-{T_coh}-{gamma1}-{gamma2}-{N}-{depth}.pdf")
            
            # plt.close(fig)
            
        else:
            print("Warning: No valid mismatch calculations possible")
    else:
        print("Warning: Insufficient data for mismatch analysis")
    
    
    # 保存config和mismatch数据到JSON
    if len(mismatches) > 0:
        json_filename = f"config_mismatch_run_{run_number:04d}.json"
        # 构造要保存的数据 - 确保所有值都是JSON可序列化的
        data_to_save = {
            'config': {
                'mf': float(config['mf']),
                'mf1': float(config['mf1']), 
                'mf2': float(config['mf2']),
                'T_coh': int(config['T_coh']),
                'tStack': float(config['tStack']),
                'nStacks': int(config['nStacks']),
                'gamma1': int(config['gamma1']),
                'gamma2': int(config['gamma2']),
                'gamma1_upper': float(config['gamma1_upper']),
                'gamma2_upper': float(config['gamma2_upper'])
            },
            'mismatch_list': [float(x) for x in mismatches],  # 转换列表中的每个元素
            'statistics': {
                'mean_mismatch': float(mean_mismatch),
                'std_mismatch': float(std_mismatch),
                'min_mismatch': float(np.min(mismatches)),
                'max_mismatch': float(np.max(mismatches)),
                'valid_samples': int(len(mismatches)),
                'total_runs': int(N)
            },
            'perfect_2F': float(perfect_2F)
        }
        
        json_filepath = os.path.join("LAL_example_data", label, json_filename)
        
        # 保存JSON文件
        with open(json_filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"Config and mismatch data saved to {json_filepath}")
    
    
    
    return successful_runs


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # GW数据属性
    sqrtSX = 1e-22
    tstart = 1126051217
    duration = 120 * 86400
    tend = tstart + duration
    tref = 0.5 * (tstart + tend)
    IFO = "H1"
    
    # 注入信号参数
    depth = 0.6
    h0 = sqrtSX / depth
    F0_inj = 151.5
    F1_inj = -1e-10
    F2_inj = -1e-20
    Alpha_inj = 0.5
    Delta_inj = 1
    cosi_inj = 1
    psi_inj = 0.0
    phi0_inj = 0.0
    label = "LALSemiCoherentF0F1F2_aggressive_memory"
    
    # 设置内存优化环境
    memory_workspace, memory_sft_dir, memory_dats_dir = setup_aggressive_memory_optimization()
    
    # 注册清理函数
    atexit.register(cleanup_memory_workspace, memory_workspace)
    
    try:
        # 生成SFT数据一次
        memory_sft_pattern, sft_time = generate_sft_data_once(
            memory_workspace, memory_sft_dir, memory_dats_dir, 
            sqrtSX, tstart, duration, F0_inj, F1_inj, F2_inj, 
            Alpha_inj, Delta_inj, cosi_inj, psi_inj, phi0_inj, 
            tref, h0, IFO
        )
        
        # 检查已完成的运行次数
        last_completed = get_last_completed_run("LAL_example_data", label)
        start_run = last_completed + 1
        total_runs = 2000
        
        if last_completed > 0:
            print(f"Resuming from run {start_run} (found {last_completed} completed runs)")
        else:
            print(f"Starting fresh Monte Carlo simulation")
        
        # 运行Monte Carlo循环
        for i in range(start_run - 1, total_runs):  # Adjust range to start from last completed
            current_run = i + 1
            print(f"\n=== Monte Carlo Run {current_run}/{total_runs} ===")
            config = sample_parameters(duration)
            
            print(f"Config: mf={config['mf']}, mf1={config['mf1']}, mf2={config['mf2']}")
            print(f"        T_coh={config['T_coh']}, nStacks={config['nStacks']}")
            print(f"        gamma1={config['gamma1']}, gamma2={config['gamma2']}")
            
            successful_runs = one_config(
                config, sqrtSX, tstart, duration, tend, tref, IFO, depth, h0, 
                F0_inj, F1_inj, F2_inj, Alpha_inj, Delta_inj, cosi_inj, 
                psi_inj, phi0_inj, memory_sft_pattern, memory_dats_dir, label, current_run
            )
            
            if successful_runs == 0:
                print(f"Run {current_run}: No valid results, skipping further processing")
                continue
            
            print(f"Run {current_run}: {successful_runs} valid results")
            
        print(f"Program completed successfully!")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理内存工作空间
        cleanup_memory_workspace(memory_workspace)