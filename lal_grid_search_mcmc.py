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

# ============================================================================
# 激进内存优化配置
# ============================================================================

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
    memory_workspace = os.path.join(memory_base, f"lal_mcmc_{os.getpid()}_{int(time.time())}")
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

def get_aggressive_worker_count():
    """计算激进优化的工作线程数"""
    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # 每个worker预估需要0.5-1GB内存
    max_workers_by_memory = max(1, int(memory_gb / 1.5))
    
    # CPU限制：保留1-2个核心给系统
    max_workers_by_cpu = max(1, cpu_count - 2)
    
    # 激进设置：在内存允许的情况下使用更多线程
    optimal_workers = min(max_workers_by_memory, max_workers_by_cpu, 8)
    
    print(f"Using {optimal_workers} worker threads (CPU: {cpu_count}, Memory: {memory_gb:.1f}GB)")
    return optimal_workers

# ============================================================================
# MCMC参数采样
# ============================================================================

def sample_parameters(duration):
    """从参数空间随机采样"""
    
    # 参数范围定义
    mf_range = [0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.7, 2.0]
    mf1_range = [0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
    mf2_range = [0.0001, 0.0002, 0.0005, 0.0008, 0.001, 0.002, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1.0]
    T_coh_range = [5, 10, 15, 20, 30, 40, 60]
    
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
    gamma2_upper = min(gamma2_upper, 50)
    
    # 随机采样gamma值（在1 to gamma1_upper范围内的单数，整数）
    gamma1 = np.random.choice(np.arange(1, int(gamma1_upper) + 1, 2))
    gamma2 = np.random.choice(np.arange(5, int(gamma2_upper) + 1, 2))
    
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

def save_checkpoint(config_id, results_data, checkpoint_dir):
    """保存检查点数据"""
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{config_id:04d}.json")
    
    # Convert numpy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    converted_data = convert_numpy_types(results_data)
    
    with open(checkpoint_file, 'w') as f:
        json.dump(converted_data, f, indent=2)

def load_checkpoint(checkpoint_dir):
    """加载已完成的检查点"""
    completed_configs = {}
    if os.path.exists(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".json"):
                config_id = int(filename.split("_")[1].split(".")[0])
                filepath = os.path.join(checkpoint_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        completed_configs[config_id] = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load checkpoint {filepath}: {e}")
    return completed_configs

# ============================================================================
# 主程序
# ============================================================================

def main():
    # MCMC参数设置
    N_CONFIGS = 2000  # 配置数量
    N_SIMS_PER_CONFIG = 500  # 每个配置的模拟次数
    label = "LAL_MCMC_SemiCoherent"
    
    # 设置内存优化环境
    memory_workspace, memory_sft_dir, memory_dats_dir = setup_aggressive_memory_optimization()
    
    # 注册清理函数
    atexit.register(cleanup_memory_workspace, memory_workspace)
    
    # 创建输出目录结构
    results_dir = os.path.join("MCMC_LAL_Results", label)
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    data_dir = os.path.join(results_dir, "data")
    plots_dir = os.path.join(results_dir, "plots")
    
    for dir_path in [results_dir, checkpoint_dir, data_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # GW数据属性
    sqrtSX = 1e-22
    tstart = 1126051217
    duration = 120 * 86400
    tend = tstart + duration
    tref = 0.5 * (tstart + tend)
    IFO = "H1"
    
    # 注入信号参数
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
    
    memory_sft_pattern = os.path.join(memory_sft_dir, "*.sft")
    
    print(f"Starting MCMC search with {N_CONFIGS} configurations, {N_SIMS_PER_CONFIG} simulations each")
    print(f"Total simulations: {N_CONFIGS * N_SIMS_PER_CONFIG}")
    print(f"Signal depth: {depth}, h0: {h0:.2e}")
    print(f"Duration: {duration/86400:.1f} days")
    
    # 加载已完成的检查点
    completed_configs = load_checkpoint(checkpoint_dir)
    print(f"Found {len(completed_configs)} completed configurations")
    
    # ========================================================================
    # Step 1: 生成SFT数据（只需要一次）
    # ========================================================================
    
    sft_files = [f for f in os.listdir(memory_sft_dir) if f.endswith('.sft')] if os.path.exists(memory_sft_dir) else []
    
    if len(sft_files) == 0:
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
            f"--fmin={F0_inj - 0.2:.15g}",
            f"--Band=0.4",
            "--Tsft=1800",
            f"--outSFTdir={memory_sft_dir}",
            f"--outLabel=MCMC",
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
        
        sft_files = [f for f in os.listdir(memory_sft_dir) if f.endswith('.sft')]
        print(f"Generated {len(sft_files)} SFT files in memory")
    else:
        print(f"Using existing {len(sft_files)} SFT files")
    
    # ========================================================================
    # Step 2: MCMC采样和搜索
    # ========================================================================
    
    # 存储所有结果
    all_results = []
    
    # 生成所有配置参数
    print("Generating parameter configurations...")
    configs = []
    for config_id in range(N_CONFIGS):
        if config_id not in completed_configs:
            config = sample_parameters(duration)
            config['config_id'] = config_id
            configs.append(config)
    
    print(f"Need to process {len(configs)} new configurations")
    
    # 全局进度条
    start_time = time.time()
    
    with Progress(
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        "[progress.completed]{task.completed}/{task.total}",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        refresh_per_second=1,
    ) as progress:
        
        main_task = progress.add_task("Processing configurations", total=len(configs))
        
        # 处理每个配置
        for config in configs:
            config_id = config['config_id']
            
            print(f"\nProcessing configuration {config_id + 1}/{N_CONFIGS}")
            print(f"Parameters: mf={config['mf']}, mf1={config['mf1']}, mf2={config['mf2']}")
            print(f"T_coh={config['T_coh']}d, nStacks={config['nStacks']}, γ1={config['gamma1']:.2f}, γ2={config['gamma2']:.2f}")
            
            # 计算网格参数
            mf, mf1, mf2 = config['mf'], config['mf1'], config['mf2']
            tStack, gamma1, gamma2 = config['tStack'], config['gamma1'], config['gamma2']
            
            
            dF0 = np.sqrt(12 * mf) / (np.pi * tStack)
            dF1 = np.sqrt(180 * mf1) / (np.pi * tStack**2)
            df2 = np.sqrt(25200 * mf2) / (np.pi * tStack**3)
            
            N1, N2, N3 = 2, 3, 3
            DeltaF0 = N1 * dF0
            DeltaF1 = N2 * dF1
            DeltaF2 = N3 * df2
            
            # 预生成随机数
            F0_randoms = np.random.uniform(-dF0/2.0, dF0/2.0, size=N_SIMS_PER_CONFIG)
            F1_randoms = np.random.uniform(-dF1/2.0, dF1/2.0, size=N_SIMS_PER_CONFIG)
            F2_randoms = np.random.uniform(-df2/2.0, df2/2.0, size=N_SIMS_PER_CONFIG)
            
            # 共享命令参数
            shared_cmd = [
                f"--DataFiles1={memory_sft_pattern}",
                f"--assumeSqrtSX={sqrtSX:.15e}",
                "--gridType1=3",
                f"--skyGridFile={{{Alpha_inj} {Delta_inj}}}",
                f"--refTime={tref:.15f}",
                f"--tStack={tStack:.15g}",
                f"--nStacksMax={config['nStacks']}",
                "--nCand1=10",
                "--printCand1",
                "--semiCohToplist",
                f"--minStartTime1={int(tstart)}",
                f"--maxStartTime1={int(tend)}",
                "--recalcToplistStats=TRUE",
                "--FstatMethod=DemodBest",
                "--FstatMethodRecalc=DemodOptC",
            ]
            
            # 定义单次搜索函数
            def single_run_mcmc(i):
                try:
                    F0_min = F0_inj - DeltaF0/2.0 + F0_randoms[i]
                    F1_min = F1_inj - DeltaF1/2.0 + F1_randoms[i]
                    F2_min = F2_inj - DeltaF2/2.0 + F2_randoms[i]
                    
                    memory_output_file = os.path.join(memory_dats_dir, f"config_{config_id}_run_{i}.dat")
                    
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
                        f"--df2dot={df2:.15e}",
                        f"--gammaRefine={gamma1:.15g}",
                        f"--gamma2Refine={gamma2:.15g}",
                    ] + shared_cmd
                    
                    result = subprocess.run(
                        hierarchsearch_cmd, 
                        capture_output=True, 
                        text=True,
                        timeout=180,
                        env=os.environ.copy()
                    )
                    
                    if result.returncode != 0:
                        return None
                    
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
                        
                        os.remove(memory_output_file)
                        return max_twoF
                    
                    return None
                    
                except Exception as e:
                    return None
            
            # 并行执行搜索
            max_twoFs = []
            optimal_workers = get_aggressive_worker_count()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(single_run_mcmc, i) for i in range(N_SIMS_PER_CONFIG)]
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            max_twoFs.append(result)
                    except Exception as e:
                        pass
            
            # 计算完美匹配的2F值
            perfect_output_file = os.path.join(memory_dats_dir, f"perfect_match_config_{config_id}.dat")
            
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
                f"--gammaRefine={gamma1}",
                f"--gamma2Refine={gamma2}",
            ] + shared_cmd
            
            result = subprocess.run(perfect_search_cmd, capture_output=True, text=True, timeout=300)
            
            perfect_2F = 0.0
            if result.returncode == 0 and os.path.exists(perfect_output_file):
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
                
                os.remove(perfect_output_file)
            
            # 计算失配
            mismatches = []
            if len(max_twoFs) > 0 and perfect_2F > 4:
                for max_twoF in max_twoFs:
                    if max_twoF > 0:
                        mismatch = (perfect_2F - max_twoF) / (perfect_2F - 4)
                        mismatches.append(max(0, min(1, mismatch)))
            
            # 保存结果
            config_results = {
                'config_id': config_id,
                'parameters': config,
                'grid_spacing': {'dF0': dF0, 'dF1': dF1, 'df2': df2},
                'search_bands': {'DeltaF0': DeltaF0, 'DeltaF1': DeltaF1, 'DeltaF2': DeltaF2},
                'perfect_2F': perfect_2F,
                'max_twoFs': max_twoFs,
                'mismatches': mismatches,
                'n_successful_runs': len(max_twoFs),
                'mean_mismatch': np.mean(mismatches) if mismatches else None,
                'std_mismatch': np.std(mismatches) if mismatches else None,
                'timestamp': time.time()
            }
            
            # 保存检查点
            save_checkpoint(config_id, config_results, checkpoint_dir)
            all_results.append(config_results)
            
            print(f"Config {config_id}: {len(max_twoFs)}/{N_SIMS_PER_CONFIG} successful, "
                  f"perfect_2F={perfect_2F:.3f}, mean_mismatch={np.mean(mismatches) if mismatches else 'N/A'}")
            
            progress.advance(main_task, 1)
    
    # ========================================================================
    # Step 3: 保存汇总结果
    # ========================================================================
    
    # 合并已完成的配置
    for config_id, config_results in completed_configs.items():
        all_results.append(config_results)
    
    # 保存所有结果
    all_results_file = os.path.join(data_dir, "all_results.json")
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 创建汇总DataFrame
    summary_data = []
    for result in all_results:
        if result['mismatches']:
            summary_data.append({
                'config_id': result['config_id'],
                'mf': result['parameters']['mf'],
                'mf1': result['parameters']['mf1'],
                'mf2': result['parameters']['mf2'],
                'T_coh': result['parameters']['T_coh'],
                'nStacks': result['parameters']['nStacks'],
                'gamma1': result['parameters']['gamma1'],
                'gamma2': result['parameters']['gamma2'],
                'perfect_2F': result['perfect_2F'],
                'n_successful_runs': result['n_successful_runs'],
                'mean_mismatch': result['mean_mismatch'],
                'std_mismatch': result['std_mismatch']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(data_dir, "summary.csv"), index=False)
        
        # 创建详细的失配数据
        detailed_mismatches = []
        for result in all_results:
            if result['mismatches']:
                for i, mismatch in enumerate(result['mismatches']):
                    detailed_mismatches.append({
                        'config_id': result['config_id'],
                        'run_id': i,
                        'mismatch': mismatch,
                        'mf': result['parameters']['mf'],
                        'mf1': result['parameters']['mf1'],
                        'mf2': result['parameters']['mf2'],
                        'T_coh': result['parameters']['T_coh'],
                        'gamma1': result['parameters']['gamma1'],
                        'gamma2': result['parameters']['gamma2']
                    })
        
        detailed_df = pd.DataFrame(detailed_mismatches)
        detailed_df.to_csv(os.path.join(data_dir, "detailed_mismatches.csv"), index=False)
        
        print(f"Saved {len(summary_data)} configuration summaries")
        print(f"Saved {len(detailed_mismatches)} individual mismatch measurements")
    
    # 清理内存工作空间
    cleanup_memory_workspace(memory_workspace)
    
    total_time = time.time() - start_time
    print(f"\nMCMC search completed!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {results_dir}")
    
    return len(all_results)

# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    try:
        total_configs = main()
        print(f"Program completed successfully with {total_configs} configurations processed")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program failed with error: {e}")
        import traceback
        traceback.print_exc()