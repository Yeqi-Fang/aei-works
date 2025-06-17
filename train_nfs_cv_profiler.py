# train_bottleneck_analyzer.py - 智能瓶颈分析器

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import os
from datetime import datetime
import time
import threading
import subprocess
import psutil
import gc
from collections import deque
import json

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, ProfilerActivity, record_function
import matplotlib.pyplot as plt

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import RandomPermutation

# -----------------------------------------------------------------------------
# 高级瓶颈分析器
# -----------------------------------------------------------------------------

class BottleneckAnalyzer:
    def __init__(self, log_dir: Path, sample_interval: float = 0.5):
        self.log_dir = log_dir
        self.profiler_dir = log_dir / "bottleneck_analysis"
        self.profiler_dir.mkdir(exist_ok=True)
        
        self.sample_interval = sample_interval
        self.monitoring = False
        self.monitor_thread = None
        
        # 性能数据收集
        self.gpu_metrics = deque(maxlen=1000)
        self.cpu_metrics = deque(maxlen=1000)
        self.io_metrics = deque(maxlen=1000)
        self.memory_metrics = deque(maxlen=1000)
        
        # 瓶颈检测阈值
        self.thresholds = {
            'gpu_util_low': 70,      # GPU利用率低于70%认为是瓶颈
            'cpu_util_high': 80,     # CPU利用率高于80%认为是瓶颈
            'memory_util_high': 85,  # 内存使用率高于85%认为是瓶颈
            'io_wait_high': 10,      # IO等待时间高于10%认为是瓶颈
        }
        
        # 时间测量
        self.timings = {}
        self.current_phase = None
        
    def start_monitoring(self):
        """开始后台监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 开始实时性能监控...")
        
    def stop_monitoring(self):
        """停止后台监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("⏹️  停止性能监控")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # GPU监控
                gpu_metrics = self._get_gpu_metrics()
                if gpu_metrics:
                    gpu_metrics['timestamp'] = timestamp
                    gpu_metrics['phase'] = self.current_phase
                    self.gpu_metrics.append(gpu_metrics)
                
                # CPU监控
                cpu_metrics = self._get_cpu_metrics()
                cpu_metrics['timestamp'] = timestamp
                cpu_metrics['phase'] = self.current_phase
                self.cpu_metrics.append(cpu_metrics)
                
                # 内存监控
                memory_metrics = self._get_memory_metrics()
                memory_metrics['timestamp'] = timestamp
                memory_metrics['phase'] = self.current_phase
                self.memory_metrics.append(memory_metrics)
                
                # IO监控
                io_metrics = self._get_io_metrics()
                io_metrics['timestamp'] = timestamp
                io_metrics['phase'] = self.current_phase
                self.io_metrics.append(io_metrics)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                print(f"监控出错: {e}")
                
    def _get_gpu_metrics(self) -> Optional[Dict]:
        """获取GPU指标"""
        if not torch.cuda.is_available():
            return None
            
        try:
            # PyTorch GPU内存
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # 尝试获取GPU利用率 (需要nvidia-ml-py或nvidia-smi)
            gpu_utilization = 0
            gpu_memory_util = 0
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=1)
                
                if result.returncode == 0:
                    values = result.stdout.strip().split(',')
                    gpu_utilization = float(values[0])
                    gpu_memory_util = float(values[1])
                    gpu_temperature = float(values[2])
                else:
                    gpu_temperature = 0
            except:
                gpu_temperature = 0
                
            return {
                'utilization': gpu_utilization,
                'memory_allocated': gpu_memory_allocated,
                'memory_reserved': gpu_memory_reserved,
                'memory_total': gpu_memory_total,
                'memory_utilization': gpu_memory_util,
                'temperature': gpu_temperature
            }
        except Exception:
            return None
            
    def _get_cpu_metrics(self) -> Dict:
        """获取CPU指标"""
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        
        # 获取负载平均值
        try:
            load_avg = os.getloadavg()
        except:
            load_avg = [0, 0, 0]
            
        return {
            'utilization': cpu_percent,
            'count': cpu_count,
            'load_1min': load_avg[0],
            'load_5min': load_avg[1],
            'load_15min': load_avg[2]
        }
        
    def _get_memory_metrics(self) -> Dict:
        """获取内存指标"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / 1024**3,
            'used_gb': memory.used / 1024**3,
            'available_gb': memory.available / 1024**3,
            'percent': memory.percent
        }
        
    def _get_io_metrics(self) -> Dict:
        """获取IO指标"""
        try:
            io_counters = psutil.disk_io_counters()
            if io_counters:
                return {
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes,
                    'read_time': io_counters.read_time,
                    'write_time': io_counters.write_time
                }
        except:
            pass
            
        return {
            'read_bytes': 0,
            'write_bytes': 0,
            'read_time': 0,
            'write_time': 0
        }
        
    def set_phase(self, phase_name: str):
        """设置当前阶段"""
        self.current_phase = phase_name
        print(f"📊 进入阶段: {phase_name}")
        
    def time_phase(self, phase_name: str):
        """测量阶段执行时间的上下文管理器"""
        class PhaseTimer:
            def __init__(self, analyzer, name):
                self.analyzer = analyzer
                self.name = name
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                self.analyzer.set_phase(self.name)
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.analyzer.timings[self.name] = duration
                print(f"⏱️  {self.name}: {duration:.2f}s")
                self.analyzer.set_phase(None)
                
        return PhaseTimer(self, phase_name)
        
    def analyze_bottlenecks(self) -> Dict:
        """分析性能瓶颈"""
        bottlenecks = {
            'gpu_bottleneck': False,
            'cpu_bottleneck': False,
            'memory_bottleneck': False,
            'io_bottleneck': False,
            'data_loading_bottleneck': False,
            'recommendations': []
        }
        
        # 分析GPU瓶颈
        if self.gpu_metrics:
            avg_gpu_util = np.mean([m['utilization'] for m in self.gpu_metrics if m.get('utilization', 0) > 0])
            max_gpu_memory = max([m['memory_allocated'] for m in self.gpu_metrics])
            
            if avg_gpu_util < self.thresholds['gpu_util_low']:
                bottlenecks['gpu_bottleneck'] = True
                bottlenecks['recommendations'].append(f"🔥 GPU利用率低 ({avg_gpu_util:.1f}%) - 建议增大batch_size或模型复杂度")
                
            if max_gpu_memory < 2:  # 小于2GB
                bottlenecks['recommendations'].append(f"💾 GPU内存使用偏低 ({max_gpu_memory:.1f}GB) - 可以增大模型或batch_size")
                
        # 分析CPU瓶颈
        if self.cpu_metrics:
            avg_cpu_util = np.mean([m['utilization'] for m in self.cpu_metrics])
            max_cpu_util = max([m['utilization'] for m in self.cpu_metrics])
            
            if avg_cpu_util > self.thresholds['cpu_util_high']:
                bottlenecks['cpu_bottleneck'] = True
                bottlenecks['recommendations'].append(f"🖥️  CPU利用率高 ({avg_cpu_util:.1f}%) - 可能是数据加载瓶颈")
                
        # 分析内存瓶颈
        if self.memory_metrics:
            max_memory_percent = max([m['percent'] for m in self.memory_metrics])
            
            if max_memory_percent > self.thresholds['memory_util_high']:
                bottlenecks['memory_bottleneck'] = True
                bottlenecks['recommendations'].append(f"💾 内存使用率高 ({max_memory_percent:.1f}%) - 考虑减小batch_size")
                
        # 分析数据加载瓶颈
        training_time = self.timings.get('训练', 0)
        data_loading_time = self.timings.get('数据加载', 0)
        
        if data_loading_time > 0 and training_time > 0:
            data_loading_ratio = data_loading_time / (training_time + data_loading_time)
            if data_loading_ratio > 0.2:  # 数据加载占总时间20%以上
                bottlenecks['data_loading_bottleneck'] = True
                bottlenecks['recommendations'].append(f"📁 数据加载慢 ({data_loading_ratio*100:.1f}%) - 增加num_workers或使用pin_memory")
                
        # 综合分析
        if not any([bottlenecks['gpu_bottleneck'], bottlenecks['cpu_bottleneck'], 
                   bottlenecks['memory_bottleneck'], bottlenecks['data_loading_bottleneck']]):
            bottlenecks['recommendations'].append("✅ 未发现明显瓶颈，性能配置较为合理")
            
        return bottlenecks
        
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        bottlenecks = self.analyze_bottlenecks()
        
        report_path = self.log_dir / "bottleneck_report.txt"
        with open(report_path, 'w') as f:
            f.write("=== 🔍 智能瓶颈分析报告 ===\n")
            f.write(f"生成时间: {datetime.now()}\n\n")
            
            # 瓶颈总结
            f.write("📊 瓶颈检测结果:\n")
            f.write(f"  GPU瓶颈: {'是' if bottlenecks['gpu_bottleneck'] else '否'}\n")
            f.write(f"  CPU瓶颈: {'是' if bottlenecks['cpu_bottleneck'] else '否'}\n")
            f.write(f"  内存瓶颈: {'是' if bottlenecks['memory_bottleneck'] else '否'}\n")
            f.write(f"  数据加载瓶颈: {'是' if bottlenecks['data_loading_bottleneck'] else '否'}\n\n")
            
            # 性能指标统计
            if self.gpu_metrics:
                gpu_utils = [m['utilization'] for m in self.gpu_metrics if m.get('utilization', 0) > 0]
                if gpu_utils:
                    f.write("🎯 GPU性能统计:\n")
                    f.write(f"  平均利用率: {np.mean(gpu_utils):.1f}%\n")
                    f.write(f"  最大利用率: {max(gpu_utils):.1f}%\n")
                    f.write(f"  最小利用率: {min(gpu_utils):.1f}%\n")
                    
                gpu_memory = [m['memory_allocated'] for m in self.gpu_metrics]
                f.write(f"  最大GPU内存: {max(gpu_memory):.2f}GB\n\n")
                
            if self.cpu_metrics:
                cpu_utils = [m['utilization'] for m in self.cpu_metrics]
                f.write("🖥️  CPU性能统计:\n")
                f.write(f"  平均利用率: {np.mean(cpu_utils):.1f}%\n")
                f.write(f"  最大利用率: {max(cpu_utils):.1f}%\n\n")
                
            # 时间分析
            f.write("⏱️  时间分析:\n")
            total_time = sum(self.timings.values())
            for phase, duration in self.timings.items():
                percentage = (duration / total_time * 100) if total_time > 0 else 0
                f.write(f"  {phase}: {duration:.2f}s ({percentage:.1f}%)\n")
            f.write(f"  总时间: {total_time:.2f}s\n\n")
            
            # 优化建议
            f.write("🚀 优化建议:\n")
            for i, rec in enumerate(bottlenecks['recommendations'], 1):
                f.write(f"  {i}. {rec}\n")
                
        print(f"📋 详细分析报告保存到: {report_path}")
        
        # 保存原始数据
        self._save_raw_data()
        
        # 生成可视化图表
        self._generate_plots()
        
        return bottlenecks
        
    def _save_raw_data(self):
        """保存原始监控数据"""
        data = {
            'gpu_metrics': list(self.gpu_metrics),
            'cpu_metrics': list(self.cpu_metrics),
            'memory_metrics': list(self.memory_metrics),
            'io_metrics': list(self.io_metrics),
            'timings': self.timings
        }
        
        with open(self.profiler_dir / "raw_metrics.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    def _generate_plots(self):
        """生成性能可视化图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # GPU利用率图
            if self.gpu_metrics:
                gpu_times = [m['timestamp'] for m in self.gpu_metrics if m.get('utilization', 0) > 0]
                gpu_utils = [m['utilization'] for m in self.gpu_metrics if m.get('utilization', 0) > 0]
                if gpu_times and gpu_utils:
                    axes[0, 0].plot(gpu_times, gpu_utils, 'g-', linewidth=2)
                    axes[0, 0].axhline(y=70, color='r', linestyle='--', label='目标线 (70%)')
                    axes[0, 0].set_title('GPU利用率')
                    axes[0, 0].set_ylabel('利用率 (%)')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True)
            
            # CPU利用率图
            if self.cpu_metrics:
                cpu_times = [m['timestamp'] for m in self.cpu_metrics]
                cpu_utils = [m['utilization'] for m in self.cpu_metrics]
                axes[0, 1].plot(cpu_times, cpu_utils, 'b-', linewidth=2)
                axes[0, 1].axhline(y=80, color='r', linestyle='--', label='警告线 (80%)')
                axes[0, 1].set_title('CPU利用率')
                axes[0, 1].set_ylabel('利用率 (%)')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # 内存使用图
            if self.memory_metrics:
                mem_times = [m['timestamp'] for m in self.memory_metrics]
                mem_utils = [m['percent'] for m in self.memory_metrics]
                axes[1, 0].plot(mem_times, mem_utils, 'orange', linewidth=2)
                axes[1, 0].axhline(y=85, color='r', linestyle='--', label='警告线 (85%)')
                axes[1, 0].set_title('内存使用率')
                axes[1, 0].set_ylabel('使用率 (%)')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # GPU内存使用图
            if self.gpu_metrics:
                gpu_mem_times = [m['timestamp'] for m in self.gpu_metrics]
                gpu_mem_usage = [m['memory_allocated'] for m in self.gpu_metrics]
                axes[1, 1].plot(gpu_mem_times, gpu_mem_usage, 'purple', linewidth=2)
                axes[1, 1].set_title('GPU内存使用')
                axes[1, 1].set_ylabel('内存 (GB)')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.profiler_dir / "performance_charts.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 性能图表保存到: {self.profiler_dir}/performance_charts.png")
            
        except Exception as e:
            print(f"生成图表时出错: {e}")

# -----------------------------------------------------------------------------
# 简化的训练脚本
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_log_directory(base_dir: str = "logs") -> Path:
    """创建日志目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir) / f"bottleneck_analysis_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def load_dataset(x_path: str, y_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """加载数据集"""
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)
    
    n = min(len(x_df), len(y_df))
    x_df = x_df.iloc[:n]
    y_df = y_df.iloc[:n]
    
    x_tensor = torch.tensor(x_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_df.values.reshape(-1, 1), dtype=torch.float32)
    
    return x_tensor, y_tensor

def build_nfs_model(context_features: int, flow_features: int = 1, 
                   hidden_features: int = 128, n_layers: int = 10) -> Flow:
    """构建归一化流模型"""
    base_dist = StandardNormal([flow_features])
    transforms: List = []
    for _ in range(n_layers):
        transforms += [
            RandomPermutation(features=flow_features),
            MaskedAffineAutoregressiveTransform(
                features=flow_features,
                hidden_features=hidden_features,
                context_features=context_features,
            ),
        ]
    return Flow(CompositeTransform(transforms), base_dist)

def train_with_bottleneck_analysis(model: Flow, train_loader: DataLoader, 
                                 analyzer: BottleneckAnalyzer, epochs: int = 50, 
                                 lr: float = 1e-3) -> List[float]:
    """带瓶颈分析的训练函数"""
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    losses = []
    
    print(f"🎯 训练配置:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   Batches per epoch: {len(train_loader)}")
    print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   混合精度: {'启用' if scaler else '禁用'}")
    
    with analyzer.time_phase("训练"):
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            # 每10个epoch分析一次瓶颈
            if epoch % 10 == 1:
                print(f"\n📊 Epoch {epoch} - 实时瓶颈分析:")
                current_bottlenecks = analyzer.analyze_bottlenecks()
                for rec in current_bottlenecks['recommendations'][:3]:  # 显示前3个建议
                    print(f"   {rec}")
            
            for batch_idx, (cx, y) in enumerate(train_loader):
                # 数据传输到GPU
                cx = cx.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                # 前向传播和反向传播
                if scaler:
                    with torch.amp.autocast('cuda'):
                        loss = -model.log_prob(inputs=y, context=cx).mean()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = -model.log_prob(inputs=y, context=cx).mean()
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            epoch_loss /= batch_count
            losses.append(epoch_loss)
            scheduler.step()
            
            # 定期输出进度
            if epoch % 10 == 0 or epoch == 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:3d}/{epochs} | Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}")
                
                # 显示当前资源使用
                if analyzer.gpu_metrics:
                    latest_gpu = analyzer.gpu_metrics[-1]
                    print(f"   GPU: {latest_gpu.get('utilization', 0):.1f}% | "
                          f"Memory: {latest_gpu['memory_allocated']:.1f}GB")
                    
                if analyzer.cpu_metrics:
                    latest_cpu = analyzer.cpu_metrics[-1]
                    print(f"   CPU: {latest_cpu['utilization']:.1f}% | "
                          f"RAM: {analyzer.memory_metrics[-1]['percent']:.1f}%")
    
    return losses

def main():
    parser = argparse.ArgumentParser(description="智能瓶颈分析训练")
    parser.add_argument("--x_csv", type=str, default="data/x_data.csv", help="输入特征文件")
    parser.add_argument("--y_csv", type=str, default="data/y_data.csv", help="目标值文件")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2048, help="批次大小")
    parser.add_argument("--hidden_features", type=int, default=128, help="隐藏层大小")
    parser.add_argument("--n_layers", type=int, default=10, help="模型层数")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载进程数")
    
    args = parser.parse_args()
    
    # 创建日志目录和分析器
    log_dir = create_log_directory()
    analyzer = BottleneckAnalyzer(log_dir, sample_interval=0.5)
    
    print(f"📁 分析结果将保存到: {log_dir}")
    print(f"🖥️  系统信息: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total/1024**3:.1f}GB RAM")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎯 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    try:
        # 开始监控
        analyzer.start_monitoring()
        
        # 加载数据
        with analyzer.time_phase("数据加载"):
            x, y = load_dataset(args.x_csv, args.y_csv)
            print(f"📊 数据集: {len(x)} 样本, {x.shape[1]} 特征")
            
            # 划分训练集和测试集
            n_train = int(len(x) * args.train_ratio)
            indices = torch.randperm(len(x))
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
            
            x_train, y_train = x[train_indices], y[train_indices]
            x_test, y_test = x[test_indices], y[test_indices]
            
            print(f"📦 训练集: {len(x_train)} 样本, 测试集: {len(x_test)} 样本")
        
        # 创建数据加载器
        with analyzer.time_phase("数据加载器创建"):
            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True if args.num_workers > 0 else False
            )
            
            print(f"📦 DataLoader配置:")
            print(f"   Batch size: {args.batch_size}")
            print(f"   Num workers: {args.num_workers}")
            print(f"   Pin memory: {torch.cuda.is_available()}")
        
        # 构建模型
        with analyzer.time_phase("模型构建"):
            model = build_nfs_model(
                context_features=x.shape[1],
                hidden_features=args.hidden_features,
                n_layers=args.n_layers
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"🧠 模型信息:")
            print(f"   总参数: {total_params:,}")
            print(f"   可训练参数: {trainable_params:,}")
            print(f"   模型大小: {total_params * 4 / 1024**2:.1f}MB")
        
        # 训练模型
        print(f"\n🚀 开始训练...")
        losses = train_with_bottleneck_analysis(
            model, train_loader, analyzer,
            epochs=args.epochs
        )
        
        print(f"✅ 训练完成! 最终损失: {losses[-1]:.4f}")
        
        # 停止监控
        analyzer.stop_monitoring()
        
        # 生成综合分析报告
        print(f"\n🔍 生成瓶颈分析报告...")
        bottlenecks = analyzer.generate_comprehensive_report()
        
        # 输出关键发现
        print(f"\n=== 🎯 瓶颈分析结果 ===")
        print(f"GPU瓶颈: {'⚠️  是' if bottlenecks['gpu_bottleneck'] else '✅ 否'}")
        print(f"CPU瓶颈: {'⚠️  是' if bottlenecks['cpu_bottleneck'] else '✅ 否'}")
        print(f"内存瓶颈: {'⚠️  是' if bottlenecks['memory_bottleneck'] else '✅ 否'}")
        print(f"数据加载瓶颈: {'⚠️  是' if bottlenecks['data_loading_bottleneck'] else '✅ 否'}")
        
        print(f"\n🚀 优化建议:")
        for i, rec in enumerate(bottlenecks['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # 保存模型
        model_path = log_dir / "trained_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'losses': losses,
            'config': {
                'context_features': x.shape[1],
                'hidden_features': args.hidden_features,
                'n_layers': args.n_layers
            },
            'bottleneck_analysis': bottlenecks
        }, model_path)
        
        print(f"\n💾 模型保存到: {model_path}")
        print(f"📋 完整报告: {log_dir}/bottleneck_report.txt")
        print(f"📈 性能图表: {log_dir}/bottleneck_analysis/performance_charts.png")
        
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.stop_monitoring()

if __name__ == "__main__":
    main()