import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import matplotlib as mpl
print(mpl.get_cachedir())
 
plt.rcParams["font.sans-serif"]=["simhei"]
plt.rcParams["axes.unicode_minus"]=False
class GravitationalWaveFilter:
    """引力波匹配滤波器"""
    
    def __init__(self, sampling_rate=4096, duration=4):
        """
        初始化滤波器
        
        Parameters:
        sampling_rate: 采样率 (Hz)
        duration: 数据持续时间 (秒)
        """
        self.fs = sampling_rate
        self.duration = duration
        self.N = int(sampling_rate * duration)
        self.time = np.linspace(0, duration, self.N)
        self.freqs = fftfreq(self.N, 1/self.fs)
        
    def chirp_template(self, M=30, tc=2.0, phi0=0):
        """
        生成啁啾信号模板（简化的引力波信号）
        
        Parameters:
        M: 总质量 (太阳质量)
        tc: 合并时间 (秒)
        phi0: 初始相位
        """
        # 简化的啁啾参数
        f0 = 35  # 起始频率 (Hz)
        tau = tc - self.time
        
        # 避免除零和负数
        tau = np.where(tau > 0.01, tau, 0.01)
        
        # 频率演化 (简化模型)
        freq = f0 * (tau / tc) ** (-3/8)
        freq = np.where(freq < 250, freq, 250)  # 频率上限
        
        # 振幅演化
        amplitude = (tau / tc) ** (-1/4)
        amplitude = np.where(self.time < tc, amplitude, 0)
        
        # 相位演化
        phase = phi0 + 2 * np.pi * np.cumsum(freq) / self.fs
        
        # 两个偏振分量
        h_plus = amplitude * np.cos(phase)
        h_cross = amplitude * np.sin(phase)
        
        return h_plus, h_cross
    
    def add_noise(self, signal, noise_level=1.0):
        """添加高斯白噪声"""
        noise = np.random.normal(0, noise_level, len(signal))
        return signal + noise
    
    def power_spectral_density(self, data):
        """计算功率谱密度"""
        freqs, psd = signal.welch(data, self.fs, nperseg=self.N//4)
        return freqs, psd
    
    def matched_filter(self, data, template):
        """
        执行匹配滤波
        
        Parameters:
        data: 观测数据
        template: 模板信号
        
        Returns:
        snr: 信噪比时间序列
        optimal_snr: 最优信噪比
        """
        # 转到频域
        data_fft = fft(data)
        template_fft = fft(template)
        
        # 估计噪声功率谱密度
        _, psd = self.power_spectral_density(data)
        
        # 插值PSD到完整频率网格
        psd_interp = np.interp(np.abs(self.freqs), 
                              np.linspace(0, self.fs/2, len(psd)), psd)
        psd_interp[psd_interp == 0] = np.inf  # 避免除零
        
        # 噪声加权
        weight = 1.0 / psd_interp
        weight[np.abs(self.freqs) > self.fs/2] = 0  # 只保留有效频率
        
        # 匹配滤波在频域
        numerator = data_fft * np.conj(template_fft) * weight
        denominator = np.sum(np.abs(template_fft)**2 * weight)
        
        # 转回时域得到SNR
        snr_complex = ifft(numerator) * self.N
        snr = np.abs(snr_complex) / np.sqrt(denominator)
        
        optimal_snr = np.max(snr)
        
        return snr, optimal_snr
    
    def demonstrate_matched_filtering(self):
        """演示匹配滤波过程"""
        print("=== 引力波匹配滤波演示 ===\n")
        
        # 1. 生成真实信号模板
        print("1. 生成引力波信号模板...")
        h_plus_true, _ = self.chirp_template(M=30, tc=2.0)
        
        # 2. 创建观测数据（信号 + 噪声）
        print("2. 创建观测数据（真实信号 + 噪声）...")
        noise_level = 10
        observed_data = self.add_noise(h_plus_true, noise_level)
        
        # 3. 生成搜索模板（略有不同的参数）
        print("3. 生成搜索模板...")
        h_plus_template, _ = self.chirp_template(M=30, tc=2.05)  # 稍微不同的合并时间
        
        # 4. 执行匹配滤波
        print("4. 执行匹配滤波...")
        snr, optimal_snr = self.matched_filter(observed_data, h_plus_template)
        
        # 5. 寻找峰值
        peak_idx = np.argmax(snr)
        peak_time = self.time[peak_idx]
        
        print(f"\n=== 结果 ===")
        print(f"最优信噪比: {optimal_snr:.2f}")
        print(f"检测时间: {peak_time:.3f} 秒")
        print(f"真实合并时间: 2.000 秒")
        print(f"时间误差: {abs(peak_time - 2.0)*1000:.1f} 毫秒")
        
        # 绘图
        self.plot_results(observed_data, h_plus_true, h_plus_template, 
                         snr, peak_idx)
        
        return snr, optimal_snr
    
    def plot_results(self, data, true_signal, template, snr, peak_idx):
        """绘制结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 观测数据
        axes[0,0].plot(self.time, data, 'b-', alpha=0.7, label='观测数据')
        axes[0,0].plot(self.time, true_signal, 'r-', label='真实信号')
        axes[0,0].set_xlabel('时间 (秒)')
        axes[0,0].set_ylabel('应变')
        axes[0,0].set_title('观测数据 vs 真实信号')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # 模板
        axes[0,1].plot(self.time, template, 'g-', label='搜索模板')
        axes[0,1].plot(self.time, true_signal, 'r--', alpha=0.7, label='真实信号')
        axes[0,1].set_xlabel('时间 (秒)')
        axes[0,1].set_ylabel('应变')
        axes[0,1].set_title('搜索模板 vs 真实信号')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 信噪比时间序列
        axes[1,0].plot(self.time, snr, 'purple', linewidth=2)
        axes[1,0].axvline(self.time[peak_idx], color='red', linestyle='--', 
                         label=f'峰值时间: {self.time[peak_idx]:.3f}s')
        axes[1,0].set_xlabel('时间 (秒)')
        axes[1,0].set_ylabel('信噪比')
        axes[1,0].set_title('匹配滤波输出 (SNR)')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # 频域分析
        data_fft = np.abs(fft(data))
        template_fft = np.abs(fft(template))
        freqs_plot = self.freqs[:self.N//2]
        
        axes[1,1].loglog(freqs_plot[1:], data_fft[1:self.N//2], 
                        alpha=0.7, label='观测数据')
        axes[1,1].loglog(freqs_plot[1:], template_fft[1:self.N//2], 
                        label='模板')
        axes[1,1].set_xlabel('频率 (Hz)')
        axes[1,1].set_ylabel('振幅')
        axes[1,1].set_title('频域对比')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()

# 演示匹配滤波
if __name__ == "__main__":
    # 创建滤波器实例
    gw_filter = GravitationalWaveFilter(sampling_rate=4096, duration=4)
    
    # 运行演示
    snr, optimal_snr = gw_filter.demonstrate_matched_filtering()
    
    print(f"\n=== 匹配滤波评估 ===")
    if optimal_snr > 8:
        print("✓ 强信号检测 (SNR > 8)")
    elif optimal_snr > 5:
        print("✓ 可信检测 (SNR > 5)")
    else:
        print("× 信号太弱，需要更好的模板或更低噪声")
    
    # 展示不同质量参数的影响
    print("\n=== 模板匹配测试 ===")
    masses = [25, 30, 35, 40]
    snr_values = []
    
    # 固定的"真实"信号
    h_true, _ = gw_filter.chirp_template(M=30, tc=2.0)
    observed = gw_filter.add_noise(h_true, 0.5)
    
    for mass in masses:
        h_template, _ = gw_filter.chirp_template(M=mass, tc=2.0)
        snr_test, optimal_snr_test = gw_filter.matched_filter(observed, h_template)
        snr_values.append(optimal_snr_test)
        print(f"质量 {mass} M☉: SNR = {optimal_snr_test:.2f}")
    
    best_mass = masses[np.argmax(snr_values)]
    print(f"\n最佳匹配质量: {best_mass} M☉ (真实值: 30 M☉)")