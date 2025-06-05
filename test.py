
import lal
import lalsimulation as lalsim
import numpy as np
import matplotlib.pyplot as plt


# 设置双星系统参数
m1 = 30.0 * lal.MSUN_SI  # 天体1质量 (30倍太阳质量)
m2 = 30.0 * lal.MSUN_SI  # 天体2质量 (30倍太阳质量)
spin1x = 0.0            # 天体1自旋x分量
spin1y = 0.0            # 天体1自旋y分量  
spin1z = 0.0            # 天体1自旋z分量
spin2x = 0.0            # 天体2自旋x分量
spin2y = 0.0            # 天体2自旋y分量
spin2z = 0.0            # 天体2自旋z分量
distance = 100.0 * lal.PC_SI * 1e6  # 距离 (100 Mpc)
inclination = 0.0       # 轨道倾角
phiref = 0.0           # 参考相位
longAscNodes = 0.0     # 升交点经度
eccentricity = 0.0     # 偏心率
meanPerAno = 0.0       # 平近点角

# 时间参数
deltaT = 1.0/4096.0    # 采样间隔
f_min = 20.0          # 起始频率 (Hz)
f_ref = 20.0          # 参考频率 (Hz)

# 选择波形近似模型
approximant = lalsim.IMRPhenomD

# 生成时域波形
hp, hc = lalsim.SimInspiralChooseTDWaveform(
    m1, m2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
    distance, inclination, phiref, longAscNodes, eccentricity, meanPerAno,
    deltaT, f_min, f_ref, None, approximant
)


# 频域参数
deltaF = 1.0/4.0      # 频率分辨率
f_max = 1024.0        # 最大频率

# 生成频域波形  
hp_f, hc_f = lalsim.SimInspiralChooseFDWaveform(
    m1, m2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
    distance, inclination, phiref, longAscNodes, eccentricity, meanPerAno,
    deltaF, f_min, f_max, f_ref, None, approximant
)


# 从LAL时间序列提取数据
time = np.arange(hp.data.length) * hp.deltaT + hp.epoch
strain_plus = hp.data.data
strain_cross = hc.data.data

# 绘制波形
plt.figure(figsize=(12, 6))
plt.plot(time, strain_plus, label='h_plus')
plt.plot(time, strain_cross, label='h_cross')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.legend()
plt.title('Gravitational Wave Strain')
plt.show()


