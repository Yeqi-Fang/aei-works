import numpy as np
import math
import numba

# def cal_cost_overall_700yr_new_timing_model(fmin, fmax, fdotmin, fdotmax, fsp, fdotsp,
#                                            fband_cell, fdotband_cell, m0, m1, m2, r1, r2,
#                                            Tcoh_day, nSFT, nSeg, tau, n, fbandwidth):
#     """
#     计算700年引力波搜索的总体计算成本

#     参数:
#     fmin, fmax: 频率搜索范围 (Hz)
#     fdotmin, fdotmax: 频率导数搜索范围 (Hz/s)
#     fsp, fdotsp: 特定频率和频率导数值
#     fband_cell, fdotband_cell: 频率和频率导数的网格单元大小
#     m0, m1, m2: 搜索参数的失配参数
#     r1, r2: 精化参数
#     Tcoh_day: 相干时间 (天)
#     nSFT, nSeg: SFT数量和段数
#     tau: 年龄 (年)
#     n: 制动指数
#     fbandwidth: 固定带宽

#     返回:
#     CPU_year: 每个网格点的CPU年数
#     CPU_year_sp: 总CPU年数
#     co_cost_rate: 相干成本率
#     co_cost_rate_sp: 总相干成本率
#     """

#     # 时间常数 (秒)
#     tau_FFT = 3.3e-8      # FFT时间常数
#     tau_spin = 7.54e-8    # 自旋时间常数
#     tau_Fbin = 6.0e-8     # 频率bin时间常数

#     # 探测器数量
#     N_det = 2

#     # 时间转换
#     age_year = tau
#     age_sec = age_year * 86400 * 365.24  # 年龄转换为秒
#     f_fixband = fbandwidth

#     # 观测时间跨度
#     T_span_day = 183
#     T_span = T_span_day * 86400  # 转换为秒

#     # 计算搜索参数
#     R_value_rs = np.sqrt(12 * m0) / np.pi

#     # 确保r1, r2为奇数
#     r1 = int(np.floor(r1/2)) * 2 + 1
#     r2 = int(np.floor(r2/2)) * 2 + 1

#     # 时间转换
#     Tcoh = Tcoh_day * 3600 * 24  # 相干时间转换为秒
#     age = age_year * 3600 * 24 * 365.24  # 年龄转换为秒
#     N_seg = nSeg

#     # 计算边带频率和相关参数
#     f_sideband = 0.5 * (T_span * np.sqrt(180 * m1) / np.pi / Tcoh**2 +
#                        T_span**2 * np.sqrt(25200 * m2) / np.pi / Tcoh**3)

#     N_sb = math.ceil(f_sideband * np.pi * Tcoh / np.sqrt(12 * m0))
#     Nfc = math.ceil(f_fixband * np.pi * Tcoh / np.sqrt(12 * m0))
#     N_F_total = Nfc + N_sb
#     DDD = math.ceil(np.sqrt(12 * m0) / np.pi / Tcoh * Tcoh)

#     # 段数截止
#     if N_seg <= 12:
#         N_seg_cutoff = N_seg
#     else:
#         N_seg_cutoff = 12

#     # 创建频率和频率导数网格
#     fband_k = np.arange(fmin + fband_cell/2, fmax - fband_cell/2 + fband_cell, fband_cell)
#     fdotband_k = np.arange(fdotmin + fdotband_cell/2, fdotmax - fdotband_cell/2 + fdotband_cell, fdotband_cell)

#     # 初始化结果数组
#     cost_co = np.zeros((len(fband_k), len(fdotband_k)))
#     cost_inco = np.zeros((len(fband_k), len(fdotband_k)))
#     cost_Bayes = np.zeros((len(fband_k), len(fdotband_k)))
#     computingcost = np.zeros((len(fband_k), len(fdotband_k)))
#     CPU_year = np.zeros((len(fband_k), len(fdotband_k)))
#     tau_Rs_sp = np.zeros((len(fband_k), len(fdotband_k)))

#     # 双重循环计算每个网格点的成本
#     for ik in range(len(fband_k)):
#         for jk in range(len(fdotband_k)):
#             # 检查频率导数是否在有效范围内
#             if fdotband_k[jk] > fband_k[ik] / age_sec:
#                 # 超出范围时设为0
#                 cost_co_sp = 0
#                 cost_inco_sp = 0
#                 cost_Bayes_sp = 0
#                 cost_sec_in_in_WUband = 0
#                 cost_sec_in_in_cell = 0
#                 tau_Rs = 0
#             else:
#                 # 计算二阶频率导数
#                 fddot = n * fband_k[ik] / age_sec**2

#                 # 计算各种参数
#                 Nddfc = math.ceil(5 * fband_k[ik] * np.pi * Tcoh**3 / np.sqrt(25200 * m2) / age**2)
#                 Ndfc = math.ceil(fdotband_cell * np.pi * Tcoh**2 / np.sqrt(180 * m1))

#                 # 计算采样频段
#                 f_band_samp = (f_sideband + f_fixband + 2e-4 * fband_k[ik] +
#                               abs(fdotband_k[jk]) * T_span + fddot * T_span**2 + 16/1800)

#                 N_samp = f_band_samp / (np.sqrt(12 * m0) / np.pi / Tcoh / DDD)
#                 N_samp = 2**math.ceil(math.log2(N_samp))  # 向上取到2的幂次

#                 # 计算模板和搜索参数
#                 Ninc = r1 * r2 * Ndfc * Nfc * Nddfc
#                 Ncoh = Ndfc * N_F_total * Nddfc

#                 N_Fbin = N_F_total

#                 # 计算各种时间成本
#                 tau_Rs = (tau_FFT + R_value_rs * tau_spin) * N_samp / N_Fbin + tau_Fbin
#                 tau_F0 = tau_Rs
#                 tau_D0 = -3.7208e-10 * N_seg_cutoff + 7.2835e-09
#                 tau_Bayes = 4.4e-8

#                 # 计算各种成本组件
#                 cost_co_sp = N_det * N_seg * Ncoh * tau_F0 * fband_cell / f_fixband
#                 cost_inco_sp = N_seg * tau_D0 * Ninc * fband_cell / f_fixband
#                 cost_Bayes_sp = Ninc * tau_Bayes * fband_cell / f_fixband
#                 cost_sec_in_in_WUband = (N_det * N_seg * Ncoh * tau_F0 +
#                                         N_seg * tau_D0 * Ninc + Ninc * tau_Bayes)
#                 cost_sec_in_in_cell = cost_sec_in_in_WUband * fband_cell / f_fixband

#             # 存储结果
#             cost_co[ik, jk] = cost_co_sp
#             cost_inco[ik, jk] = cost_inco_sp
#             cost_Bayes[ik, jk] = cost_Bayes_sp
#             computingcost[ik, jk] = cost_sec_in_in_cell
#             CPU_year[ik, jk] = computingcost[ik, jk] / (365.24 * 24 * 3600)
#             tau_Rs_sp[ik, jk] = tau_Rs

#     # 计算总计成本
#     computingcost_sp = np.sum(computingcost)
#     co_cost_rate_sp = np.sum(cost_co) / np.sum(computingcost) if np.sum(computingcost) > 0 else 0
#     CPU_year_sp = np.sum(CPU_year)

#     # 计算网格数量
#     nn_f = int(np.floor((fmax - fmin) / fband_cell))
#     nn_fdot = int(np.floor((fdotmax - fdotmin) / fdotband_cell))

#     # 原始代码最后设置这些为0，但这可能是错误的
#     # CPU_year = 0
#     # co_cost_rate = 0

#     # 返回有意义的值
#     co_cost_rate = co_cost_rate_sp

#     return CPU_year, CPU_year_sp, co_cost_rate, co_cost_rate_sp


# import numpy as np, math

SECONDS_PER_YEAR = 365.24 * 24 * 3600.0  # used repeatedly


@numba.jit(nopython=True, nogil=True)  # for numba performance
def cal_cost_overall_700yr_new_timing_model(  # ← same name!
    fmin,
    fmax,
    fsp,
    fdotsp,  # (unused)
    fband_cell,
    m0,
    m1,
    m2,
    r1,
    r2,
    Tcoh_day,
    nSFT,
    nSeg,
    tau,  # source age [yr]
    n,  # braking index
    fbandwidth,
):
    """
    Return only the *scalar* totals to avoid huge memory use.

    Returns
    -------
    CPU_year_sp        : total CPU-years for the whole grid
    co_cost_rate_sp    : overall coherent-cost fraction (0–1)

    (The old per-cell outputs are omitted to save RAM.)
    """

    # ------------------------------------------------------------------ #
    # constants
    tau_FFT = 3.3e-8
    tau_spin = 7.54e-8
    tau_Fbin = 6.0e-8
    N_det = 2

    fdotmax = 0.0  # unused, but required by the function signature

    # time conversions
    age_sec = tau * SECONDS_PER_YEAR
    Tcoh = Tcoh_day * 86400.0
    T_span = 183 * 86400.0

    # derived grid constants
    r1 = (r1 // 2) * 2 + 1  # make odd
    r2 = (r2 // 2) * 2 + 1
    R_value_rs = np.sqrt(12 * m0) / np.pi
    f_sideband = 0.5 * (
        T_span * np.sqrt(180 * m1) / (np.pi * Tcoh**2)
        + T_span**2 * np.sqrt(25200 * m2) / (np.pi * Tcoh**3)
    )
    N_sb = np.ceil(f_sideband * np.pi * Tcoh / np.sqrt(12 * m0))
    Nfc = np.ceil(fbandwidth * np.pi * Tcoh / np.sqrt(12 * m0))
    N_F_total = N_sb + Nfc
    DDD = np.ceil(np.sqrt(12 * m0) / np.pi)

    # segments
    N_seg = nSeg
    N_seg_cutoff = N_seg if N_seg <= 12 else 12

    # frequency / fdot grids (centres)
    fband_k = np.arange(
        fmin + fband_cell / 2, fmax - fband_cell / 2 + fband_cell, fband_cell
    )

    # ------------------------------------------------------------------ #
    # scalars we will accumulate
    CPU_year_sp = 0.0
    cost_co_sum = 0.0
    computingcost_sum = 0.0

    # nested loops but *without* storing per-cell arrays
    for f in fband_k:
        fdotmax = 0.0  # unused, but required by the function signature
        fdotmin = -f / (2 - 1) / age_sec  # fdotmin for this f
        fdotband_cell = (fdotmax - fdotmin) / 200
        fdot_k = np.arange(
            fdotmin + fdotband_cell / 2,
            fdotmax - fdotband_cell / 2 + fdotband_cell,
            fdotband_cell,
        )
        
        for fd in fdot_k:

            if fd > f / age_sec:
                # cell ruled out by age prior → contributes zero
                continue

            fddot = n * f / age_sec**2
            Nddfc = np.ceil(5 * f * np.pi * Tcoh**3 / np.sqrt(25200 * m2) / age_sec**2)
            Ndfc = np.ceil(fdotband_cell * np.pi * Tcoh**2 / np.sqrt(180 * m1))
            Ninc = r1 * r2 * Ndfc * Nfc * Nddfc
            Ncoh = Ndfc * N_F_total * Nddfc

            f_band_samp = (
                f_sideband
                + fbandwidth
                + 2e-4 * f
                + abs(fd) * T_span
                + fddot * T_span**2
                + 16 / 1800
            )
            N_samp = f_band_samp / (np.sqrt(12 * m0) / np.pi / Tcoh / DDD)
            N_samp = 1 << int(np.ceil(np.log2(N_samp)))  # next power of 2

            tau_Rs = (tau_FFT + R_value_rs * tau_spin) * N_samp / N_F_total + tau_Fbin
            tau_D0 = -3.7208e-10 * N_seg_cutoff + 7.2835e-09

            # costs (seconds)
            cost_co = N_det * N_seg * Ncoh * tau_Rs * fband_cell / fbandwidth
            cost_inco = N_seg * tau_D0 * Ninc * fband_cell / fbandwidth
            cost_bayes = Ninc * 4.4e-8 * fband_cell / fbandwidth
            cost_total = cost_co + cost_inco + cost_bayes  # seconds

            # accumulate
            CPU_year_sp += cost_total / SECONDS_PER_YEAR
            cost_co_sum += cost_co
            computingcost_sum += cost_total

    # overall coherent fraction
    co_cost_rate_sp = cost_co_sum / computingcost_sum if computingcost_sum else 0.0

    # del fband_k, fdot_k  # free memory

    return CPU_year_sp, co_cost_rate_sp




@numba.jit(nopython=True, nogil=True)
def cal_cost_overall_700yr_new_timing_model_optimized(
    fmin,
    fmax,
    fsp,
    fdotsp,  # (unused)
    fband_cell,
    fdotband_cell,
    m0,
    m1,
    m2,
    r1,
    r2,
    Tcoh_day,
    nSFT,
    nSeg,
    tau,  # source age [yr]
    n,  # braking index
    fbandwidth,
):
    """
    内存优化版本：使用循环计数器而不是创建大数组
    """
    # ------------------------------------------------------------------ #
    # constants
    tau_FFT = 3.3e-8
    tau_spin = 7.54e-8
    tau_Fbin = 6.0e-8
    N_det = 2
    SECONDS_PER_YEAR = 365.25 * 24 * 3600

    fdotmax = 0.0  # unused, but required by the function signature

    # time conversions
    age_sec = tau * SECONDS_PER_YEAR
    Tcoh = Tcoh_day * 86400.0
    T_span = 183 * 86400.0

    # derived grid constants
    r1 = (r1 // 2) * 2 + 1  # make odd
    r2 = (r2 // 2) * 2 + 1
    R_value_rs = np.sqrt(12 * m0) / np.pi
    f_sideband = 0.5 * (
        T_span * np.sqrt(180 * m1) / (np.pi * Tcoh**2)
        + T_span**2 * np.sqrt(25200 * m2) / (np.pi * Tcoh**3)
    )
    N_sb = np.ceil(f_sideband * np.pi * Tcoh / np.sqrt(12 * m0))
    Nfc = np.ceil(fbandwidth * np.pi * Tcoh / np.sqrt(12 * m0))
    N_F_total = N_sb + Nfc
    DDD = np.ceil(np.sqrt(12 * m0) / np.pi)

    # segments
    N_seg = nSeg
    N_seg_cutoff = N_seg if N_seg <= 12 else 12

    # ------------------------------------------------------------------ #
    # scalars we will accumulate
    CPU_year_sp = 0.0
    cost_co_sum = 0.0
    computingcost_sum = 0.0

    # 计算频率网格的数量，避免创建数组
    n_freq_points = int(np.ceil((fmax - fmin) / fband_cell))
    
    # 使用循环计数器而不是数组
    for i_freq in range(n_freq_points):
        f = fmin + (i_freq + 0.5) * fband_cell
        
        # 检查频率是否在范围内
        if f > fmax:
            break
            
        fdotmax = 0.0  # unused, but required by the function signature
        fdotmin = -f / (2 - 1) / age_sec  # fdotmin for this f
        
        # 计算fdot网格的数量，避免创建数组
        if fdotmax <= fdotmin:
            continue
            
        n_fdot_points = int(np.ceil((fdotmax - fdotmin) / fdotband_cell))
        
        for i_fdot in range(n_fdot_points):
            fd = fdotmin + (i_fdot + 0.5) * fdotband_cell
            
            # 检查fdot是否在范围内
            if fd > fdotmax:
                break
                
            if fd > f / age_sec:
                # cell ruled out by age prior → contributes zero
                continue

            fddot = n * f / age_sec**2
            Nddfc = np.ceil(5 * f * np.pi * Tcoh**3 / np.sqrt(25200 * m2) / age_sec**2)
            Ndfc = np.ceil(fdotband_cell * np.pi * Tcoh**2 / np.sqrt(180 * m1))
            Ninc = r1 * r2 * Ndfc * Nfc * Nddfc
            Ncoh = Ndfc * N_F_total * Nddfc

            f_band_samp = (
                f_sideband
                + fbandwidth
                + 2e-4 * f
                + abs(fd) * T_span
                + fddot * T_span**2
                + 16 / 1800
            )
            N_samp = f_band_samp / (np.sqrt(12 * m0) / np.pi / Tcoh / DDD)
            N_samp = 1 << int(np.ceil(np.log2(N_samp)))  # next power of 2

            tau_Rs = (tau_FFT + R_value_rs * tau_spin) * N_samp / N_F_total + tau_Fbin
            tau_D0 = -3.7208e-10 * N_seg_cutoff + 7.2835e-09

            # costs (seconds)
            cost_co = N_det * N_seg * Ncoh * tau_Rs * fband_cell / fbandwidth
            cost_inco = N_seg * tau_D0 * Ninc * fband_cell / fbandwidth
            cost_bayes = Ninc * 4.4e-8 * fband_cell / fbandwidth
            cost_total = cost_co + cost_inco + cost_bayes  # seconds

            # accumulate
            CPU_year_sp += cost_total / SECONDS_PER_YEAR
            cost_co_sum += cost_co
            computingcost_sum += cost_total

    # overall coherent fraction
    co_cost_rate_sp = cost_co_sum / computingcost_sum if computingcost_sum else 0.0

    return CPU_year_sp, co_cost_rate_sp








# 示例使用
if __name__ == "__main__":
    # 示例参数
    fmin, fmax = 20.0, 1000.0  # 频率范围 (Hz)
    fdotmin, fdotmax = -1e-8, 1e-8  # 频率导数范围 (Hz/s)
    fsp, fdotsp = 100.0, 0.0  # 特定值
    fband_cell, fdotband_cell = 0.1, 1e-10  # 网格大小
    m0, m1, m2 = 0.3, 0.3, 0.3  # 失配参数
    r1, r2 = 3, 3  # 精化参数
    Tcoh_day = 30  # 相干时间 (天)
    nSFT, nSeg = 1000, 10  # SFT和段数
    tau = 700  # 年龄 (年)
    n = 7  # 制动指数
    fbandwidth = 1.0  # 带宽

    CPU_year_sp, co_cost_rate_sp = cal_cost_overall_700yr_new_timing_model(
        fmin,
        fmax,
        fdotmin,
        fdotmax,
        fsp,
        fdotsp,
        fband_cell,
        fdotband_cell,
        m0,
        m1,
        m2,
        r1,
        r2,
        Tcoh_day,
        nSFT,
        nSeg,
        tau,
        n,
        fbandwidth,
    )

    print(f"Total runtime   (CPU-years): {CPU_year_sp:.3e}")
    print(f"Coherent fraction            {co_cost_rate_sp:.3f}")
