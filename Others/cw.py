from lalpulsar import simulateCW
# 定义或导入函数waveform(dt)，在时间dt返回(dphi, ap, ac)
cw = simulateCW.CWSimulator(
    tref=1000000000, tstart=1000000000, Tdata=1e6, 
    waveform=my_waveform, dt_wf=1.0,
    phi0=0, psi=0, alpha=1.0, delta=0.5, det_name="L1"
)
# 生成CW信号的SFTs，频率高达fmax
cw.write_sft_files(fmax=500, Tsft=1800, out_dir="sim_SFTs")
