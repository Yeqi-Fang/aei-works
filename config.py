import numpy as np
import os
from runtime_calculation import cal_cost_overall_700yr_new_timing_model

sky = False
plot = False
numbers = 2000
num_workers = 14
tstart = 1000000000
duration = 120 * 86400
T_coh = 15 * 86400  # coherence time for the MCMC
nsegs = int(duration / T_coh)  # number of segments for the MCMC
Tsft = 1800
detectors = "H1,L1"
sqrtSX = 1e-22

label = "PyFstatExampleSimpleMCMCvsGridComparisonSemi"
outdir = os.path.join("PyFstat_example_data", label)


# inj = {
#     "tref": tstart,
#     "F0": 30.0,
#     "F1": -1e-10,
#     "F2": 0,
#     "Alpha": 0.5,
#     "Delta": 1,
#     "h0": 0.05 * sqrtSX,
#     "cosi": 1.0,
# }

inj = {
    "tref": tstart,
    "F0": 151.5,
    "F1": -1e-10,
    "F2": -1e-20,
    "Alpha": 0.5,
    "Delta": 1,
    "h0": 0.05 * sqrtSX,
    "cosi": 1.0,
}


# latex-formatted plotting labels
labels = {
    "F0": "$f$ [Hz]",
    "F1": "$\\dot{f}$ [Hz/s]",
    "2F": "$2\\mathcal{F}$",
    "Alpha": "$\\alpha$",
    "Delta": "$\\delta$",
}
labels["max2F"] = "$\\max\\,$" + labels["2F"]

mf = 0.15
mf1 = 0.3
mf2 = 0.003
gamma1 = 8
gamma2 = 20
dF0 = np.sqrt(12 * mf) / (np.pi * T_coh)
dF1 = np.sqrt(180 * mf1) / (np.pi * T_coh**2)
dF2 = np.sqrt(25200 * mf2) / (np.pi * T_coh**3)


dF1_refined = dF1 / gamma1
dF2_refined = dF2 / gamma2


DeltaF0 = 10 * dF0  # 500
DeltaF1 = 10 * dF1_refined  # 200
DeltaF2 = 10 * dF2_refined  # 60

if sky:
    # cover less range to keep runtime down
    DeltaF0 /= 10
    DeltaF1 /= 10
    DeltaF2 /= 5


DeltaF0_fixed = 9.885590880794127e-06
DeltaF1_fixed = 3.481585082097677e-12
DeltaF2_fixed = 6.357202196709655e-19

Nf0 = DeltaF0_fixed / dF0
Nf1 = DeltaF1_fixed / dF1
Nf2 = DeltaF2_fixed / dF2

N_det = 2
N_coh = Nf0 * Nf1 * Nf2

N_can = 0

tau_Fbin = 6e-8
tau_fft = 3.3e-8
tau_spin = 7.5e-8
tau_bayes = 4.4e-8
tau_recalc = 0


ratio = 2

R = 1

N_inc = N_coh * gamma1 * gamma2

tau_sumF = 7.28e-9 - 3.72e-10 * nsegs

tau_RS = tau_Fbin + ratio * (tau_fft + R * tau_spin)

# tau_total = nsegs * N_det * N_coh * tau_RS + nsegs * N_inc * tau_sumF + \
#     N_inc * tau_bayes + N_can * tau_recalc


# new: compute with the 700-yr timing model


# if you still want tau_total in CPU‐seconds rather than CPU‐years:

if __name__ == "__main__":
    
    CPU_year_total, co_cost_rate = (
        cal_cost_overall_700yr_new_timing_model(
            # frequency range (centered on your injection F0 ± ΔF0/2)
            fmin=20,
            fmax=150,
            # spin-down range (injection F1 ± ΔF1/2)
            fdotmin=inj["F1"] - DeltaF1_fixed / 2,
            fdotmax=inj["F1"] + DeltaF1_fixed / 2,
            # “special” points (just the injected values)
            fsp=inj["F0"],
            fdotsp=inj["F1"],
            # grid cell spacings from your config
            fband_cell=dF0,
            fdotband_cell=dF1_refined,
            # mismatch & refinement parameters
            m0=mf,
            m1=mf1,
            m2=mf2,
            r1=gamma1,
            r2=gamma2,
            # coherence time in days
            Tcoh_day=T_coh / 86400,
            # these two aren’t used inside the function but it needs them
            nSFT=numbers,
            nSeg=nsegs,
            # age in years (700 yr model)
            tau=70,
            # braking index
            n=5,
            # band­width normalization (you can tweak as needed)
            fbandwidth=1,
        )
    )
    
    
    tau_total = CPU_year_total * 365.24 * 24 * 3600
    print(f"CPU_year_total: {CPU_year_total}")
    print(f"co_cost_rate: {co_cost_rate}")
    print(f"tau_total (in CPU-seconds): {tau_total}")
