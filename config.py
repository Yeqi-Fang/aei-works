import numpy as np
import os 


sky = False
plot = False
numbers = 500
num_workers = 10
tstart = 1000000000
duration = 120 * 86400
T_coh = 15 * 86400  # coherence time for the MCMC
nsegs = int(duration / T_coh)  # number of segments for the MCMC
Tsft = 1800
detectors = "H1,L1"
sqrtSX = 1e-22

label = "PyFstatExampleSimpleMCMCvsGridComparisonSemi"
outdir = os.path.join("PyFstat_example_data", label)


inj = {
    "tref": tstart,
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
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


dF1_refined = dF1 / gamma2
dF2_refined = dF2 / gamma2


DeltaF0 = 30 * dF0 # 500 
DeltaF1 = 20 * dF1_refined # 200
DeltaF2 = 10 * dF2_refined # 60

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

tau_total = nsegs * N_det * N_coh * tau_RS + nsegs * N_inc * tau_sumF + \
    N_inc * tau_bayes + N_can * tau_recalc