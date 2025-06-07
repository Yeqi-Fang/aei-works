#!/usr/bin/env python3
"""
Compute a mismatch-distribution for a continuous-wave (CW) search **using only
LALSuite tools**, i.e. without relying on PyFstat convenience wrappers.

The script follows the same logic as the earlier PyFstat example:

1.  Generate synthetic SFTs with an injected CW signal by calling the
    `lalapps_Makefakedata_v5` executable.
2.  For a rectangular parameter grid around the signal, compute the fully
    coherent $2\mathcal F$ statistic with `lalapps_ComputeFstatistic_v2` and
    retain the loudest template.
3.  Compare the loudest-template $2\mathcal F$ with the perfect-match value to
    obtain an empirical mismatch $\mu$.
4.  Repeat the procedure *N* times (default *N = 100*) with small random offsets
    of the search grid in order to build up a distribution of mismatches.
5.  Save the $\mu$ values to a CSV file and plot their histogram as PDF.

The script is designed for clarity rather than speed: it launches the
individual **lalapps** programs via *subprocess* and parses their stdout.
At the end you will obtain two files inside `outdir/`:

* `mismatches.csv` A one-column CSV with the empirical mismatch values.
* `mismatch_distribution.pdf` A histogram of the distribution.

The run takes only a few minutes on a recent laptop for the default settings
(`N = 100`, 2-D grid in $f$ and $\dot f$).  To extend the search to the sky you
can set `SKY = True`, but expect a substantially longer runtime.
"""

import os
import subprocess
import shutil
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# User-adjustable high-level switches
# ──────────────────────────────────────────────────────────────────────────────
N_ITERATIONS = 100          # how many random grid realisations to draw
SKY = False                 # set True to include (α,δ) dimensions
KEEP_SFTS = False           # set True to keep all generated SFTs on disk

# General directories / labels -------------------------------------------------
LABEL  = "LALSuiteMismatchDistribution"
BASEDIR = os.path.join("LAL_example_data", LABEL)
os.makedirs(BASEDIR, exist_ok=True)

# Convenience: absolute paths for the lalapps binaries -------------------------
MFD_BIN = shutil.which("lalapps_Makefakedata_v5")
FS_BIN  = shutil.which("lalapps_ComputeFstatistic_v2")
if MFD_BIN is None or FS_BIN is None:
    raise RuntimeError("Could not find lalapps binaries in $PATH$. Make sure "
                       "LALSuite is installed and its 'bin' directory is on "
                       "your shell PATH before running this script.")

# ──────────────────────────────────────────────────────────────────────────────
# Injection parameters and search set-up (mirrors the PyFstat example)
# ──────────────────────────────────────────────────────────────────────────────
TSTART   = 1_000_000_000        # GPS • integer for simplicity
DURATION = 30 * 86400           # 30 days of data
TSFT     = 1800                 # 30-min SFTs
IFOS     = "H1,L1"              # detectors
SQRT_SX  = 1e-22                # (flat) noise PSD

INJ_PAR = {
    "F0":    30.0,     # Hz
    "F1":   -1e-10,   # Hz/s
    "F2":    0.0,     # Hz/s²
    "Alpha": 0.5,     # rad
    "Delta": 1.0,     # rad
    "h0":    0.5 * SQRT_SX,
    "cosi":  1.0,
}

# Fixed search-grid mismatch and resulting spacings (Prix 2009) ---------------
m        = 0.3   # nominal maximum mismatch

dF0      = np.sqrt(12  * m) / (np.pi * DURATION)
dF1      = np.sqrt(180 * m) / (np.pi * DURATION ** 2)
ΔF0      = 500 * dF0
ΔF1      = 200 * dF1
if SKY:
    ΔF0 /= 10  # keep run-time reasonable
    ΔF1 /= 10

# Histogram parameters ---------------------------------------------------------
N_BINS   = 20

# Helper regex to capture the "2F" result line of lalapps_ComputeFstatistic_v2
re_twoF  = re.compile(r"^twoF\s*=\s*([0-9eE+\-.]+)")


def run_makefakedata(sftdir: str) -> str:
    """Generate SFTs containing the injected signal and return a glob pattern."""
    os.makedirs(sftdir, exist_ok=True)

    cmd = [
        MFD_BIN,
        "--outSingleSFT=1",
        f"--outputSFTbase={sftdir}/SFT",
        f"--fmin={INJ_PAR['F0'] - 0.5}",      # narrow band suffices
        "--Band=1",
        f"--Tsft={TSFT}",
        f"--IFOs={IFOS}",
        f"--sqrtSX={SQRT_SX}",
        f"--tstart={TSTART}",
        f"--duration={DURATION}",
        # injection parameters -------------------------------------------------
        f"--F0={INJ_PAR['F0']}",
        f"--F1={INJ_PAR['F1']}",
        f"--F2={INJ_PAR['F2']}",
        f"--Alpha={INJ_PAR['Alpha']}",
        f"--Delta={INJ_PAR['Delta']}",
        f"--h0={INJ_PAR['h0']}",
        f"--cosi={INJ_PAR['cosi']}",
        # misc ---------------------------------------------------------------
        "--EphemEarth=DE421",
        "--EphemSun=DE421",
    ]

    subprocess.run(cmd, check=True, capture_output=True)

    return os.path.join(sftdir, "*.sft")


def compute_twoF(sft_glob: str, f0: float, f1: float,
                 alpha: float = INJ_PAR["Alpha"],
                 delta: float = INJ_PAR["Delta"],
                 f2: float = INJ_PAR["F2"],
                 tref: int = TSTART):
    """Run lalapps_ComputeFstatistic_v2 for a single template and return 2F."""

    cmd = [
        FS_BIN,
        f"--Alpha={alpha}",
        f"--Delta={delta}",
        f"--F0={f0}",
        f"--f1dot={f1}",
        f"--f2dot={f2}",
        f"--refTime={tref}",
        "--singleFstat=1",          # ask for one template only
        f"--SFTFiles={sft_glob}",
    ]

    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        m = re_twoF.match(line.strip())
        if m:
            return float(m.group(1))
    raise RuntimeError("Could not parse twoF from lalapps_ComputeFstatistic_v2 output.")


def build_grid(f0_center: float, f1_center: float):
    """Return arrays for the rectangular (F0, F1) grid."""
    f0_vals = np.arange(f0_center - ΔF0/2, f0_center + ΔF0/2 + dF0/2, dF0)
    f1_vals = np.arange(f1_center - ΔF1/2, f1_center + ΔF1/2 + dF1/2, dF1)
    return f0_vals, f1_vals


# ──────────────────────────────────────────────────────────────────────────────
# Main loop over random search-grid offsets
# ──────────────────────────────────────────────────────────────────────────────

mismatches = []

for iteration in range(N_ITERATIONS):

    print(f"Iteration {iteration+1}/{N_ITERATIONS} …")

    # ── 1. Create a temporary directory for this iteration's SFTs -------------
    sftdir = os.path.join(BASEDIR, f"SFTs_{iteration:03d}")
    sft_glob = run_makefakedata(sftdir)

    # ── 2. Randomly offset the grid centre inside one elementary cell ---------
    f0_random_offset = np.random.uniform(-dF0, dF0)
    f1_random_offset = np.random.uniform(-dF1, dF1)

    f0_grid_centre = INJ_PAR['F0'] + f0_random_offset
    f1_grid_centre = INJ_PAR['F1'] + f1_random_offset

    f0_vals, f1_vals = build_grid(f0_grid_centre, f1_grid_centre)

    # ── 3. Brute-force scan the grid and track the loudest template ----------
    loudest_twoF = -np.inf
    loudest_params = (None, None)

    for f0 in f0_vals:
        for f1 in f1_vals:
            twoF = compute_twoF(sft_glob, f0, f1)
            if twoF > loudest_twoF:
                loudest_twoF = twoF
                loudest_params = (f0, f1)

    # ── 4. Compute perfect-match 2F (one call only) --------------------------
    twoF_inj = compute_twoF(sft_glob, INJ_PAR['F0'], INJ_PAR['F1'])

    # ρ² = 2F − 4 (Prix+2007, Eq. 63) ------------------------------------------
    rho2_no  = twoF_inj       - 4.0
    rho2_mis = loudest_twoF   - 4.0

    mu_empirical = (rho2_no - rho2_mis) / rho2_no
    mismatches.append(mu_empirical)

    print(f"    twoF(inj)     = {twoF_inj:8.3f}")
    print(f"    twoF(loudest) = {loudest_twoF:8.3f}")
    print(f"    μ             = {mu_empirical:8.2e}\n")

    # ── 5. Clean-up -----------------------------------------------------------
    if not KEEP_SFTS:
        shutil.rmtree(sftdir)

# ──────────────────────────────────────────────────────────────────────────────
# Save results and make histogram
# ──────────────────────────────────────────────────────────────────────────────

mismatch_file = os.path.join(BASEDIR, "mismatches.csv")
np.savetxt(mismatch_file, np.array(mismatches), delimiter=",",
           header="Empirical mismatch μ", comments="")
print(f"Mismatch values written to → {mismatch_file}")

plt.figure(figsize=(10, 6))
plt.hist(mismatches, bins=N_BINS, density=True, alpha=0.7)
plt.xlabel("Empirical mismatch $\\mu$")
plt.ylabel("Probability density")
plt.title("Mismatch distribution from LALSuite grid searches")
plt.grid(True)
plot_path = os.path.join(BASEDIR, "mismatch_distribution.pdf")
plt.savefig(plot_path)
plt.show()
print(f"Histogram written to        → {plot_path}")
