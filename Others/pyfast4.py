import os
import numpy as np
import pyfstat
import matplotlib.pyplot as plt
from utils import (
    calculate_mismatch_from_fstats,
    calculate_f0_f1_mismatch_metric,
    calculate_sky_mismatch_metric,
    plot_2F_scatter,
    MismatchAnalyzer,
)
import pyfstat


# Configuration settings
plt.style.use('default')  # Ensure consistent plotting style

# Flip this switch for a more expensive 4D (F0,F1,Alpha,Delta) run
# instead of just (F0,F1)
# (still only a few minutes on current laptops)
sky = False

# General setup
label = "PyFstatExampleSimpleMCMCvsGridComparison"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

if sky:
    outdir += "AlphaDelta"

# Parameters for the data set to generate
tstart = 1000000000
duration = 30 * 86400
Tsft = 1800
detectors = "H1,L1"
sqrtSX = 1e-22

# Parameters for injected signals
inj = {
    "tref": tstart,
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": 0.5,
    "Delta": 1,
    "h0": 0.5 * sqrtSX,
    "cosi": 1.0,
}

# LaTeX-formatted plotting labels
labels = {
    "F0": "$f$ [Hz]",
    "F1": "$\\dot{f}$ [Hz/s]",
    "F2": "$\\ddot{f}$ [Hz/s^2]",
    "2F": "$2\\mathcal{F}$",
    "Alpha": "$\\alpha$",
    "Delta": "$\\delta$",
}

labels["max2F"] = "$\\max\\,$" + labels["2F"]

# Create SFT files
logger.info("Generating SFTs with injected signal...")
writer = pyfstat.Writer(
    label=label + "SimulatedSignal",
    outdir=outdir,
    tstart=tstart,
    duration=duration,
    detectors=detectors,
    sqrtSX=sqrtSX,
    Tsft=Tsft,
    **inj,
    Band=1,  # default band estimation would be too narrow for a wide grid/prior
)
writer.make_data()

# Set up square search grid with fixed (F0,F1) mismatch
# and (optionally) some ad-hoc sky coverage
m = 0.5
dF0 = np.sqrt(12 * m) / (np.pi * duration) 
dF1 = np.sqrt(180 * m) / (np.pi * duration**2) 
DeltaF0 = 500 * dF0
DeltaF1 = 200 * dF1

if sky:
    # Cover less range to keep runtime down
    DeltaF0 /= 10
    DeltaF1 /= 10

F0s = [inj["F0"] - DeltaF0 / 2.0, inj["F0"] + DeltaF0 / 2.0, dF0]
F1s = [inj["F1"] - DeltaF1 / 2.0, inj["F1"] + DeltaF1 / 2.0, dF1]
F2s = [inj["F2"]]
search_keys = ["F0", "F1"]  # only the ones that aren't 0-width

if sky:
    dSky = 0.01  # rather coarse to keep runtime down
    DeltaSky = 10 * dSky
    Alphas = [inj["Alpha"] - DeltaSky / 2.0, inj["Alpha"] + DeltaSky / 2.0, dSky]
    Deltas = [inj["Delta"] - DeltaSky / 2.0, inj["Delta"] + DeltaSky / 2.0, dSky]
    search_keys += ["Alpha", "Delta"]
else:
    Alphas = [inj["Alpha"]]
    Deltas = [inj["Delta"]]

search_keys_label = "".join(search_keys)

# Run the grid search
logger.info("Performing GridSearch...")
gridsearch = pyfstat.GridSearch(
    label="GridSearch" + search_keys_label,
    outdir=outdir,
    sftfilepattern=writer.sftfilepath,
    F0s=F0s,
    F1s=F1s,
    F2s=F2s,
    Alphas=Alphas,
    Deltas=Deltas,
    tref=inj["tref"],
)
gridsearch.run()
gridsearch.print_max_twoF()
gridsearch.generate_loudest()



# Do some plots of the GridSearch results
if not sky:  # this plotter can't currently deal with too large result arrays
    logger.info("Plotting 1D 2F distributions...")
    for key in search_keys:
        gridsearch.plot_1D(xkey=key, xlabel=labels[key], ylabel=labels["2F"])

logger.info("Making GridSearch {:s} corner plot...".format("-".join(search_keys)))
vals = [np.unique(gridsearch.data[key]) - inj[key] for key in search_keys]
twoF = gridsearch.data["twoF"].reshape([len(kval) for kval in vals])

corner_labels = [
    "$f - f_0$ [Hz]",
    "$\\dot{f} - \\dot{f}_0$ [Hz/s]",
]
if sky:
    corner_labels.append("$\\alpha - \\alpha_0$")
    corner_labels.append("$\\delta - \\delta_0$")

corner_labels.append(labels["2F"])

gridcorner_fig, gridcorner_axes = pyfstat.gridcorner(
    twoF, vals, projection="log_mean", labels=corner_labels,
    whspace=0.1, factor=1.8
)
gridcorner_fig.savefig(os.path.join(outdir, gridsearch.label + "_corner.pdf"))
plt.figure(gridcorner_fig.number)  # Make the figure current
# plt.show()

# Define zoom parameters
zoom = {
    "F0": [inj["F0"] - 10 * dF0, inj["F0"] + 10 * dF0],
    "F1": [inj["F1"] - 5 * dF1, inj["F1"] + 5 * dF1],
}

# Get grid results for plotting
grid_res = {
    "F0": gridsearch.data["F0"],
    "F1": gridsearch.data["F1"],
    "twoF": gridsearch.data["twoF"]
}

if sky:
    grid_res["Alpha"] = gridsearch.data["Alpha"]
    grid_res["Delta"] = gridsearch.data["Delta"]

# We'll use the two local plotting functions defined above
# to avoid code duplication in the sky case
plot_2F_scatter(grid_res, "grid", "F0", "F1", labels, inj, outdir)

if sky:
    plot_2F_scatter(grid_res, "grid", "Alpha", "Delta", labels, inj, outdir)

print("Analysis completed! Check the output directory for results:")
print(f"Output directory: {outdir}")

# -----------------------------------------------------------
# Mismatch diagnosis: compare loudest grid point vs injection
# -----------------------------------------------------------

# Identify loudest template (we already stored its index earlier)
grid_maxidx = np.argmax(grid_res["twoF"])

# Parameter offsets
delta_f0 = grid_res["F0"][grid_maxidx] - inj["F0"]
delta_f1 = grid_res["F1"][grid_maxidx] - inj["F1"]

# ============================================================================
# METHOD 1: Direct F-statistic Mismatch Calculation
# ============================================================================

print("\n" + "="*80)
print("METHOD 1: Direct F-statistic Mismatch Calculation")
print("="*80)



# Get 2F values
twoF_at_loudest = grid_res["twoF"][grid_maxidx]

search_ranges = {
    "F0":    [inj["F0"]],         # a single value ⇒ zero width,
    "Alpha": [inj["Alpha"]],
    "Delta": [inj["Delta"]],
}
fs = pyfstat.ComputeFstat(
        tref           = inj["tref"],
        sftfilepattern = writer.sftfilepath,
        minStartTime   = tstart,
        maxStartTime   = tstart + duration,
        search_ranges  = search_ranges,
)

# template exactly at the injected parameters
inj_pars = {k: inj[k] for k in ("F0", "F1", "F2", "Alpha", "Delta")}

twoF_inj = fs.get_fullycoherent_detstat(params=inj_pars)

# Calculate mismatch using direct F-stat method
mismatch_fstat = calculate_mismatch_from_fstats(twoF_inj, twoF_at_loudest)

print(f"2F at loudest grid point: {twoF_at_loudest:.6f}")
print(f"2F at injection (approx): {twoF_inj:.6f}")
print(f"Mismatch (F-stat method): {mismatch_fstat:.6f}")

# Find the actual closest point to injection for better comparison
if sky:
    distances = np.sqrt((grid_res["F0"] - inj["F0"])**2 + 
                       (grid_res["F1"] - inj["F1"])**2 +
                       (grid_res["Alpha"] - inj["Alpha"])**2 + 
                       (grid_res["Delta"] - inj["Delta"])**2)
else:
    distances = np.sqrt((grid_res["F0"] - inj["F0"])**2 + 
                       (grid_res["F1"] - inj["F1"])**2)

closest_idx = np.argmin(distances)
twoF_at_closest = grid_res["twoF"][closest_idx]
mismatch_fstat_closest = calculate_mismatch_from_fstats(twoF_at_closest, twoF_at_loudest)

print(f"2F at closest grid point to injection: {twoF_at_closest:.6f}")
print(f"Mismatch (F-stat method, using closest): {mismatch_fstat_closest:.6f}")

# ============================================================================
# METHOD 2: Parameter Space Metric Mismatch Calculation
# ============================================================================

print("\n" + "="*80)
print("METHOD 2: Parameter Space Metric Mismatch Calculation")
print("="*80)



# Calculate coherent integration time
Tseg = duration  # Total observation time

# Calculate frequency mismatch using metric
mismatch_freq, g_00, g_11, g_01 = calculate_f0_f1_mismatch_metric(delta_f0, delta_f1, Tseg)

print(f"Coherent integration time: {Tseg/86400:.2f} days")
print(f"Parameter offsets:")
print(f"  ΔF0 = {delta_f0:.6e} Hz")
print(f"  ΔF1 = {delta_f1:.6e} Hz/s")

print(f"\nMetric components:")
print(f"  g_00 (F0-F0): {g_00:.6e}")
print(f"  g_11 (F1-F1): {g_11:.6e}")
print(f"  g_01 (F0-F1): {g_01:.6e}")

print(f"\nMismatch contributions:")
print(f"  F0 contribution: {g_00 * delta_f0**2:.6f}")
print(f"  F1 contribution: {g_11 * delta_f1**2:.6f}")
print(f"  Cross contribution: {2 * g_01 * delta_f0 * delta_f1:.6f}")
print(f"  Total frequency mismatch: {mismatch_freq:.6f}")

# Sky mismatch if applicable
if sky:
    delta_alpha = grid_res["Alpha"][grid_maxidx] - inj["Alpha"]
    delta_delta = grid_res["Delta"][grid_maxidx] - inj["Delta"]
    
    mismatch_sky, g_alpha, g_delta = calculate_sky_mismatch_metric(delta_alpha, delta_delta, Tseg)
    
    print(f"\nSky parameter offsets:")
    print(f"  ΔAlpha = {delta_alpha:.6f} rad = {np.degrees(delta_alpha):.3f} deg")
    print(f"  ΔDelta = {delta_delta:.6f} rad = {np.degrees(delta_delta):.3f} deg")
    print(f"  Sky mismatch: {mismatch_sky:.6f}")
    
    total_mismatch_metric = mismatch_freq + mismatch_sky
    print(f"  Total mismatch (freq + sky): {total_mismatch_metric:.6f}")
else:
    total_mismatch_metric = mismatch_freq
    print(f"  Total mismatch: {total_mismatch_metric:.6f}")

# ============================================================================
# METHOD 3: Comprehensive PyFstat Integration Analysis
# ============================================================================

print("\n" + "="*80)
print("METHOD 3: Comprehensive PyFstat Integration Analysis")
print("="*80)


# Run comprehensive analysis
analyzer = MismatchAnalyzer(Tseg=duration)
analysis = analyzer.analyze_grid_search_mismatch(gridsearch, inj)

print(f"Comprehensive Analysis Results:")
print(f"  Coherent time: {analysis['coherent_time']/86400:.2f} days")
print(f"  Total grid points: {len(analysis['total_mismatch'])}")

print(f"\nLoudest Point:")
loudest = analysis['loudest_point']
print(f"  2F: {loudest['twoF']:.6f}")
print(f"  Mismatch: {loudest['mismatch']:.6f}")
print(f"  F0: {loudest['parameters']['F0']:.6f} Hz")
print(f"  F1: {loudest['parameters']['F1']:.6e} Hz/s")

print(f"\nBest Mismatch Point:")
best_mm = analysis['best_mismatch_point']
print(f"  2F: {best_mm['twoF']:.6f}")
print(f"  Mismatch: {best_mm['mismatch']:.6f}")
print(f"  F0: {best_mm['parameters']['F0']:.6f} Hz")
print(f"  F1: {best_mm['parameters']['F1']:.6e} Hz/s")

# Mismatch statistics
mm_stats = {
    'mean': np.mean(analysis['total_mismatch']),
    'median': np.median(analysis['total_mismatch']),
    'std': np.std(analysis['total_mismatch']),
    'min': np.min(analysis['total_mismatch']),
    'max': np.max(analysis['total_mismatch'])
}

print(f"\nMismatch Statistics:")
for stat, value in mm_stats.items():
    print(f"  {stat}: {value:.6f}")

# Generate plots
analyzer.plot_mismatch_vs_twoF(analysis)

# ============================================================================
# COMPARISON OF ALL THREE METHODS
# ============================================================================

print("\n" + "="*80)
print("COMPARISON OF ALL THREE METHODS")
print("="*80)

print(f"Method 1 (F-statistic):     {mismatch_fstat}")
print(f"Method 2 (Metric):          {total_mismatch_metric:.6f}")
print(f"Method 3 (Comprehensive):   {loudest['mismatch']:.6f}")

# print(f"\nRelative differences:")
# print(f"Method 2 vs Method 1: {(total_mismatch_metric/mismatch_fstat - 1)*100:.2f}%")
# print(f"Method 3 vs Method 1: {(loudest['mismatch']/mismatch_fstat - 1)*100:.2f}%")
# print(f"Method 3 vs Method 2: {(loudest['mismatch']/total_mismatch_metric - 1)*100:.2f}%")

# Additional insights
print(f"\nAdditional Insights:")
print(f"Grid spacing (dF0): {dF0:.6e} Hz")
print(f"Grid spacing (dF1): {dF1:.6e} Hz/s")
print(f"Target mismatch (m): {m:.6f}")

# Check if the loudest point is close to the target mismatch
# if abs(loudest['mismatch'] - m) < 0.5 * m:
#     print(f"✓ Grid search performed well: loudest mismatch ≈ target mismatch")
# else:
#     print(f"⚠ Grid search mismatch differs significantly from target")

# print(f"\nAnalysis complete! All mismatch calculation methods have been applied.")
# print(f"Check the plots and output files in: {outdir}")

