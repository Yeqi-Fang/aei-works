import os
import numpy as np
import pyfstat
import matplotlib.pyplot as plt

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
    "h0": 0.1 * sqrtSX,
    "cosi": 1.0,
}

# LaTeX-formatted plotting labels
labels = {
    "F0": "$f$ [Hz]",
    "F1": "$\\dot{f}$ [Hz/s]",
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
m = 0.001
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

# Some plotting helper functions
def plot_grid_vs_samples(grid_res, mcmc_res, xkey, ykey):
    """Local plotting function to avoid code duplication in the 4D case"""
    plt.figure(figsize=(10, 8))
    plt.plot(grid_res[xkey], grid_res[ykey], ".", label="grid")
    plt.plot(mcmc_res[xkey], mcmc_res[ykey], ".", label="mcmc")
    plt.plot(inj[xkey], inj[ykey], "*k", label="injection")
    
    grid_maxidx = np.argmax(grid_res["twoF"])
    mcmc_maxidx = np.argmax(mcmc_res["twoF"])
    
    plt.plot(
        grid_res[xkey][grid_maxidx],
        grid_res[ykey][grid_maxidx],
        "+g",
        label=labels["max2F"] + "(grid)",
    )
    plt.plot(
        mcmc_res[xkey][mcmc_maxidx],
        mcmc_res[ykey][mcmc_maxidx],
        "xm",
        label=labels["max2F"] + "(mcmc)",
    )
    plt.xlabel(labels[xkey])
    plt.ylabel(labels[ykey])
    plt.legend()
    
    plotfilename_base = os.path.join(outdir, "grid_vs_mcmc_{:s}_{:s}".format(xkey, ykey))
    plt.savefig(plotfilename_base + ".pdf")
    
    if xkey == "F0" and ykey == "F1":
        plt.xlim(zoom[xkey])
        plt.ylim(zoom[ykey])
        plt.savefig(plotfilename_base + "_zoom.pdf")
    
    # plt.show()

def plot_2F_scatter(res, label_name, xkey, ykey):
    """Local plotting function to avoid code duplication in the 4D case"""
    markersize = 1 if label_name == "grid" else 0.5
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(res[xkey], res[ykey], c=res["twoF"], s=markersize)
    cb = plt.colorbar(sc)
    plt.xlabel(labels[xkey])
    plt.ylabel(labels[ykey])
    cb.set_label(labels["2F"])
    plt.title(label_name)
    plt.plot(inj[xkey], inj[ykey], "*k", label="injection")
    
    maxidx = np.argmax(res["twoF"])
    plt.plot(
        res[xkey][maxidx],
        res[ykey][maxidx],
        "+r",
        label=labels["max2F"],
    )
    plt.legend(loc='upper right')
    
    plotfilename_base = os.path.join(
        outdir, "{:s}_{:s}{:s}_2F".format(label_name, xkey, ykey)
    )
    plt.xlim([min(res[xkey]), max(res[xkey])])
    plt.ylim([min(res[ykey]), max(res[ykey])])
    plt.savefig(plotfilename_base + ".pdf")
    # plt.show()

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

# print(gridsearch.data)
# # save the grid results to a file
# grid_res_file = os.path.join(outdir, "grid_results.npy")
# np.save(grid_res_file, grid_res)
# # save to text file for easy reading
# grid_res_txt_file = os.path.join(outdir, "grid_results.txt")
# np.savetxt(
#     grid_res_txt_file,
#     np.column_stack((grid_res["F0"], grid_res["F1"], grid_res["twoF"])),
#     header="F0 F1 twoF",
#     fmt="%10.6e %10.6e %10.6e"
# )
if sky:
    grid_res["Alpha"] = gridsearch.data["Alpha"]
    grid_res["Delta"] = gridsearch.data["Delta"]

# We'll use the two local plotting functions defined above
# to avoid code duplication in the sky case
plot_2F_scatter(grid_res, "grid", "F0", "F1")

if sky:
    plot_2F_scatter(grid_res, "grid", "Alpha", "Delta")

# Note: To complete the comparison with MCMC, you would need to add
# an MCMC search here and then call plot_grid_vs_samples()
# This would require additional pyfstat.MCMCSearch setup

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

# Metric-based mismatch (Brady+JKS diagonal approximation)
# m_calc = (np.pi * duration * delta_f0)**2 / 12.0 + (np.pi * duration**2 * delta_f1)**2 / 180.0


# -----------------------------------------------------------
#  Mismatch diagnosis (API-safe version, PyFstat ≥ 2.x)
# -----------------------------------------------------------
import pyfstat


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
rho2_no  = twoF_inj - 4.0      # ρ²_no-mismatch

# --- 2) loudest point from the grid you already ran ------------
grid_maxidx = np.argmax(grid_res["twoF"])
twoF_mis    = grid_res["twoF"][grid_maxidx]
rho2_mis    = twoF_mis - 4.0   # ρ²_mismatch

# --- 3) empirical mismatch -------------------------------------
mu_empirical = (rho2_no - rho2_mis) / rho2_no

print("\n--------- mismatch check (ρ-based) ---------")
print(f"2F(injection)  = {twoF_inj:10.3f}")
print(f"2F(loudest)    = {twoF_mis:10.3f}")
print(f"ρ²_no-mismatch = {rho2_no:10.3f}")
print(f"ρ²_mismatch    = {rho2_mis:10.3f}")
print(f"μ  (empirical) = {mu_empirical:10.3e}")
print("-------------------------------------------")
