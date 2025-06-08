import os
import numpy as np
import pyfstat
# make sure to put these after the pyfstat import, to not break notebook inline plots
import matplotlib.pyplot as plt
# %matplotlib inline

import concurrent.futures
# flip this switch for a more expensive 4D (F0,F1,Alpha,Delta) run
# instead of just (F0,F1)
# (still only a few minutes on current laptops)
sky = False
plot = False
# log = 


# general setup
label = "PyFstatExampleSimpleMCMCvsGridComparisonSemi"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir, log_level="WARNING")
if sky:
    outdir += "AlphaDelta"
printout = False
# parameters for the data set to generate
tstart = 1000000000
duration = 120 * 86400
T_coh = 15 * 86400  # coherence time for the MCMC
nsegs = int(duration / T_coh)  # number of segments for the MCMC
Tsft = 1800
detectors = "H1,L1"
sqrtSX = 1e-22

# parameters for injected signals
inj = {
    "tref": tstart,
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": 0.5,
    "Delta": 1,
    "h0": 1 * sqrtSX,
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

# create SFT files
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

# set up square search grid with fixed (F0,F1) mismatch
# and (optionally) some ad-hoc sky coverage
mf = 0.15
mf1 = 0.3
mf2 = 0.003
gamma1 = 8
gamma2 = 20
dF0 = np.sqrt(12 * mf) / (np.pi * T_coh)
dF1 = np.sqrt(180 * mf1) / (np.pi * T_coh**2) 
dF2 = np.sqrt(25200 * mf2) / (np.pi * T_coh**3)

dF1_refined = dF1 / gamma1
dF2_refined = dF2 / gamma1

DeltaF0 = 30 * dF0 # 500 
DeltaF1 = 20 * dF1_refined # 200
DeltaF2 = 10 * dF2_refined # 60


print(DeltaF0, DeltaF1, DeltaF2)


if sky:
    # cover less range to keep runtime down
    DeltaF0 /= 10
    DeltaF1 /= 10
    DeltaF2 /= 5
    
    
zoom = {
        "F0": [inj["F0"] - 10 * dF0, inj["F0"] + 10 * dF0],
        "F1": [inj["F1"] - 5 * dF1_refined, inj["F1"] + 5 * dF1_refined],
    }


# some plotting helper functions
def plot_grid_vs_samples(grid_res, mcmc_res, xkey, ykey):
    """local plotting function to avoid code duplication in the 4D case"""
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
    plotfilename_base = os.path.join(outdir, "grid_vs_mcmc_{:s}{:s}".format(xkey, ykey))
    plt.savefig(plotfilename_base + ".png")
    if xkey == "F0" and ykey == "F1":
        plt.xlim(zoom[xkey])
        plt.ylim(zoom[ykey])
        plt.savefig(plotfilename_base + "_zoom.png")
    # plt.show()

def plot_2F_scatter(res, label, xkey, ykey):
    """local plotting function to avoid code duplication in the 4D case"""
    markersize = 1 if label == "grid" else 0.5
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(res[xkey], res[ykey], c=res["twoF"], s=markersize)
    cb = plt.colorbar(sc)
    plt.xlabel(labels[xkey])
    plt.ylabel(labels[ykey])
    cb.set_label(labels["2F"])
    plt.title(label)
    plt.plot(inj[xkey], inj[ykey], "*k", label="injection")
    maxidx = np.argmax(res["twoF"])
    plt.plot(
        res[xkey][maxidx],
        res[ykey][maxidx],
        "+r",
        label=labels["max2F"],
    )
    plt.legend(loc='upper right')
    plotfilename_base = os.path.join(outdir, "{:s}_{:s}{:s}_2F".format(label, xkey, ykey))
    plt.xlim([min(res[xkey]), max(res[xkey])])
    plt.ylim([min(res[ykey]), max(res[ykey])])
    plt.savefig(plotfilename_base + ".png")
    # plt.show()


numbers = 500
mismatches = []
F0s_random = np.random.uniform(-dF0, dF0, size=numbers)
F1s_random = np.random.uniform(-dF1_refined, dF1_refined, size=numbers)
F2s_random = np.random.uniform(-dF2_refined, dF2_refined, size=numbers)


def calculate_mismatch(i):

    
    import pyfstat
    import numpy as np
    import os
    


    F0s = [inj["F0"] - DeltaF0 / 2.0 + F0s_random[i], inj["F0"] + DeltaF0 / 2.0 + F0s_random[i], dF0]
    F1s = [inj["F1"] - DeltaF1 / 2.0 + F1s_random[i], inj["F1"] + DeltaF1 / 2.0 + F1s_random[i], dF1_refined]
    F2s = [inj["F2"] - DeltaF2 / 2.0 + F2s_random[i], inj["F2"] + DeltaF2 / 2.0 + F2s_random[i], dF2_refined]
    
    
    search_keys = ["F0", "F1", "F2"]  # only the ones that aren't 0-width

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

    # run the grid search
    logger.info("Performing GridSearch...")
    gridsearch = pyfstat.GridSearch(
        label=f"GridSearch_iter_{i}" + search_keys_label,
        outdir=outdir,
        sftfilepattern=writer.sftfilepath,
        F0s=F0s,
        F1s=F1s,
        F2s=F2s,
        Alphas=Alphas,
        Deltas=Deltas,
        tref=inj["tref"],
        nsegs=nsegs,
    )
    gridsearch.run()
    gridsearch.print_max_twoF()
    gridsearch.generate_loudest()


    # do some plots of the GridSearch results
    if not sky:  # this plotter can't currently deal with too large result arrays
        logger.info("Plotting 1D 2F distributions...")
        if plot:
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
        if plot:
            gridcorner_fig, gridcorner_axes = pyfstat.gridcorner(
                twoF, vals, projection="log_mean", labels=corner_labels,
                whspace=0.1, factor=1.8
            )
            gridcorner_fig.savefig(os.path.join(outdir, gridsearch.label + "_corner.png"))
            # plt.show()



    # we'll use the two local plotting functions defined above
    # to avoid code duplication in the sky case
    if plot:
        plot_2F_scatter(gridsearch.data, "grid", "F0", "F1")
        if sky:
            plot_2F_scatter(gridsearch.data, "grid", "Alpha", "Delta")
        
        
        
    # -----------------------------------------------------------
    #  Mismatch diagnosis (API-safe version, PyFstat ≥ 2.x)
    # -----------------------------------------------------------


    search_ranges = {
        "F0":    [inj["F0"]],         # a single value ⇒ zero width,
        "Alpha": [inj["Alpha"]],
        "Delta": [inj["Delta"]],
    }
    fs = pyfstat.SemiCoherentSearch(
        label="MismatchTest",  # SemiCoherentSearch需要label
        outdir=outdir,         # 需要outdir
        tref=inj["tref"],
        nsegs=nsegs,              # 添加分段数
        sftfilepattern=writer.sftfilepath,
        minStartTime=tstart,
        maxStartTime=tstart + duration,
        search_ranges=search_ranges,
    )

    grid_res = gridsearch.data

    # template exactly at the injected parameters
    inj_pars = {k: inj[k] for k in ("F0", "F1", "F2", "Alpha", "Delta")}

    twoF_inj = fs.get_semicoherent_det_stat(params=inj_pars)

    rho2_no  = twoF_inj - 4.0      # ρ²_no-mismatch

    # --- 2) loudest point from the grid you already ran ------------
    grid_maxidx = np.argmax(grid_res["twoF"])
    twoF_mis    = grid_res["twoF"][grid_maxidx]
    rho2_mis    = twoF_mis - 4.0   # ρ²_mismatch

    # --- 3) empirical mismatch -------------------------------------
    mu_empirical = (rho2_no - rho2_mis) / rho2_no
    
    if printout:
        print("\n--------- mismatch check (ρ-based) ---------")
        print(f"2F(injection)  = {twoF_inj:10.3f}")
        print(f"2F(loudest)    = {twoF_mis:10.3f}")
        print(f"ρ²_no-mismatch = {rho2_no:10.3f}")
        print(f"ρ²_mismatch    = {rho2_mis:10.3f}")
        print(f"μ  (empirical) = {mu_empirical:10.3e}")
        print("-------------------------------------------")
        
    # mismatches.append(mu_empirical)
    del gridsearch            # 1️⃣ free Python references
    del fs                    # 2️⃣ free ComputeFstat object
    import gc; gc.collect()   # 3️⃣ force GC inside the worker

    return mu_empirical

if __name__ == "__main__":
    # run the mismatch calculation in parallel
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        futures = []
        for i in range(numbers):
            futures.append(executor.submit(calculate_mismatch, i))
        mismatches = [future.result() for future in concurrent.futures.as_completed(futures)]

    # save the mismatch results to a csv file
    mismatch_file = os.path.join(outdir, "mismatches.csv")
    np.savetxt(mismatch_file, mismatches, delimiter=",", header="Empirical Mismatch (μ)", comments='')


    # plot the mismatch distribution


    plt.figure(figsize=(10, 6))
    plt.hist(mismatches, bins=10, density=True, alpha=0.7, color='blue')
    plt.xlabel("Empirical Mismatch (μ)")
    plt.ylabel("Density")
    plt.title("Mismatch Distribution from Grid Search")
    plt.grid()
    plt.savefig(os.path.join(outdir, f"mismatch_distribution-max-mismatch:{mf}.pdf"))
    plt.show()

    
    # rumtime
    DeltaF0_fixed = 9.885590880794127e-06
    DeltaF1_fixed = 3.481585082097677e-12
    DeltaF2_fixed = 6.357202196709655e-19
    
    Nf0 = DeltaF0_fixed / dF0
    Nf1 = DeltaF1_fixed / dF1
    Nf2 = DeltaF2_fixed / dF2
    
    N_det = 2
    N_coh = (Nf0 + 1) * (Nf1 +1) * (Nf2 + 1)
    
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
        
    print("runtime: ", tau_total)
    
    