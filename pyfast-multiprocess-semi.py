import os
import numpy as np
import pyfstat
import config
# make sure to put these after the pyfstat import, to not break notebook inline plots
import matplotlib.pyplot as plt
# %matplotlib inline
from utils import plot_grid_vs_samples, plot_2F_scatter


import concurrent.futures
# flip this switch for a more expensive 4D (F0,F1,Alpha,Delta) run
# instead of just (F0,F1)
# (still only a few minutes on current laptops)

# log = 


# general setup

logger = pyfstat.set_up_logger(label=config.label, outdir=config.outdir, log_level="WARNING")
if config.sky:
    config.outdir += "AlphaDelta"
printout = False
# parameters for the data set to generate


# parameters for injected signals

# create SFT files
logger.info("Generating SFTs with injected signal...")
writer = pyfstat.Writer(
    label=config.label + "SimulatedSignal",
    outdir=config.outdir,
    tstart=config.tstart,
    duration=config.duration,
    detectors=config.detectors,
    sqrtSX=config.sqrtSX,
    Tsft=config.Tsft,
    **config.inj,
    Band=1,  # default band estimation would be too narrow for a wide grid/prior
)
writer.make_data()

# set up square search grid with fixed (F0,F1) mismatch
# and (optionally) some ad-hoc sky coverage

print(config.DeltaF0, config.DeltaF1, config.DeltaF2)
    

mismatches = []
F0s_random = np.random.uniform(-config.dF0, config.dF0, size=config.numbers)
F1s_random = np.random.uniform(-config.dF1_refined, config.dF1_refined, size=config.numbers)
F2s_random = np.random.uniform(-config.dF2_refined, config.dF2_refined, size=config.numbers)


def calculate_mismatch(i):

    
    import pyfstat
    import numpy as np
    import os
    

    F0s = [config.inj["F0"] - config.DeltaF0 / 2.0 + F0s_random[i], config.inj["F0"] + config.DeltaF0 / 2.0 + F0s_random[i], config.dF0]
    F1s = [config.inj["F1"] - config.DeltaF1 / 2.0 + F1s_random[i], config.inj["F1"] + config.DeltaF1 / 2.0 + F1s_random[i], config.dF1_refined]
    F2s = [config.inj["F2"] - config.DeltaF2 / 2.0 + F2s_random[i], config.inj["F2"] + config.DeltaF2 / 2.0 + F2s_random[i], config.dF2_refined]
    
    
    search_keys = ["F0", "F1", "F2"]  # only the ones that aren't 0-width

    if config.sky:
        dSky = 0.01  # rather coarse to keep runtime down
        DeltaSky = 10 * dSky
        Alphas = [config.inj["Alpha"] - DeltaSky / 2.0, config.inj["Alpha"] + DeltaSky / 2.0, dSky]
        Deltas = [config.inj["Delta"] - DeltaSky / 2.0, config.inj["Delta"] + DeltaSky / 2.0, dSky]
        search_keys += ["Alpha", "Delta"]
    else:
        Alphas = [config.inj["Alpha"]]
        Deltas = [config.inj["Delta"]]

    search_keys_label = "".join(search_keys)

    # run the grid search
    logger.info("Performing GridSearch...")
    gridsearch = pyfstat.GridSearch(
        label=f"GridSearch_iter_{i}" + search_keys_label,
        outdir=config.outdir,
        sftfilepattern=writer.sftfilepath,
        F0s=F0s,
        F1s=F1s,
        F2s=F2s,
        Alphas=Alphas,
        Deltas=Deltas,
        tref=config.inj["tref"],
        nsegs=config.nsegs,
    )
    gridsearch.run()
    gridsearch.print_max_twoF()
    gridsearch.generate_loudest()


    # do some plots of the GridSearch results
    if not config.sky:  # this plotter can't currently deal with too large result arrays
        logger.info("Plotting 1D 2F distributions...")
        if config.plot:
            for key in search_keys:
                gridsearch.plot_1D(xkey=key, xlabel=config.labels[key], ylabel=config.labels["2F"])

        logger.info("Making GridSearch {:s} corner plot...".format("-".join(search_keys)))
        vals = [np.unique(gridsearch.data[key]) - config.inj[key] for key in search_keys]
        twoF = gridsearch.data["twoF"].reshape([len(kval) for kval in vals])
        corner_labels = [
            "$f - f_0$ [Hz]",
            "$\\dot{f} - \\dot{f}_0$ [Hz/s]",
        ]
        if config.sky:
            corner_labels.append("$\\alpha - \\alpha_0$")
            corner_labels.append("$\\delta - \\delta_0$")
        corner_labels.append(config.labels["2F"])
        if config.plot:
            gridcorner_fig, gridcorner_axes = pyfstat.gridcorner(
                twoF, vals, projection="log_mean", labels=corner_labels,
                whspace=0.1, factor=1.8
            )
            gridcorner_fig.savefig(os.path.join(config.outdir, gridsearch.label + "_corner.png"))
            # plt.show()



    # we'll use the two local plotting functions defined above
    # to avoid code duplication in the sky case
    if config.plot:
        plot_2F_scatter(gridsearch.data, "grid", "F0", "F1")
        if config.sky:
            plot_2F_scatter(gridsearch.data, "grid", "Alpha", "Delta")
        
        
        
    # -----------------------------------------------------------
    #  Mismatch diagnosis (API-safe version, PyFstat ≥ 2.x)
    # -----------------------------------------------------------


    search_ranges = {
        "F0":    [config.inj["F0"]],         # a single value ⇒ zero width,
        "Alpha": [config.inj["Alpha"]],
        "Delta": [config.inj["Delta"]],
    }
    
    fs = pyfstat.SemiCoherentSearch(
        label="MismatchTest",  # SemiCoherentSearch需要label
        outdir=config.outdir,         # 需要outdir
        tref=config.inj["tref"],
        nsegs=config.nsegs,              # 添加分段数
        sftfilepattern=writer.sftfilepath,
        minStartTime=config.tstart,
        maxStartTime=config.tstart + config.duration,
        search_ranges=search_ranges,
    )

    grid_res = gridsearch.data

    # template exactly at the injected parameters
    inj_pars = {k: config.inj[k] for k in ("F0", "F1", "F2", "Alpha", "Delta")}

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
    with concurrent.futures.ProcessPoolExecutor(config.num_workers) as executor:
        futures = []
        for i in range(config.numbers):
            futures.append(executor.submit(calculate_mismatch, i))
        mismatches = [future.result() for future in concurrent.futures.as_completed(futures)]

    # save the mismatch results to a csv file
    mismatch_file = os.path.join(config.outdir, "mismatches.csv")
    np.savetxt(mismatch_file, mismatches, delimiter=",", header="Empirical Mismatch (μ)", comments='')


    # plot the mismatch distribution


    plt.figure(figsize=(10, 6))
    plt.hist(mismatches, bins=10, density=True, alpha=0.7, color='blue')
    plt.xlabel("Empirical Mismatch (μ)")
    plt.ylabel("Density")
    plt.title("Mismatch Distribution from Grid Search")
    plt.grid()
    plt.savefig(os.path.join(config.outdir, f"mismatch_distribution-max-mismatch:{config.mf}.pdf"))
    plt.show()

    
    # rumtime    
    print("runtime: ", config.tau_total)
    
    