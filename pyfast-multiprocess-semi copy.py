# pyfast-multiprocess-semi.py
import os
import numpy as np
import pyfstat
import config
from typing import Dict, Any
import pandas as pd
import matplotlib
# make sure to put these after the pyfstat import, to not break notebook inline plots
import matplotlib.pyplot as plt

# %matplotlib inline
from utils import plot_grid_vs_samples, plot_2F_scatter, CalculationParams
import matplotlib as mpl
mpl.use('Agg')

# --- one-off style tweaks ---------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": "serif",  # Times/Computer Modern-style text
        "mathtext.fontset": "cm",
        "axes.spines.top": False,  # hide unnecessary spines
        "axes.spines.right": False,
        "axes.linewidth": 1.2,  # make the remaining spines a little bolder
    }
)

import concurrent.futures

# flip this switch for a more expensive 4D (F0,F1,Alpha,Delta) run
# instead of just (F0,F1)
# (still only a few minutes on current laptops)

# log =


# general setup

logger = pyfstat.set_up_logger(
    label=config.label, outdir=config.outdir, log_level="WARNING"
)
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


def calculate_mismatch(i: int, params: CalculationParams, random_offsets: Dict[str, float]) -> float:

    import pyfstat
    import numpy as np
    import os

    F0s = [
        params.inj_params["F0"] - params.DeltaF0 / 2.0 + random_offsets["F0"],
        params.inj_params["F0"] + params.DeltaF0 / 2.0 + random_offsets["F0"],
        params.dF0
    ]
    F1s = [
        params.inj_params["F1"] - params.DeltaF1 / 2.0 + random_offsets["F1"],
        params.inj_params["F1"] + params.DeltaF1 / 2.0 + random_offsets["F1"],
        params.dF1_refined
    ]
    F2s = [
        params.inj_params["F2"] - params.DeltaF2 / 2.0 + random_offsets["F2"],
        params.inj_params["F2"] + params.DeltaF2 / 2.0 + random_offsets["F2"],
        params.dF2_refined
    ]

    search_keys = ["F0", "F1", "F2"]  # only the ones that aren't 0-width

    if params.sky:
        dSky = 0.01
        DeltaSky = 10 * dSky
        Alphas = [
            params.inj_params["Alpha"] - DeltaSky / 2.0,
            params.inj_params["Alpha"] + DeltaSky / 2.0,
            dSky
        ]
        Deltas = [
            params.inj_params["Delta"] - DeltaSky / 2.0,
            params.inj_params["Delta"] + DeltaSky / 2.0,
            dSky
        ]
        search_keys += ["Alpha", "Delta"]
    else:
        Alphas = [params.inj_params["Alpha"]]
        Deltas = [params.inj_params["Delta"]]

    search_keys_label = "".join(search_keys)

    # run the grid search
    # logger.info("Performing GridSearch...")
    gridsearch = pyfstat.GridSearch(
        label=f"GridSearch_iter_{i}" + search_keys_label,
        outdir=params.outdir,
        sftfilepattern=params.sftfilepath,
        F0s=F0s,
        F1s=F1s,
        F2s=F2s,
        Alphas=Alphas,
        Deltas=Deltas,
        tref=params.tref,
        nsegs=params.nsegs,
    )
    gridsearch.run()
    gridsearch.print_max_twoF()
    gridsearch.generate_loudest()

    # do some plots of the GridSearch results
    if not params.sky:  # this plotter can't currently deal with too large result arrays
        # logger.info("Plotting 1D 2F distributions...")
        if params.plot:
            for key in search_keys:
                gridsearch.plot_1D(
                    xkey=key, xlabel=params.labels[key], ylabel=params.labels["2F"]
                )

        # logger.info(
        #     "Making GridSearch {:s} corner plot...".format("-".join(search_keys))
        # )
        vals = [
            np.unique(gridsearch.data[key]) - params.inj_params[key] for key in search_keys
        ]
        twoF = gridsearch.data["twoF"].reshape([len(kval) for kval in vals])
        corner_labels = [
            "$f - f_0$ [Hz]",
            "$\\dot{f} - \\dot{f}_0$ [Hz/s]",
        ]
        if params.sky:
            corner_labels.append("$\\alpha - \\alpha_0$")
            corner_labels.append("$\\delta - \\delta_0$")
        corner_labels.append(params.labels["2F"])
        if params.plot:
            gridcorner_fig, gridcorner_axes = pyfstat.gridcorner(
                twoF,
                vals,
                projection="log_mean",
                labels=corner_labels,
                whspace=0.1,
                factor=1.8,
            )
            gridcorner_fig.savefig(
                os.path.join(params.outdir, gridsearch.label + "_corner.png")
            )
            # plt.show()

    # we'll use the two local plotting functions defined above
    # to avoid code duplication in the sky case
    if params.plot:
        plot_2F_scatter(gridsearch.data, "grid", "F0", "F1")
        if params.sky:
            plot_2F_scatter(gridsearch.data, "grid", "Alpha", "Delta")

    # -----------------------------------------------------------
    #  Mismatch diagnosis (API-safe version, PyFstat ≥ 2.x)
    # -----------------------------------------------------------

    search_ranges = {
        "F0": [params.inj_params["F0"]],  # a single value ⇒ zero width,
        "Alpha": [params.inj_params["Alpha"]],
        "Delta": [params.inj_params["Delta"]],
    }

    fs = pyfstat.SemiCoherentSearch(
        label=f"MismatchTest_{i}",  # SemiCoherentSearch需要label
        outdir=params.outdir,  # 需要outdir
        tref=params.tref,
        nsegs=params.nsegs,  # 添加分段数
        sftfilepattern=params.sftfilepath,
        minStartTime=params.tstart,
        maxStartTime=params.tstart + params.duration,
        search_ranges=search_ranges,
    )

    grid_res = gridsearch.data

    # template exactly at the injected parameters
    inj_pars = {k: params.inj_params[k] for k in ("F0", "F1", "F2", "Alpha", "Delta")}

    twoF_inj = fs.get_semicoherent_det_stat(params=inj_pars)

    rho2_no = twoF_inj - 4.0  # ρ²_no-mismatch

    # --- 2) loudest point from the grid you already ran ------------
    grid_maxidx = np.argmax(grid_res["twoF"])
    twoF_mis = grid_res["twoF"][grid_maxidx]
    rho2_mis = twoF_mis - 4.0  # ρ²_mismatch

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
    del gridsearch  # 1️⃣ free Python references
    del fs  # 2️⃣ free ComputeFstat object
    import gc

    gc.collect()  # 3️⃣ force GC inside the worker

    return mu_empirical


if __name__ == "__main__":
    
    grids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    for grid in grids:
        config.DeltaF0 = grid * config.dF0  # 500
        config.DeltaF1 = grid * config.dF1_refined  # 200
        config.DeltaF2 = grid * config.dF2_refined  # 60
        
        
        all_random_offsets = []
        for i in range(config.numbers):
            random_offsets = {
                "F0": np.random.uniform(-config.dF0 / 2, config.dF0 / 2),
                "F1": np.random.uniform(-config.dF1_refined / 2, config.dF1_refined / 2),
                "F2": np.random.uniform(-config.dF2_refined / 2, config.dF2_refined / 2)
            }
            all_random_offsets.append(random_offsets)

        
        params = CalculationParams(
            inj_params=config.inj,
            DeltaF0=config.DeltaF0,
            DeltaF1=config.DeltaF1,
            DeltaF2=config.DeltaF2,
            dF0=config.dF0,
            dF1_refined=config.dF1_refined,
            dF2_refined=config.dF2_refined,
            sky=config.sky,
            outdir=config.outdir,
            sftfilepath=writer.sftfilepath,  # This needs to be available
            tref=config.inj["tref"],
            nsegs=config.nsegs,
            plot=config.plot,
            labels=config.labels,
            tstart=config.tstart,
            duration=config.duration
        )
        
        # run the mismatch calculation in parallel
        with concurrent.futures.ProcessPoolExecutor(config.num_workers) as executor:
            futures = []
            for i in range(config.numbers):
                futures.append(executor.submit(calculate_mismatch, i, params, all_random_offsets[i]))
                
            mismatches = [future.result() for future in concurrent.futures.as_completed(futures)]

        # save the mismatch results to a csv file
        mismatch_file = os.path.join(config.outdir, f"new-mismatches-{config.inj['h0'] / config.sqrtSX}-{grid}.csv")
        np.savetxt(
            mismatch_file,
            mismatches,
            delimiter=",",
            header="Empirical Mismatch (μ)",
            comments="",
        )

        # plot the mismatch distribution

        fig, ax = plt.subplots(figsize=(10, 6))
        # choose bin edges so the last bar ends at 1.0, like in the photo
        # bins = np.linspace(0, 1, 11)        # 10 equal-width bins → 11 edges
        ax.hist(
            mismatches,
            bins=10,
            density=True,
            color="#5B9BD5",  # pleasant blue
            alpha=0.85,
            edgecolor="black",
            linewidth=1.0,
        )

        # axis labels & limits
        ax.set_xlabel(r"mismatch $\mu$", fontsize=20)
        ax.set_ylabel("normalized histogram", fontsize=20)
        ax.set_xlim(0, 1)
        # ax.set_ylim(0, 0.25)

        # ticks & grid
        ax.tick_params(axis="both", which="major", labelsize=14, length=6)
        ax.grid(axis="y", linewidth=0.6, alpha=0.35)

        fig.tight_layout()
        fig.savefig(os.path.join(config.outdir, f"mismatch_distribution-max-mismatch:{config.mf}-grid:{grid}.pdf"))
        # plt.show()

        # rumtime
        # print("runtime: ", config.tau_total)

        print("mean mismatch: ", np.mean(mismatches))
        