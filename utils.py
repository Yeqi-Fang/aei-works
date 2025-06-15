import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any
from dataclasses import dataclass


def get_zoom(cfg):
    return {
        "F0": [cfg.inj["F0"] - 10*cfg.dF0, cfg.inj["F0"] + 10*cfg.dF0],
        "F1": [cfg.inj["F1"] -  5*cfg.dF1_refined, cfg.inj["F1"] +  5*cfg.dF1_refined],
    }

# some plotting helper functions
def plot_grid_vs_samples(grid_res, mcmc_res, xkey, ykey, config):
    """local plotting function to avoid code duplication in the 4D case"""
    plt.plot(grid_res[xkey], grid_res[ykey], ".", label="grid")
    plt.plot(mcmc_res[xkey], mcmc_res[ykey], ".", label="mcmc")
    plt.plot(config.inj[xkey], config.inj[ykey], "*k", label="injection")
    grid_maxidx = np.argmax(grid_res["twoF"])
    mcmc_maxidx = np.argmax(mcmc_res["twoF"])
    plt.plot(
        grid_res[xkey][grid_maxidx],
        grid_res[ykey][grid_maxidx],
        "+g",
        label=config.labels["max2F"] + "(grid)",
    )
    plt.plot(
        mcmc_res[xkey][mcmc_maxidx],
        mcmc_res[ykey][mcmc_maxidx],
        "xm",
        label=config.labels["max2F"] + "(mcmc)",
    )
    plt.xlabel(config.labels[xkey])
    plt.ylabel(config.labels[ykey])
    plt.legend()
    plotfilename_base = os.path.join(config.outdir, "grid_vs_mcmc_{:s}{:s}".format(xkey, ykey))
    plt.savefig(plotfilename_base + ".png")
    zoom = get_zoom(config)
    if xkey == "F0" and ykey == "F1":
        plt.xlim(zoom[xkey])
        plt.ylim(zoom[ykey])
        plt.savefig(plotfilename_base + "_zoom.png")
    # plt.show()

def plot_2F_scatter(res, label, xkey, ykey, config):
    """local plotting function to avoid code duplication in the 4D case"""
    markersize = 1 if label == "grid" else 0.5
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(res[xkey], res[ykey], c=res["twoF"], s=markersize)
    cb = plt.colorbar(sc)
    plt.xlabel(config.labels[xkey])
    plt.ylabel(config.labels[ykey])
    cb.set_label(config.labels["2F"])
    plt.title(label)
    plt.plot(config.inj[xkey], config.inj[ykey], "*k", label="injection")
    maxidx = np.argmax(res["twoF"])
    plt.plot(
        res[xkey][maxidx],
        res[ykey][maxidx],
        "+r",
        label=config.labels["max2F"],
    )
    plt.legend(loc='upper right')
    plotfilename_base = os.path.join(config.outdir, "{:s}_{:s}{:s}_2F".format(label, xkey, ykey))
    plt.xlim([min(res[xkey]), max(res[xkey])])
    plt.ylim([min(res[ykey]), max(res[ykey])])
    plt.savefig(plotfilename_base + ".pdf")
    plt.close("all")
    # plt.show()


@dataclass
class CalculationParams:
    """Structure to hold all parameters needed for calculation"""
    inj_params: Dict[str, Any]
    DeltaF0: float
    DeltaF1: float
    DeltaF2: float
    dF0: float
    dF1_refined: float
    dF2_refined: float
    sky: bool
    outdir: str
    sftfilepath: str
    tref: int
    nsegs: int
    plot: bool
    labels: Dict[str, str]
    tstart: int
    duration: int
    @property
    def inj(self):
        return self.inj_params     # <-- makes existing helpers happy
# The next step : 1. make each run return a list rather than a mismatch value (mf, mfdot)
# change the mf, mfdot and get different point
# train the model