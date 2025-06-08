import numpy as np
import matplotlib.pyplot as plt
import os
import config


zoom = {
        "F0": [config.inj["F0"] - 10 * config.dF0, config.inj["F0"] + 10 * config.dF0],
        "F1": [config.inj["F1"] - 5 * config.dF1, config.inj["F1"] + 5 * config.dF1],
}

# some plotting helper functions
def plot_grid_vs_samples(grid_res, mcmc_res, xkey, ykey):
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
    plt.savefig(plotfilename_base + ".png")
    # plt.show()
