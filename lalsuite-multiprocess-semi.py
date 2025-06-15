# lalsuite-multiprocess-semi.py
import os
import glob            # ➊  add to the imports at the top of the file
import numpy as np
import lal
import lalpulsar
import config
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import logging
import sys
import subprocess
from utils import plot_grid_vs_samples, plot_2F_scatter, CalculationParams

# --- Setup logging directly ---
def setup_logger(label=None, outdir=None, log_level="INFO"):
    """Setup logger using standard Python logging instead of pyfstat"""
    logger = logging.getLogger("mismatch_calc")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(name)s %(levelname)-8s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if outdir and label provided
    if outdir and label:
        os.makedirs(outdir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(outdir, f"{label}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# --- Get ephemeris files ---
def get_ephemeris_files():
    """Get default ephemeris file paths"""
    config_file = os.path.join(os.path.expanduser("~"), ".pyfstat.conf")
    ephem_version = "DE405"
    earth_ephem = f"earth00-40-{ephem_version}.dat.gz"
    sun_ephem = f"sun00-40-{ephem_version}.dat.gz"
    
    if os.path.isfile(config_file):
        d = {}
        with open(config_file, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    d[k] = v
        earth_ephem = d.get("earth_ephem", earth_ephem)
        sun_ephem = d.get("sun_ephem", sun_ephem)
    
    return earth_ephem, sun_ephem

# --- Direct LAL SFT Writer replacement ---
class LALWriter:
    """Direct LAL replacement for pyfstat.Writer"""
    
    def __init__(self, label, outdir, tstart, duration, detectors, sqrtSX, Tsft, 
                 Band=1, **inj_params):
        self.label = label
        self.outdir = outdir
        self.tstart = tstart
        self.duration = duration
        self.detectors = detectors.split(',') if isinstance(detectors, str) else detectors
        self.sqrtSX = sqrtSX
        self.Tsft = Tsft
        self.Band = Band
        self.inj_params = inj_params
        
        # Set up ephemeris files
        self.earth_ephem, self.sun_ephem = get_ephemeris_files()
        
        # Create output directory
        os.makedirs(outdir, exist_ok=True)
        
    def make_data(self):
        """Create SFT files with injected signal using LAL directly"""
        # Build the Makefakedata_v5 command line
        cl_mfd = ["lalpulsar_Makefakedata_v5"]
        cl_mfd.append("--outSingleSFT=TRUE")
        cl_mfd.append(f'--outSFTdir="{self.outdir}"')
        cl_mfd.append(f'--outLabel="{self.label}"')
        # cl_mfd.append(f'--IFOs={",".join([f\'"{d}\' for d in self.detectors])}')
        cl_mfd.append(f'--IFOs="{",".join(self.detectors)}"')
        
        if self.sqrtSX:
            cl_mfd.append(f'--sqrtSX="{self.sqrtSX}"')
        
        cl_mfd.append(f"--startTime={self.tstart}")
        cl_mfd.append(f"--duration={self.duration}")
        cl_mfd.append(f"--Tsft={self.Tsft}")
        
        # Calculate frequency range
        F0 = self.inj_params.get('F0', 100)
        fmin = F0 - self.Band/2
        cl_mfd.append(f"--fmin={fmin:.16g}")
        cl_mfd.append(f"--Band={self.Band:.16g}")
        
        # Create injection config file if needed
        if self.inj_params.get('h0', 0) > 0:
            config_file = os.path.join(self.outdir, f"{self.label}.cff")
            with open(config_file, 'w') as f:
                f.write("[TS0]\n")
                f.write(f"Alpha = {self.inj_params.get('Alpha', 0):.18e}\n")
                f.write(f"Delta = {self.inj_params.get('Delta', 0):.18e}\n")
                f.write(f"h0 = {self.inj_params.get('h0', 0):.18e}\n")
                f.write(f"cosi = {self.inj_params.get('cosi', 0):.18e}\n")
                f.write(f"psi = {self.inj_params.get('psi', 0):.18e}\n")
                f.write(f"phi0 = {self.inj_params.get('phi', 0):.18e}\n")
                f.write(f"Freq = {F0:.18e}\n")
                f.write(f"f1dot = {self.inj_params.get('F1', 0):.18e}\n")
                f.write(f"f2dot = {self.inj_params.get('F2', 0):.18e}\n")
                f.write(f"refTime = {self.inj_params.get('tref', self.tstart):.9f}\n")
            cl_mfd.append(f'--injectionSources="{config_file}"')
        
        cl_mfd.append(f'--ephemEarth="{self.earth_ephem}"')
        cl_mfd.append(f'--ephemSun="{self.sun_ephem}"')
        
        # Run Makefakedata_v5
        cmd = " ".join(cl_mfd)
        logger.info(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        
        # Build expected SFT file paths
        # self.sftfilepath = ";".join([
        #     os.path.join(self.outdir, f"{det}-1_{self.label}.sft") 
        #     for det in self.detectors
        # ])

        pattern = os.path.join(self.outdir, f"*{self.label}*.sft")
        files   = sorted(glob.glob(pattern))
        if not files:
            raise RuntimeError(f"No SFTs found that match {pattern}")
        self.sftfilepath = ";".join(files)                  # ➌ <- now a ;-separated list
        logger.info(f"SFTs found ({len(files)}): {self.sftfilepath}")

# --- Direct LAL Grid Search replacement ---
class LALGridSearch:
    """Direct LAL replacement for pyfstat.GridSearch"""
    
    def __init__(self, label, outdir, sftfilepattern, F0s, F1s, F2s, 
                 Alphas, Deltas, tref, nsegs=1):
        self.label = label
        self.outdir = outdir
        self.sftfilepattern = sftfilepattern
        self.F0s = self._parse_range(F0s)
        self.F1s = self._parse_range(F1s) 
        self.F2s = self._parse_range(F2s)
        self.Alphas = self._parse_range(Alphas)
        self.Deltas = self._parse_range(Deltas)
        self.tref = tref
        self.nsegs = nsegs
        
        # Setup LAL F-stat computation
        self._setup_fstat()
        
    # 2. FIX: Grid point generation to match PyFstat exactly
    def _parse_range(self, param_range):
        """Parse parameter range [min, max, step] into array - FIXED VERSION"""
        if len(param_range) == 1:
            return np.array(param_range)
        elif len(param_range) == 3:
            start, stop, step = param_range
            # FIXED: Use same calculation as PyFstat
            if step == 0:
                return np.array([start])
            npoints = int(np.round((stop - start) / step)) + 1
            return np.linspace(start, stop, npoints)
        else:
            return np.array(param_range)
            
    def _setup_fstat(self):
        """Setup LAL F-statistic computation"""
        # Load SFT catalog
        constraints = lalpulsar.SFTConstraints()
        self.catalog = lalpulsar.SFTdataFind(self.sftfilepattern, constraints)
        
        # Setup ephemeris
        earth_ephem, sun_ephem = get_ephemeris_files()
        self.ephems = lalpulsar.InitBarycenter(earth_ephem, sun_ephem)
        
        df          = fbin_from_catalog(self.catalog)
        FREQ_BUFFER = 20 * df          # 20 guard bins  ≈ 0.01 Hz for 1800 s SFTs
        fMin        = np.min(self.F0s) - FREQ_BUFFER
        fMax        = np.max(self.F0s) + FREQ_BUFFER
        # print(fMax)
        # print(fMin)
        
        # Setup F-stat input
        self.FstatInput = lalpulsar.CreateFstatInput(
            self.catalog, fMin, fMax, 0, self.ephems, None
        )
        
        # Setup Doppler parameters
        self.PulsarDopplerParams = lalpulsar.PulsarDopplerParams()
        self.PulsarDopplerParams.refTime = self.tref
        self.PulsarDopplerParams.fkdot = np.zeros(lalpulsar.PULSAR_MAX_SPINS)
        
        # Setup F-stat results
        self.FstatResults = lalpulsar.FstatResults()
        
    def run(self):
        """Run the grid search"""
        results = []
        
        for F0 in self.F0s:
            for F1 in self.F1s:
                for F2 in self.F2s:
                    for Alpha in self.Alphas:
                        for Delta in self.Deltas:
                            # Set parameters
                            self.PulsarDopplerParams.fkdot[0] = F0
                            self.PulsarDopplerParams.fkdot[1] = F1  
                            self.PulsarDopplerParams.fkdot[2] = F2
                            self.PulsarDopplerParams.Alpha = Alpha
                            self.PulsarDopplerParams.Delta = Delta
                            
                            # Compute F-statistic
                            lalpulsar.ComputeFstat(
                                Fstats=self.FstatResults,
                                input=self.FstatInput,
                                doppler=self.PulsarDopplerParams,
                                numFreqBins=1,
                                whatToCompute=lalpulsar.FSTATQ_2F
                            )
                            
                            twoF = float(self.FstatResults.twoF[0])
                            results.append({
                                'F0': F0, 'F1': F1, 'F2': F2,
                                'Alpha': Alpha, 'Delta': Delta,
                                'twoF': twoF
                            })
        
        # Convert to structured array similar to pyfstat
        self.data = np.array([(r['F0'], r['F1'], r['F2'], r['Alpha'], r['Delta'], r['twoF']) 
                             for r in results],
                            dtype=[('F0', 'f8'), ('F1', 'f8'), ('F2', 'f8'),
                                   ('Alpha', 'f8'), ('Delta', 'f8'), ('twoF', 'f8')])
    
    def print_max_twoF(self):
        """Print maximum 2F value and parameters"""
        idx = np.argmax(self.data['twoF'])
        max_result = self.data[idx]
        print(f"Grid point with max(twoF) for {self.label}:")
        for key in ['F0', 'F1', 'F2', 'Alpha', 'Delta', 'twoF']:
            print(f"  {key}={max_result[key]}")
    
    def generate_loudest(self):
        """Generate loudest file - simplified version"""
        pass


def fbin_from_catalog(cat):
    if cat.length == 0:
        raise ValueError("Empty SFT catalog")
    return cat.data[0].fbin



# 3. FIX: Semi-coherent search - complete rewrite
class LALSemiCoherentSearch:
    def __init__(self, label, outdir, tref, nsegs, sftfilepattern, 
                 minStartTime, maxStartTime, search_ranges):
        self.label = label
        self.outdir = outdir
        self.tref = tref
        self.nsegs = nsegs
        self.sftfilepattern = sftfilepattern
        self.minStartTime = minStartTime
        self.maxStartTime = maxStartTime
        self.search_ranges = search_ranges
        self.F0s           = np.array(search_ranges["F0"])  # NEW
        
        # Setup segments first
        self._setup_segments()
        # Setup F-stat computation for each segment
        self._setup_fstat_segments()
        
    def _setup_segments(self):
        """Set up time segments for semi-coherent search"""
        self.segment_boundaries = np.linspace(
            self.minStartTime, self.maxStartTime, self.nsegs + 1
        )
        self.Tcoh = self.segment_boundaries[1] - self.segment_boundaries[0]
        
    def _setup_fstat_segments(self):
        """Setup F-stat computation for each segment - FIXED VERSION"""
        # Setup ephemeris once
        earth_ephem, sun_ephem = get_ephemeris_files()
        self.ephems = lalpulsar.InitBarycenter(earth_ephem, sun_ephem)
        

       
        df          = fbin_from_catalog(self.catalog)
        FREQ_BUFFER = 20 * df          # 20 guard bins  ≈ 0.01 Hz for 1800 s SFTs
        fMin        = np.min(self.F0s) - FREQ_BUFFER
        fMax        = np.max(self.F0s) + FREQ_BUFFER
        
        # FIXED: Pre-load SFT catalogs for each segment to avoid memory issues
        self.segment_catalogs = []
        self.segment_inputs = []
        
        for i in range(self.nsegs):
            # Set time constraints for this segment
            seg_constraints = lalpulsar.SFTConstraints()
            seg_constraints.minStartTime = lal.LIGOTimeGPS(self.segment_boundaries[i])
            seg_constraints.maxStartTime = lal.LIGOTimeGPS(self.segment_boundaries[i+1])
            
            # Load SFTs for this segment
            seg_catalog = lalpulsar.SFTdataFind(self.sftfilepattern, seg_constraints)
            self.segment_catalogs.append(seg_catalog)
            
            # Create F-stat input for segment if SFTs exist
            if seg_catalog.length > 0:
                seg_input = lalpulsar.CreateFstatInput(
                    seg_catalog, fMin, fMax, 0, self.ephems, None
                )
                self.segment_inputs.append(seg_input)
            else:
                self.segment_inputs.append(None)
        
        # Setup Doppler parameters
        self.PulsarDopplerParams = lalpulsar.PulsarDopplerParams()
        self.PulsarDopplerParams.refTime = self.tref
        self.PulsarDopplerParams.fkdot = np.zeros(lalpulsar.PULSAR_MAX_SPINS)
        
    def get_semicoherent_det_stat(self, params):
        """Compute semi-coherent detection statistic - FIXED VERSION"""
        # Set parameters
        self.PulsarDopplerParams.fkdot[0] = params['F0']
        self.PulsarDopplerParams.fkdot[1] = params['F1'] 
        self.PulsarDopplerParams.fkdot[2] = params['F2']
        self.PulsarDopplerParams.Alpha = params['Alpha']
        self.PulsarDopplerParams.Delta = params['Delta']
        
        # FIXED: Sum F-stats over segments using pre-loaded inputs
        twoF_sum = 0
        valid_segments = 0
        
        for i in range(self.nsegs):
            if self.segment_inputs[i] is None:
                continue
                
            # Compute F-stat for segment using pre-loaded input
            seg_results = lalpulsar.FstatResults()
            lalpulsar.ComputeFstat(
                Fstats=seg_results,
                input=self.segment_inputs[i],
                doppler=self.PulsarDopplerParams,
                numFreqBins=1,
                whatToCompute=lalpulsar.FSTATQ_2F
            )
            
            twoF_sum += float(seg_results.twoF[0])
            valid_segments += 1
            
        self.twoF = twoF_sum
        return twoF_sum

# --- Main script logic ---
logger = setup_logger(
    label=config.label, outdir=config.outdir, log_level="WARNING"
)
if config.sky:
    config.outdir += "AlphaDelta"
printout = False

# create SFT files
logger.info("Generating SFTs with injected signal...")
writer = LALWriter(
    label=config.label + "SimulatedSignal",
    outdir=config.outdir,
    tstart=config.tstart,
    duration=config.duration,
    detectors=config.detectors,
    sqrtSX=config.sqrtSX,
    Tsft=config.Tsft,
    **config.inj,
    Band = max(1.0, config.DeltaF0 + 2 * 20 / config.Tsft)
)
writer.make_data()

# set up square search grid with fixed (F0,F1) mismatch
print(config.DeltaF0, config.DeltaF1, config.DeltaF2)

mismatches = []

def calculate_mismatch(i: int, params: CalculationParams, random_offsets: Dict[str, float]) -> float:
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

    search_keys = ["F0", "F1", "F2"]

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

    # run the grid search using LAL directly
    gridsearch = LALGridSearch(
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
    if not params.sky:
        if params.plot:
            for key in search_keys:
                # Simple 1D plot
                x = np.unique(gridsearch.data[key])
                y = [np.mean(gridsearch.data['twoF'][gridsearch.data[key] == xi]) for xi in x]
                plt.figure()
                plt.plot(x - params.inj_params[key], y)
                plt.xlabel(f"{key} - {key}_0")
                plt.ylabel("2F")
                plt.savefig(os.path.join(params.outdir, f"{gridsearch.label}_1D_{key}.png"))
                plt.close()

        # Create corner plot
        vals = [
            np.unique(gridsearch.data[key]) - params.inj_params[key] for key in search_keys
        ]
        twoF_shape = [len(val) for val in vals]
        twoF = gridsearch.data["twoF"].reshape(twoF_shape)
        corner_labels = [
            "$f - f_0$ [Hz]",
            "$\\dot{f} - \\dot{f}_0$ [Hz/s]", 
        ]
        if params.sky:
            corner_labels.append("$\\alpha - \\alpha_0$")
            corner_labels.append("$\\delta - \\delta_0$")
        corner_labels.append("2F")
        if params.plot:
            # Simple corner plot replacement
            fig, axes = plt.subplots(figsize=(10, 8))
            if len(vals) >= 2:
                X, Y = np.meshgrid(vals[0], vals[1])
                Z = np.log(np.mean(twoF, axis=tuple(range(2, len(twoF.shape)))))
                im = axes.contour(X, Y, Z.T)
                axes.set_xlabel(corner_labels[0])
                axes.set_ylabel(corner_labels[1])
            fig.savefig(os.path.join(params.outdir, gridsearch.label + "_corner.png"))
            plt.close(fig)

    # we'll use the two local plotting functions defined above
    if params.plot:
        plot_2F_scatter(gridsearch.data, "grid", "F0", "F1")
        if params.sky:
            plot_2F_scatter(gridsearch.data, "grid", "Alpha", "Delta")

    # -----------------------------------------------------------
    #  Mismatch diagnosis using LAL directly
    # -----------------------------------------------------------

    search_ranges = {
        "F0": [params.inj_params["F0"]],
        "Alpha": [params.inj_params["Alpha"]],
        "Delta": [params.inj_params["Delta"]],
    }

    fs = LALSemiCoherentSearch(
        label=f"MismatchTest_{i}",
        outdir=params.outdir,
        tref=params.tref,
        nsegs=params.nsegs,
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

    # Clean up memory
    del gridsearch
    del fs
    import gc
    gc.collect()

    return mu_empirical


if __name__ == "__main__":
    
    all_random_offsets = []
    for i in range(config.numbers):
        random_offsets = {
            "F0": np.random.uniform(-config.dF0, config.dF0),
            "F1": np.random.uniform(-config.dF1_refined, config.dF1_refined),
            "F2": np.random.uniform(-config.dF2_refined, config.dF2_refined)
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
        sftfilepath=writer.sftfilepath,
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
    mismatch_file = os.path.join(config.outdir, "mismatches.csv")
    np.savetxt(
        mismatch_file,
        mismatches,
        delimiter=",",
        header="Empirical Mismatch (μ)",
        comments="",
    )

    # plot the mismatch distribution
    fig, ax = plt.subplots(figsize=(10, 6))
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

    # ticks & grid
    ax.tick_params(axis="both", which="major", labelsize=14, length=6)
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)

    fig.tight_layout()
    fig.savefig(os.path.join(config.outdir, f"mismatch_distribution-max-mismatch:{config.mf}.pdf"))
    plt.show()

    # runtime
    print("runtime: ", config.tau_total)
    print("mean mismatch: ", np.mean(mismatches))