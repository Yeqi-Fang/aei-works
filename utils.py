import numpy as np
import matplotlib.pyplot as plt
import os



def calculate_mismatch_from_fstats(twoF_signal, twoF_template):
    """
    Calculate mismatch from F-statistic values.
    
    This uses the relationship: mismatch ≈ 1 - (2F_template - 4)/(2F_signal - 4)
    since 2F = 4 + SNR² for matched filtering.
    """
    # Convert 2F to SNR²: SNR² = 2F - 4
    snr2_signal = twoF_signal - 4.0
    snr2_template = twoF_template - 4.0
    print(f"2F signal: {twoF_signal:.6f}, 2F template: {twoF_template:.6f}")
    print(f"SNR² signal: {snr2_signal:.6f}, SNR² template: {snr2_template:.6f}")
    # Mismatch = 1 - (recovered power / optimal power)
    # if snr2_signal <= 0:
    #     return np.nan
    
    mismatch = 1.0 - (snr2_template / snr2_signal)
    return mismatch  # Mismatch should be non-negative



def calculate_f0_f1_mismatch_metric(delta_F0, delta_F1, Tseg):
    """
    Calculate mismatch using the parameter space metric for F0 and F1.
    
    For F0 and F1, the metric components are:
    g_00 = (π * T)² / 12  (for F0)
    g_11 = (π * T²)² / 720  (for F1) 
    g_01 = (π² * T³) / 120  (cross term)
    """
    # Metric components for F0-F1 parameter space
    g_00 = (np.pi * Tseg)**2 / 12.0          # F0-F0 component
    g_11 = (np.pi * Tseg**2)**2 / 720.0      # F1-F1 component  
    g_01 = (np.pi**2 * Tseg**3) / 120.0      # F0-F1 cross component
    
    # Quadratic form: m = δλᵀ g δλ
    mismatch = (g_00 * delta_F0**2 + 
                g_11 * delta_F1**2 + 
                2 * g_01 * delta_F0 * delta_F1)
    
    return mismatch, g_00, g_11, g_01

def calculate_sky_mismatch_metric(delta_Alpha, delta_Delta, Tseg):
    """
    Calculate mismatch for sky position parameters using approximate metric.
    """
    # Simplified sky metric (this is approximate)
    # For more precise calculations, use the full LALSuite metric functions
    
    # Typical scale for sky position metric
    # This varies with detector motion and sky position
    g_alpha = (2 * np.pi / (1 * 86400))**2 * Tseg**2  # Rough approximation
    g_delta = g_alpha  # Simplified assumption
    
    mismatch = g_alpha * delta_Alpha**2 + g_delta * delta_Delta**2
    
    return mismatch, g_alpha, g_delta



def plot_2F_scatter(res, label_name, xkey, ykey, labels, inj, outdir):
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



class MismatchAnalyzer:
    """
    Class to analyze mismatch from PyFstat search results.
    """
    
    def __init__(self, Tseg=None, duration=1.0, outdir='.'):
        self.Tseg = Tseg
        self.duration = duration  # Default duration if Tseg is not provided
        self.outdir = outdir
    
    def analyze_grid_search_mismatch(self, gridsearch_obj, injection_params):
        """
        Comprehensive mismatch analysis for a PyFstat GridSearch object.
        """
        if not hasattr(gridsearch_obj, 'data'):
            raise ValueError("GridSearch must be run first")
        
        # Get the best point from the search
        max_point = gridsearch_obj.get_max_twoF()
        
        # Determine coherent time
        Tseg = self.Tseg or self.duration
        
        # Calculate mismatch for each grid point
        analysis = {}
        
        # Parameter offsets from injection
        for param in ['F0', 'F1', 'Alpha', 'Delta']:
            if param in gridsearch_obj.data.dtype.names and param in injection_params:
                offsets = gridsearch_obj.data[param] - injection_params[param]
                analysis[f'{param}_offsets'] = offsets
        
        # Mismatch calculations
        mismatches = np.zeros(len(gridsearch_obj.data))
        
        # F0 contribution
        if 'F0' in injection_params and 'F0' in gridsearch_obj.data.dtype.names:
            F0_offsets = gridsearch_obj.data['F0'] - injection_params['F0']
            F0_mm = (np.pi * Tseg * F0_offsets) ** 2 / 12.0
            mismatches += F0_mm
            analysis['F0_mismatch'] = F0_mm
        
        # F1 contribution  
        if 'F1' in injection_params and 'F1' in gridsearch_obj.data.dtype.names:
            F1_offsets = gridsearch_obj.data['F1'] - injection_params['F1']
            F1_mm = (np.pi * Tseg**2 * F1_offsets) ** 2 / 720.0
            mismatches += F1_mm
            analysis['F1_mismatch'] = F1_mm
        
        # Sky position (simplified)
        if all(p in injection_params and p in gridsearch_obj.data.dtype.names 
               for p in ['Alpha', 'Delta']):
            # Approximate sky mismatch
            alpha_offset = gridsearch_obj.data['Alpha'] - injection_params['Alpha']
            delta_offset = gridsearch_obj.data['Delta'] - injection_params['Delta']
            
            # Rough sky metric
            sky_scale = (2 * np.pi / 86400)**2 * Tseg**2  
            sky_mm = sky_scale * (alpha_offset**2 + delta_offset**2)
            mismatches += sky_mm
            analysis['sky_mismatch'] = sky_mm
        
        analysis['total_mismatch'] = mismatches
        analysis['twoF_values'] = gridsearch_obj.data['twoF']
        analysis['coherent_time'] = Tseg
        analysis['injection_params'] = injection_params
        analysis['max_point'] = max_point
        
        # Find best mismatch point
        best_mm_idx = np.argmin(mismatches)
        analysis['best_mismatch_point'] = {
            'index': best_mm_idx,
            'mismatch': mismatches[best_mm_idx],
            'twoF': gridsearch_obj.data['twoF'][best_mm_idx],
            'parameters': {param: gridsearch_obj.data[param][best_mm_idx] 
                          for param in gridsearch_obj.data.dtype.names}
        }
        
        # Find loudest point mismatch
        loudest_idx = np.argmax(gridsearch_obj.data['twoF'])
        analysis['loudest_point'] = {
            'index': loudest_idx,
            'mismatch': mismatches[loudest_idx],
            'twoF': gridsearch_obj.data['twoF'][loudest_idx],
            'parameters': {param: gridsearch_obj.data[param][loudest_idx] 
                          for param in gridsearch_obj.data.dtype.names}
        }
        
        return analysis
    
    def plot_mismatch_vs_twoF(self, analysis, savefig=True, filename=None):
        """
        Plot mismatch vs 2F to visualize the relationship.
        """
        if filename is None:
            filename = os.path.join(self.outdir, "mismatch_vs_twoF_analysis.png")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Mismatch vs 2F scatter
        scatter = ax1.scatter(analysis['total_mismatch'], analysis['twoF_values'], 
                           c=analysis['twoF_values'], cmap='viridis', alpha=0.7, s=20)
        
        # Highlight important points
        loudest = analysis['loudest_point']
        best_mm = analysis['best_mismatch_point']
        
        ax1.scatter(loudest['mismatch'], loudest['twoF'],
                   c='red', s=100, marker='*', label='Loudest Point', zorder=5)
        ax1.scatter(best_mm['mismatch'], best_mm['twoF'],
                   c='orange', s=100, marker='s', label='Best Mismatch', zorder=5)
        
        ax1.set_xlabel('Mismatch')
        ax1.set_ylabel('2F')
        ax1.set_title('Mismatch vs 2F Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax1, label='2F')
        
        # Plot 2: Mismatch histogram
        ax2.hist(analysis['total_mismatch'], bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(loudest['mismatch'], color='red', linestyle='--', 
                   label=f'Loudest: {loudest["mismatch"]:.4f}')
        ax2.axvline(best_mm['mismatch'], color='orange', linestyle='--',
                   label=f'Best: {best_mm["mismatch"]:.4f}')
        
        ax2.set_xlabel('Mismatch')
        ax2.set_ylabel('Count')
        ax2.set_title('Mismatch Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Mismatch analysis plot saved to: {filename}")
        
        # plt.show()
        return fig, (ax1, ax2)
