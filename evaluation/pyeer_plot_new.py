# Code for plotting results of matching after corrections (or no correction)
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pyeer.eer_info import get_eer_stats
from pyeer.plot import plot_eer_stats
import pyeer.plot

# USER CONFIGURATION - SET YOUR INPUT FILES AND OUTPUT DIRECTORY HERE
# 1. INPUT SCORES: Text files containing the raw matching scores for Left and Right eyes.
GEN_L_FILE = "gen_L_Unirnet_balanced.txt"
IMP_L_FILE = "imp_L_Unirnet_balanced.txt"
GEN_R_FILE = "gen_R_Unirnet_balanced.txt"
IMP_R_FILE = "imp_R_Unirnet_balanced.txt"

# 2. OUTPUT: Folder to save the generated PyEER plots (ROC, DET, FMR/FNMR).
OUTPUT_DIR = "./Plots_PyEER_Output/"


plt.rcParams.update({
    'font.size': 14,          
    'axes.labelsize': 16,     
    'axes.titlesize': 16,     
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,
    'figure.autolayout': True 
})

def _custom_error_curves(stats, ids, line_width=3, lgf_size=15, save_plots=True, dpi=None, save_path='', ext='.png'):
    for i, st in enumerate(stats):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_ylabel('EER')
        ax.set_xlabel('Matching Scores')
        ax.set_title('FMR and FNMR Curves')
        
        ax.plot(st.thrs, st.fmr, linewidth=line_width, label=f"{ids[i]} (FMR)")
        ax.plot(st.thrs, st.fnmr, linewidth=line_width, label=f"{ids[i]} (FNMR)")
        
        ax.set_ylim(0.0, 1.0)
        
        eer_2dec = round(st.eer, 2)
        ax.axhline(y=st.eer, color='red', linestyle='--', label=f'EER ({eer_2dec:.2f})')
        
        current_ticks = [t for t in ax.get_yticks() if abs(t - eer_2dec) > 0.04]
        ax.set_yticks(sorted(set(current_ticks + [eer_2dec])))
        
        ax.legend(loc='lower right', prop={'size': 12})
        
        if save_plots:
            fname = os.path.join(save_path, f"FMR and FNMR curves ({ids[i]}){ext}")
            fig.savefig(fname, dpi=dpi)
            plt.close('all')

_old_plt_roc_curve = pyeer.plot.plt_roc_curve

def _custom_roc_curve(stats, ids, line_width=3, lgf_size=15, save_plots=True, dpi=None, save_path='', ext='.png', resampling=True):
    _old_plt_roc_curve(stats, ids, line_width, lgf_size, False, dpi, save_path, ext, resampling)
    
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend:
                for text_obj in legend.get_texts():
                    old_text = text_obj.get_text()
                    new_text = re.sub(r'AUC = (\d+\.\d+)', lambda m: f"AUC = {float(m.group(1)):.2f}", old_text)
                    text_obj.set_text(new_text)
        
        if save_plots:
            title = fig.axes[0].get_title()
            fname = "ROC (Log)" if "Log" in title else "ROC"
            fig.savefig(os.path.join(save_path, f"{fname}{ext}"), dpi=dpi)
    plt.close('all')

pyeer.plot.plt_error_curves = _custom_error_curves
pyeer.plot.plt_roc_curve = _custom_roc_curve
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Left Eye Balanced Scores...")
gen_L = np.loadtxt(GEN_L_FILE)
imp_L = np.loadtxt(IMP_L_FILE)

print("Loading Right Eye Balanced Scores...")
gen_R = np.loadtxt(GEN_R_FILE)
imp_R = np.loadtxt(IMP_R_FILE)

print("Fusing Left and Right for the Global Combined Baseline...")
gen_combined = np.concatenate((gen_L, gen_R))
imp_combined = np.concatenate((imp_L, imp_R))

print("Crunching PyEER math for Left, Right, and Combined (ds_scores=True applied!)...")
stats_L = get_eer_stats(gen_L, imp_L, ds_scores=False)
stats_R = get_eer_stats(gen_R, imp_R, ds_scores=False)
stats_combined = get_eer_stats(gen_combined, imp_combined, ds_scores=False)

print("Generating Multi-Curve ROC and DET Graphs...")
plot_eer_stats(
    [stats_combined, stats_L, stats_R],
    ["Combined (Both Eyes)", "Left Eye Only", "Right Eye Only"],
    lgf_size=16, 
    save_plots=True,
    save_path=OUTPUT_DIR,
    ext=".png"
)

print("Graphs successfully generated and saved!")
