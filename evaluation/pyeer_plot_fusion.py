# Code for plotting results of matching after fusion
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
from pyeer.eer_info import get_eer_stats
from pyeer.plot import plot_eer_stats


# 1. INPUT: The master CSV file containing the paired matcher scores.
CSV_PATH = "master_fused_scores.csv"

# 2. OUTPUT - WEIGHT COMPARISONS: Folder to save ROC/DET plots for different fusion ratios.
DIR_WEIGHTS = "./Plots_Fusion/"

# 3. OUTPUT - EYE SIDEDNESS: Folder to save the Left vs. Right eye performance plots.
DIR_LR = "./Plots_L_R_Fusion/"
# ==============================================================================

print(f"Loading Master Dataset: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print(f"Total Pairs Before Filtering: {len(df)}")

df = df[(df['score_iris'] > 0.0) & (df['score_peri'] > 0.0)].dropna().copy()

print(f"Total Pairs After Filtering (Intersection): {len(df)}")

print("Parsing Left and Right eyes...")
df['eye_side'] = df['probe'].apply(lambda x: os.path.basename(x).split('_')[2][0])

gen_L = df[(df['label'] == 1) & (df['eye_side'] == 'L')]
gen_R = df[(df['label'] == 1) & (df['eye_side'] == 'R')]
imp_L = df[(df['label'] == 0) & (df['eye_side'] == 'L')]
imp_R = df[(df['label'] == 0) & (df['eye_side'] == 'R')]

print("\n--- PURE POPULATION COUNTS ---")
print(f"Genuine Left:  {len(gen_L)}")
print(f"Genuine Right: {len(gen_R)}")
print(f"Imposter Left: {len(imp_L)}")
print(f"Imposter Right:{len(imp_R)}")
print("------------------------------\n")

print("Calculating Static Fusion Weights...")
s_iris = df['score_iris'].values
s_peri = df['score_peri'].values

df['fused_50_50'] = (0.50 * s_iris) + (0.50 * s_peri)
df['fused_75_25'] = (0.75 * s_iris) + (0.25 * s_peri) 
df['fused_25_75'] = (0.25 * s_iris) + (0.75 * s_peri) 
df['fused_40_60'] = (0.40 * s_iris) + (0.60 * s_peri) 

def get_stats(dataframe, score_column):
    gen_scores = dataframe[dataframe['label'] == 1][score_column].values
    imp_scores = dataframe[dataframe['label'] == 0][score_column].values
    return get_eer_stats(gen_scores, imp_scores, ds_scores=False)

def get_d_prime(gen_scores, imp_scores):
    mu_gen = np.mean(gen_scores)
    mu_imp = np.mean(imp_scores)
    var_gen = np.var(gen_scores)
    var_imp = np.var(imp_scores)
    
    d_prime = abs(mu_gen - mu_imp) / np.sqrt(0.5 * (var_gen + var_imp))
    return d_prime

gen_df = df[df['label'] == 1]
imp_df = df[df['label'] == 0]

print("\n--- D-PRIME SCORES ---")
print(f"Iris Only d':      {get_d_prime(gen_df['score_iris'], imp_df['score_iris']):.4f}")
print(f"Periocular Only d':{get_d_prime(gen_df['score_peri'], imp_df['score_peri']):.4f}")
print(f"Fusion (50/50) d': {get_d_prime(gen_df['fused_50_50'], imp_df['fused_50_50']):.4f}")
print(f"Fusion (75/25) d': {get_d_prime(gen_df['fused_75_25'], imp_df['fused_75_25']):.4f}")
print(f"Fusion (25/75) d': {get_d_prime(gen_df['fused_25_75'], imp_df['fused_25_75']):.4f}")
print(f"Fusion (40/60) d': {get_d_prime(gen_df['fused_40_60'], imp_df['fused_40_60']):.4f}")
print("----------------------\n")

os.makedirs(DIR_WEIGHTS, exist_ok=True)
os.makedirs(DIR_LR, exist_ok=True)

print("Crunching PyEER math for Global Weight Comparisons...")
stats_iris = get_stats(df, 'score_iris')
stats_peri = get_stats(df, 'score_peri')
stats_5050 = get_stats(df, 'fused_50_50')
stats_7525 = get_stats(df, 'fused_75_25')
stats_2575 = get_stats(df, 'fused_25_75')
stats_4060 = get_stats(df, 'fused_40_60')

print("\n--- EQUAL ERROR RATES (EER) ---")
print(f"Iris Only EER:      {stats_iris.eer:.6f}")
print(f"Peri Only EER:      {stats_peri.eer:.6f}")
print(f"50/50 Fusion EER:   {stats_5050.eer:.6f}")
print(f"75/25 Fusion EER:   {stats_7525.eer:.6f}")
print(f"25/75 Fusion EER:   {stats_2575.eer:.6f}")
print(f"40/60 Fusion EER:   {stats_4060.eer:.6f}")
print("-------------------------------\n")

print(f"Plotting Graph Set 1 (Saving all standard PyEER plots to {DIR_WEIGHTS})...")
plot_eer_stats(
    [stats_iris, stats_peri, stats_5050, stats_7525, stats_2575, stats_4060],
    ["Iris Only", "Peri Only", "50/50", "75/25 (Iris Heavy)", "25/75 (Peri Heavy)", "40/60 (VR Optimal)"],
    save_plots=True,
    save_path=DIR_WEIGHTS,
    ext=".png"
)

print("\nCrunching PyEER math for L/R/Combined (50/50 Baseline)...")
df_L = df[df['eye_side'] == 'L']
df_R = df[df['eye_side'] == 'R']

stats_L = get_stats(df_L, 'fused_50_50')
stats_R = get_stats(df_R, 'fused_50_50')
stats_Combined = get_stats(df, 'fused_50_50')

print(f"Plotting Graph Set 2 (Saving all standard PyEER plots to {DIR_LR})...")
plot_eer_stats(
    [stats_Combined, stats_L, stats_R],
    ["Combined (Both Eyes)", "Left Eye Only", "Right Eye Only"],
    save_plots=True,
    save_path=DIR_LR,
    ext=".png"
)

print("\n--- ALL FUSIONS AND PLOTS SUCCESSFULLY GENERATED ---")