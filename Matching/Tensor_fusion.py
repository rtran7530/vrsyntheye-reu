# Getting scores for 1-1 verification using templates folder (fusion)
import os
import torch
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(".")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates_fusion")
CSV_PATH = "pairs_with_similarity_scores_normalized.csv"
OUTPUT_CSV_PATH = "master_fused_scores.csv"

print(f"Loading Peer's CSV into RAM from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

iris_scores = []
fte_count = 0

print("Calculating Iris scores for exact CSV pairs...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for index, row in df.iterrows():
    probe_png = row['probe']
    ref_png = row['reference']
    
    probe_npz = os.path.splitext(probe_png)[0] + ".npz"
    ref_npz = os.path.splitext(ref_png)[0] + ".npz"
    
    probe_path = os.path.join(TEMPLATES_DIR, probe_npz)
    ref_path = os.path.join(TEMPLATES_DIR, ref_npz)
    
    if not os.path.exists(probe_path) or not os.path.exists(ref_path):
        fte_count += 1
        iris_scores.append(0.0)
        continue 
        
    v1 = np.load(probe_path)['arr_0']
    v2 = np.load(ref_path)['arr_0']
    
    t1 = torch.tensor(v1).unsqueeze(0).float().to(device)
    t2 = torch.tensor(v2).unsqueeze(0).float().to(device)
    
    t1 = torch.nn.functional.normalize(t1, p=2, dim=1)
    t2 = torch.nn.functional.normalize(t2, p=2, dim=1)
    
    sim = torch.mm(t1, t2.t()).item()
    final_score = (sim + 1.0) / 2.0 
    
    iris_scores.append(final_score)
    
    if index % 5000 == 0 and index > 0:
        print(f"Processed {index} pairs...")

print(f"\nMatching Complete! Missing/FTE pairs penalized with 0.0: {fte_count}")

print("Appending new column and saving master file...")

df['score_iris'] = iris_scores

if 'score' in df.columns:
    df.rename(columns={'score': 'score_peri'}, inplace=True)

df.to_csv(OUTPUT_CSV_PATH, index=False)

print("\n--- EXTRACTION COMPLETE ---")
print(f"Master file saved to: {OUTPUT_CSV_PATH}")
print(f"Total pairs processed: {len(df)}")