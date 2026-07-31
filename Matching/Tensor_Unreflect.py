# Getting scores for 1-1 verification using templates folder 
import sys
import os
import torch
import numpy as np

# USER CONFIGURATION - SET YOUR DIRECTORIES AND SAMPLING RATES HERE
BASE_DIR = os.path.abspath(".")

# 1. INPUT TEMPLATES: Folder containing the extracted .npz feature vectors.
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates_Unreflect")

# 2. MATCHING THRESHOLDS: Define the maximum number of scores to save per eye.
# This ensures balanced pools and prevents memory overflow during evaluation.
TARGET_GEN_PER_EYE = 15000
TARGET_IMP_PER_EYE = 1500000

# 3. IMPOSTER SAMPLING RATE: Probability of keeping a calculated imposter score.
# For massive VR biometric datasets, keeping all imposters can cause RAM crashes.
SAMPLE_RATE = 0.002 

left_vectors, left_names, left_subjs = [], [], []
right_vectors, right_names, right_subjs = [], [], []

print("Gathering .npz templates and pre-slicing names...")
for root, dirs, files in os.walk(TEMPLATES_DIR):
    for filename in files:
        if not filename.endswith(".npz"):
            continue
            
        filepath = os.path.join(root, filename)
        vector = np.load(filepath)['arr_0']
        
        parts = filename.split('_')
        subj_id = parts[1]
        eye_side = parts[2][0]
        
        if eye_side == 'L':
            left_vectors.append(vector)
            left_names.append(filename)
            left_subjs.append(subj_id)
        elif eye_side == 'R':
            right_vectors.append(vector)
            right_names.append(filename)
            right_subjs.append(subj_id)

print(f"Loaded {len(left_vectors)} Left templates and {len(right_vectors)} Right templates.")
print("Moving Data to the GPU...")

X_left = torch.tensor(np.array(left_vectors)).cuda().squeeze()
X_left = torch.nn.functional.normalize(X_left, p=2, dim=1)

X_right = torch.tensor(np.array(right_vectors)).cuda().squeeze()
X_right = torch.nn.functional.normalize(X_right, p=2, dim=1)

def process_arena(arena_name, X_matrix, arena_subjs):
    if X_matrix is None or len(X_matrix) == 0:
        return np.array([]), np.array([])
        
    print(f"\n--- OPENING {arena_name} ---")
    N = X_matrix.shape[0] 
    
    all_genuines = []
    sampled_imposters = []
    
    subjs_arr = np.array(arena_subjs)
    
    for i in range(N - 1):
        query_vector = X_matrix[i:i+1] 
        gallery_matrix = X_matrix[i+1:] 
        
        similarities = torch.mm(query_vector, gallery_matrix.t())
        scores = (similarities + 1.0) / 2.0 
        
        subj1 = subjs_arr[i]
        gallery_subjs = subjs_arr[i+1:]
        is_genuine = torch.tensor((gallery_subjs == subj1), dtype=torch.bool, device='cuda')
        
        gen_scores = scores[0, is_genuine]
        all_genuines.extend(gen_scores.cpu().tolist())
        
        imp_scores = scores[0, ~is_genuine]
        keep_mask = torch.rand(imp_scores.shape, device='cuda') < SAMPLE_RATE
        sampled_imposters.extend(imp_scores[keep_mask].cpu().tolist())
        
        if i % 1000 == 0:
            print(f"Processed {i} / {N} vectors... (Gen Pool: {len(all_genuines)}, Imp Pool: {len(sampled_imposters)})")
            
    print(f"\nShuffling and trimming {arena_name} pools...")
    all_genuines = np.array(all_genuines)
    sampled_imposters = np.array(sampled_imposters)
    
    np.random.shuffle(all_genuines)
    np.random.shuffle(sampled_imposters)
    
    actual_gen_target = min(len(all_genuines), TARGET_GEN_PER_EYE)
    actual_imp_target = min(len(sampled_imposters), TARGET_IMP_PER_EYE)
    
    final_gen = all_genuines[:actual_gen_target]
    final_imp = sampled_imposters[:actual_imp_target]
    
    print(f"{arena_name} Complete -> Genuine: {len(final_gen)}, Imposter: {len(final_imp)}")
    return final_gen, final_imp

gen_L, imp_L = process_arena("LEFT ARENA", X_left, left_subjs)
if len(gen_L) > 0:
    np.savetxt("gen_L_Unreflect_balanced.txt", gen_L, fmt='%.6f')
    np.savetxt("imp_L_Unreflect_balanced.txt", imp_L, fmt='%.6f')

gen_R, imp_R = process_arena("RIGHT ARENA", X_right, right_subjs)
if len(gen_R) > 0:
    np.savetxt("gen_R_Unreflect_balanced.txt", gen_R, fmt='%.6f')
    np.savetxt("imp_R_Unreflect_balanced.txt", imp_R, fmt='%.6f')

print("\n--- Generating Global Combined Files (30K / 3M) ---")
gen_global = np.concatenate((gen_L, gen_R))
imp_global = np.concatenate((imp_L, imp_R))

np.savetxt("genuine_scores_Unreflect.txt", gen_global, fmt='%.6f')
np.savetxt("imposter_scores_Unreflect.txt", imp_global, fmt='%.6f')

print(f"Global Genuine count: {len(gen_global)}")
print(f"Global Imposter count: {len(imp_global)}")
print("\n--- Zero-Disk Extraction Complete! Ready for PyEER. ---")