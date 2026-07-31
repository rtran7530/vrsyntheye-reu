# Getting templates (.npz files) for 1-1 verification (fusion)
import sys
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

# USER CONFIGURATION - SET YOUR DIRECTORIES HERE
BASE_DIR = os.path.abspath(".")

# 1. INPUT CSV: A CSV file containing the specific probe and reference pairs you need.
# The script will parse this file and extract a unique list of images to process.
CSV_PATH = "pairs_with_similarity_scores_normalized.csv"

# 2. INPUT DATA: Folder containing your raw or processed images referenced in the CSV.
RAW_DATA_DIR = os.path.join(BASE_DIR, "processed_data") 

# 3. OUTPUT: Folder where the targeted feature templates (.npz) will be saved.
# The original nested directory structure will be preserved.
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates_fusion")

# 4. DEPENDENCY: Path to the CVRL 'ArcIris' Python directory.
# You must clone https://github.com/CVRL/OpenSourceIrisRecognition in your base directory.
ARCIRIS_DIR = os.path.join(BASE_DIR, "OpenSourceIrisRecognition", "methods", "ArcIris", "Python")
sys.path.append(ARCIRIS_DIR)

from modules.irisRecognition import irisRecognition

cfg = {
    "cuda": True,
    "polar_height": 64,
    "polar_width": 512,
    "circle_model_path": os.path.join(ARCIRIS_DIR, "models", "resnet18-1543-0.047488-maskIoU-0.934494.pth"),
    "nn_model_path": os.path.join(ARCIRIS_DIR, "models", "ResNet100_154000.pt"),
    "minimum_pupil_radius": 12,
    "minimum_iris_radius": 16
}

print("Reading Peer's CSV to find required images...")
df = pd.read_csv(CSV_PATH)
unique_images = pd.concat([df['probe'], df['reference']]).unique()
print(f"Found {len(unique_images)} unique images required for this evaluation.")

print("Waking up ArcIris Template Generator...")
irisRec = irisRecognition(cfg)
print("AI loaded successfully! Starting the Targeted Assembly Line...")

for relative_img_path in unique_images:
    raw_image_path = os.path.join(RAW_DATA_DIR, relative_img_path)
    
    relative_dir = os.path.dirname(relative_img_path)
    filename = os.path.basename(relative_img_path)
    
    template_folder = os.path.join(TEMPLATES_DIR, relative_dir)
    os.makedirs(template_folder, exist_ok=True)
    
    template_filename = os.path.splitext(filename)[0] + ".npz"
    template_save_path = os.path.join(template_folder, template_filename)

    if os.path.exists(template_save_path):
        continue

    if not os.path.exists(raw_image_path):
        print(f"[WARNING] Raw image not found on drive: {raw_image_path}")
        continue

    try:
        im_pil = Image.open(raw_image_path).convert("L")
        im_fixed = irisRec.fix_image(im_pil)
        pupil_xyr, iris_xyr = irisRec.circApprox(im_fixed)

        if irisRec.checkQuality(pupil_xyr, iris_xyr):
            im_polar = irisRec.cartToPol_torch(im_fixed, pupil_xyr, iris_xyr)
            vector = irisRec.extractVector(im_polar)
            np.savez_compressed(template_save_path, vector)
            print(f"[TEMPLATE GENERATED] Saved: {template_filename}")
        else:
            print(f"[QUALITY FAILED] ArcIris rejected: {filename}")

    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {e}")

print("\n--- Targeted Template Manufacturing Complete! ---")