# Code for filtering out the data before matching
import sys
import os
import cv2
import shutil
import torch
import csv 
from PIL import Image

# USER CONFIGURATION - SET YOUR DIRECTORIES HERE
BASE_DIR = os.path.abspath(".")

# 1. INPUT: Folder containing your unconstrained images (e.g., OAC output)
# The script will recursively scan all subfolders automatically.
INPUT_DIR = os.path.join(BASE_DIR, "OAC_data") 

# 2. OUTPUT - VALID: Folder for images that pass the geometric quality check.
VALID_IRIS_DIR = os.path.join(BASE_DIR, "Valid_Iris_Data")

# 3. OUTPUT - REJECTED: Folder for images that fail (periocular/degraded data).
PERIOCULAR_DIR = os.path.join(BASE_DIR, "Periocular_Data")

# 4. OUTPUT - TRACKING: CSV file to cache the successful boundary coordinates.
CSV_PATH = os.path.join(BASE_DIR, "cvrl_gold_standard_coordinates.csv")

# 5. DEPENDENCY: Path to the CVRL 'ArcIris' Python directory.
# You must clone https://github.com/CVRL/OpenSourceIrisRecognition in your base directory.
ARCIRIS_DIR = os.path.join(BASE_DIR, "OpenSourceIrisRecognition", "methods", "ArcIris", "Python")

# Initialize the coordinate tracker CSV if it doesn't already exist
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "px", "py", "pr", "ix", "iy", "ir"])
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

print("Waking up ArcIris Geometric Regressor...")
irisRec = irisRecognition(cfg)
print("AI loaded successfully! Starting the Sorting Hat Conveyor Belt...")

for root, dirs, files in os.walk(INPUT_DIR):
    for filename in files:
        if not filename.endswith(".png"):
            continue
            
        raw_image_path = os.path.join(root, filename)
        
        relative_path = os.path.relpath(root, INPUT_DIR)
        valid_folder = os.path.join(VALID_IRIS_DIR, relative_path)
        periocular_folder = os.path.join(PERIOCULAR_DIR, relative_path)
        
        valid_path = os.path.join(valid_folder, filename)
        periocular_path = os.path.join(periocular_folder, filename)
        
        if os.path.exists(valid_path) or os.path.exists(periocular_path):
            continue
        
        try:
            im_pil = Image.open(raw_image_path).convert("L")
            
            im_fixed = irisRec.fix_image(im_pil)
            
            pupil_xyr, iris_xyr = irisRec.circApprox(im_fixed)
            
            is_valid = irisRec.checkQuality(pupil_xyr, iris_xyr)
            
            if is_valid:
                os.makedirs(valid_folder, exist_ok=True)
                shutil.copy2(raw_image_path, valid_path)
                
                with open(CSV_PATH, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        filename, 
                        pupil_xyr[0], pupil_xyr[1], pupil_xyr[2], 
                        iris_xyr[0], iris_xyr[1], iris_xyr[2]
                    ])
                
                print(f"[VALID] Saved & Cached: {filename}") 
            else:
                os.makedirs(periocular_folder, exist_ok=True)
                shutil.copy2(raw_image_path, periocular_path)
                print(f"[PERIOCULAR] Dropped: {filename}")
                
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

print("\n--- Biometric Sorting Complete! ---")