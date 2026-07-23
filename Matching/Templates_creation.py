# Getting templates (.npz files) for 1-1 verification
import sys
import os
import torch
import numpy as np
from PIL import Image

BASE_DIR = os.path.abspath(".")
VALID_IRIS_DIR = os.path.join(BASE_DIR, "Valid_Iris_Data")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

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

print("Waking up ArcIris Template Generator...")
irisRec = irisRecognition(cfg)
print("AI loaded successfully! Starting the Assembly Line...")

for root, dirs, files in os.walk(VALID_IRIS_DIR):
    for filename in files:
        if not filename.endswith(".png"):
            continue
            
        raw_image_path = os.path.join(root, filename)
        
        relative_path = os.path.relpath(root, VALID_IRIS_DIR)
        template_folder = os.path.join(TEMPLATES_DIR, relative_path)
        os.makedirs(template_folder, exist_ok=True)
        
        template_filename = os.path.splitext(filename)[0] + ".npz"
        template_save_path = os.path.join(template_folder, template_filename)
        
        if os.path.exists(template_save_path):
            continue
            
        try:
            im_pil = Image.open(raw_image_path).convert("L")
            im_fixed = irisRec.fix_image(im_pil)
            
            pupil_xyr, iris_xyr = irisRec.circApprox(im_fixed)
            
            im_polar = irisRec.cartToPol_torch(im_fixed, pupil_xyr, iris_xyr)
            
            vector = irisRec.extractVector(im_polar)
            
            np.savez_compressed(template_save_path, vector)
            print(f"[TEMPLATE GENERATED] Saved: {template_filename} in {relative_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

print("\n--- Template Manufacturing Complete! ---")