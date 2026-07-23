# Performing Off-angular correction on available dataset
import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.abspath(".")
INPUT_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "OAC_data")

CVRL_PYTHON_DIR = os.path.join(BASE_DIR, "OpenSourceIrisRecognition", "methods", "iris-fm-tools", "Python")
CVRL_SCRIPTS_DIR = os.path.join(CVRL_PYTHON_DIR, "scripts")

sys.path.append(CVRL_PYTHON_DIR)
sys.path.append(CVRL_SCRIPTS_DIR)

from inference import load_dino, load_model, predict_single, scale_to_original

DINO_REPO = os.path.join(CVRL_PYTHON_DIR, "modules", "dinov3")
DINO_WEIGHTS = os.path.join(CVRL_PYTHON_DIR, "models", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
H8NET_WEIGHTS = os.path.join(CVRL_PYTHON_DIR, "models", "h8net.pth")

print("Waking up DINOv3 and H8Net...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dino_model = load_dino(DINO_REPO, DINO_WEIGHTS, device)
h8_model, h8_info = load_model(H8NET_WEIGHTS, device)

print(f"AI loaded successfully on {device}! Starting Conveyor Belt...")

for root, dirs, files in os.walk(INPUT_DIR):
    for filename in files:
        if not filename.endswith(".png"):
            continue
           
        raw_image_path = os.path.join(root, filename)
       
        relative_path = os.path.relpath(root, INPUT_DIR)
        target_folder = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(target_folder, exist_ok=True)
        final_image_path = os.path.join(target_folder, filename)

        img_cv2 = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = img_cv2.shape[:2]

        img_pil = Image.open(raw_image_path).convert("RGB")

        params_train = predict_single(img_pil, h8_model, dino_model, h8_info, device)
        params_orig = scale_to_original(params_train, h8_info, orig_w, orig_h)

        h11, h12, h13, h21, h22, h23, h31, h32 = params_orig

        homography_matrix = np.array([
            [h11, h12, h13],
            [h21, h22, h23],
            [h31, h32, 1.0]
        ], dtype=np.float32)
       
        dewarped_img = cv2.warpPerspective(img_cv2, homography_matrix, (orig_w, orig_h))

        cv2.imwrite(final_image_path, dewarped_img)
        print(f"Flattened and Saved: {filename}")

print("\n--- Geometric Correction Complete! ---")