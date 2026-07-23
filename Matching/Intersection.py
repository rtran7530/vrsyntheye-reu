# Code for intersecting the same frames for all corrections (or no correction) before matching
import os
import shutil


VALID_OAC_DIR = os.path.abspath("Valid_Iris_Data") 
RAW_DATA_DIR = os.path.abspath("processed_data") 

OUTPUT_DIR = os.path.abspath("Valid_Iris_Data_nooac")

print(f"Scanning '{VALID_OAC_DIR}' for valid file names...")

copy_count = 0
missing_count = 0

for root, dirs, files in os.walk(VALID_OAC_DIR):
    for filename in files:
        # Skip hidden files or system files
        if not filename.endswith((".png", ".jpg")): 
            continue

        relative_path = os.path.relpath(root, VALID_OAC_DIR)
        
        raw_image_path = os.path.join(RAW_DATA_DIR, relative_path, filename)
        
        target_folder = os.path.join(OUTPUT_DIR, relative_path)
        final_image_path = os.path.join(target_folder, filename)
        
        if os.path.exists(raw_image_path):
            os.makedirs(target_folder, exist_ok=True)
            shutil.copy2(raw_image_path, final_image_path) 
            copy_count += 1
        else:
            print(f"Warning: Could not find raw equivalent for {filename} in {RAW_DATA_DIR}")
            missing_count += 1

print("\n--- Synchronization Complete! ---")
print(f"Successfully copied: {copy_count} raw images into '{OUTPUT_DIR}'")
if missing_count > 0:
    print(f"Failed to find: {missing_count} images.")