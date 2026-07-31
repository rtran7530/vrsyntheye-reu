# Code for extracting frames
import os
import cv2

# USER CONFIGURATION - SET YOUR DIRECTORIES AND PARAMETERS HERE
# 1. INPUT: Directory containing the raw .avi video files, organized by subject.
INPUT_DIR = os.path.abspath("data")

# 2. OUTPUT: Directory where the extracted grayscale .png frames will be saved.
# The original subject and video folder hierarchy will be preserved.
OUTPUT_DIR = os.path.abspath("processed_data")

# 3. FRAME SKIPPING: Number of initial frames to discard per video.
# Useful for bypassing headset initialization or sensor stabilization periods.
SKIP_FRAMES = 75

print("Starting Frame Extraction (Cloning Original Directory Structure)...")

for root, dirs, files in os.walk(INPUT_DIR):
    for filename in files:

        if not filename.endswith(".avi"):
            continue
            
        video_path = os.path.join(root, filename)

        subject_id = os.path.basename(root)

        video_name = os.path.splitext(filename)[0]
 
        target_folder = os.path.join(OUTPUT_DIR, subject_id, video_name)

        os.makedirs(target_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break 
                
            frame_count += 1

            if frame_count <= SKIP_FRAMES:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_filename = f"{video_name}_F{frame_count:04d}.png"
            frame_path = os.path.join(target_folder, frame_filename)

            cv2.imwrite(frame_path, gray)
            
        cap.release()
        print(f"Extracted: {filename} -> {target_folder}")

print("\n--- Extraction Complete ---")