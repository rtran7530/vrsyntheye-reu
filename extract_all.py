# Code for extracting frames
import os
import cv2

INPUT_DIR = "data"
OUTPUT_DIR = "processed_data"

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

            if frame_count <= 75:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_filename = f"{video_name}_F{frame_count:04d}.png"
            frame_path = os.path.join(target_folder, frame_filename)

            cv2.imwrite(frame_path, gray)
            
        cap.release()
        print(f"Extracted: {filename} -> {target_folder}")

print("\n--- Extraction Complete ---")