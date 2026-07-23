# Getting mean and standard deviation of evaluated .json files
import os
import json
import numpy as np

BASE_DIR = os.path.abspath(".")
JSON_DIR = os.path.join(BASE_DIR, "processed_json_data") 

metrics_tracker = {
    "USABLE_IRIS_AREA": [],
    "IRIS_SCLERA_CONTRAST": [],
    "IRIS_PUPIL_CONTRAST": [],
    "PUPIL_BOUNDARY_CIRCULARITY": [],
    "GREY_SCALE_UTILISATION": [],
    "IRIS_RADIUS": [],
    "PUPIL_IRIS_RATIO": [],
    "IRIS_PUPIL_CONCENTRICITY": [],
    "MARGIN_ADEQUACY": [],
    "SHARPNESS": [],
    "OVERALL_QUALITY": []
}

processed_count = 0

print(f"Starting targeted metric extraction from {JSON_DIR}...")

for root, dirs, files in os.walk(JSON_DIR):
    for filename in files:
        if not filename.endswith(".json"):
            continue
        
        json_path = os.path.join(root, filename)

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if not data or not isinstance(data, dict):
                continue
            
            first_key = list(data.keys())[0]
            image_data = data.get(first_key, {})
            biqt_list = image_data.get("BIQTIris", [])
            
            if not biqt_list or not isinstance(biqt_list, list):
                continue
                
            eval_data = biqt_list[0]
            features = eval_data.get("features", {})
            metrics = eval_data.get("metrics", {})

            metrics_tracker["USABLE_IRIS_AREA"].append(metrics.get("iso_usable_iris_area", 0))
            metrics_tracker["IRIS_SCLERA_CONTRAST"].append(metrics.get("iso_iris_sclera_contrast", 0))
            metrics_tracker["IRIS_PUPIL_CONTRAST"].append(metrics.get("iso_iris_pupil_contrast", 0))
            metrics_tracker["PUPIL_BOUNDARY_CIRCULARITY"].append(metrics.get("iso_pupil_boundary_circularity", 0))
            metrics_tracker["GREY_SCALE_UTILISATION"].append(metrics.get("iso_greyscale_utilization", 0))
            metrics_tracker["IRIS_RADIUS"].append(features.get("iris_diameter", 0) / 2.0)
            metrics_tracker["PUPIL_IRIS_RATIO"].append(metrics.get("iso_iris_pupil_ratio", 0))
            metrics_tracker["IRIS_PUPIL_CONCENTRICITY"].append(metrics.get("iso_iris_pupil_concentricity", 0))
            metrics_tracker["MARGIN_ADEQUACY"].append(metrics.get("iso_margin_adequacy", 0))
            metrics_tracker["SHARPNESS"].append(metrics.get("iso_sharpness", metrics.get("sharpness", 0)))
            metrics_tracker["OVERALL_QUALITY"].append(metrics.get("iso_overall_quality", 0))

            processed_count += 1
            if processed_count % 500 == 0:
                print(f"Files processed: {processed_count}...", end="\r", flush=True)

        except Exception as e:
            pass 

print(f"\n\n--- FINAL DATASET SCORES ({processed_count} files read) ---")

for metric_name, values in metrics_tracker.items():
    if values:
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name}: {mean_val:.2f} \\pm {std_val:.2f}")
    else:
        print(f"{metric_name}: No data found.")

print("\n--- Extraction Complete! ---")