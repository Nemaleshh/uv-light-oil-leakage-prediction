import os
import csv
import cv2
import pickle
from ultralytics import YOLO
import numpy as np

# Import functions from retrain_classifier
from retrain_classifier import extract_features, get_crop

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
CSV_PATH = os.path.join(BASE, "new_data", "reports_4", "reports", "inspection_log.csv")
IMAGES_DIR = os.path.join(BASE, "new_data", "detected_images_4", "detected_images")
YOLO_PATH = os.path.join(BASE, "app_v2", "models", "best.pt")
CLF_PATH = os.path.join(BASE, "app_v2", "models", "gb_classifier.pkl")

def main():
    print("Loading YOLO model...")
    model = YOLO(YOLO_PATH)
    
    print("Loading new GB Classifier...")
    with open(CLF_PATH, "rb") as f:
        clf = pickle.load(f)

    old_oil = 0
    new_oil = 0
    old_noleak = 0
    new_noleak = 0
    
    changed_to_noleak = 0
    changed_to_oilleak = 0

    print("\nRunning new model on pending images in reports_4...")
    
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id"):
                continue
            
            auto_res = row["auto_result"]
            image_path_str = row["image_path"]
            
            if auto_res == "NO_PART" or image_path_str == "—" or not image_path_str:
                continue
                
            filename = os.path.basename(image_path_str)
            full_path = os.path.join(IMAGES_DIR, filename)
            
            if not os.path.exists(full_path):
                continue
                
            if auto_res == "OIL LEAK":
                old_oil += 1
            elif auto_res == "NO LEAK":
                old_noleak += 1
                
            # Predict with new model
            crop, method = get_crop(full_path, model)
            if crop is not None:
                feats = extract_features(crop).reshape(1, -1)
                pred = clf.predict(feats)[0]
                
                pred_label = "OIL LEAK" if pred == 1 else "NO LEAK"
                
                if pred_label == "OIL LEAK":
                    new_oil += 1
                else:
                    new_noleak += 1
                    
                if auto_res == "OIL LEAK" and pred_label == "NO LEAK":
                    changed_to_noleak += 1
                elif auto_res == "NO LEAK" and pred_label == "OIL LEAK":
                    changed_to_oilleak += 1

    print("\n--- NEW MODEL PERFORMANCE ON REPORTS 4 PENDING IMAGES ---")
    print(f"Total evaluated images: {old_oil + old_noleak}")
    print(f"\nOLD Auto Result counts:")
    print(f"  OIL LEAK : {old_oil}")
    print(f"  NO LEAK  : {old_noleak}")
    print(f"\nNEW Model Predictions:")
    print(f"  OIL LEAK : {new_oil}")
    print(f"  NO LEAK  : {new_noleak}")
    
    print(f"\nChanges:")
    print(f"  Old 'OIL LEAK' corrected to 'NO LEAK' (False Positives fixed): {changed_to_noleak}")
    print(f"  Old 'NO LEAK' changed to 'OIL LEAK': {changed_to_oilleak}")

if __name__ == "__main__":
    main()
