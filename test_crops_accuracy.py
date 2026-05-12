import os
import csv
import cv2
import pickle
import numpy as np
import sys

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
sys.path.insert(0, BASE)

from retrain_classifier import extract_features

CSV_PATH = os.path.join(BASE, "new_data", "reports6", "reports", "inspection_log.csv")
CROPS_DIR = os.path.join(BASE, "new_data", "detected_images6", "detected_images", "crops")
CLF_PATH = os.path.join(BASE, "app_v2", "models", "gb_classifier.pkl")

def main():
    print("Loading GB Classifier...")
    with open(CLF_PATH, "rb") as f:
        clf = pickle.load(f)
        
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    skipped = 0
    valid = 0
    not_found = 0
    
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id"):
                continue
                
            manual = row.get("manual_label", "").strip()
            crop_path_str = row.get("yolo_crop_path", "").strip()
            
            if manual in ["Pending", "-", "—", ""]:
                skipped += 1
                continue
                
            if not crop_path_str or crop_path_str == "—":
                not_found += 1
                continue
                
            filename = os.path.basename(crop_path_str.replace("\\", "/"))
            full_crop_path = os.path.join(CROPS_DIR, filename)
            
            if not os.path.exists(full_crop_path):
                not_found += 1
                continue
                
            crop = cv2.imread(full_crop_path)
            if crop is None:
                not_found += 1
                continue
                
            # Extract features and predict
            feats = extract_features(crop).reshape(1, -1)
            pred = clf.predict(feats)[0]
            
            # 1 == OIL LEAK, 0 == NO LEAK
            pred_label = "OIL LEAK" if pred == 1 else "NO LEAK"
            manual_label = "OIL LEAK" if manual == "Engine Oil" else "NO LEAK"
            
            valid += 1
            
            if pred_label == "OIL LEAK" and manual_label == "OIL LEAK":
                TP += 1
            elif pred_label == "NO LEAK" and manual_label == "NO LEAK":
                TN += 1
            elif pred_label == "OIL LEAK" and manual_label == "NO LEAK":
                FP += 1
            elif pred_label == "NO LEAK" and manual_label == "OIL LEAK":
                FN += 1
                
    accuracy = (TP + TN) / valid * 100 if valid > 0 else 0
    
    print("\n========================================")
    print(" CROP-ONLY ACCURACY EVALUATION")
    print("========================================")
    print(f"Total Valid Images Evaluated: {valid}")
    print(f"Skipped (Pending): {skipped}")
    print(f"Crops Not Found: {not_found}")
    print("\n--- CONFUSION MATRIX ---")
    print(f"TP (Correctly identified Oil Leak): {TP}")
    print(f"TN (Correctly identified No Leak) : {TN}")
    print(f"FP (False Alarm: No Leak -> Oil)  : {FP}")
    print(f"FN (Missed Leak: Oil -> No Leak)  : {FN}")
    print(f"\nOVERALL ACCURACY: {accuracy:.2f}%")
    print("========================================")

if __name__ == "__main__":
    main()
