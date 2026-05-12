import os
import cv2
import csv
import pickle
import numpy as np

# Import from retrain_classifier
from retrain_classifier import extract_features

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
CSV_PATH = os.path.join(BASE, "new_data", "reports 3", "reports", "inspection_log.csv")
CROPS_DIR = os.path.join(BASE, "new_data", "detected_images", "crops")
CROPS_DIR_3 = os.path.join(BASE, "new_data", "detected_images 3", "detected_images", "crops")
CLF_PATH = os.path.join(BASE, "app_v2", "models", "gb_classifier.pkl")

def main():
    print("Loading new classifier model...")
    with open(CLF_PATH, "rb") as f:
        clf = pickle.load(f)

    results = []
    correct_count = 0
    total_count = 0

    vin_labels = {}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row["manual_label"].strip()
            if lbl not in ["Pending", "—", ""]:
                vin_labels[row["vin_id"]] = lbl

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["vin_id"]
            # Ground truth based on user logic:
            manual_label = row["manual_label"].strip()
            if manual_label in ["Pending", "—", ""]:
                if vid in vin_labels:
                    manual_label = vin_labels[vid]
                else:
                    continue

            if manual_label == "Engine Oil":
                ground_truth = "OIL LEAK"
            elif manual_label == "No Leak":
                ground_truth = "NO LEAK"
            else:
                ground_truth = manual_label.upper()

            yolo_crop_path = row["yolo_crop_path"]
            filename = os.path.basename(yolo_crop_path)
            
            # The actual path to the crop in our directory
            crop_file = os.path.join(CROPS_DIR, filename)
            if not os.path.exists(crop_file):
                crop_file = os.path.join(CROPS_DIR_3, filename)

            if not os.path.exists(crop_file):
                print(f"File not found: {crop_file}")
                continue
                
            crop = cv2.imread(crop_file)
            if crop is None:
                print(f"Could not read image: {crop_file}")
                continue

            feats = extract_features(crop).reshape(1, -1)
            pred = clf.predict(feats)[0]
            proba = clf.predict_proba(feats)[0]
            conf = max(proba) * 100
            
            pred_label = "OIL LEAK" if pred == 1 else "NO LEAK"

            total_count += 1
            is_correct = (pred_label == ground_truth)
            if is_correct:
                correct_count += 1
                
            print(f"ID: {row['id']} | VIN: {vid}")
            print(f"  Old Auto Result: {row['auto_result']}")
            print(f"  Ground Truth   : {ground_truth}")
            print(f"  NEW PREDICTION : {pred_label} ({conf:.1f}%)")
            print(f"  Correct?       : {'YES' if is_correct else 'NO'}\n")

    print(f"Summary: {correct_count}/{total_count} correct ({(correct_count/total_count)*100:.1f}%)")

if __name__ == "__main__":
    main()
