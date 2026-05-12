import os
import csv
import cv2
import pickle
from ultralytics import YOLO

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

    updated_rows = []
    headers = []

    print("Updating reports_4 CSV with new model predictions...")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            if not row.get("id"):
                continue
            
            auto_res = row["auto_result"]
            image_path_str = row.get("image_path", "")
            
            if auto_res == "NO_PART" or image_path_str == "—" or not image_path_str:
                updated_rows.append(row)
                continue
                
            filename = os.path.basename(image_path_str)
            full_path = os.path.join(IMAGES_DIR, filename)
            
            if os.path.exists(full_path):
                # Predict with new model
                crop, method = get_crop(full_path, model)
                if crop is not None:
                    feats = extract_features(crop).reshape(1, -1)
                    pred = clf.predict(feats)[0]
                    proba = clf.predict_proba(feats)[0]
                    conf = max(proba) * 100
                    
                    pred_label = "OIL LEAK" if pred == 1 else "NO LEAK"
                    
                    # Update row with new prediction and confidence
                    row["auto_result"] = pred_label
                    row["confidence"] = f"{conf:.1f}%"
            
            updated_rows.append(row)

    # Write back the updated CSV
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Updated CSV with new predictions successfully.")

if __name__ == "__main__":
    main()
