import os
import shutil
import csv

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
CSV_PATH = os.path.join(BASE, "new_data", "reports", "inspection_log.csv")
CROPS_DIR = os.path.join(BASE, "new_data", "detected_images", "crops")

OLD_OIL_CROPS = os.path.join(BASE, "new_pipeline", "crops", "oil_leak")
OLD_NOLEAK_CROPS = os.path.join(BASE, "new_pipeline", "crops", "no_leak")

def main():
    print("Copying new hard examples to training dataset...")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = os.path.basename(row["yolo_crop_path"])
            src = os.path.join(CROPS_DIR, filename)
            
            if not os.path.exists(src):
                print(f"Skipping missing file: {filename}")
                continue
                
            if row["id"] == "1" or row["id"] == "8":
                dst = os.path.join(OLD_OIL_CROPS, "NEW_" + filename)
                print(f"Adding OIL LEAK: {filename}")
            else:
                dst = os.path.join(OLD_NOLEAK_CROPS, "NEW_" + filename)
                print(f"Adding NO LEAK (False Positive Fix): {filename}")
                
            shutil.copy2(src, dst)
            
    print("Done copying. Ready to retrain.")

if __name__ == "__main__":
    main()
