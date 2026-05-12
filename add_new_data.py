import os
import csv
import shutil

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
CSV_PATH = os.path.join(BASE, "new_data", "reports 3", "reports", "inspection_log.csv")
IMAGES_DIR = os.path.join(BASE, "new_data", "detected_images")
IMAGES_DIR_3 = os.path.join(BASE, "new_data", "detected_images 3", "detected_images")

SUMMARY_OIL = os.path.join(BASE, "..", "summary", "data", "engine_oil")
SUMMARY_NO_LEAK = os.path.join(BASE, "..", "summary", "data", "no_leak")

def main():
    vin_labels = {}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row["manual_label"].strip()
            if lbl not in ["Pending", "—", ""]:
                vin_labels[row["vin_id"]] = lbl

    count_copied = 0
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["vin_id"]
            manual_label = row["manual_label"].strip()
            
            if manual_label in ["Pending", "—", ""]:
                if vid in vin_labels:
                    manual_label = vin_labels[vid]
                else:
                    continue

            # Determine destination folder
            if manual_label == "Engine Oil":
                dest_dir = SUMMARY_OIL
            elif manual_label == "No Leak":
                dest_dir = SUMMARY_NO_LEAK
            else:
                dest_dir = None
            
            if dest_dir:
                # The full frame path from CSV might have a different directory, so extract basename
                filename = os.path.basename(row["full_frame_path"])
                src_path = os.path.join(IMAGES_DIR, filename)
                if not os.path.exists(src_path):
                    src_path = os.path.join(IMAGES_DIR_3, filename)

                dest_path = os.path.join(dest_dir, filename)
                
                if os.path.exists(src_path):
                    if not os.path.exists(dest_path):
                        shutil.copy2(src_path, dest_path)
                        count_copied += 1
                        print(f"Copied {filename} to {os.path.basename(dest_dir)}")
                    else:
                        print(f"Already exists: {filename}")
                else:
                    print(f"Source file not found: {src_path}")

    print(f"Copied {count_copied} new images for retraining.")

if __name__ == "__main__":
    main()
