import os
import csv
import shutil

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
DEST_DIR = os.path.join(BASE, "verified_oil_leaks")

CSV_PATHS = [
    os.path.join(BASE, "app_v2", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports 2", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports 3", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports_4", "reports", "inspection_log.csv")
]

# Possible directories where images might be stored
IMAGE_DIRS = [
    os.path.join(BASE, "new_data", "detected_images"),
    os.path.join(BASE, "new_data", "detected_images 3", "detected_images"),
    os.path.join(BASE, "new_data", "detected_images_4", "detected_images"),
    os.path.join(BASE, "app_v2", "detected_images")
]

def find_image(filename):
    for d in IMAGE_DIRS:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

def main():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    # First pass: Build a dictionary of VIN -> verified manual label
    vin_manual_labels = {}
    for path in CSV_PATHS:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vin = row.get("vin_id", "").strip()
                label = row.get("manual_label", row.get("manual_confirm", "")).strip()
                if label and label not in ["Pending", "—", ""]:
                    vin_manual_labels[vin] = label

    # Second pass: Extract images where manual label indicates an oil leak
    copied_count = 0
    seen_files = set()

    for path in CSV_PATHS:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vin = row.get("vin_id", "").strip()
                
                # Get label, resolving pending ones using the dictionary
                label = row.get("manual_label", row.get("manual_confirm", "")).strip()
                if label in ["Pending", "—", ""]:
                    label = vin_manual_labels.get(vin, "")

                # Check if the manual label is verified as an oil leak
                if label.upper() in ["ENGINE OIL", "OIL LEAK"]:
                    # Try to get full frame or image path
                    image_path_str = row.get("full_frame_path", row.get("image_path", ""))
                    if not image_path_str or image_path_str == "—":
                        continue

                    filename = os.path.basename(image_path_str)
                    
                    if filename in seen_files:
                        continue # Skip duplicates
                        
                    src_path = find_image(filename)
                    if src_path:
                        dest_path = os.path.join(DEST_DIR, filename)
                        if not os.path.exists(dest_path):
                            shutil.copy2(src_path, dest_path)
                        copied_count += 1
                        seen_files.add(filename)
                        print(f"Copied: {filename}")
                    else:
                        print(f"File not found on disk: {filename}")

    print("=" * 50)
    print(f"Successfully extracted {copied_count} verified oil leak images to: {DEST_DIR}")

if __name__ == "__main__":
    main()
