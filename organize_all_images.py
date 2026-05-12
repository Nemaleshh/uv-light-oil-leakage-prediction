import os
import csv
import shutil

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
DEST_DIR = os.path.join(BASE, "all_inspected_images")

OIL_DIR = os.path.join(DEST_DIR, "OIL_LEAK")
NO_LEAK_DIR = os.path.join(DEST_DIR, "NO_LEAK")

CSV_PATHS = [
    os.path.join(BASE, "app_v2", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports 2", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports 3", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports_4", "reports", "inspection_log.csv")
]

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
    if not os.path.exists(OIL_DIR):
        os.makedirs(OIL_DIR)
    if not os.path.exists(NO_LEAK_DIR):
        os.makedirs(NO_LEAK_DIR)

    # Resolve pending labels
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

    oil_count = 0
    noleak_count = 0
    seen_files = set()

    for path in CSV_PATHS:
        if not os.path.exists(path):
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path_str = row.get("full_frame_path", row.get("image_path", ""))
                auto_res = row.get("auto_result", "").strip()
                vin = row.get("vin_id", "").strip()
                
                # Try to use manual label
                label = row.get("manual_label", row.get("manual_confirm", "")).strip()
                if label in ["Pending", "—", ""]:
                    label = vin_manual_labels.get(vin, "") 
                
                # If there is no strict human label, skip entirely
                if not label or label in ["Pending", "—"]:
                    continue
                
                if not image_path_str or image_path_str == "—" or auto_res == "NO_PART":
                    continue

                filename = os.path.basename(image_path_str)
                if filename in seen_files:
                    continue
                    
                src_path = find_image(filename)
                if src_path:
                    # Decide folder
                    if label.upper() in ["ENGINE OIL", "OIL LEAK"]:
                        dest_path = os.path.join(OIL_DIR, filename)
                        oil_count += 1
                    else:
                        dest_path = os.path.join(NO_LEAK_DIR, filename)
                        noleak_count += 1
                        
                    if not os.path.exists(dest_path):
                        shutil.copy2(src_path, dest_path)
                        
                    # Clean up the old un-categorized file if it exists in the root
                    old_path = os.path.join(DEST_DIR, filename)
                    if os.path.exists(old_path) and not os.path.isdir(old_path):
                        os.remove(old_path)
                        
                    seen_files.add(filename)

    print(f"Organized images into:")
    print(f" - OIL LEAK : {oil_count} images")
    print(f" - NO LEAK  : {noleak_count} images")

if __name__ == "__main__":
    main()
