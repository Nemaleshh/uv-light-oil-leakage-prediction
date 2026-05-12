import os
import csv
import shutil

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
DEST_DIR = os.path.join(BASE, "all_inspected_images")

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
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    copied_count = 0
    seen_files = set()

    print(f"Gathering all images from all reports...")
    
    for path in CSV_PATHS:
        if not os.path.exists(path):
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path_str = row.get("full_frame_path", row.get("image_path", ""))
                
                # Skip if no image path or part wasn't found
                if not image_path_str or image_path_str == "—" or row.get("auto_result") == "NO_PART":
                    continue

                filename = os.path.basename(image_path_str)
                
                if filename in seen_files:
                    continue
                    
                src_path = find_image(filename)
                if src_path:
                    dest_path = os.path.join(DEST_DIR, filename)
                    if not os.path.exists(dest_path):
                        shutil.copy2(src_path, dest_path)
                    copied_count += 1
                    seen_files.add(filename)

    print("=" * 50)
    print(f"Successfully aggregated {copied_count} total images to: {DEST_DIR}")

if __name__ == "__main__":
    main()
