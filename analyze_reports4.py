import os
import csv
from collections import defaultdict
from datetime import datetime

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
CSV_PATH = os.path.join(BASE, "new_data", "reports_4", "reports", "inspection_log.csv")
IMAGES_DIR = os.path.join(BASE, "new_data", "detected_images_4", "detected_images")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found at: {CSV_PATH}")
        return

    total_inspections = 0
    test_system_inspections = 0
    cars_by_day = defaultdict(set)
    test_by_day = defaultdict(set)
    parts_not_found = 0
    auto_results = defaultdict(int)

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id"):
                continue  # skip empty rows
            
            total_inspections += 1
            vin = row["vin_id"]
            ts = row["timestamp"]
            auto_res = row["auto_result"]
            
            # Extract day
            day = ts.split(" ")[0] if ts else "Unknown"

            if vin.startswith("TEST") or vin.startswith("MANUAL"):
                test_system_inspections += 1
                test_by_day[day].add(vin)
            else:
                cars_by_day[day].add(vin)

            if auto_res == "NO_PART":
                parts_not_found += 1
            
            auto_results[auto_res] += 1

    print(f"--- REPORTS 4 SUMMARY ---")
    print(f"Total Log Entries: {total_inspections}")
    print(f"Parts Not Found (NO_PART): {parts_not_found}")
    print(f"System Test Entries ('TEST' / 'MANUAL' VINs): {test_system_inspections}")
    print(f"Real Production Entries: {total_inspections - test_system_inspections}")
    print("\n--- CARS INSPECTED PER DAY (Unique Real VINs) ---")
    for day, vins in sorted(cars_by_day.items()):
        print(f"  {day}: {len(vins)} unique cars")
        
    print("\n--- SYSTEM TESTED PER DAY (Unique Test VINs) ---")
    for day, vins in sorted(test_by_day.items()):
        print(f"  {day}: {len(vins)} unique system tests")

    print("\n--- AUTO RESULT DISTRIBUTION ---")
    for res, count in auto_results.items():
        print(f"  {res}: {count}")

    if os.path.exists(IMAGES_DIR):
        img_count = len([f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")])
        print(f"\nTotal Detected Images in 'detected_images_4/detected_images': {img_count}")

if __name__ == "__main__":
    main()
