import csv
import os
from collections import defaultdict

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
CSV_PATHS = [
    os.path.join(BASE, "app_v2", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports 2", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports 3", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports_4", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports6", "reports", "inspection_log.csv")
]

def main():
    # This automatically deduplicates VINs per day, keeping the *last* logged result
    date_vin_results = defaultdict(dict)
    
    # Track System Tests separately
    date_test_results = defaultdict(dict)

    for path in CSV_PATHS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get("id"):
                        continue
                        
                    vin = row.get("vin_id", "")
                    ts = row.get("timestamp", "")
                    auto_res = row.get("auto_result", "")
                    
                    if not vin or not ts:
                        continue
                    
                    # Extract YYYY-MM-DD
                    day = ts.split(" ")[0] if ts else "Unknown"
                    
                    if vin.startswith("TEST") or vin.startswith("MANUAL"):
                        date_test_results[day][vin] = auto_res
                    else:
                        date_vin_results[day][vin] = auto_res

    print("=" * 60)
    print(" DATE-WISE INSPECTION REPORT (Duplicates Removed) ")
    print("=" * 60)

    for day in sorted(date_vin_results.keys()):
        vins = date_vin_results[day]
        total_unique = len(vins)
        
        oil_leak = sum(1 for res in vins.values() if res == "OIL LEAK")
        no_leak = sum(1 for res in vins.values() if res == "NO LEAK")
        no_part = sum(1 for res in vins.values() if res == "NO_PART")
        pending = total_unique - (oil_leak + no_leak + no_part)

        print(f"\n--- {day} ---")
        print(f"Total Unique Cars Inspected : {total_unique}")
        print(f"  NO LEAK               : {no_leak}")
        print(f"  OIL LEAK              : {oil_leak}")
        print(f"  PARTS NOT FOUND       : {no_part}")
        if pending > 0:
            print(f"  OTHER                 : {pending}")

        if day in date_test_results:
            test_vins = date_test_results[day]
            print(f"  [+] Plus {len(test_vins)} System Test/Manual Runs")

if __name__ == "__main__":
    main()
