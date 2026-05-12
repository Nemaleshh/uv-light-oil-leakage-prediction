import os
import csv
from datetime import datetime

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
CSV_PATH = os.path.join(BASE, "data_v2", "reports", "reports", "inspection_log.csv")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}")
        return

    target_date = "2026-05-11"
    
    total_today = 0
    valid_today = 0
    skipped_today = 0
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp", "")
            if not ts.startswith(target_date):
                continue
            
            total_today += 1
            auto_res = row.get("auto_result", "").strip().upper()
            manual = row.get("manual_label", "").strip()

            if manual in ["Pending", "-", "—", ""]:
                skipped_today += 1
                continue
                
            valid_today += 1
            
            # Map labels
            system_is_positive = (auto_res == "OIL LEAK")
            manual_is_positive = (manual == "Engine Oil")
            
            if system_is_positive and manual_is_positive:
                TP += 1
            elif not system_is_positive and not manual_is_positive:
                TN += 1
            elif system_is_positive and not manual_is_positive:
                FP += 1
            elif not system_is_positive and manual_is_positive:
                FN += 1

    accuracy = (TP + TN) / valid_today * 100 if valid_today > 0 else 0
    
    print("\n========================================")
    print(f" DATA_V2 ACCURACY REPORT FOR {target_date}")
    print("========================================")
    print(f"Total Cars Inspected Today: {total_today}")
    print(f"Valid for Accuracy Check  : {valid_today}")
    print(f"Skipped (Pending)         : {skipped_today}")
    print("\n--- CONFUSION MATRIX ---")
    print(f"TP (Correctly identified Oil Leak): {TP}")
    print(f"TN (Correctly identified No Leak) : {TN}")
    print(f"FP (False Alarm: No Leak -> Oil)  : {FP}")
    print(f"FN (Missed Leak: Oil -> No Leak)  : {FN}")
    print(f"\nOVERALL ACCURACY: {accuracy:.2f}%")
    print(f"CORRECT PREDICTIONS: {TP + TN} out of {valid_today}")
    print("========================================")

if __name__ == "__main__":
    main()
