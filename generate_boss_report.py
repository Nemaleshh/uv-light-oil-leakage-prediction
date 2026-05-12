import os
import csv
import sys

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
sys.path.insert(0, os.path.join(BASE, "app_v2"))

from core.report_generator import generate_summary_report

CSV_PATHS = [
    os.path.join(BASE, "app_v2", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports 2", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports 3", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports_4", "reports", "inspection_log.csv")
]

MERGED_CSV = os.path.join(BASE, "merged_boss_log.csv")
OUT_DIR = BASE  # output directory for HTML report

def main():
    print("Merging CSV files...")
    rows = []
    all_headers = []

    for path in CSV_PATHS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for h in reader.fieldnames:
                        if h not in all_headers:
                            all_headers.append(h)
                    for row in reader:
                        # Skip test cases
                        vin = row.get("vin_id", "")
                        if vin.startswith("TEST"):
                            continue
                        rows.append(row)

    if not rows:
        print("No data found in any CSV!")
        return

    # Write MERGED CSV using DictWriter to correctly align columns
    with open(MERGED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_headers)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Merged {len(rows)} records into {MERGED_CSV}")

    print("Generating HTML Summary Report...")
    report_path = generate_summary_report(
        csv_path=MERGED_CSV,
        reports_dir=OUT_DIR,
        base_dir=os.path.join(BASE, "app_v2"),
        target_date=None # All-time
    )

    if report_path:
        print(f"\nSUCCESS! Boss report generated at: {report_path}")
        os.remove(MERGED_CSV)
    else:
        print("Failed to generate report.")

if __name__ == "__main__":
    main()
