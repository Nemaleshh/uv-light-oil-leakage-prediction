import csv
from collections import defaultdict
import os

csv_path = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil\new_data\reports6\reports\inspection_log.csv"

# Stellantis standard report string format
report_template = """
================================================================================
                    STELLANTIS - AI MODEL ACCURACY REPORT
================================================================================
Date Analyzed: {date}
Dataset Location: {dataset}
--------------------------------------------------------------------------------
1. INSPECTION VOLUME
Total Inspections Logged : {total}
Valid for Accuracy Check : {valid} (Manual label != 'Pending' or '-')
Skipped (Pending/Unlabeled): {skipped}

2. CONFUSION MATRIX (System vs Manual)
                      Manual: ENGINE OIL    Manual: NO LEAK
System: OIL LEAK         {TP:<14}      {FP:<14}
System: NO LEAK          {FN:<14}      {TN:<14}

3. KEY PERFORMANCE INDICATORS (KPIs)
Accuracy (Overall)       : {accuracy:.2f}%
Precision (Oil Leak)     : {precision:.2f}%
Recall (Oil Leak)        : {recall:.2f}%
Specificity (No Leak)    : {specificity:.2f}%
F1 Score                 : {f1:.2f}%

False Positive Rate      : {fpr:.2f}%
False Negative Rate      : {fnr:.2f}%

4. OBSERVATIONS
{observations}
================================================================================
"""

def main():
    target_date = "2026-05-11"
    
    total = 0
    valid = 0
    skipped = 0
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp", "")
            if not ts.startswith(target_date):
                continue
                
            total += 1
            auto_res = row.get("auto_result", "").strip().upper()
            manual = row.get("manual_label", "").strip().upper()
            
            # Map manual to standard classes
            if manual in ["PENDING", "-", "—", ""]:
                skipped += 1
                continue
                
            valid += 1
            
            system_is_positive = (auto_res == "OIL LEAK")
            manual_is_positive = (manual == "ENGINE OIL")
            
            if system_is_positive and manual_is_positive:
                TP += 1
            elif not system_is_positive and not manual_is_positive:
                TN += 1
            elif system_is_positive and not manual_is_positive:
                FP += 1
            elif not system_is_positive and manual_is_positive:
                FN += 1

    accuracy = (TP + TN) / valid * 100 if valid > 0 else 0
    precision = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else 0
    recall = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
    specificity = (TN / (TN + FP)) * 100 if (TN + FP) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = (FP / (FP + TN)) * 100 if (FP + TN) > 0 else 0
    fnr = (FN / (FN + TP)) * 100 if (FN + TP) > 0 else 0
    
    observations = "- System is performing well."
    if FP > 0:
         observations += f"\n- {FP} False Positives: System called an oil leak, but manual confirmed NO LEAK."
    if FN > 0:
         observations += f"\n- {FN} False Negatives: System missed an oil leak that was present."
         
    report = report_template.format(
        date=target_date,
        dataset="reports6",
        total=total,
        valid=valid,
        skipped=skipped,
        TP=TP, FP=FP, FN=FN, TN=TN,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        fpr=fpr,
        fnr=fnr,
        observations=observations
    )
    
    report_out_path = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil\stellantis_accuracy_report_11_05_2026.txt"
    with open(report_out_path, "w") as f:
        f.write(report)
        
    print(report)

if __name__ == "__main__":
    main()
