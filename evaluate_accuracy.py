"""
evaluate_accuracy.py
====================
Evaluates YOLO + GB classifier accuracy against the two image folders
used by the live application:
  - nonoil leak/  → ground truth: NO LEAK
  - oilleak/      → ground truth: OIL LEAK

Uses the EXACT same pipeline (LeakPipeline) and config.json as the app,
including the ROI filter, YOLO_CONF, crop-size guard, and feature extraction.

Run:
    python evaluate_accuracy.py

Outputs:
  - Console summary (accuracy, precision, recall, F1, confusion matrix)
  - evaluate_accuracy_report.txt  (full per-image log)
"""

import os
import sys
import json
import cv2
import datetime
import textwrap
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
APP_DIR      = BASE_DIR / "app_v2"
CONFIG_PATH  = APP_DIR / "config.json"
NONOIL_DIR   = BASE_DIR / "nonoil leak"
OILLEAK_DIR  = BASE_DIR / "oilleak"
REPORT_PATH  = BASE_DIR / "evaluate_accuracy_report.txt"

# Add app_v2 to path so we can import LeakPipeline
sys.path.insert(0, str(APP_DIR))

# ─── Load config ───────────────────────────────────────────────────────────────
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

MODEL_PATH = cfg["model_path"]
CLF_PATH   = cfg["clf_path"]
ROI        = cfg.get("roi", None)

# ─── Import pipeline ───────────────────────────────────────────────────────────
from core.pipeline import LeakPipeline

pipeline = LeakPipeline(MODEL_PATH, CLF_PATH, roi=ROI)
print("Loading models …")
pipeline.load()
print("Models loaded.\n")

# ─── Image categories ──────────────────────────────────────────────────────────
CATEGORIES = [
    {"folder": NONOIL_DIR, "gt_label": "NO LEAK",  "display": "nonoil leak"},
    {"folder": OILLEAK_DIR,"gt_label": "OIL LEAK", "display": "oilleak"},
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Counters ──────────────────────────────────────────────────────────────────
TP = TN = FP = FN = 0   # oil-leak is the POSITIVE class
errors     = 0
no_part    = 0

lines = []   # full per-image log

def log(msg):
    lines.append(msg)
    print(msg)

log("=" * 70)
log(f"  YOLO + GB Classifier Accuracy Evaluation")
log(f"  Started : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"  Model   : {MODEL_PATH}")
log(f"  Clf     : {CLF_PATH}")
log(f"  ROI     : {ROI}")
log("=" * 70)

# ─── Run inference ─────────────────────────────────────────────────────────────
for cat in CATEGORIES:
    folder   = cat["folder"]
    gt_label = cat["gt_label"]   # "NO LEAK" or "OIL LEAK"
    display  = cat["display"]

    image_files = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ])

    log(f"\n{'─'*70}")
    log(f"  Folder : {display}  ({len(image_files)} images)  GT = {gt_label}")
    log(f"{'─'*70}")

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            log(f"  [SKIP]  {img_path.name}  — could not read image")
            errors += 1
            continue

        out = pipeline.run(frame)

        pred   = out["result"]      # "NO LEAK" | "OIL LEAK" | "NO_PART" | "ERROR"
        yconf  = out["yolo_conf"]
        gconf  = out["confidence"]
        err    = out["error"]

        # ── Classify outcome ──────────────────────────────────────────────────
        if pred == "ERROR":
            status = "ERROR"
            errors += 1
        elif pred == "NO_PART":
            # YOLO found nothing (or detection outside ROI)
            # Treat as NO LEAK prediction for accuracy purposes
            status  = "NO_PART→NO LEAK"
            pred_eff = "NO LEAK"
            no_part += 1
            if gt_label == "NO LEAK":
                TN += 1
            else:
                FN += 1
            log(f"  [NO_PART] {img_path.name:<50s}  GT={gt_label}  err={err}")
            continue
        else:
            pred_eff = pred

        if gt_label == "OIL LEAK" and pred_eff == "OIL LEAK":
            status = "TP ✓"; TP += 1
        elif gt_label == "NO LEAK"  and pred_eff == "NO LEAK":
            status = "TN ✓"; TN += 1
        elif gt_label == "NO LEAK"  and pred_eff == "OIL LEAK":
            status = "FP ✗"; FP += 1
        else:
            status = "FN ✗"; FN += 1

        mark = "✓" if "✓" in status else "✗"
        log(
            f"  [{status:<6}] {img_path.name:<50s} "
            f"GT={gt_label:<9} PRED={pred_eff:<9} "
            f"YOLO={yconf:.3f}  CLF={gconf:.1f}%"
        )

# ─── Summary metrics ───────────────────────────────────────────────────────────
total        = TP + TN + FP + FN
correct      = TP + TN
accuracy     = correct / total * 100 if total else 0

precision    = TP / (TP + FP) * 100 if (TP + FP) else 0
recall       = TP / (TP + FN) * 100 if (TP + FN) else 0
f1           = (2 * precision * recall / (precision + recall)
                if (precision + recall) else 0)
specificity  = TN / (TN + FP) * 100 if (TN + FP) else 0
fpr          = FP / (FP + TN) * 100 if (FP + TN) else 0

log(f"\n{'='*70}")
log(f"  RESULTS SUMMARY")
log(f"{'='*70}")
log(f"  Total images evaluated : {total}")
log(f"  NO_PART  (YOLO miss)   : {no_part}")
log(f"  Errors   (read/crash)  : {errors}")
log(f"")
log(f"  Confusion Matrix  (Positive = OIL LEAK)")
log(f"  ┌──────────────────┬──────────────────┐")
log(f"  │  TP (oil→oil)  {TP:>4} │  FN (oil→noleak){FN:>3} │")
log(f"  │  FP (no→oil)   {FP:>4} │  TN (no→noleak){TN:>4} │")
log(f"  └──────────────────┴──────────────────┘")
log(f"")
log(f"  Accuracy    : {accuracy:>6.2f}%   ({correct}/{total} correct)")
log(f"  Precision   : {precision:>6.2f}%   (of OIL LEAK predictions, how many were right)")
log(f"  Recall      : {recall:>6.2f}%   (of real OIL LEAK images, how many detected)")
log(f"  Specificity : {specificity:>6.2f}%   (of real NO LEAK images, how many correct)")
log(f"  F1 Score    : {f1:>6.2f}%")
log(f"  False Pos.  : {fpr:>6.2f}%   (NO LEAK wrongly called OIL LEAK)")
log(f"{'='*70}")
log(f"  Report saved → {REPORT_PATH}")

# ─── Write report file ─────────────────────────────────────────────────────────
with open(REPORT_PATH, "w", encoding="utf-8") as rf:
    rf.write("\n".join(lines))

print(f"\nDone.  Report → {REPORT_PATH}")
