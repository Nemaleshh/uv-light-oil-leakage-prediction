"""
retrain_with_datav2.py
======================
Safely adds NEW labeled crops from data_v2 into the training pipeline
and retrains the GB classifier.

DUPLICATE PROTECTION:
- Reads all previously-trained CSVs (reports 1–4) to collect crop filenames
- Skips any crop already in those CSVs — zero duplicate training
- Only copies GENUINELY NEW crops from data_v2 into new_pipeline/crops/

Run:
    python retrain_with_datav2.py
"""

import os, sys, csv, glob, shutil, pickle
import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

BASE = r"C:\Users\hnema\OneDrive\Desktop\stellatis\newengil2\newengil"
sys.path.insert(0, BASE)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_V2_CSV   = os.path.join(BASE, "data_v2", "reports", "reports", "inspection_log.csv")
DATA_V2_CROPS = os.path.join(BASE, "data_v2", "detected_images", "detected_images", "crops")

OLD_OIL_CROPS    = os.path.join(BASE, "new_pipeline", "crops", "oil_leak")
OLD_NOLEAK_CROPS = os.path.join(BASE, "new_pipeline", "crops", "no_leak")

CLF_OUT = os.path.join(BASE, "app_v2", "models", "gb_classifier.pkl")

# CSVs whose crops are ALREADY in the training set — used for duplicate check
TRAINED_CSVS = [
    os.path.join(BASE, "app_v2", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports 2", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports 3", "reports", "inspection_log.csv"),
    os.path.join(BASE, "new_data", "reports_4", "reports", "inspection_log.csv"),
]

# ── Feature extraction (same as pipeline) ─────────────────────────────────────
def suppress_background(img):
    hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg_mask = cv2.inRange(hsv, (0, 0, 215), (180, 25, 255))
    if cv2.countNonZero(bg_mask) == 0:
        return img
    fg_mask = cv2.bitwise_not(bg_mask)
    median_bgr = (np.median(img[fg_mask > 0], axis=0).astype(np.uint8)
                  if cv2.countNonZero(fg_mask) > 0 else np.zeros(3, dtype=np.uint8))
    result = img.copy()
    result[bg_mask > 0] = median_bgr
    return result

def extract_features(crop):
    crop = suppress_background(crop)
    r    = cv2.resize(crop, (128, 128))

    hsv_raw = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    h_raw, s_raw, v_raw = cv2.split(hsv_raw)
    uv_mean_brightness = float(np.mean(v_raw))
    uv_max_brightness  = float(np.max(v_raw))
    uv_bright_ratio    = float(np.mean(v_raw > 180))
    uv_sat_spike_ratio = float(np.mean(s_raw > 150))

    uv_mask     = cv2.inRange(hsv_raw, (100, 30, 30), (160, 255, 255))
    uv_hue_hist = cv2.calcHist([hsv_raw], [0], uv_mask, [60], [100, 160])
    cv2.normalize(uv_hue_hist, uv_hue_hist)
    uv_hue_flat = uv_hue_hist.flatten()

    lab = cv2.cvtColor(r, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    limg  = cv2.merge((clahe.apply(l), a, b))
    norm  = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    hist_hs = cv2.calcHist([hsv], [0, 1], None, [64, 32], [0, 180, 0, 256])
    cv2.normalize(hist_hs, hist_hs)

    l_ch, a_ch, b_ch = cv2.split(cv2.cvtColor(norm, cv2.COLOR_BGR2LAB))

    def chan_stats(c):
        flat = c.flatten().astype(np.float32)
        return [float(np.mean(flat)), float(np.std(flat)),
                float(np.percentile(flat, 25)), float(np.percentile(flat, 75))]

    stats = (chan_stats(h_ch) + chan_stats(s_ch) + chan_stats(v_ch) +
             chan_stats(l_ch) + chan_stats(a_ch) + chan_stats(b_ch))

    gray    = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return np.concatenate([
        [uv_mean_brightness, uv_max_brightness, uv_bright_ratio, uv_sat_spike_ratio],
        uv_hue_flat, hist_hs.flatten(), stats, [lap_var],
    ])

# ── Augmentation (fixed-camera safe) ──────────────────────────────────────────
def augment_crop(img):
    h, w = img.shape[:2]
    variants = [img]
    variants.append(np.clip(img.astype(np.int16) + 25, 0, 255).astype(np.uint8))
    variants.append(np.clip(img.astype(np.int16) - 25, 0, 255).astype(np.uint8))
    M1 = cv2.getRotationMatrix2D((w//2, h//2), 5, 1.0)
    variants.append(cv2.warpAffine(img, M1, (w, h), borderMode=cv2.BORDER_REFLECT))
    M2 = cv2.getRotationMatrix2D((w//2, h//2), -5, 1.0)
    variants.append(cv2.warpAffine(img, M2, (w, h), borderMode=cv2.BORDER_REFLECT))
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    variants.append(np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    return variants

def load_crops_with_augment(folder, label_name):
    feats = []
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                   glob.glob(os.path.join(folder, "*.png")))
    print(f"  Loading {len(files)} crops from {os.path.basename(folder)} [{label_name}]")
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        for variant in augment_crop(img):
            feats.append(extract_features(variant))
    return feats

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RETRAIN: Adding data_v2 NEW crops + retraining GB")
    print("=" * 60)

    # Step 1: Build set of already-trained crop filenames
    print("\n[1] Checking already-trained crop filenames...")
    already_trained = set()
    for csv_path in TRAINED_CSVS:
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cp = row.get("yolo_crop_path", "").strip()
                if cp and cp not in ["--", "—"]:
                    already_trained.add(os.path.basename(cp))
    # Also collect existing filenames in the crop folders themselves
    for folder in [OLD_OIL_CROPS, OLD_NOLEAK_CROPS]:
        for f in glob.glob(os.path.join(folder, "*.*")):
            already_trained.add(os.path.basename(f))
    print(f"  Total already-trained filenames tracked: {len(already_trained)}")

    # Step 2: Collect NEW labeled crops from data_v2
    print("\n[2] Scanning data_v2 for new labeled crops...")
    new_oil_paths   = []
    new_noleak_paths = []
    skipped = 0

    with open(DATA_V2_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = row.get("manual_label", "").strip()
            if m not in ["Engine Oil", "No Leak"]:
                continue
            cp = row.get("yolo_crop_path", "").strip()
            if not cp or cp in ["--", "—"]:
                continue
            fname = os.path.basename(cp)
            full  = os.path.join(DATA_V2_CROPS, fname)
            if not os.path.exists(full):
                continue
            if fname in already_trained:
                skipped += 1
                continue
            if m == "Engine Oil":
                new_oil_paths.append(full)
            else:
                new_noleak_paths.append(full)

    print(f"  New Engine Oil crops : {len(new_oil_paths)}")
    print(f"  New No Leak crops    : {len(new_noleak_paths)}")
    print(f"  Skipped (duplicates) : {skipped}")

    # Step 3: Copy new crops into training folders
    print("\n[3] Copying new crops into training folders...")
    for src in new_oil_paths:
        dst = os.path.join(OLD_OIL_CROPS, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"  [OIL]    -> {os.path.basename(src)}")
    for src in new_noleak_paths:
        dst = os.path.join(OLD_NOLEAK_CROPS, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"  [NOLEAK] -> {os.path.basename(src)}")

    # Step 4: Load ALL crops (old + new) with augmentation
    print("\n[4] Loading ALL training crops with augmentation...")
    oil_feats    = load_crops_with_augment(OLD_OIL_CROPS,    "OIL LEAK")
    noleak_feats = load_crops_with_augment(OLD_NOLEAK_CROPS, "NO LEAK")

    print(f"\n  OIL LEAK feature vectors (augmented) : {len(oil_feats)}")
    print(f"  NO LEAK  feature vectors (augmented) : {len(noleak_feats)}")

    # Step 5: Train
    print("\n[5] Training Gradient Boosting Classifier...")
    X = np.array(noleak_feats + oil_feats)
    y = np.array([0] * len(noleak_feats) + [1] * len(oil_feats))

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=4,
                        min_samples_split=3,
                        subsample=0.8,
                        random_state=42))
    ])

    n_splits = min(5, min(len(noleak_feats), len(oil_feats)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy":  "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall":    make_scorer(recall_score,    zero_division=0),
        "f1":        make_scorer(f1_score,        zero_division=0),
    }
    cv_res = cross_validate(clf, X, y, cv=cv, scoring=scoring)

    print(f"\n  {n_splits}-Fold Cross-Validation Results:")
    print(f"  Accuracy  : {cv_res['test_accuracy'].mean()*100:.1f}%  ± {cv_res['test_accuracy'].std()*100:.1f}%")
    print(f"  Precision : {cv_res['test_precision'].mean()*100:.1f}% ± {cv_res['test_precision'].std()*100:.1f}%")
    print(f"  Recall    : {cv_res['test_recall'].mean()*100:.1f}%    ± {cv_res['test_recall'].std()*100:.1f}%")
    print(f"  F1-Score  : {cv_res['test_f1'].mean()*100:.1f}%    ± {cv_res['test_f1'].std()*100:.1f}%")

    # Final fit on ALL data
    clf.fit(X, y)
    with open(CLF_OUT, "wb") as pf:
        pickle.dump(clf, pf)

    print(f"\n✅ New model saved → {CLF_OUT}")
    print(f"   Oil Leak raw crops in training : {len(glob.glob(os.path.join(OLD_OIL_CROPS,'*.*')))}")
    print(f"   No Leak  raw crops in training : {len(glob.glob(os.path.join(OLD_NOLEAK_CROPS,'*.*')))}")
    print("\nDone! Retraining complete.\n")

if __name__ == "__main__":
    main()
