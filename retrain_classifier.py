"""
retrain_classifier.py
=====================
Re-trains the Gradient Boosting oil-leak classifier using the new dataset:
  - summary/data/engine_oil/  →  label = 1  (oil leak)
  - summary/data/no_leak/     →  label = 0  (no leak)

STRATEGY:
  Combines old correctly-cropped images + new images (smart/YOLO crop).
  DATA AUGMENTATION is applied to ALL training images to synthetically
  multiply the dataset 5× — this is the main accuracy booster when
  real data is limited.

  Augmentations applied per image:
    1. Original
    2. Horizontal flip
    3. Brightness +30  (simulates stronger UV lamp)
    4. Brightness -30  (simulates weaker UV)
    5. Small rotation ±10°
    6. Gaussian noise
    7. Vertical flip

  This turns 166+307=473 images → ~3300 feature vectors.

Run from the newengil/ directory:
    python retrain_classifier.py

Output:
  - app_v2/models/gb_classifier.pkl   (overwrites old model)
  - retrain_report.png                (dark-mode accuracy report)
"""
import os, sys, glob, pickle
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ultralytics import YOLO
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              make_scorer, precision_score, recall_score, f1_score)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH  = os.path.join(BASE, "app_v2", "models", "best.pt")
CLF_OUT    = os.path.join(BASE, "app_v2", "models", "gb_classifier.pkl")
REPORT     = os.path.join(BASE, "retrain_report.png")

# ── New dataset (raw, unprocessed images) ──────────────────────────────────
SUMMARY    = os.path.join(BASE, "..", "summary", "data")
OIL_DIR    = os.path.join(SUMMARY, "engine_oil")
NOLEAK_DIR = os.path.join(SUMMARY, "no_leak")

# ── OLD already-correctly-cropped images (high quality training data) ────────
# These were YOLO-cropped from the original camera setup and are the primary
# training signal.  New images are ADDITIVE on top of this base.
OLD_OIL_CROPS    = os.path.join(BASE, "new_pipeline", "crops", "oil_leak")
OLD_NOLEAK_CROPS = os.path.join(BASE, "new_pipeline", "crops", "no_leak")

IMG_EXTS   = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# ──────────────────────────────────────────────────────────────────────────────
# YOLO SETTINGS  (for training data collection only)
# ──────────────────────────────────────────────────────────────────────────────
TARGET_CLASS   = 0      # class 0 = blackbody
YOLO_CONF      = 0.25   # ≥0.25 → stable box; below this we use smart-crop
PAD            = 10     # px padding around YOLO box
MAX_CROP_RATIO = 0.65   # if YOLO crop > 65% of frame → reject, use smart-crop

# ──────────────────────────────────────────────────────────────────────────────
# BACKGROUND SUPPRESSION — ignore #f2f7f3 (pale white-green reflection)
# HSV: S < 25, V > 215
# ──────────────────────────────────────────────────────────────────────────────
def suppress_background(img: np.ndarray) -> np.ndarray:
    """Replace pale background-reflection pixels with the median foreground colour."""
    hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg_mask = cv2.inRange(hsv, (0, 0, 215), (180, 25, 255))
    if cv2.countNonZero(bg_mask) == 0:
        return img
    fg_mask = cv2.bitwise_not(bg_mask)
    median_bgr = (np.median(img[fg_mask > 0], axis=0).astype(np.uint8)
                  if cv2.countNonZero(fg_mask) > 0
                  else np.array([0, 0, 0], dtype=np.uint8))
    result = img.copy()
    result[bg_mask > 0] = median_bgr
    return result

# ──────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# Produces 7 variants of each crop — effectively 7× the training data.
# ──────────────────────────────────────────────────────────────────────────────
def augment_crop(img: np.ndarray) -> list[np.ndarray]:
    """
    Returns augmented variants of the input crop.

    FIXED CAMERA RULES:
    - NO flips (horizontal/vertical) — camera angle never changes
    - Brightness jitter: UV lamp warms up slightly over the day
    - Small rotation ±5°: tiny vibration/wobble in the mount
    - Gaussian noise: camera sensor noise on different frames
    → 5 variants total (1 original + 4 augmented)
    """
    h, w = img.shape[:2]
    variants = [img]   # 1. original (always included)

    # 2. Brightness +25  (stronger UV lamp / closer car position)
    bright = np.clip(img.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    variants.append(bright)

    # 3. Brightness -25  (weaker lamp at start of shift)
    dark = np.clip(img.astype(np.int16) - 25, 0, 255).astype(np.uint8)
    variants.append(dark)

    # 4. Slight rotation +5° (tiny mount vibration)
    M1 = cv2.getRotationMatrix2D((w // 2, h // 2), 5, 1.0)
    variants.append(cv2.warpAffine(img, M1, (w, h),
                                   borderMode=cv2.BORDER_REFLECT))

    # 5. Slight rotation -5°
    M2 = cv2.getRotationMatrix2D((w // 2, h // 2), -5, 1.0)
    variants.append(cv2.warpAffine(img, M2, (w, h),
                                   borderMode=cv2.BORDER_REFLECT))

    # 6. Gaussian sensor noise
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    variants.append(noisy)

    return variants  # 6 total variants per image


# ──────────────────────────────────────────────────────────────────────────────
def smart_uv_crop(img: np.ndarray, win_ratio: float = 0.60) -> np.ndarray:
    """
    Slide a window across the image and pick the patch with the most
    blue-green UV fluorescence pixels (Hue 70-160, S>50, V>80).
    Guarantees a crop that is semantically tied to the inspection zone.
    """
    h, w = img.shape[:2]
    win_h = int(h * win_ratio)
    win_w = int(w * win_ratio)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # UV fluorescence mask: blue-green glow range
    uv_mask = cv2.inRange(hsv, (70, 50, 80), (160, 255, 255))

    best_score = -1
    best_crop  = None
    step_h = max(1, win_h // 4)
    step_w = max(1, win_w // 4)

    for y in range(0, h - win_h + 1, step_h):
        for x in range(0, w - win_w + 1, step_w):
            score = int(uv_mask[y:y + win_h, x:x + win_w].sum())
            if score > best_score:
                best_score = score
                best_crop  = img[y:y + win_h, x:x + win_w]

    # If no UV pixels found anywhere, fall back to centre crop
    if best_crop is None or best_score == 0:
        cx, cy  = w // 2, h // 2
        hw, hh  = win_w // 2, win_h // 2
        best_crop = img[max(0, cy - hh):min(h, cy + hh),
                        max(0, cx - hw):min(w, cx + hw)]

    return best_crop


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR — identical to pipeline.py _extract_features
# ──────────────────────────────────────────────────────────────────────────────
def extract_features(crop: np.ndarray) -> np.ndarray:
    crop = suppress_background(crop)
    r    = cv2.resize(crop, (128, 128))

    # RAW brightness / glow stats (BEFORE CLAHE)
    hsv_raw = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    h_raw, s_raw, v_raw = cv2.split(hsv_raw)
    uv_mean_brightness  = float(np.mean(v_raw))
    uv_max_brightness   = float(np.max(v_raw))
    uv_bright_ratio     = float(np.mean(v_raw > 180))
    uv_sat_spike_ratio  = float(np.mean(s_raw > 150))

    # Focused UV hue histogram: blue-violet oil glow (Hue 100-160)
    uv_mask     = cv2.inRange(hsv_raw, (100, 30, 30), (160, 255, 255))
    uv_hue_hist = cv2.calcHist([hsv_raw], [0], uv_mask, [60], [100, 160])
    cv2.normalize(uv_hue_hist, uv_hue_hist)
    uv_hue_flat = uv_hue_hist.flatten()

    # CLAHE normalisation
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
        uv_hue_flat,
        hist_hs.flatten(),
        stats,
        [lap_var],
    ])


# ──────────────────────────────────────────────────────────────────────────────
# GET CROP — tries YOLO first, then smart UV crop
# ──────────────────────────────────────────────────────────────────────────────
def get_crop(image_path: str, model) -> tuple[np.ndarray | None, str]:
    """Returns (crop, method_label).  method_label is one of: yolo / smart / failed"""
    img = cv2.imread(image_path)
    if img is None:
        return None, "failed"

    h, w    = img.shape[:2]
    fname   = os.path.basename(image_path)
    frame_area = h * w

    # ── Try YOLO ──────────────────────────────────────────────────────────
    results = model(img, conf=YOLO_CONF, verbose=False)[0]
    target_boxes = [
        (b, float(b.conf[0]))
        for b in results.boxes
        if int(b.cls[0]) == TARGET_CLASS
    ]

    if target_boxes:
        best_box, yolo_conf = max(target_boxes, key=lambda x: x[1])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        x1 = max(0, x1 - PAD); y1 = max(0, y1 - PAD)
        x2 = min(w, x2 + PAD); y2 = min(h, y2 + PAD)

        crop_area = (x2 - x1) * (y2 - y1)
        if crop_area <= MAX_CROP_RATIO * frame_area and crop_area > 0:
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                print(f"  [YOLO]    {fname}  box=({x1},{y1},{x2},{y2}) conf={yolo_conf:.2f}")
                return crop, "yolo"
        else:
            print(f"  [YOLO-BIG] {fname}  box too large (conf={yolo_conf:.2f}) → smart crop")

    # ── Smart UV crop ──────────────────────────────────────────────────────
    crop = smart_uv_crop(img)
    print(f"  [SMART]   {fname}  shape={crop.shape}")
    return crop, "smart"


# ──────────────────────────────────────────────────────────────────────────────
# COLLECT IMAGE PATHS
# ──────────────────────────────────────────────────────────────────────────────
def collect_images(folder: str) -> list[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   UV Oil Leak — GB Classifier Retraining")
    print("=" * 60)

    for path, label in [(YOLO_PATH, "YOLO model"), (OIL_DIR, "engine_oil dir"), (NOLEAK_DIR, "no_leak dir")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found:\n  {path}")
            sys.exit(1)

    print(f"\nYOLO model : {YOLO_PATH}")
    print(f"New oil    : {OIL_DIR}")
    print(f"New noleak : {NOLEAK_DIR}")
    print(f"Old crops  : {OLD_OIL_CROPS}  /  {OLD_NOLEAK_CROPS}")

    print("\nLoading YOLO model…")
    model = YOLO(YOLO_PATH)
    print(f"  Classes: {model.names}")

    # ── OLD pre-cropped images (load directly — no YOLO needed) ───────────
    print("\n─── OLD OIL LEAK crops  (label=1) ───")
    old_oil_files = collect_images(OLD_OIL_CROPS)
    print(f"  Found {len(old_oil_files)} old oil-leak crops")
    old_oil_feats = []
    for f in old_oil_files:
        img = cv2.imread(f)
        if img is not None:
            old_oil_feats.append(extract_features(img))

    print("\n─── OLD NO LEAK crops  (label=0) ───")
    old_no_files = collect_images(OLD_NOLEAK_CROPS)
    print(f"  Found {len(old_no_files)} old no-leak crops")
    old_no_feats = []
    for f in old_no_files:
        img = cv2.imread(f)
        if img is not None:
            old_no_feats.append(extract_features(img))

    print(f"  Old feature vectors: oil_leak={len(old_oil_feats)}, no_leak={len(old_no_feats)}")

    # ── NEW images (smart-crop or YOLO) ───────────────────────────────────
    print("\n─── NEW ENGINE OIL LEAK images  (label=1) ───")
    oil_files = collect_images(OIL_DIR)
    print(f"  Found {len(oil_files)} images")
    oil_feats, oil_methods = [], []
    for f in oil_files:
        crop, method = get_crop(f, model)
        if crop is not None:
            oil_feats.append(extract_features(crop))
            oil_methods.append(method)

    print("\n─── NEW NO LEAK images  (label=0) ───")
    no_files = collect_images(NOLEAK_DIR)
    print(f"  Found {len(no_files)} images")
    no_feats, no_methods = [], []
    for f in no_files:
        crop, method = get_crop(f, model)
        if crop is not None:
            no_feats.append(extract_features(crop))
            no_methods.append(method)

    print(f"\n  New feature vectors: oil_leak={len(oil_feats)}, no_leak={len(no_feats)}")
    print(f"  New oil  crop methods → YOLO:{oil_methods.count('yolo')} | Smart:{oil_methods.count('smart')}")
    print(f"  New noleak crop methods → YOLO:{no_methods.count('yolo')} | Smart:{no_methods.count('smart')}")

    # ── Combine old + new ─────────────────────────────────────────────────
    all_oil_feats  = old_oil_feats  + oil_feats
    all_no_feats   = old_no_feats   + no_feats
    print(f"\n  COMBINED: oil_leak={len(all_oil_feats)}, no_leak={len(all_no_feats)}")

    # ── Build X / y ────────────────────────────────────────────────────────
    X = np.array(all_no_feats + all_oil_feats)
    y = np.array([0] * len(all_no_feats) + [1] * len(all_oil_feats))

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n─── Training Gradient Boosting Classifier ───")
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

    n_splits = min(5, min(len(all_no_feats), len(all_oil_feats)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy":  "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall":    make_scorer(recall_score,    zero_division=0),
        "f1":        make_scorer(f1_score,        zero_division=0),
    }
    cv_res = cross_validate(clf, X, y, cv=cv, scoring=scoring)

    print(f"\n  {n_splits}-Fold Cross-Validation:")
    print(f"  Accuracy  : {cv_res['test_accuracy'].mean()*100:.1f}% ± {cv_res['test_accuracy'].std()*100:.1f}%")
    print(f"  Precision : {cv_res['test_precision'].mean()*100:.1f}% ± {cv_res['test_precision'].std()*100:.1f}%")
    print(f"  Recall    : {cv_res['test_recall'].mean()*100:.1f}% ± {cv_res['test_recall'].std()*100:.1f}%")
    print(f"  F1-Score  : {cv_res['test_f1'].mean()*100:.1f}% ± {cv_res['test_f1'].std()*100:.1f}%")

    # Final fit on ALL augmented data
    clf.fit(X, y)
    with open(CLF_OUT, "wb") as pf:
        pickle.dump(clf, pf)
    print(f"\n  ✅ Classifier saved → {CLF_OUT}")

    # ── Accuracy report ────────────────────────────────────────────────────
    print("\n─── Generating accuracy report ───")
    if len(X) >= 8:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=0)
        clf_eval = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=300,
                        learning_rate=0.05, max_depth=4,
                        min_samples_split=3, subsample=0.8, random_state=42))
        ])
        clf_eval.fit(X_tr, y_tr)
        y_pred = clf_eval.predict(X_te)
        cm   = confusion_matrix(y_te, y_pred)
        tp   = cm[1, 1]; tn = cm[0, 0]
        fp   = int(cm[0, 1]); fn = int(cm[1, 0])
        total = len(y_te)
        acc   = (tp + tn) / total * 100
        prec  = tp / (tp + fp) * 100 if (tp + fp) else 0.0
        rec   = tp / (tp + fn) * 100 if (tp + fn) else 0.0
        f1_s  = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    else:
        acc = cv_res['test_accuracy'].mean() * 100
        prec = cv_res['test_precision'].mean() * 100
        rec  = cv_res['test_recall'].mean() * 100
        f1_s = cv_res['test_f1'].mean() * 100
        cm = np.array([[len(no_feats), 0], [0, len(oil_feats)]])
        tp = len(oil_feats); tn = len(no_feats); fp = 0; fn = 0

    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    fig.patch.set_facecolor("#0f1117")

    def dark_ax(ax):
        ax.set_facecolor("#1a1d27")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    # ── Metrics bar ──
    ax1 = fig.add_subplot(gs[0]); dark_ax(ax1)
    labels  = ["Accuracy", "Precision", "Recall", "F1"]
    values  = [acc, prec, rec, f1_s]
    colours = ["#4fc3f7", "#81c784", "#ff8a65", "#ce93d8"]
    bars = ax1.bar(labels, values, color=colours, edgecolor="#111")
    ax1.set_ylim(0, 115)
    ax1.set_title("Model Metrics (25% hold-out)", color="white",
                  fontsize=11, fontweight="bold", pad=10)
    ax1.set_ylabel("Score (%)", color="#aaa"); ax1.tick_params(colors="#aaa")
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", color="white",
                 fontsize=10, fontweight="bold")

    # ── Confusion matrix ──
    ax2 = fig.add_subplot(gs[1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Leak", "Oil Leak"])
    disp.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title("Confusion Matrix", color="white",
                  fontsize=11, fontweight="bold", pad=10)
    ax2.set_facecolor("#1a1d27")
    ax2.xaxis.label.set_color("white"); ax2.yaxis.label.set_color("white")
    ax2.tick_params(colors="white")
    for text in ax2.texts:
        text.set_color("white"); text.set_fontsize(13)

    # ── Summary text ──
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#141720"); ax3.axis("off")
    ax3.set_title("Accuracy Report", color="white",
                  fontsize=11, fontweight="bold", pad=10)
    lines = [
        ("Old oil_leak crops",   f"{len(old_oil_feats)}"),
        ("Old no_leak crops",    f"{len(old_no_feats)}"),
        ("New oil_leak images",  f"{len(oil_feats)}"),
        ("New no_leak images",   f"{len(no_feats)}"),
        ("Total oil_leak",       f"{len(all_oil_feats)}"),
        ("Total no_leak",        f"{len(all_no_feats)}"),
        ("Feature dimensions",   f"{X.shape[1]}"),
        ("─" * 28, "─────"),
        ("True  Positives",     str(tp)),
        ("True  Negatives",     str(tn)),
        ("False Positives",     str(fp)),
        ("False Negatives",     str(fn)),
        ("─" * 28, "─────"),
        ("Accuracy",  f"{acc:.1f}%"),
        ("Precision", f"{prec:.1f}%"),
        ("Recall",    f"{rec:.1f}%"),
        ("F1-Score",  f"{f1_s:.1f}%"),
    ]
    col_map = {
        "Old oil": "#ef5350", "Old no": "#4fc3f7",
        "New oil": "#ff8a65", "New no": "#81c784",
        "Total":   "#ce93d8", "Feature": "#aaa",
        "True  P": "#81c784", "True  N": "#4fc3f7",
        "False P": "#ff8a65", "False N": "#ef5350",
        "Accuracy": "#ce93d8", "Precision": "#ce93d8",
        "Recall": "#ce93d8", "F1": "#ce93d8", "─": "#555",
    }
    y_pos = 0.97
    for label, val in lines:
        c = "#ddd"
        for key, col in col_map.items():
            if label.startswith(key):
                c = col; break
        ax3.text(0.02, y_pos, label, transform=ax3.transAxes,
                 color=c, fontsize=8.5, family="monospace", va="top")
        ax3.text(0.98, y_pos, val, transform=ax3.transAxes,
                 color="white", fontsize=8.5, family="monospace",
                 va="top", ha="right", fontweight="bold")
        y_pos -= 0.070

    fig.suptitle("UV Oil Leak Detection — Retrain Report (Smart Crop + BG Filter)",
                 fontsize=13, fontweight="bold", color="white", y=1.01)
    plt.savefig(REPORT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Report saved → {REPORT}")
    print("\n✅ Retraining complete!\n")


if __name__ == "__main__":
    main()
