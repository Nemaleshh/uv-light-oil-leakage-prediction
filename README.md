# UV Light Oil Leakage Prediction System

An **industrial-grade UV oil leak detection system** built for Stellantis engine assembly lines. The system uses a two-stage AI pipeline to automatically detect engine oil leaks under UV illumination in real time.

---

## 🔬 How It Works

```
Live Camera Frame
      │
      ▼
 ┌──────────────────────────────┐
 │  Stage 1: YOLOv8 Detection  │  ← Detects & crops the engine blackbody region
 └──────────────────────────────┘
      │  ROI-filtered crop
      ▼
 ┌──────────────────────────────────────────────────────┐
 │  Stage 2: Gradient Boosting Classifier               │
 │  2137 hand-crafted UV colour features:               │
 │   • UV brightness stats (raw HSV)                    │
 │   • UV hue histogram (Hue 100–160, oil glow range)   │
 │   • 2D Hue×Saturation histogram (post-CLAHE)         │
 │   • Per-channel stats (H,S,V,L,A,B)                  │
 │   • Laplacian variance (texture)                     │
 └──────────────────────────────────────────────────────┘
      │
      ▼
 "OIL LEAK" | "NO LEAK" + Confidence %
```

---

## 📁 Project Structure

```
newengil/
├── app_v2/                      # Main application
│   ├── core/
│   │   ├── pipeline.py          # YOLO + GB inference engine
│   │   ├── camera_thread.py     # Live camera frame capture thread
│   │   ├── presence_detector.py # Vehicle presence detection
│   │   ├── report_generator.py  # CSV + image inspection reports
│   │   └── cloud_sync.py        # Optional cloud sync
│   ├── ui/                      # PyQt5 dashboard UI
│   ├── models/                  # (not tracked) best.pt + gb_classifier.pkl
│   ├── config.json              # Camera, ROI, model path config
│   └── main.py                  # Application entry point
│
├── retrain_classifier.py        # Full retrain from scratch
├── retrain_with_datav2.py       # Safe incremental retrain (dedup protected)
├── evaluate_accuracy.py         # Real-world accuracy evaluation
├── add_to_training.py           # Utility: add specific crops to training set
│
├── roi_calibrator.py            # GUI tool to calibrate inspection ROI
├── _debug_crops.py              # Debug crop extraction visually
├── test_single_image.py         # Quick single-image inference test
│
├── requirements.txt             # Python dependencies
└── .gitignore
```

---

## 🤖 Model Architecture

### Gradient Boosting Classifier
```
StandardScaler → GradientBoostingClassifier
  n_estimators    = 300
  learning_rate   = 0.05
  max_depth       = 4
  min_samples_split = 3
  subsample       = 0.8
```

### Training Data
- **Oil Leak (label=1):** ~155 YOLO-cropped engine images under UV
- **No Leak (label=0):** ~318 YOLO-cropped clean engine images
- **Augmentation:** 6× per image (brightness ±25, rotation ±5°, Gaussian noise)
- **Effective training size:** ~2,800+ feature vectors

### Cross-Validation Results
```
5-Fold Stratified CV:
  Accuracy  : 94.3% ± 2.2%
  Precision : 93.7% ± 1.9%
  Recall    : 89.8% ± 6.0%
  F1-Score  : 91.6% ± 3.5%
```

---

## 🚀 Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Model Files
```
app_v2/models/best.pt          # YOLOv8 detection model
app_v2/models/gb_classifier.pkl  # Trained GB classifier
```

### 3. Configure
Edit `app_v2/config.json`:
```json
{
  "camera_index": 0,
  "model_path": "models/best.pt",
  "clf_path": "models/gb_classifier.pkl",
  "roi": { "x1_pct": 0.1, "y1_pct": 0.1, "x2_pct": 0.9, "y2_pct": 0.9 }
}
```

### 4. Run Application
```bash
cd app_v2
python main.py
```

### 5. Retrain Classifier
```bash
# Full retrain (from scratch)
python retrain_classifier.py

# Incremental retrain (add new data safely)
python retrain_with_datav2.py
```

### 6. Evaluate Accuracy
```bash
python evaluate_accuracy.py
```

---

## 🔧 Key Features

| Feature | Detail |
|---|---|
| **ROI Filtering** | Rejects YOLO detections outside the calibrated inspection zone |
| **Low-light Fix** | CLAHE + gamma boost for YOLO pre-processing; original frame used for GB |
| **Background Suppression** | Removes pale white-green reflections (S<25, V>215) before feature extraction |
| **Smart Crop Fallback** | If YOLO fails, UV-max sliding window crop ensures a valid engine region |
| **Dedup Protection** | Incremental retraining tracks all previously trained filenames |
| **Augmentation** | 6× augmentation for fixed-camera conditions |

---

## 📦 Requirements

```
opencv-python
ultralytics
scikit-learn
numpy
matplotlib
PyQt5
```

---

## 🏭 Deployment Context

Deployed on standalone factory hardware at a Stellantis engine assembly line.  
Operates in **read-only, offline mode** — no cloud dependency, no data leaves the factory network.

---

## 📝 License

Internal use — Stellantis UV Leak Detection Project
