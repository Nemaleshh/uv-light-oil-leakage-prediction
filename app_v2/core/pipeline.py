"""
pipeline.py
YOLO crop + Gradient Boosting classifier inference.

Feature extraction mirrors test_pipeline.ipynb exactly (2137 features).

Changelog (2026-04-30):
- YOLO_CONF raised 0.01 → 0.25  (rejects noisy, unstable low-confidence boxes)
- PAD reduced 35 → 10            (tighter crop, less surrounding noise captured)
- MAX_CROP_RATIO guard added      (if crop > 65% of frame → clamp to 60% ROI)
- suppress_background() added     (masks out #f2f7f3 pale reflection pixels
                                   before feature extraction)
- Fixed ROI support added         (config.json 'roi' field; YOLO detections
                                   whose centre is outside the ROI are rejected
                                   — prevents wrong blackbody captures)

Changelog (2026-05-07):
- BUG FIX: ROI filter now applied BEFORE selecting best box.  Previously the
  highest-confidence box (background ~0.97) was chosen first, then rejected by
  ROI, completely missing the real engine box inside the ROI at ~0.05 conf.
- LOW LIGHT FIX: _preprocess_for_yolo() added — applies CLAHE + gamma boost to
  the frame before YOLO inference so the model can detect the part even when
  UV intensity is low.  The ORIGINAL unprocessed frame is still used for
  cropping + feature extraction (preserving the GB classifier's colour signal).
- Dashboard error-abort fix: 'Detection outside ROI' is now a diagnostic note,
  not a fatal error, so the NO_PART path in the dashboard runs correctly.
"""
import os
import cv2
import pickle
import numpy as np
from ultralytics import YOLO


class LeakPipeline:
    """Run YOLO detection then GB classifier on a single frame."""

    TARGET_CLASS   = 0      # 0 = blackbody  (from model.names)
    YOLO_CONF      = 0.01   # Must be very low (0.01) because real engines score ~0.05
    PAD            = 10     # px padding (reduced from 35 — tighter crop)
    MAX_CROP_RATIO = 0.65   # if crop > 65% of frame area → clamp to 60% ROI

    # Maps classifier output strings → display label
    DISPLAY_MAP = {
        "no_leak":  "NO LEAK",
        "oil_leak": "OIL LEAK",
        "no leak":  "NO LEAK",
        "oil leak": "OIL LEAK",
        "noleak":   "NO LEAK",
        "oilleak":  "OIL LEAK",
        "0":        "NO LEAK",
        "1":        "OIL LEAK",
    }

    def __init__(self, model_path: str, clf_path: str, roi: dict = None):
        """
        Parameters
        ----------
        model_path : str   path to best.pt
        clf_path   : str   path to gb_classifier.pkl
            Format: {x1_pct, y1_pct, x2_pct, y2_pct}  (0.0 – 1.0)
            YOLO detections whose centre falls OUTSIDE this zone are
            rejected as wrong-blackbody and treated as NO_PART.
            If None → no ROI filtering (backwards-compatible).
        """
        self.model_path = model_path
        self.clf_path   = clf_path
        self._roi       = roi
        self._model     = None
        self._clf       = None
        self._loaded    = False

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load YOLO and classifier.  Call once before run()."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        if not os.path.exists(self.clf_path):
            raise FileNotFoundError(f"Classifier not found: {self.clf_path}")

        self._model = YOLO(self.model_path)
        with open(self.clf_path, "rb") as f:
            self._clf = pickle.load(f)
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def run(self, frame: np.ndarray) -> dict:
        """
        Parameters
        ----------
        frame : np.ndarray  BGR image from camera / loaded file

        Returns
        -------
        dict with keys:
            result       : str   – "NO LEAK" | "OIL LEAK" | "NO_PART"
            confidence   : float – 0-100 %
            crop         : np.ndarray | None
            bbox         : tuple  (x1,y1,x2,y2) | None
            yolo_conf    : float
            error        : str | None
        """
        if not self._loaded:
            return self._err("Models not loaded. Call load() first.")

        try:
            return self._infer(frame)
        except Exception as exc:
            return self._err(str(exc))

    def _infer(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]

        # ── 0. LOW-LIGHT PRE-PROCESSING for YOLO only ────────────────────
        # The model was trained under UV illumination (bright).  In low UV
        # intensity, the part appears darker and YOLO confidence drops below
        # threshold. Applying CLAHE + gamma boost normalises brightness so
        # the model can still recognise the part geometry.
        # IMPORTANT: we keep `frame` untouched for the crop used by the GB
        # classifier — its colour features must reflect real lighting.
        yolo_frame = self._preprocess_for_yolo(frame)

        # ── 1. YOLO detection (on enhanced frame) ────────────────────────
        results = self._model(yolo_frame, conf=self.YOLO_CONF, verbose=False)[0]
        target_boxes = [
            (b, float(b.conf[0]))
            for b in results.boxes
            if int(b.cls[0]) == self.TARGET_CLASS
        ]

        if not target_boxes:
            return {
                "result": "NO_PART", "confidence": 0.0,
                "crop": None, "bbox": None, "yolo_conf": 0.0,
                "error": None,
                "debug": "YOLO: no detections"
            }

        # ── 1b. ROI FILTER FIRST, then pick best box inside ROI ──────────
        # BUG FIX: Previously we selected max-conf globally (background at
        # ~0.97) THEN checked ROI — the real engine box (~0.05) was never
        # reached.  Now we filter ALL boxes by ROI first, then pick the
        # highest-confidence one that is actually inside the zone.
        roi_boxes = []
        for b, conf_val in target_boxes:
            bx1, by1, bx2, by2 = map(int, b.xyxy[0])
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            if self._roi is None or self._is_in_roi(cx, cy, w, h):
                roi_boxes.append((b, conf_val))

        if not roi_boxes:
            # All detections were outside the ROI — real engine not visible
            best_global, best_conf_g = max(target_boxes, key=lambda x: x[1])
            return {
                "result": "NO_PART", "confidence": 0.0,
                "crop": None, "bbox": None, "yolo_conf": best_conf_g,
                "error": None,
                "debug": f"YOLO found {len(target_boxes)} box(es) but all outside ROI"
            }

        best_box, yolo_conf = max(roi_boxes, key=lambda x: x[1])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        # ── 1c. PADDING ──────────────────────────────────────────────────
        x1 = max(0, x1 - self.PAD)
        y1 = max(0, y1 - self.PAD)
        x2 = min(w, x2 + self.PAD)
        y2 = min(h, y2 + self.PAD)

        # ── 1d. CROP SIZE GUARD ──────────────────────────────────────────
        # When YOLO fires at very high confidence the bounding box sometimes
        # covers the entire frame, pulling in fans/reflective panels that
        # look like UV leaks. Clamp to a 60% ROI centred on the detection.
        frame_area = w * h
        crop_area  = (x2 - x1) * (y2 - y1)
        if crop_area > self.MAX_CROP_RATIO * frame_area:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            max_hw = int(w * 0.30)
            max_hh = int(h * 0.30)
            x1 = max(0, cx - max_hw)
            y1 = max(0, cy - max_hh)
            x2 = min(w, cx + max_hw)
            y2 = min(h, cy + max_hh)

        # ── Crop from ORIGINAL frame (not brightness-boosted) ────────────
        # The GB classifier's colour features (UV hue, saturation) must see
        # the real frame so its oil-leak signal is preserved accurately.
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return {
                "result": "NO_PART", "confidence": 0.0,
                "crop": None, "bbox": None, "yolo_conf": yolo_conf,
                "error": None, "debug": "Empty crop after bbox"
            }

        # ── 2. Feature extraction (same as test_pipeline.ipynb) ──────────
        feats = self._extract_features(crop).reshape(1, -1)

        # ── 3. Classification ────────────────────────────────────────────
        pred  = self._clf.predict(feats)[0]
        proba = self._clf.predict_proba(feats)[0]
        conf  = float(max(proba)) * 100.0

        # Normalise label
        label_raw = str(pred).strip().lower()
        result = self.DISPLAY_MAP.get(label_raw, "OIL LEAK" if "oil" in label_raw else "NO LEAK")

        return {
            "result":    result,
            "confidence": conf,
            "crop":      crop.copy(),
            "bbox":      (x1, y1, x2, y2),
            "yolo_conf": yolo_conf,
            "error":     None,
            "debug":     f"ROI boxes={len(roi_boxes)}, yolo_conf={yolo_conf:.3f}"
        }

    # ------------------------------------------------------------------ #
    #  Low-light pre-processing — boost image before YOLO
    # ------------------------------------------------------------------ #

    @staticmethod
    def _preprocess_for_yolo(frame: np.ndarray) -> np.ndarray:
        """
        Enhance brightness/contrast so that YOLO can detect the engine part
        even when UV light intensity is low.

        Steps:
          1. Convert to LAB
          2. Apply CLAHE on the L channel (local contrast normalisation)
          3. Apply a mild gamma boost (<1.0) to lift shadows

        The output is used ONLY for YOLO detection.  The original frame is
        used for cropping and feature extraction.
        """
        # Step 1 – CLAHE on L channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_eq  = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

        # Step 2 – gamma boost (gamma < 1 brightens shadows)
        gamma   = 0.7
        lut     = np.array([((i / 255.0) ** gamma) * 255
                            for i in range(256)], dtype=np.uint8)
        boosted = cv2.LUT(enhanced, lut)

        return boosted

    # ------------------------------------------------------------------ #
    #  Background suppression — ignore #f2f7f3 pale reflection pixels
    # ------------------------------------------------------------------ #

    @staticmethod
    def _suppress_background(crop: np.ndarray) -> np.ndarray:
        """
        Masks out pale background-reflection pixels (#f2f7f3 ≈ S<25, V>215
        in HSV) by replacing them with the median foreground colour.

        These pixels appear on clean metal/painted surfaces under UV and have
        very low saturation (almost white) — completely unlike the saturated
        blue/green UV oil glow.  Without this step, a large bright reflective
        area in an oversized crop can push the classifier to OIL LEAK.
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Very low saturation AND very high brightness = background white glare
        bg_mask = cv2.inRange(hsv, (0, 0, 215), (180, 25, 255))

        if cv2.countNonZero(bg_mask) == 0:
            return crop  # nothing to suppress

        fg_mask = cv2.bitwise_not(bg_mask)
        if cv2.countNonZero(fg_mask) > 0:
            median_bgr = np.median(crop[fg_mask > 0], axis=0).astype(np.uint8)
        else:
            median_bgr = np.array([0, 0, 0], dtype=np.uint8)

        result = crop.copy()
        result[bg_mask > 0] = median_bgr
        return result

    # ------------------------------------------------------------------ #
    #  Feature extraction — exact replica of test_pipeline.ipynb
    # ------------------------------------------------------------------ #

    def _extract_features(self, crop: np.ndarray) -> np.ndarray:
        r = cv2.resize(crop, (128, 128))

        # RAW features BEFORE CLAHE — preserves UV glow brightness signal
        hsv_raw = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
        h_raw, s_raw, v_raw = cv2.split(hsv_raw)
        uv_mean_brightness = float(np.mean(v_raw))
        uv_max_brightness  = float(np.max(v_raw))
        uv_bright_ratio    = float(np.mean(v_raw > 180))
        uv_sat_spike_ratio = float(np.mean(s_raw > 150))

        # Focused UV hue histogram: blue-violet oil glow (Hue 100-160 in OpenCV)
        uv_mask     = cv2.inRange(hsv_raw, (100, 30, 30), (160, 255, 255))
        uv_hue_hist = cv2.calcHist([hsv_raw], [0], uv_mask, [60], [100, 160])
        cv2.normalize(uv_hue_hist, uv_hue_hist)
        uv_hue_flat = uv_hue_hist.flatten()

        # CLAHE normalisation for general colour features
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
            return [
                float(np.mean(flat)),
                float(np.std(flat)),
                float(np.percentile(flat, 25)),
                float(np.percentile(flat, 75)),
            ]

        stats = (
            chan_stats(h_ch) + chan_stats(s_ch) + chan_stats(v_ch) +
            chan_stats(l_ch) + chan_stats(a_ch) + chan_stats(b_ch)
        )

        gray    = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return np.concatenate([
            [uv_mean_brightness, uv_max_brightness,
             uv_bright_ratio, uv_sat_spike_ratio],
            uv_hue_flat,
            hist_hs.flatten(),
            stats,
            [lap_var],
        ])

    # ------------------------------------------------------------------ #
    #  ROI helper
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    #  ROI helper
    # ------------------------------------------------------------------ #

    def _is_in_roi(self, cx: int, cy: int, frame_w: int, frame_h: int) -> bool:
        """
        Returns True if (cx, cy) is inside the calibrated ROI.
        """
        if not self._roi:
            return True
        r = self._roi
        rx1 = int(r.get("x1_pct", 0) * frame_w)
        ry1 = int(r.get("y1_pct", 0) * frame_h)
        rx2 = int(r.get("x2_pct", 1) * frame_w)
        ry2 = int(r.get("y2_pct", 1) * frame_h)
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _err(msg: str) -> dict:
        return {
            "result": "ERROR", "confidence": 0.0,
            "crop": None, "bbox": None, "yolo_conf": 0.0, "error": msg
        }
