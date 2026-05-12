"""
UV Fluorescent Oil Leak Detector
Detects oil leakage based on the light-blue UV fluorescent oil sheen.

Target leak color (sampled from reference images):
  #85b5f6  →  RGB(133, 181, 246)  →  OpenCV HSV ≈ (107, 117, 246)

Detection range (hue 100–120, OpenCV 0-180 scale):
  - Captures the bright light-blue/sky-blue oil sheen (OCV hue ~107)
  - Covers paler washed-out patches and slightly darker deep-blue areas
  - Excludes lime/yellow-green sticker labels (hue 25–85)
  - Excludes violet/purple UV background reflections (hue 125+)

Black engine surfaces are ignored naturally by the Value threshold.
Tiny single-pixel noise is filtered via min_area.
"""

import cv2
import numpy as np


class UVOilDetector:
    def __init__(self):
        # ── TARGET: light-blue UV fluorescent oil sheen ──────────────────────
        # #85b5f6 (RGB 133,181,246) → OpenCV HSV (107, 117, 246)
        # Primary range — strictly centred on the measured hue, demanding high brightness and moderate saturation
        self.hsv_lower1 = np.array([100,  85, 160], dtype=np.uint8)
        self.hsv_upper1 = np.array([114, 255, 255], dtype=np.uint8)

        # Secondary range — catches slightly more washed-out/paler blue but ONLY if extremely bright
        self.hsv_lower2 = np.array([ 95,  60, 200], dtype=np.uint8)
        self.hsv_upper2 = np.array([115, 255, 255], dtype=np.uint8)

        # ── EXCLUSION: lime/yellow-green sticker labels ────────────────────
        # Hue 25-90 → sticker / green tape false positives
        self.excl_lower = np.array([25,  50,  60], dtype=np.uint8)
        self.excl_upper = np.array([90, 255, 255], dtype=np.uint8)

        # ── EXCLUSION: violet/purple UV background reflections ─────────────
        # Hue 115-165 → chrome edge glare & black plastic reflections
        self.excl_violet_lower = np.array([115,  30,  50], dtype=np.uint8)
        self.excl_violet_upper = np.array([165, 255, 255], dtype=np.uint8)

        self.kernel = np.ones((5, 5), np.uint8)

        # Minimum blob area in px² — raised to ignore reflections
        self.min_area   = 800
        self.min_aspect = 0.15
        self.max_aspect = 8.0

        # Annotation color: RED (BGR) for detected leaks
        self.leak_color = (0, 0, 255)      # Red in BGR  ← was blue/custom
        self.ok_color   = (0, 200, 80)     # Green for no leak

    def detect(self, frame):
        """
        Run UV oil leak detection on a BGR frame using HSV color space.

        Returns:
            leak_detected (bool): Whether a leak was found
            annotated_frame: Frame with bounding boxes drawn
            mask: Binary mask of detected oil regions
            contour_count (int): Number of valid contours found
        """
        annotated = frame.copy()

        # 1. Convert BGR to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. Build detection mask (two overlapping ranges for the light-blue sheen)
        mask1 = cv2.inRange(hsv_frame, self.hsv_lower1, self.hsv_upper1)
        mask2 = cv2.inRange(hsv_frame, self.hsv_lower2, self.hsv_upper2)
        oil_mask = cv2.bitwise_or(mask1, mask2)

        # 3a. Remove lime/yellow-green sticker colors
        excl_mask = cv2.inRange(hsv_frame, self.excl_lower, self.excl_upper)
        oil_mask = cv2.bitwise_and(oil_mask, cv2.bitwise_not(excl_mask))

        # 3b. Remove violet/purple UV background reflections (chrome edges)
        excl_violet = cv2.inRange(hsv_frame,
                                  self.excl_violet_lower, self.excl_violet_upper)
        oil_mask = cv2.bitwise_and(oil_mask, cv2.bitwise_not(excl_violet))

        # 4. Morphological cleanup (remove tiny noise, close small holes)
        oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_CLOSE, self.kernel,
                                    iterations=2)
        oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN,  self.kernel,
                                    iterations=1)

        # 5. Find connected blobs (contours)
        contours, _ = cv2.findContours(oil_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        leak_detected = False
        valid_count   = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue                        # skip tiny single-point noise

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h) if h > 0 else 0

            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            leak_detected = True
            valid_count  += 1

            # Draw RED bounding box around the leak region
            cv2.rectangle(annotated, (x, y), (x + w, y + h),
                          self.leak_color, 2)
            cv2.putText(annotated, f"LEAK {int(area)}px", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.leak_color, 2)

        # Overlay status text (top-left corner)
        if leak_detected:
            status = "UV FLUORESCENT OIL LEAK - NOT OK"
            color  = self.leak_color          # RED
        else:
            status = "NO OIL LEAK - OK"
            color  = self.ok_color            # Green

        cv2.putText(annotated, status, (14, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        # Draw timestamp (bottom-left)
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(annotated, ts, (14, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return leak_detected, annotated, oil_mask, valid_count
