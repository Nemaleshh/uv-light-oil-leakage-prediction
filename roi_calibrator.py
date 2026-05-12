"""
roi_calibrator.py
=================
One-time tool to define the inspection ROI (Region of Interest).

Since the camera is fixed, the target blackbody is ALWAYS in the same
zone of the frame.  This tool lets you draw that zone once, saves it
to config.json, and from then on pipeline.py will REJECT any YOLO
detection whose center falls outside that zone.

Usage:
    python roi_calibrator.py
    python roi_calibrator.py "path/to/specific/image.jpg"

Controls inside the window:
    - Click + drag  : draw the ROI rectangle (green)
    - R             : reset / redraw
    - SPACE / ENTER : confirm and save ROI to config.json
    - ESC           : exit without saving

After saving, restart the app to apply the new ROI.
"""
import cv2
import json
import os
import sys
import glob

# ── Config path ────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "app_v2", "config.json")

# ── Drawing state ──────────────────────────────────────────────────────────────
drawing   = False
ix, iy    = -1, -1
fx, fy    = -1, -1
roi_ready = False
canvas    = None          # the image shown in the window
base_img  = None          # clean copy (no overlay)


def draw_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, roi_ready, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing   = True
        roi_ready = False
        ix, iy    = x, y
        fx, fy    = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx, fy = x, y
        canvas = base_img.copy()
        cv2.rectangle(canvas, (ix, iy), (fx, fy), (0, 255, 0), 2)
        _draw_instructions(canvas)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing   = False
        roi_ready = True
        fx, fy = x, y
        canvas = base_img.copy()
        x1, y1 = min(ix, fx), min(iy, fy)
        x2, y2 = max(ix, fx), max(iy, fy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, "ROI SET — press SPACE to save, R to redo",
                    (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        _draw_instructions(canvas)


def _draw_instructions(img):
    lines = [
        "Draw ROI: click + drag",
        "R = reset  |  SPACE = save  |  ESC = cancel",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)


def grab_frame(camera_index: int = 0) -> "np.ndarray | None":
    """Try to grab one frame from the camera."""
    import numpy as np
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    frame = None
    for _ in range(5):                    # discard a few frames to let camera warm up
        ret, f = cap.read()
        if ret:
            frame = f
    cap.release()
    return frame


def load_sample_image() -> "np.ndarray | None":
    """Fall back to the first available image in detected_images/ or new_pipeline/crops/."""
    import numpy as np
    search_dirs = [
        os.path.join(BASE, "app_v2", "detected_images"),
        os.path.join(BASE, "new_pipeline", "crops", "oil_leak"),
        os.path.join(BASE, "new_pipeline", "crops", "no_leak"),
        os.path.join(BASE, "..", "summary", "data", "engine_oil"),
        os.path.join(BASE, "..", "summary", "data", "no_leak"),
    ]
    for d in search_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files = glob.glob(os.path.join(d, ext))
            if files:
                img = cv2.imread(files[0])
                if img is not None:
                    print(f"  Using sample image: {files[0]}")
                    return img
    return None


def save_roi(x1, y1, x2, y2, frame_w, frame_h):
    """Save ROI as percentages of frame dimensions to config.json."""
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: config.json not found at {CONFIG_PATH}")
        return False

    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    cfg["roi"] = {
        "x1_pct": round(x1 / frame_w, 4),
        "y1_pct": round(y1 / frame_h, 4),
        "x2_pct": round(x2 / frame_w, 4),
        "y2_pct": round(y2 / frame_h, 4),
        "note":   "Drawn by roi_calibrator.py — YOLO detections outside this zone are rejected"
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n  ROI saved to config.json:")
    print(f"    Pixels  : ({x1},{y1}) → ({x2},{y2})")
    print(f"    %% frame : x={cfg['roi']['x1_pct']:.1%}–{cfg['roi']['x2_pct']:.1%}"
          f"  y={cfg['roi']['y1_pct']:.1%}–{cfg['roi']['y2_pct']:.1%}")
    return True


def main():
    global base_img, canvas, roi_ready

    print("=" * 55)
    print("   UV Inspection — ROI Calibrator")
    print("=" * 55)

    frame = None

    # 1. Check if user provided an image path
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if os.path.exists(img_path):
            frame = cv2.imread(img_path)
            if frame is not None:
                print(f"\n  Loaded user image: {img_path}")
            else:
                print(f"\nERROR: Could not read image at {img_path}")
        else:
            print(f"\nERROR: File not found: {img_path}")

    # 2. If no valid user image, try camera
    if frame is None:
        camera_index = 0
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            camera_index = cfg.get("camera_index", 0)

        print(f"\nOpening camera {camera_index}…")
        frame = grab_frame(camera_index)

    # 3. If camera fails, try sample images
    if frame is None:
        print("  Camera not available — trying sample images…")
        frame = load_sample_image()

    if frame is None:
        print("ERROR: No camera and no sample images found. Cannot calibrate.")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"  Frame size: {w} x {h}")

    # If ROI already set, show it
    existing_roi = None
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        existing_roi = cfg.get("roi")
        if existing_roi:
            ex1 = int(existing_roi["x1_pct"] * w)
            ey1 = int(existing_roi["y1_pct"] * h)
            ex2 = int(existing_roi["x2_pct"] * w)
            ey2 = int(existing_roi["y2_pct"] * h)
            print(f"\n  Existing ROI found: ({ex1},{ey1}) → ({ex2},{ey2})")
            print("  Draw a new rectangle to replace it, or press SPACE to keep.")

    base_img = frame.copy()
    if existing_roi:
        ex1 = int(existing_roi["x1_pct"] * w)
        ey1 = int(existing_roi["y1_pct"] * h)
        ex2 = int(existing_roi["x2_pct"] * w)
        ey2 = int(existing_roi["y2_pct"] * h)
        cv2.rectangle(base_img, (ex1, ey1), (ex2, ey2), (0, 165, 255), 2)
        cv2.putText(base_img, "Existing ROI (orange)",
                    (ex1, ey1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    canvas = base_img.copy()
    _draw_instructions(canvas)

    cv2.namedWindow("ROI Calibrator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Calibrator", min(w, 1280), min(h, 720))
    cv2.setMouseCallback("ROI Calibrator", draw_callback)

    print("\n  Window opened — draw the inspection zone rectangle.")
    print("  SPACE/ENTER = save  |  R = reset  |  ESC = cancel\n")

    while True:
        cv2.imshow("ROI Calibrator", canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:                       # ESC — cancel
            print("Cancelled — no changes saved.")
            break

        elif key in (32, 13):              # SPACE or ENTER — save
            if roi_ready and abs(fx - ix) > 20 and abs(fy - iy) > 20:
                x1, y1 = min(ix, fx), min(iy, fy)
                x2, y2 = max(ix, fx), max(iy, fy)
                if save_roi(x1, y1, x2, y2, w, h):
                    print("\n✅ ROI saved! Restart the app to apply.\n")
            elif existing_roi:
                print("Keeping existing ROI — no changes.")
            else:
                print("No ROI drawn yet. Draw a rectangle first.")
            break

        elif key == ord("r") or key == ord("R"):
            roi_ready = False
            canvas    = frame.copy()
            if existing_roi:
                ex1 = int(existing_roi["x1_pct"] * w)
                ey1 = int(existing_roi["y1_pct"] * h)
                ex2 = int(existing_roi["x2_pct"] * w)
                ey2 = int(existing_roi["y2_pct"] * h)
                cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (0, 165, 255), 2)
            base_img = canvas.copy()
            _draw_instructions(canvas)
            print("  Reset — draw a new ROI.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
