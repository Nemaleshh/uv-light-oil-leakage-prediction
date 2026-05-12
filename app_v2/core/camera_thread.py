"""
camera_thread.py
QThread that:
  • Opens USB camera and streams frames to UI (~10 fps)
  • Tracks UV light on-time and emits uv_usage_update
  • Manual-only mode: NO auto-capture, NO presence detection
"""
import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


# ═══════════════════════════════════════════════════════════════════ #
#   Camera Scan Thread — enumerates devices without blocking UI
# ═══════════════════════════════════════════════════════════════════ #

class CameraScanThread(QThread):
    """Scans for connected USB cameras in background."""
    scan_complete = pyqtSignal(list)   # list of (index, label) tuples

    MAX_INDEX = 10

    def run(self):
        found = []
        for i in range(self.MAX_INDEX):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap is not None and cap.isOpened():
                    found.append((i, f"Camera {i}"))
                    cap.release()
            except Exception:
                pass
        self.scan_complete.emit(found)


# ═══════════════════════════════════════════════════════════════════ #
#   Main Camera Thread — frame streaming only
# ═══════════════════════════════════════════════════════════════════ #

class CameraThread(QThread):
    """
    Continuously reads from a USB camera and streams frames.
    Manual-capture only — no presence detection or auto-trigger.

    Signals
    -------
    frame_ready(np.ndarray)
        Throttled live frame for the UI feed (~10 fps).

    uv_usage_update(float)
        Seconds the UV light has been continuously on.

    camera_error(str)
        Emitted on irrecoverable camera errors.
    """
    frame_ready     = pyqtSignal(np.ndarray)
    uv_usage_update = pyqtSignal(float)
    camera_error    = pyqtSignal(str)

    FRAME_SKIP = 3   # emit 1 UI frame every N captured frames (~10 fps)

    def __init__(self, cam_index: int = 0, **kwargs):
        super().__init__()
        self.cam_index     = cam_index
        self._running      = False
        self.uv_on_seconds = 0.0
        self._last_time    = None

    # ------------------------------------------------------------------ #
    #  Thread entry point
    # ------------------------------------------------------------------ #

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            self.camera_error.emit(
                f"Cannot open camera at index {self.cam_index}"
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_count = 0

        while self._running:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                else:
                    self.camera_error.emit("Camera lost. Trying to reconnect...")
                    time.sleep(2.0)
                    continue

            ret, frame = cap.read()
            if not ret or frame is None:
                # Frame read failed (USB drop)
                cap.release()
                cap = None
                self.camera_error.emit("Frame read failed — attempting auto-reconnect...")
                time.sleep(1.0)
                continue

            current_time = time.time()
            dt = current_time - (self._last_time or current_time)
            self._last_time = current_time

            # ── UV Light Tracking ──────────────────────────────────────
            if self._is_uv_on(frame):
                self.uv_on_seconds += dt
            else:
                self.uv_on_seconds = 0.0

            # Emit UV usage ~1x per second (UI doesn't need 30fps updates)
            if frame_count % (self.FRAME_SKIP * 10) == 0:
                self.uv_usage_update.emit(self.uv_on_seconds)

            # ── UI frame (throttled) ───────────────────────────────────
            frame_count += 1
            if frame_count % self.FRAME_SKIP == 0:
                self.frame_ready.emit(frame.copy())

        if cap:
            cap.release()

    # ------------------------------------------------------------------ #
    #  Control methods (called from UI thread)
    # ------------------------------------------------------------------ #

    def stop(self):
        self._running = False
        self.wait(3000)

    def _is_uv_on(self, frame: np.ndarray) -> bool:
        """Heuristic: Check if > 2% of the frame has strong violet/purple hues."""
        try:
            small = cv2.resize(frame, (160, 120))
            hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask  = cv2.inRange(hsv, (125, 70, 60), (160, 255, 255))
            return float(np.mean(mask)) > 5.0
        except Exception:
            return False
