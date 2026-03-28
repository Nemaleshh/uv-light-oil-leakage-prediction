"""
Camera capture thread — grabs frames from IP camera and emits them via Qt signals.
"""

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import time


class CameraThread(QThread):
    frame_ready = pyqtSignal(object)   # emits BGR numpy array
    connection_status = pyqtSignal(bool, str)  # (connected, message)

    def __init__(self, url: str, is_usb: bool = False):
        super().__init__()
        self.url = url
        self.is_usb = is_usb
        self._running = False
        self.cap = None

    def run(self):
        self._running = True
        self.connection_status.emit(False, "Connecting…")

        if self.is_usb:
            try:
                cam_id = int(self.url)
            except ValueError:
                cam_id = 0
            # Use default backend for better compatibility with built-in webcams
            self.cap = cv2.VideoCapture(cam_id)
        else:
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.connection_status.emit(False, "Failed to connect to camera")
            return

        self.connection_status.emit(True, f"Connected: {self.url}")

        consecutive_fails = 0
        while self._running:
            if not self.cap.grab():
                consecutive_fails += 1
                if consecutive_fails > 60:
                    self.connection_status.emit(False, "Camera disconnected — retrying…")
                    self.cap.release()
                    if self.is_usb:
                        try:
                            cam_id = int(self.url)
                        except ValueError:
                            cam_id = 0
                        self.cap = cv2.VideoCapture(cam_id)
                    else:
                        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    consecutive_fails = 0
                continue

            ret, frame = self.cap.retrieve()
            if not ret:
                consecutive_fails += 1
                continue

            consecutive_fails = 0
            
            # Limit to ~30 FPS to prevent UI thread from freezing
            now = time.time()
            if not hasattr(self, '_last_emit_time'):
                self._last_emit_time = 0
                
            if now - self._last_emit_time > 0.033:
                frame = cv2.resize(frame, (640, 480))
                self.frame_ready.emit(frame)
                self._last_emit_time = now

        if self.cap:
            self.cap.release()

    def stop(self):
        self._running = False
        self.wait(3000)
