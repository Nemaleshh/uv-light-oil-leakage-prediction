"""
Main IoT Dashboard — UV Engine Oil Leak Detection System
Changes v2:
  - One-shot inspection: 5s countdown → single frame capture → auto-stop
  - Camera LIVE / IDLE / SCANNING badges on feed
  - "LAST RESULT" replaces "LEAKS THIS CAR" stat card
  - Full Python 3.9-compatible type hints (Optional instead of X | Y)
  - Error-safe report generation
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QProgressBar, QStatusBar, QSizePolicy,
    QHeaderView, QMessageBox, QFrame, QSplitter, QComboBox,
    QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QSize
from PyQt5.QtGui import QImage, QPixmap, QColor

from app.ui.camera_thread import CameraThread
from app.core.detector import UVOilDetector
from app.core.reporter import ExcelReporter

CONFIG_FILE = "config.json"
SNAP_DIR    = "detected_images"
os.makedirs(SNAP_DIR, exist_ok=True)


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"camera_url": ""}


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


# ================================================================ #

class Dashboard(QMainWindow):
    COUNTDOWN_SECS = 5          # gap between Start click and capture

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UV Engine Oil Leak Detection System — Industrial IoT v2")
        self.setMinimumSize(1300, 840)

        self.config   = load_config()
        self.detector = UVOilDetector()
        self.reporter = ExcelReporter()

        self.cam_thread: Optional[CameraThread] = None
        self._cam_connected = False

        # Inspection state
        self._phase         = "IDLE"    # IDLE | COUNTDOWN | CAPTURE | DONE
        self._countdown_val = 0
        self._current_win   = ""
        self._latest_frame: Optional[np.ndarray] = None
        self._last_snap_path: Optional[str]      = None

        # Session stats
        self._total_inspected = 0
        self._total_passes    = 0
        self._total_fails     = 0
        self._last_result     = "—"

        # FPS tracking
        self._frame_count    = 0
        self._fps_ts: Optional[datetime] = None

        self._build_ui()

        # 1-second clock + LED blink
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick)
        self._clock_timer.start(1000)

        # LED blink state
        self._led_blink = False

        # Countdown timer (fires every 1 s)
        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._countdown_tick)

        if self.config.get("camera_type") == "USB":
            self.cam_type_combo.setCurrentText("USB Web Camera")
            if self.config.get("camera_id") is not None:
                self.cam_url_input.setText(str(self.config["camera_id"]))
        else:
            self.cam_type_combo.setCurrentText("IPv4 IP Camera")
            if self.config.get("camera_url"):
                self.cam_url_input.setText(self.config["camera_url"])

    # ================================================================ #
    #   UI CONSTRUCTION
    # ================================================================ #

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vlay = QVBoxLayout(root)
        vlay.setContentsMargins(10, 8, 10, 8)
        vlay.setSpacing(8)

        vlay.addWidget(self._make_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.addWidget(self._make_left())
        splitter.addWidget(self._make_centre())
        splitter.addWidget(self._make_right())
        splitter.setSizes([280, 750, 270])
        vlay.addWidget(splitter, stretch=1)

        vlay.addWidget(self._make_stats_bar())
        vlay.addWidget(self._make_log_table())

        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._sb.showMessage("Ready — enter camera URL and WIN number to begin")

    # ---- Header ---- #
    def _make_header(self):
        f = QFrame()
        f.setFixedHeight(66)
        f.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #060614,stop:0.5 #0a1830,stop:1 #060614);"
            "border-bottom:2px solid #00558a;"
        )
        h = QHBoxLayout(f)
        h.setContentsMargins(18, 4, 18, 4)

        title = QLabel("⚙   UV ENGINE OIL LEAK DETECTION SYSTEM")
        title.setObjectName("header_title")
        h.addWidget(title)
        h.addStretch()

        # Camera status badge
        self._cam_badge = QLabel("● CAM OFFLINE")
        self._cam_badge.setStyleSheet(
            "color:#ff3333; font-size:12px; font-weight:bold;"
            "background:#1a0000; border:1px solid #550000;"
            "border-radius:4px; padding:3px 10px;"
        )
        h.addWidget(self._cam_badge)

        # Inspection status badge
        self._insp_badge = QLabel("  IDLE  ")
        self._insp_badge.setStyleSheet(
            "color:#778899; font-size:12px; font-weight:bold;"
            "background:#111122; border:1px solid #223344;"
            "border-radius:4px; padding:3px 10px;"
        )
        h.addWidget(self._insp_badge)

        sep = QLabel("  |  ")
        sep.setStyleSheet("color:#334455;")
        h.addWidget(sep)

        self._clock_lbl = QLabel()
        self._clock_lbl.setObjectName("clock_label")
        self._clock_lbl.setText(datetime.now().strftime("  %Y-%m-%d   %H:%M:%S"))
        h.addWidget(self._clock_lbl)
        return f

    # ---- Left Panel ---- #
    def _make_left(self):
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(4, 4, 4, 4)
        ly.setSpacing(10)

        # --- Camera ---
        cg = QGroupBox("📡  CAMERA SETUP")
        cl = QVBoxLayout(cg)
        
        self.cam_type_combo = QComboBox()
        self.cam_type_combo.addItems(["IPv4 IP Camera", "USB Web Camera"])
        self.cam_type_combo.currentTextChanged.connect(self._on_cam_type_changed)
        cl.addWidget(self._hint("Select camera source type:"))
        cl.addWidget(self.cam_type_combo)
        
        self.url_hint_lbl = self._hint("IP Camera URL — saved automatically")
        cl.addWidget(self.url_hint_lbl)
        self.cam_url_input = QLineEdit()
        self.cam_url_input.setPlaceholderText("http://192.168.x.x:8080/video")
        cl.addWidget(self.cam_url_input)

        br = QHBoxLayout()
        self.btn_connect = QPushButton("🔗  Connect")
        self.btn_connect.clicked.connect(self._on_connect)
        br.addWidget(self.btn_connect)
        self.btn_disconnect = QPushButton("✖  Disconnect")
        self.btn_disconnect.setEnabled(False)
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        br.addWidget(self.btn_disconnect)
        cl.addLayout(br)

        # Camera live indicator strip
        self._cam_strip = QLabel("  NO SIGNAL  ")
        self._cam_strip.setAlignment(Qt.AlignCenter)
        self._cam_strip.setStyleSheet(
            "background:#110000; color:#440000; font-size:11px; font-weight:bold;"
            "border-radius:4px; padding:4px; letter-spacing:2px;"
        )
        cl.addWidget(self._cam_strip)
        ly.addWidget(cg)

        # --- WIN Entry ---
        wg = QGroupBox("🚗  VEHICLE WIN / VIN NUMBER")
        wl = QVBoxLayout(wg)
        wl.addWidget(self._hint("Scan barcode or type — press Enter to start"))
        self.win_input = QLineEdit()
        self.win_input.setPlaceholderText("WIN-2024-001  or scan barcode…")
        self.win_input.returnPressed.connect(self._on_start)
        wl.addWidget(self.win_input)

        self.btn_start = QPushButton("▶   START INSPECTION")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        wl.addWidget(self.btn_start)

        # Countdown display inside WIN box
        self._countdown_lbl = QLabel("")
        self._countdown_lbl.setAlignment(Qt.AlignCenter)
        self._countdown_lbl.setStyleSheet(
            "color:#ffaa00; font-size:30px; font-weight:bold;"
        )
        wl.addWidget(self._countdown_lbl)

        ly.addWidget(wg)

        # --- Reports ---
        rg = QGroupBox("📊  REPORTS")
        rl = QVBoxLayout(rg)
        self.btn_report = QPushButton("📥  Generate / Open Report")
        self.btn_report.setObjectName("btn_report")
        self.btn_report.clicked.connect(self._on_report)
        rl.addWidget(self.btn_report)

        self.btn_test_image = QPushButton("🖼  Test Image (Upload)")
        self.btn_test_image.setObjectName("btn_test_image")
        self.btn_test_image.setToolTip("Upload an image file and run the oil leak detector on it")
        self.btn_test_image.clicked.connect(self._on_test_image)
        rl.addWidget(self.btn_test_image)

        self._report_lbl = QLabel("No report generated yet")
        self._report_lbl.setStyleSheet("font-size:10px; color:#445566;")
        self._report_lbl.setWordWrap(True)
        rl.addWidget(self._report_lbl)
        ly.addWidget(rg)

        ly.addStretch()
        return w

    # ---- Centre Panel ---- #
    def _make_centre(self):
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(4, 4, 4, 4)
        ly.setSpacing(6)

        fg = QGroupBox("📷  LIVE CAMERA FEED")
        fl = QVBoxLayout(fg)
        fl.setContentsMargins(2, 2, 2, 2)  # smaller border gap

        # Container for the feed and the overlay
        feed_container = QWidget()
        feed_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # We need a layout that allows overlaying. Simplest is to make the feed_lbl stretch, 
        # and place the overlay as a child of feed_lbl.
        fc_lay = QVBoxLayout(feed_container)
        fc_lay.setContentsMargins(0, 0, 0, 0)
        fc_lay.setSpacing(0)
        
        self.feed_lbl = QLabel()
        self.feed_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.feed_lbl.setAlignment(Qt.AlignCenter)
        self.feed_lbl.setStyleSheet(
            "background:#050510; border:2px solid #1e3a5f; border-radius:4px;"
        )
        self.feed_lbl.setText("No camera feed\nConnect camera to begin")
        fc_lay.addWidget(self.feed_lbl)

        # Overlay label (child of feed_lbl)
        self._leak_overlay = QLabel(self.feed_lbl)
        self._leak_overlay.setText("OIL LEAK DETECTED")
        self._leak_overlay.setAlignment(Qt.AlignCenter)
        self._leak_overlay.setStyleSheet(
            "color: white; background: rgba(255, 0, 0, 180);"
            "font-size: 32px; font-weight: bold; border-radius: 4px;"
            "padding: 10px;"
        )
        self._leak_overlay.setVisible(False)
        # Position will be updated in resizeEvent

        fl.addWidget(feed_container)

        bottom_row = QHBoxLayout()
        self._fps_lbl = QLabel("FPS: --")
        self._fps_lbl.setStyleSheet("font-size:10px; color:#334455;")
        bottom_row.addWidget(self._fps_lbl)
        bottom_row.addStretch()

        # Live / Recording indicator
        self._live_badge = QLabel("  ● LIVE  ")
        self._live_badge.setStyleSheet(
            "color:#00ff88; font-size:11px; font-weight:bold;"
            "background:#001a00; border:1px solid #004400;"
            "border-radius:4px; padding:2px 8px;"
        )
        self._live_badge.setVisible(False)
        bottom_row.addWidget(self._live_badge)

        # Scanning badge
        self._scan_badge = QLabel("  🔍 SCANNING…  ")
        self._scan_badge.setStyleSheet(
            "color:#ffaa00; font-size:11px; font-weight:bold;"
            "background:#1a1000; border:1px solid #443300;"
            "border-radius:4px; padding:2px 8px;"
        )
        self._scan_badge.setVisible(False)
        bottom_row.addWidget(self._scan_badge)

        fl.addLayout(bottom_row)
        ly.addWidget(fg, stretch=1)
        return w

    # ---- Right Panel ---- #
    def _make_right(self):
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(4, 4, 4, 4)
        ly.setSpacing(10)

        # Detection result
        dg = QGroupBox("🔍  DETECTION RESULT")
        dl = QVBoxLayout(dg)
        dl.setAlignment(Qt.AlignCenter)
        self._big_status = QLabel("IDLE")
        self._big_status.setAlignment(Qt.AlignCenter)
        self._big_status.setWordWrap(True)
        self._big_status.setMinimumHeight(80)
        self._big_status.setStyleSheet("color:#778899; font-size:20px; font-weight:bold;")
        dl.addWidget(self._big_status)
        self._win_disp = QLabel("WIN: —")
        self._win_disp.setStyleSheet("color:#445566; font-size:11px;")
        self._win_disp.setAlignment(Qt.AlignCenter)
        dl.addWidget(self._win_disp)
        ly.addWidget(dg)

        # Snapshot
        sg = QGroupBox("📸  LAST CAPTURE")
        sl = QVBoxLayout(sg)
        self._snap_lbl = QLabel()
        self._snap_lbl.setFixedSize(270, 165)
        self._snap_lbl.setAlignment(Qt.AlignCenter)
        self._snap_lbl.setStyleSheet(
            "background:#050510; border:1px solid #1e3a5f; border-radius:4px;"
        )
        self._snap_lbl.setText("No snapshot yet")
        sl.addWidget(self._snap_lbl, alignment=Qt.AlignCenter)
        self._snap_info = QLabel("—")
        self._snap_info.setStyleSheet("font-size:10px; color:#445566;")
        self._snap_info.setAlignment(Qt.AlignCenter)
        sl.addWidget(self._snap_info)

        # Download last captured image button
        self._btn_download = QPushButton("💾  Download Last Image")
        self._btn_download.setObjectName("btn_download")
        self._btn_download.setEnabled(False)
        self._btn_download.setToolTip("Save the last captured inspection image to a chosen location")
        self._btn_download.clicked.connect(self._on_download_snap)
        sl.addWidget(self._btn_download)

        ly.addWidget(sg)

        # Mask
        mg = QGroupBox("🟢  UV FLUORESCENT MASK")
        ml = QVBoxLayout(mg)
        self._mask_lbl = QLabel()
        self._mask_lbl.setFixedSize(270, 145)
        self._mask_lbl.setAlignment(Qt.AlignCenter)
        self._mask_lbl.setStyleSheet(
            "background:#050510; border:1px solid #1e3a5f; border-radius:4px;"
        )
        self._mask_lbl.setText("Mask preview")
        ml.addWidget(self._mask_lbl, alignment=Qt.AlignCenter)
        ly.addWidget(mg)

        ly.addStretch()
        return w

    # ---- Stats Bar ---- #
    def _make_stats_bar(self):
        f = QFrame()
        f.setFixedHeight(110)
        f.setStyleSheet(
            "background:#0d0d20; border:1px solid #1e3a5f; border-radius:6px;"
        )
        h = QHBoxLayout(f)
        h.setContentsMargins(16, 8, 16, 8)
        h.setSpacing(0)

        def card(attr, label, color):
            c = QWidget()
            c.setStyleSheet("border:none;")
            vl = QVBoxLayout(c)
            vl.setSpacing(2)
            vl.setContentsMargins(16, 4, 16, 4)
            num = QLabel("0")
            num.setStyleSheet(f"color:{color}; font-size:28px; font-weight:bold; border:none;")
            num.setAlignment(Qt.AlignCenter)
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#556677; font-size:10px; letter-spacing:1px; border:none;")
            lbl.setAlignment(Qt.AlignCenter)
            vl.addWidget(num)
            vl.addWidget(lbl)
            setattr(self, attr, num)
            return c

        def vsep():
            s = QFrame()
            s.setFrameShape(QFrame.VLine)
            s.setStyleSheet("background:#1e3a5f;")
            s.setFixedWidth(1)
            return s

        h.addWidget(card("_stat_total",   "CARS INSPECTED",   "#00c8ff"))
        h.addWidget(vsep())
        h.addWidget(card("_stat_pass",    "PASSED (OK)",      "#00ff88"))
        h.addWidget(vsep())
        h.addWidget(card("_stat_fail",    "FAILED (LEAK)",    "#ff3333"))
        h.addWidget(vsep())

        # Pass rate
        pr = QWidget()
        pr.setStyleSheet("border:none;")
        prl = QVBoxLayout(pr)
        prl.setSpacing(4)
        prl.setContentsMargins(16, 4, 16, 4)
        pr_title = QLabel("PASS RATE")
        pr_title.setStyleSheet("color:#556677; font-size:10px; letter-spacing:1px; border:none;")
        pr_title.setAlignment(Qt.AlignCenter)
        self._rate_lbl = QLabel("— %")
        self._rate_lbl.setStyleSheet("color:#00ff88; font-size:22px; font-weight:bold; border:none;")
        self._rate_lbl.setAlignment(Qt.AlignCenter)
        self._rate_bar = QProgressBar()
        self._rate_bar.setRange(0, 100)
        self._rate_bar.setValue(0)
        self._rate_bar.setFixedHeight(14)
        self._rate_bar.setTextVisible(False)
        prl.addWidget(pr_title)
        prl.addWidget(self._rate_lbl)
        prl.addWidget(self._rate_bar)
        h.addWidget(pr)
        h.addWidget(vsep())

        # Current WIN
        wc = QWidget()
        wc.setStyleSheet("border:none;")
        wcl = QVBoxLayout(wc)
        wcl.setSpacing(2)
        wcl.setContentsMargins(16, 4, 16, 4)
        wt = QLabel("CURRENT WIN")
        wt.setStyleSheet("color:#556677; font-size:10px; letter-spacing:1px; border:none;")
        wt.setAlignment(Qt.AlignCenter)
        self._stat_win = QLabel("—")
        self._stat_win.setStyleSheet("color:#ffaa00; font-size:18px; font-weight:bold; border:none;")
        self._stat_win.setAlignment(Qt.AlignCenter)
        wcl.addWidget(wt)
        wcl.addWidget(self._stat_win)
        h.addWidget(wc)
        h.addWidget(vsep())

        # Last Result
        lc = QWidget()
        lc.setStyleSheet("border:none;")
        lcl = QVBoxLayout(lc)
        lcl.setSpacing(2)
        lcl.setContentsMargins(16, 4, 16, 4)
        lt = QLabel("LAST RESULT")
        lt.setStyleSheet("color:#556677; font-size:10px; letter-spacing:1px; border:none;")
        lt.setAlignment(Qt.AlignCenter)
        self._stat_last = QLabel("—")
        self._stat_last.setStyleSheet("color:#778899; font-size:16px; font-weight:bold; border:none;")
        self._stat_last.setAlignment(Qt.AlignCenter)
        lcl.addWidget(lt)
        lcl.addWidget(self._stat_last)
        h.addWidget(lc)

        return f

    # ---- Session Log ---- #
    def _make_log_table(self):
        g = QGroupBox("📋  SESSION INSPECTION LOG")
        g.setMaximumHeight(200)
        ly = QVBoxLayout(g)
        self._log = QTableWidget(0, 5)
        self._log.setHorizontalHeaderLabels(
            ["#", "WIN / VIN", "Timestamp", "Detection Result", "Image Saved"]
        )
        self._log.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._log.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self._log.setColumnWidth(0, 40)
        self._log.setAlternatingRowColors(True)
        self._log.setEditTriggers(QTableWidget.NoEditTriggers)
        self._log.verticalHeader().setVisible(False)
        self._log.setSelectionBehavior(QTableWidget.SelectRows)
        ly.addWidget(self._log)
        return g

    def _hint(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size:10px; color:#557799;")
        return lbl

    # ================================================================ #
    #   TIMERS
    # ================================================================ #

    def _tick(self):
        """Every 1 second — clock + LED blink."""
        self._clock_lbl.setText(datetime.now().strftime("  %Y-%m-%d   %H:%M:%S"))

        # Blink LED badge when camera is connected
        if self._cam_connected:
            self._led_blink = not self._led_blink
            dot = "●" if self._led_blink else "○"
            self._cam_badge.setText(f"{dot} CAM LIVE")
            self._cam_badge.setStyleSheet(
                f"color:{'#00ff88' if self._led_blink else '#00aa55'}; font-size:12px; font-weight:bold;"
                "background:#001a00; border:1px solid #005500;"
                "border-radius:4px; padding:3px 10px;"
            )

    def _countdown_tick(self):
        """Fires every second during countdown."""
        self._countdown_val -= 1
        if self._countdown_val > 0:
            self._countdown_lbl.setText(f"Capturing in  {self._countdown_val}s…")
            self._big_status.setText(f"⏱  WAIT {self._countdown_val}s")
            self._big_status.setStyleSheet("color:#ffaa00; font-size:22px; font-weight:bold;")
        else:
            self._countdown_timer.stop()
            self._countdown_lbl.setText("📷  CAPTURING…")
            self._big_status.setText("📷 CAPTURING")
            self._big_status.setStyleSheet("color:#44aaff; font-size:22px; font-weight:bold;")
            self._scan_badge.setText("  📷 CAPTURING…  ")
            self._phase = "CAPTURE"
            # The next frame that arrives in _on_frame will be captured

    # ================================================================ #
    #   CAMERA SLOTS
    # ================================================================ #

    def _on_cam_type_changed(self, text):
        if text == "USB Web Camera":
            self.url_hint_lbl.setText("USB Camera ID (e.g., 0)")
            self.cam_url_input.setPlaceholderText("0")
            if self.config.get("camera_type") == "USB" and self.config.get("camera_id") is not None:
                self.cam_url_input.setText(str(self.config["camera_id"]))
            else:
                self.cam_url_input.clear()
        else:
            self.url_hint_lbl.setText("IP Camera URL")
            self.cam_url_input.setPlaceholderText("http://192.168.x.x:8080/video")
            if self.config.get("camera_type") != "USB" and self.config.get("camera_url"):
                self.cam_url_input.setText(self.config["camera_url"])
            else:
                self.cam_url_input.clear()

    @pyqtSlot()
    def _on_connect(self):
        val = self.cam_url_input.text().strip()
        is_usb = self.cam_type_combo.currentText() == "USB Web Camera"
        
        if not val:
            QMessageBox.warning(self, "Missing Info", "Please enter the camera URL or ID.")
            return
            
        if is_usb:
            self.config["camera_type"] = "USB"
            try:
                self.config["camera_id"] = int(val)
            except ValueError:
                self.config["camera_id"] = 0
                val = "0"
                self.cam_url_input.setText("0")
        else:
            self.config["camera_type"] = "IPv4"
            self.config["camera_url"] = val
            
        save_config(self.config)
        if self.cam_thread:
            self.cam_thread.stop()

        self.cam_thread = CameraThread(val, is_usb=is_usb)
        self.cam_thread.frame_ready.connect(self._on_frame)
        self.cam_thread.connection_status.connect(self._on_cam_status)
        self.cam_thread.start()

        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(True)
        self._sb.showMessage(f"Connecting to {val}…")

    @pyqtSlot()
    def _on_disconnect(self):
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread = None
        self._cam_connected = False
        self._phase = "IDLE"
        self._countdown_timer.stop()
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.btn_start.setEnabled(False)
        self._set_cam_offline()
        self.feed_lbl.setText("Camera disconnected")
        self._sb.showMessage("Camera disconnected")

    @pyqtSlot(bool, str)
    def _on_cam_status(self, connected: bool, msg: str):
        self._cam_connected = connected
        if connected:
            self._cam_strip.setText("  ● CAMERA LIVE  ")
            self._cam_strip.setStyleSheet(
                "background:#001800; color:#00ff88; font-size:11px; font-weight:bold;"
                "border-radius:4px; padding:4px; letter-spacing:2px;"
            )
            self._live_badge.setVisible(True)
            self.btn_start.setEnabled(True)
        else:
            self._set_cam_offline()
        self._sb.showMessage(msg)

    def _set_cam_offline(self):
        self._cam_badge.setText("● CAM OFFLINE")
        self._cam_badge.setStyleSheet(
            "color:#ff3333; font-size:12px; font-weight:bold;"
            "background:#1a0000; border:1px solid #550000;"
            "border-radius:4px; padding:3px 10px;"
        )
        self._cam_strip.setText("  NO SIGNAL  ")
        self._cam_strip.setStyleSheet(
            "background:#110000; color:#440000; font-size:11px; font-weight:bold;"
            "border-radius:4px; padding:4px; letter-spacing:2px;"
        )
        self._live_badge.setVisible(False)
        self._scan_badge.setVisible(False)

    # ================================================================ #
    #   INSPECTION LOGIC (one-shot with 5s delay)
    # ================================================================ #

    @pyqtSlot()
    def _on_start(self):
        win = self.win_input.text().strip()
        if not win:
            QMessageBox.warning(self, "No WIN", "Please enter the vehicle WIN number first.")
            return
        if not self._cam_connected:
            QMessageBox.warning(self, "No Camera", "Please connect the camera first.")
            return
        if self._phase != "IDLE":
            return  # already running

        self._current_win = win
        self._phase = "COUNTDOWN"
        self._countdown_val = self.COUNTDOWN_SECS
        
        # Reset overlay
        self._leak_overlay.setVisible(False)

        # UI update
        self._stat_win.setText(win)
        self._win_disp.setText(f"WIN:  {win}")
        self.btn_start.setEnabled(False)
        self.win_input.setEnabled(False)

        # Inspection badge
        self._insp_badge.setText("  SCANNING  ")
        self._insp_badge.setStyleSheet(
            "color:#ffaa00; font-size:12px; font-weight:bold;"
            "background:#1a1000; border:1px solid #664400;"
            "border-radius:4px; padding:3px 10px;"
        )
        self._scan_badge.setText("  🔍 SCANNING…  ")
        self._scan_badge.setVisible(True)

        # Start countdown
        self._countdown_lbl.setText(f"Capturing in  {self._countdown_val}s…")
        self._big_status.setText(f"⏱  WAIT {self._countdown_val}s")
        self._big_status.setStyleSheet("color:#ffaa00; font-size:22px; font-weight:bold;")
        self._countdown_timer.start(1000)
        self._sb.showMessage(f"Inspection started for WIN: {win} — capturing in {self.COUNTDOWN_SECS}s…")

    def _do_capture(self, frame: np.ndarray):
        """Called once when phase=CAPTURE — run detection on this single frame."""
        self._phase = "DONE"
        self._countdown_timer.stop()

        leak, annotated, mask, _ = self.detector.detect(frame)

        # Save snapshot always (even if no leak — for records)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = os.path.join(SNAP_DIR, f"WIN_{self._current_win}_{ts}.jpg")
        cv2.imwrite(snap_path, annotated)
        self._last_snap_path = snap_path

        # Show on UI
        self._show_snap(annotated)
        self._show_mask(mask)
        self._snap_info.setText(os.path.basename(snap_path))
        self._btn_download.setEnabled(True)   # enable download now

        # Result
        status = "NOT OK" if leak else "OK"
        
        # Display overlay on live feed
        if leak:
            self._leak_overlay.setVisible(True)
            # Center the overlay
            self._leak_overlay.adjustSize()
            cx = (self.feed_lbl.width() - self._leak_overlay.width()) // 2
            cy = (self.feed_lbl.height() - self._leak_overlay.height()) // 2
            self._leak_overlay.move(cx, cy)
        else:
            self._leak_overlay.setVisible(False)
        if leak:
            self._big_status.setText("⚠  OIL LEAK\nDETECTED — NOT OK")
            self._big_status.setStyleSheet("color:#ff3333; font-size:18px; font-weight:bold;")
        else:
            self._big_status.setText("✅  NO OIL LEAK\n— OK")
            self._big_status.setStyleSheet("color:#00ff88; font-size:18px; font-weight:bold;")

        # Log record
        self.reporter.add_record(
            win_number=self._current_win,
            status=status,
            image_path=snap_path,
        )
        self._total_inspected += 1
        if leak:
            self._total_fails += 1
            self._last_result = "NOT OK"
        else:
            self._total_passes += 1
            self._last_result = "OK"
        self._update_stats()
        self._add_log_row(self._current_win, status, snap_path)

        # Auto-generate report
        try:
            path = self.reporter.generate()
            self._report_lbl.setText(f"Auto-saved:\n{path}")
        except Exception as ex:
            self._report_lbl.setText(f"Report error: {ex}")

        self._finish_inspection(status)

    def _finish_inspection(self, status: str):
        """Reset UI back to IDLE ready for next car."""
        # Badges
        insp_color = "#ff3333" if status == "NOT OK" else "#00ff88"
        insp_bg    = "#1a0000" if status == "NOT OK" else "#001a00"
        insp_bdr   = "#550000" if status == "NOT OK" else "#005500"
        self._insp_badge.setText(f"  {status}  ")
        self._insp_badge.setStyleSheet(
            f"color:{insp_color}; font-size:12px; font-weight:bold;"
            f"background:{insp_bg}; border:1px solid {insp_bdr};"
            "border-radius:4px; padding:3px 10px;"
        )
        self._scan_badge.setVisible(False)
        self._countdown_lbl.setText("")
        # Keep overlay visible until a new start

        # Re-enable inputs
        self.win_input.setEnabled(True)
        self.win_input.setText("")         # auto-clear for next car
        self.win_input.setFocus()
        self.btn_start.setEnabled(True)
        self._stat_win.setText("—")
        self._phase = "IDLE"

        self._sb.showMessage(
            f"✔ Inspection COMPLETE  |  WIN: {self._current_win}  |  Result: {status}"
        )

    # ================================================================ #
    #   FRAME SLOT
    # ================================================================ #

    @pyqtSlot(object)
    def _on_frame(self, frame: np.ndarray):
        # FPS
        self._frame_count += 1
        now = datetime.now()
        if self._fps_ts is None:
            self._fps_ts = now
        else:
            elapsed = (now - self._fps_ts).total_seconds()
            if elapsed >= 1.0:
                self._fps_lbl.setText(f"FPS: {self._frame_count / elapsed:.1f}")
                self._frame_count = 0
                self._fps_ts = now

        # Always show live feed
        self._show_feed(frame)
        self._latest_frame = frame

        # One-shot capture trigger
        if self._phase == "CAPTURE":
            self._do_capture(frame)

    # ================================================================ #
    #   REPORT
    # ================================================================ #

    @pyqtSlot()
    def _on_download_snap(self):
        """Let the user choose where to save the last captured high-quality image."""
        if not self._last_snap_path or not os.path.exists(self._last_snap_path):
            QMessageBox.warning(self, "No Image",
                                "No captured image available yet.\n"
                                "Run an inspection first.")
            return

        default_name = os.path.basename(self._last_snap_path)
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save Captured Image",
            os.path.join(os.path.expanduser("~"), "Desktop", default_name),
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if not dest:
            return  # user cancelled

        try:
            import shutil
            shutil.copy2(self._last_snap_path, dest)
            QMessageBox.information(self, "Image Saved",
                                    f"Image saved to:\n\n{dest}")
            self._sb.showMessage(f"Image downloaded → {dest}")
        except Exception as ex:
            QMessageBox.critical(self, "Save Error",
                                 f"Could not save image:\n\n{ex}")

    @pyqtSlot()
    def _on_test_image(self):
        """Open a file dialog, run the detector on the chosen image, show results."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image to Test",
            os.path.expanduser("~"),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)"
        )
        if not path:
            return  # cancelled

        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.critical(self, "Load Error",
                                 f"Could not read image:\n{path}")
            return

        # Run detector
        leak, annotated, mask, count = self.detector.detect(frame)

        # Save annotated copy
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(path))[0]
        snap_path = os.path.join(SNAP_DIR, f"TEST_{base}_{ts}.jpg")
        cv2.imwrite(snap_path, annotated)
        self._last_snap_path = snap_path

        # Update right-panel displays
        self._show_snap(annotated)
        self._show_mask(mask)
        self._snap_info.setText(os.path.basename(snap_path))
        self._btn_download.setEnabled(True)

        # Show detection result
        status = "NOT OK" if leak else "OK"
        if leak:
            self._big_status.setText("⚠  OIL LEAK\nDETECTED — NOT OK")
            self._big_status.setStyleSheet("color:#ff3333; font-size:18px; font-weight:bold;")
            self._leak_overlay.setVisible(False)   # no live feed to overlay
        else:
            self._big_status.setText("✅  NO OIL LEAK\n— OK")
            self._big_status.setStyleSheet("color:#00ff88; font-size:18px; font-weight:bold;")
            self._leak_overlay.setVisible(False)

        # Also show annotated image in the main feed area
        self._show_feed(annotated)

        # Update badge
        insp_color = "#ff3333" if leak else "#00ff88"
        insp_bg    = "#1a0000" if leak else "#001a00"
        insp_bdr   = "#550000" if leak else "#005500"
        self._insp_badge.setText(f"  {status}  ")
        self._insp_badge.setStyleSheet(
            f"color:{insp_color}; font-size:12px; font-weight:bold;"
            f"background:{insp_bg}; border:1px solid {insp_bdr};"
            "border-radius:4px; padding:3px 10px;"
        )

        # Log it
        win_label = f"[IMAGE] {base}"
        self.reporter.add_record(win_number=win_label, status=status, image_path=snap_path)
        self._total_inspected += 1
        if leak:
            self._total_fails += 1
            self._last_result = "NOT OK"
        else:
            self._total_passes += 1
            self._last_result = "OK"
        self._update_stats()
        self._add_log_row(win_label, status, snap_path)

        leak_txt = f"⚠ LEAK ({count} region{'s' if count != 1 else ''})" if leak else "✅ No leak"
        self._sb.showMessage(
            f"Image test complete  |  {os.path.basename(path)}  |  {leak_txt}  |  Saved → {snap_path}"
        )

    @pyqtSlot()
    def _on_report(self):
        if not self.reporter._records:
            QMessageBox.information(self, "No Data",
                                    "No inspections recorded yet.\n"
                                    "Complete at least one inspection to generate a report.")
            return
        try:
            path = self.reporter.generate()
            self._report_lbl.setText(f"Saved:\n{path}")
            QMessageBox.information(self, "Report Saved",
                                    f"Excel report saved to:\n\n{path}")
            # Open the file
            os.startfile(path)
        except Exception as ex:
            QMessageBox.critical(self, "Report Error",
                                 f"Could not generate report:\n\n{ex}")

    # ================================================================ #
    #   HELPERS
    # ================================================================ #

    def _update_stats(self):
        self._stat_total.setText(str(self._total_inspected))
        self._stat_pass.setText(str(self._total_passes))
        self._stat_fail.setText(str(self._total_fails))
        ok = self._last_result == "OK"
        self._stat_last.setText("✅ OK" if ok else "❌ FAIL")
        self._stat_last.setStyleSheet(
            f"color:{'#00ff88' if ok else '#ff3333'}; font-size:16px; font-weight:bold; border:none;"
        )
        if self._total_inspected > 0:
            rate = int(self._total_passes / self._total_inspected * 100)
            self._rate_lbl.setText(f"{rate}%")
            self._rate_bar.setValue(rate)
            c = "#00ff88" if rate >= 80 else "#ffaa00" if rate >= 50 else "#ff3333"
            self._rate_lbl.setStyleSheet(
                f"color:{c}; font-size:22px; font-weight:bold; border:none;"
            )

    def _add_log_row(self, win: str, status: str, snap: str):
        r = self._log.rowCount()
        self._log.insertRow(r)
        ok = status == "OK"
        fg = QColor("#00ff88") if ok else QColor("#ff3333")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def cell(v, a=Qt.AlignCenter):
            i = QTableWidgetItem(str(v))
            i.setTextAlignment(a | Qt.AlignVCenter)
            return i

        self._log.setItem(r, 0, cell(r + 1))
        self._log.setItem(r, 1, cell(win))
        self._log.setItem(r, 2, cell(ts))
        sc = cell("✅  NO OIL LEAK — OK" if ok else "❌  OIL LEAK DETECTED — NOT OK")
        sc.setForeground(fg)
        self._log.setItem(r, 3, sc)
        self._log.setItem(r, 4, cell("✓ " + os.path.basename(snap)))
        self._log.scrollToBottom()

    def _show_feed(self, frame):
        self.feed_lbl.setPixmap(self._cv2qt(frame, self.feed_lbl.size()))

    def _show_snap(self, frame):
        self._snap_lbl.setPixmap(self._cv2qt(frame, self._snap_lbl.size()))

    def _show_mask(self, mask):
        cm = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cm[mask > 0] = [0, 220, 80]
        self._mask_lbl.setPixmap(self._cv2qt(cm, self._mask_lbl.size()))

    def _cv2qt(self, frame, size: QSize) -> QPixmap:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        # Use KeepAspectRatio to fit without cropping.
        return QPixmap.fromImage(qi).scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-center overlay if visible
        if hasattr(self, '_leak_overlay') and self._leak_overlay.isVisible():
            self._leak_overlay.adjustSize()
            cx = (self.feed_lbl.width() - self._leak_overlay.width()) // 2
            cy = (self.feed_lbl.height() - self._leak_overlay.height()) // 2
            self._leak_overlay.move(cx, cy)

    def closeEvent(self, event):
        if self.cam_thread:
            self.cam_thread.stop()
        if self.reporter._records:
            try:
                self.reporter.generate()
            except Exception:
                pass
        event.accept()
