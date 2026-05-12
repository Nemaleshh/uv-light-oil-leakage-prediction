"""
dashboard.py
Main application window for UV Engine Oil Leak Detection System v2.

Layout (4-column splitter):
  LEFT   — Camera Setup | VIN/WIN | Reports | Debug Test
  CENTRE — Live Camera Feed
  RIGHT  — Detection Result | Last Snapshot | Manual Confirm Buttons
  [below splitter] Stats bar | Session Log Table

Manual-only mode: NO auto-capture, NO presence detection.
"""

import os
import re
import cv2
import csv
import json
import numpy as np
from datetime import datetime
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QProgressBar, QStatusBar, QSizePolicy,
    QHeaderView, QMessageBox, QFrame, QSplitter, QComboBox,
    QFileDialog, QSpacerItem, QScrollArea, QDateEdit,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor

from core.camera_thread import CameraThread, CameraScanThread
from core.pipeline import LeakPipeline
from core.report_generator import generate_car_report, generate_summary_report

# ── Paths ── #
import sys
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE     = os.path.join(BASE_DIR, "config.json")
SNAP_DIR        = os.path.join(BASE_DIR, "detected_images")          # full frames
CROPS_DIR       = os.path.join(BASE_DIR, "detected_images", "crops") # YOLO crops
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
SESSIONS_DIR    = os.path.join(REPORTS_DIR, "sessions")
CSV_PATH        = os.path.join(REPORTS_DIR, "inspection_log.csv")
DATA_DIRS = {
    "No Leak":    os.path.join(BASE_DIR, "data", "no_leak"),
    "Engine Oil": os.path.join(BASE_DIR, "data", "engine_oil"),
    "TM Oil":     os.path.join(BASE_DIR, "data", "tm_oil_leak"),
    "Both Leaks": os.path.join(BASE_DIR, "data", "both_leaks"),
}
for _d in [SNAP_DIR, CROPS_DIR, REPORTS_DIR, SESSIONS_DIR] + list(DATA_DIRS.values()):
    os.makedirs(_d, exist_ok=True)


# ── Filename sanitizer ── #
_ILLEGAL_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

def _safe_filename(text: str, max_len: int = 40) -> str:
    """Replace all Windows-illegal filename characters with '_'."""
    safe = _ILLEGAL_CHARS.sub('_', text.strip())
    safe = safe.strip('. ')   # no leading/trailing dots or spaces
    return safe[:max_len] if safe else "unknown"

# ── Default config ── #
DEFAULT_CONFIG = {
    "camera_index":      0,
    "model_path":        os.path.join(BASE_DIR, "models", "best.pt"),
    "clf_path":          os.path.join(BASE_DIR, "models", "gb_classifier.pkl"),
    "hold_seconds":      30,
    "threshold_pct":     0.15,
    "uv_max_on_minutes": 120,
}


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            # Merge with defaults (new keys won't break old config)
            merged = {**DEFAULT_CONFIG, **cfg}
            
            # ── Fix Hardcoded Paths ──
            # If the user copied config.json from another PC, the absolute paths
            # might be broken. Always override with the local BASE_DIR.
            merged["model_path"] = os.path.join(BASE_DIR, "models", "best.pt")
            merged["clf_path"]   = os.path.join(BASE_DIR, "models", "gb_classifier.pkl")
            
            return merged
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")


# ═══════════════════════════════════════════════════════════════════ #
#   Pipeline Worker — runs inference in background thread
# ═══════════════════════════════════════════════════════════════════ #

class PipelineWorker(QThread):
    result_ready = pyqtSignal(dict)

    def __init__(self, pipeline: LeakPipeline, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self._frame    = frame

    def run(self):
        result = self._pipeline.run(self._frame)
        self.result_ready.emit(result)


# ═══════════════════════════════════════════════════════════════════ #
#   Main Dashboard Window
# ═══════════════════════════════════════════════════════════════════ #

class Dashboard(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UV Engine Oil Leak Detection System  v2")
        self.setMinimumSize(1400, 860)

        self.config   = load_config()
        self.pipeline = LeakPipeline(
            model_path=self.config.get("model_path"),
            clf_path=self.config.get("clf_path"),
            roi=self.config.get("roi")
        )

        # Thread handles
        self._cam_thread:    Optional[CameraThread]    = None
        self._scan_thread:   Optional[CameraScanThread] = None
        self._pipe_worker:   Optional[PipelineWorker]  = None

        # App state
        self._cam_connected   = False
        self._phase           = "IDLE"
        self._led_blink       = False
        self._last_frame      = None   # live camera frame
        self._last_full_frame = None   # full frame at capture time (for training)
        self._last_crop       = None   # YOLO-cropped region (for reports)
        self._last_snap_path  = None   # path of saved full frame
        self._last_crop_path  = None   # path of saved YOLO crop
        self._last_vin        = ""
        self._last_result     = {}
        self._total = self._passed = self._failed = 0

        # Build UI first, then load pipeline in background
        self._build_ui()

        # Clock + LED blink timer
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick)
        self._clock_timer.start(1000)

        # Load models (non-blocking)
        self._load_models_async()

    # ════════════════════════════════════════════════════════════════ #
    #   UI CONSTRUCTION
    # ════════════════════════════════════════════════════════════════ #

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
        splitter.setSizes([280, 700, 320])
        vlay.addWidget(splitter, stretch=1)

        vlay.addWidget(self._make_stats_bar())
        vlay.addWidget(self._make_log_table())

        self._sb = QStatusBar()
        self.setStatusBar(self._sb)
        self._sb.showMessage("Ready — scan for a USB camera and enter VIN to begin")

    # ── Header ─────────────────────────────────────────────────────── #

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

        # Camera badge
        self._cam_badge = QLabel("● CAM OFFLINE")
        self._cam_badge.setStyleSheet(
            "color:#ff3333;font-size:12px;font-weight:bold;"
            "background:#1a0000;border:1px solid #550000;"
            "border-radius:4px;padding:3px 10px;"
        )
        h.addWidget(self._cam_badge)

        # Phase badge
        self._phase_badge = QLabel("  IDLE  ")
        self._phase_badge.setStyleSheet(
            "color:#778899;font-size:12px;font-weight:bold;"
            "background:#111122;border:1px solid #223344;"
            "border-radius:4px;padding:3px 10px;"
        )
        h.addWidget(self._phase_badge)

        # Model load badge
        self._model_badge = QLabel("  MODELS: LOADING…  ")
        self._model_badge.setStyleSheet(
            "color:#ffaa00;font-size:11px;font-weight:bold;"
            "background:#1a1000;border:1px solid #443300;"
            "border-radius:4px;padding:3px 8px;"
        )
        h.addWidget(self._model_badge)

        sep = QLabel("  |  ")
        sep.setStyleSheet("color:#334455;")
        h.addWidget(sep)

        self._clock_lbl = QLabel()
        self._clock_lbl.setObjectName("clock_label")
        self._clock_lbl.setText(datetime.now().strftime("  %Y-%m-%d   %H:%M:%S"))
        h.addWidget(self._clock_lbl)

        return f

    # ── Left Panel ─────────────────────────────────────────────────── #

    def _make_left(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)

        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(4, 4, 4, 4)
        ly.setSpacing(10)

        # ── Camera Setup ──
        cg = QGroupBox("📡  CAMERA SETUP")
        cl = QVBoxLayout(cg)

        cl.addWidget(self._hint("Select USB camera device:"))
        self._cam_combo = QComboBox()
        self._cam_combo.setPlaceholderText("— click Scan first —")
        cl.addWidget(self._cam_combo)

        self._btn_scan = QPushButton("🔍  Scan Devices")
        self._btn_scan.setObjectName("btn_scan")
        self._btn_scan.clicked.connect(self._on_scan)
        cl.addWidget(self._btn_scan)

        row = QHBoxLayout()
        self._btn_connect = QPushButton("🔗  Connect")
        self._btn_connect.setObjectName("btn_connect")
        self._btn_connect.setEnabled(False)
        self._btn_connect.clicked.connect(self._on_connect)
        row.addWidget(self._btn_connect)

        self._btn_disconnect = QPushButton("✖  Disconnect")
        self._btn_disconnect.setObjectName("btn_disconnect")
        self._btn_disconnect.setEnabled(False)
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        row.addWidget(self._btn_disconnect)
        cl.addLayout(row)

        self._cam_strip = QLabel("  NO SIGNAL  ")
        self._cam_strip.setAlignment(Qt.AlignCenter)
        self._cam_strip.setStyleSheet(
            "background:#110000;color:#440000;font-size:11px;font-weight:bold;"
            "border-radius:4px;padding:4px;letter-spacing:2px;"
        )
        cl.addWidget(self._cam_strip)
        ly.addWidget(cg)

        # ── VIN / WIN ──
        vg = QGroupBox("🚗  VIN / WIN NUMBER")
        vl = QVBoxLayout(vg)
        vl.addWidget(self._hint("Type or scan barcode. Leave empty → auto timestamp ID"))
        self._vin_input = QLineEdit()
        self._vin_input.setPlaceholderText("e.g. WIN-2024-001 or scan barcode…")
        vl.addWidget(self._vin_input)
        # ── Manual Capture ──
        self._btn_manual_capture = QPushButton("📸  CAPTURE NOW")
        self._btn_manual_capture.setObjectName("btn_manual_capture")
        self._btn_manual_capture.setEnabled(False)
        self._btn_manual_capture.clicked.connect(self._on_manual_capture)
        vl.addWidget(self._btn_manual_capture)

        ly.addWidget(vg)

        # ── Reports ──
        rg = QGroupBox("📊  REPORTS")
        rl = QVBoxLayout(rg)
        self._btn_report = QPushButton("📥  Open Inspection Log (CSV)")
        self._btn_report.setObjectName("btn_report")
        self._btn_report.clicked.connect(self._on_open_report)
        rl.addWidget(self._btn_report)

        # ── Date Picker for Summary ──
        d_lay = QHBoxLayout()
        d_lbl = QLabel("Summary Date:")
        d_lbl.setStyleSheet("font-size:11px;color:#778899;")
        self._summary_date = QDateEdit()
        self._summary_date.setCalendarPopup(True)
        self._summary_date.setDate(datetime.now().date())
        self._summary_date.setStyleSheet("background:#ffffff; color:#000000; padding:2px;")
        d_lay.addWidget(d_lbl)
        d_lay.addWidget(self._summary_date, stretch=1)
        rl.addLayout(d_lay)

        self._btn_summary_report = QPushButton("📄  Generate Summary Report")
        self._btn_summary_report.setObjectName("btn_summary_report")
        self._btn_summary_report.clicked.connect(self._on_generate_summary)
        rl.addWidget(self._btn_summary_report)

        self._btn_open_data = QPushButton("📂  Open Collected Data Folder")
        self._btn_open_data.clicked.connect(self._on_open_data)
        rl.addWidget(self._btn_open_data)
        ly.addWidget(rg)

        # ── Debug Test Image ──
        dg = QGroupBox("🧪  DEBUG / TEST IMAGE")
        dl = QVBoxLayout(dg)
        dl.addWidget(self._hint("Load any image and run pipeline without camera"))

        self._test_path_lbl = QLabel("No image selected")
        self._test_path_lbl.setStyleSheet("font-size:10px;color:#445566;")
        self._test_path_lbl.setWordWrap(True)
        dl.addWidget(self._test_path_lbl)

        self._btn_load_test = QPushButton("📂  Load Test Image")
        self._btn_load_test.setObjectName("btn_load_test")
        self._btn_load_test.clicked.connect(self._on_load_test_image)
        dl.addWidget(self._btn_load_test)

        self._btn_run_test = QPushButton("▶  Run Pipeline on Image")
        self._btn_run_test.setObjectName("btn_run_test")
        self._btn_run_test.setEnabled(False)
        self._btn_run_test.clicked.connect(self._on_run_test)
        dl.addWidget(self._btn_run_test)
        ly.addWidget(dg)

        ly.addStretch()
        scroll.setWidget(w)
        return scroll

    # ── Centre Panel ───────────────────────────────────────────────── #

    def _make_centre(self):
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(4, 4, 4, 4)
        ly.setSpacing(6)

        fg = QGroupBox("📷  LIVE CAMERA FEED")
        fl = QVBoxLayout(fg)
        fl.setContentsMargins(2, 2, 2, 2)

        self._feed_lbl = QLabel()
        self._feed_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._feed_lbl.setAlignment(Qt.AlignCenter)
        self._feed_lbl.setStyleSheet(
            "background:#050510;border:2px solid #1e3a5f;border-radius:4px;"
        )
        self._feed_lbl.setText("No camera feed\nConnect camera to begin")
        fl.addWidget(self._feed_lbl, stretch=1)

        # Presence / countdown overlay (drawn on top of feed in centre of widget)
        self._overlay_lbl = QLabel(self._feed_lbl)
        self._overlay_lbl.setAlignment(Qt.AlignCenter)
        self._overlay_lbl.setStyleSheet(
            "color:white;background:rgba(0,0,0,0);"
            "font-size:22px;font-weight:bold;"
        )
        self._overlay_lbl.setVisible(False)

        # Bottom row badges
        bottom = QHBoxLayout()
        self._fps_lbl = QLabel("FPS: --")
        self._fps_lbl.setStyleSheet("font-size:10px;color:#334455;")
        bottom.addWidget(self._fps_lbl)
        bottom.addStretch()

        self._live_badge = QLabel("  ● LIVE  ")
        self._live_badge.setStyleSheet(
            "color:#00ff88;font-size:11px;font-weight:bold;"
            "background:#001a00;border:1px solid #004400;"
            "border-radius:4px;padding:2px 8px;"
        )
        self._live_badge.setVisible(False)
        bottom.addWidget(self._live_badge)

        self._state_badge = QLabel("")
        self._state_badge.setStyleSheet(
            "color:#ffaa00;font-size:11px;font-weight:bold;"
            "background:#1a1000;border:1px solid #443300;"
            "border-radius:4px;padding:2px 8px;"
        )
        self._state_badge.setVisible(False)
        bottom.addWidget(self._state_badge)

        fl.addLayout(bottom)
        ly.addWidget(fg, stretch=1)
        return w

    # ── Right Panel ────────────────────────────────────────────────── #

    def _make_right(self):
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(4, 4, 4, 4)
        ly.setSpacing(10)

        # Detection Result
        dg = QGroupBox("🔍  DETECTION RESULT")
        dl = QVBoxLayout(dg)
        dl.setAlignment(Qt.AlignCenter)
        self._big_result = QLabel("IDLE")
        self._big_result.setAlignment(Qt.AlignCenter)
        self._big_result.setWordWrap(True)
        self._big_result.setMinimumHeight(80)
        self._big_result.setStyleSheet("color:#778899;font-size:22px;font-weight:bold;")
        dl.addWidget(self._big_result)

        self._conf_lbl = QLabel("—")
        self._conf_lbl.setAlignment(Qt.AlignCenter)
        self._conf_lbl.setStyleSheet("color:#445566;font-size:11px;")
        dl.addWidget(self._conf_lbl)

        self._vin_disp = QLabel("VIN: —")
        self._vin_disp.setAlignment(Qt.AlignCenter)
        self._vin_disp.setStyleSheet("color:#445566;font-size:11px;")
        dl.addWidget(self._vin_disp)
        ly.addWidget(dg)

        # Last Capture
        sg = QGroupBox("📸  LAST CAPTURE")
        sl = QVBoxLayout(sg)
        self._snap_lbl = QLabel()
        self._snap_lbl.setFixedSize(290, 175)
        self._snap_lbl.setAlignment(Qt.AlignCenter)
        self._snap_lbl.setStyleSheet(
            "background:#050510;border:1px solid #1e3a5f;border-radius:4px;"
        )
        self._snap_lbl.setText("No capture yet")
        sl.addWidget(self._snap_lbl, alignment=Qt.AlignCenter)

        self._snap_info = QLabel("—")
        self._snap_info.setAlignment(Qt.AlignCenter)
        self._snap_info.setStyleSheet("font-size:10px;color:#445566;")
        sl.addWidget(self._snap_info)
        ly.addWidget(sg)

        # Manual Confirm Buttons
        cg = QGroupBox("✍  MANUAL CONFIRM  (saves crop for retraining)")
        cl = QVBoxLayout(cg)
        cl.addWidget(self._hint("Correct the auto result — saves crop to data folder:"))

        r1 = QHBoxLayout()
        self._btn_confirm_noleak = QPushButton("✅  No Leak")
        self._btn_confirm_noleak.setObjectName("btn_confirm_noleak")
        self._btn_confirm_noleak.setEnabled(False)
        self._btn_confirm_noleak.clicked.connect(lambda: self._on_confirm("No Leak"))
        r1.addWidget(self._btn_confirm_noleak)

        self._btn_confirm_engine = QPushButton("⚠️  Engine Oil")
        self._btn_confirm_engine.setObjectName("btn_confirm_engine")
        self._btn_confirm_engine.setEnabled(False)
        self._btn_confirm_engine.clicked.connect(lambda: self._on_confirm("Engine Oil"))
        r1.addWidget(self._btn_confirm_engine)
        cl.addLayout(r1)

        r2 = QHBoxLayout()
        self._btn_confirm_tmoil = QPushButton("🔧  TM Oil Leak")
        self._btn_confirm_tmoil.setObjectName("btn_confirm_tmoil")
        self._btn_confirm_tmoil.setEnabled(False)
        self._btn_confirm_tmoil.clicked.connect(lambda: self._on_confirm("TM Oil"))
        r2.addWidget(self._btn_confirm_tmoil)

        self._btn_confirm_both = QPushButton("⚡  Both Leaks")
        self._btn_confirm_both.setObjectName("btn_confirm_both")
        self._btn_confirm_both.setEnabled(False)
        self._btn_confirm_both.clicked.connect(lambda: self._on_confirm("Both Leaks"))
        r2.addWidget(self._btn_confirm_both)
        cl.addLayout(r2)

        self._confirm_status = QLabel("")
        self._confirm_status.setAlignment(Qt.AlignCenter)
        self._confirm_status.setStyleSheet("font-size:10px;color:#00ff88;")
        cl.addWidget(self._confirm_status)
        ly.addWidget(cg)

        # ── UV Light Tracker ──
        ug = QGroupBox("💡  UV LIGHT TRACKER")
        ul = QVBoxLayout(ug)
        ul.addWidget(self._hint("Thermal limits for 200W LED:"))
        self._uv_lbl = QLabel("0 mins / 120 mins")
        self._uv_lbl.setAlignment(Qt.AlignCenter)
        ul.addWidget(self._uv_lbl)
        self._uv_bar = QProgressBar()
        self._uv_bar.setMaximum(self.config.get("uv_max_on_minutes", 120) * 60)
        self._uv_bar.setValue(0)
        self._uv_bar.setTextVisible(False)
        ul.addWidget(self._uv_bar)
        ly.addWidget(ug)

        ly.addStretch()
        return w

    # ── Stats Bar ──────────────────────────────────────────────────── #

    def _make_stats_bar(self):
        f = QFrame()
        f.setFixedHeight(110)
        f.setStyleSheet("background:#0d0d20;border:1px solid #1e3a5f;border-radius:6px;")
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
            num.setStyleSheet(f"color:{color};font-size:28px;font-weight:bold;border:none;")
            num.setAlignment(Qt.AlignCenter)
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#556677;font-size:10px;letter-spacing:1px;border:none;")
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

        h.addWidget(card("_stat_total",  "CARS INSPECTED", "#00c8ff"))
        h.addWidget(vsep())
        h.addWidget(card("_stat_pass",   "PASSED (OK)",    "#00ff88"))
        h.addWidget(vsep())
        h.addWidget(card("_stat_fail",   "FAILED (LEAK)",  "#ff3333"))
        h.addWidget(vsep())

        # Pass rate
        pr = QWidget()
        pr.setStyleSheet("border:none;")
        prl = QVBoxLayout(pr)
        prl.setSpacing(4)
        prl.setContentsMargins(16, 4, 16, 4)
        pr_title = QLabel("PASS RATE")
        pr_title.setStyleSheet("color:#556677;font-size:10px;letter-spacing:1px;border:none;")
        pr_title.setAlignment(Qt.AlignCenter)
        prl.addWidget(pr_title)
        self._rate_lbl = QLabel("— %")
        self._rate_lbl.setStyleSheet("color:#00ff88;font-size:22px;font-weight:bold;border:none;")
        self._rate_lbl.setAlignment(Qt.AlignCenter)
        self._rate_bar = QProgressBar()
        self._rate_bar.setRange(0, 100)
        self._rate_bar.setValue(0)
        self._rate_bar.setFixedHeight(14)
        self._rate_bar.setTextVisible(False)
        prl.addWidget(self._rate_lbl)
        prl.addWidget(self._rate_bar)
        h.addWidget(pr)
        h.addWidget(vsep())

        # Current VIN
        vc = QWidget()
        vc.setStyleSheet("border:none;")
        vcl = QVBoxLayout(vc)
        vcl.setSpacing(2)
        vcl.setContentsMargins(16, 4, 16, 4)
        vin_title = QLabel("CURRENT VIN")
        vin_title.setStyleSheet("color:#556677;font-size:10px;letter-spacing:1px;border:none;")
        vin_title.setAlignment(Qt.AlignCenter)
        vcl.addWidget(vin_title)
        self._stat_vin = QLabel("—")
        self._stat_vin.setStyleSheet("color:#ffaa00;font-size:18px;font-weight:bold;border:none;")
        self._stat_vin.setAlignment(Qt.AlignCenter)
        vcl.addWidget(self._stat_vin)
        h.addWidget(vc)
        h.addWidget(vsep())

        # Last Result
        lc = QWidget()
        lc.setStyleSheet("border:none;")
        lcl = QVBoxLayout(lc)
        lcl.setSpacing(2)
        lcl.setContentsMargins(16, 4, 16, 4)
        last_title = QLabel("LAST RESULT")
        last_title.setStyleSheet("color:#556677;font-size:10px;letter-spacing:1px;border:none;")
        last_title.setAlignment(Qt.AlignCenter)
        lcl.addWidget(last_title)
        self._stat_last = QLabel("—")
        self._stat_last.setStyleSheet("color:#778899;font-size:16px;font-weight:bold;border:none;")
        self._stat_last.setAlignment(Qt.AlignCenter)
        lcl.addWidget(self._stat_last)
        h.addWidget(lc)
        return f

    # ── Session Log ────────────────────────────────────────────────── #

    def _make_log_table(self):
        g = QGroupBox("📋  SESSION INSPECTION LOG")
        g.setMinimumHeight(150)
        ly = QVBoxLayout(g)
        self._log = QTableWidget(0, 7)
        self._log.setHorizontalHeaderLabels(
            ["#", "VIN / ID", "Timestamp", "Auto Result", "Confidence",
             "Manual Confirm", "Image Saved"]
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

    # ── Helper ─────────────────────────────────────────────────────── #

    def _hint(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size:10px;color:#557799;")
        return lbl

    # ════════════════════════════════════════════════════════════════ #
    #   MODEL LOADING
    # ════════════════════════════════════════════════════════════════ #

    def _load_models_async(self):
        """Load YOLO + PKL in a background thread to avoid UI freeze."""
        class ModelLoader(QThread):
            done   = pyqtSignal(bool, str)

            def __init__(self, pipeline):
                super().__init__()
                self._p = pipeline

            def run(self):
                try:
                    self._p.load()
                    self.done.emit(True, "")
                except Exception as e:
                    self.done.emit(False, str(e))

        self._model_loader = ModelLoader(self.pipeline)
        self._model_loader.done.connect(self._on_models_loaded)
        self._model_loader.start()

    @pyqtSlot(bool, str)
    def _on_models_loaded(self, ok: bool, err: str):
        if ok:
            self._model_badge.setText("  ✅ MODELS READY  ")
            self._model_badge.setStyleSheet(
                "color:#00ff88;font-size:11px;font-weight:bold;"
                "background:#001a00;border:1px solid #004400;"
                "border-radius:4px;padding:3px 8px;"
            )
            self._sb.showMessage("Models loaded — connect camera to begin")
        else:
            self._model_badge.setText("  ❌ MODEL ERROR  ")
            self._model_badge.setStyleSheet(
                "color:#ff3333;font-size:11px;font-weight:bold;"
                "background:#1a0000;border:1px solid #550000;"
                "border-radius:4px;padding:3px 8px;"
            )
            self._sb.showMessage(f"Model load error: {err}")

    # ════════════════════════════════════════════════════════════════ #
    #   CAMERA SLOTS
    # ════════════════════════════════════════════════════════════════ #

    @pyqtSlot()
    def _on_scan(self):
        self._btn_scan.setText("⏳  Scanning…")
        self._btn_scan.setEnabled(False)
        self._cam_combo.clear()

        self._scan_thread = CameraScanThread()
        self._scan_thread.scan_complete.connect(self._on_scan_done)
        self._scan_thread.start()

    @pyqtSlot(list)
    def _on_scan_done(self, cameras: list):
        self._btn_scan.setText("🔍  Scan Devices")
        self._btn_scan.setEnabled(True)
        self._cam_combo.clear()

        if cameras:
            for idx, label in cameras:
                self._cam_combo.addItem(label, userData=idx)
            self._btn_connect.setEnabled(True)
            self._sb.showMessage(f"Found {len(cameras)} camera(s)")
        else:
            self._cam_combo.addItem("No cameras found", userData=-1)
            self._btn_connect.setEnabled(False)
            self._sb.showMessage("No USB cameras detected — check connections")

    @pyqtSlot()
    def _on_connect(self):
        idx = self._cam_combo.currentData()
        if idx is None or idx < 0:
            return

        self._cam_thread = CameraThread(cam_index=idx)
        self._cam_thread.frame_ready.connect(self._on_frame)
        self._cam_thread.uv_usage_update.connect(self._on_uv_usage)
        self._cam_thread.camera_error.connect(self._on_cam_error)
        self._cam_thread.start()

        self._cam_connected = True
        self._btn_connect.setEnabled(False)
        self._btn_disconnect.setEnabled(True)
        self._btn_manual_capture.setEnabled(True)
        self._live_badge.setVisible(True)
        self._cam_strip.setText("  LIVE  ")
        self._cam_strip.setStyleSheet(
            "background:#001a00;color:#00ff88;font-size:11px;font-weight:bold;"
            "border-radius:4px;padding:4px;letter-spacing:2px;"
        )
        self._sb.showMessage("Camera connected — press CAPTURE NOW to inspect")
        self._set_phase("READY")

    @pyqtSlot()
    def _on_disconnect(self):
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None
        self._cam_connected = False
        self._btn_connect.setEnabled(True)
        self._btn_disconnect.setEnabled(False)
        self._btn_manual_capture.setEnabled(False)
        self._live_badge.setVisible(False)
        self._state_badge.setVisible(False)
        self._overlay_lbl.setVisible(False)
        self._cam_badge.setText("● CAM OFFLINE")
        self._cam_badge.setStyleSheet(
            "color:#ff3333;font-size:12px;font-weight:bold;"
            "background:#1a0000;border:1px solid #550000;"
            "border-radius:4px;padding:3px 10px;"
        )
        self._cam_strip.setText("  NO SIGNAL  ")
        self._cam_strip.setStyleSheet(
            "background:#110000;color:#440000;font-size:11px;font-weight:bold;"
            "border-radius:4px;padding:4px;letter-spacing:2px;"
        )
        self._feed_lbl.setText("No camera feed\nConnect camera to begin")
        self._set_phase("IDLE")
        self._sb.showMessage("Camera disconnected")

    # ════════════════════════════════════════════════════════════════ #
    #   CAMERA SIGNAL HANDLERS
    # ════════════════════════════════════════════════════════════════ #

    @pyqtSlot(np.ndarray)
    def _on_frame(self, frame: np.ndarray):
        self._last_frame = frame
        self._display_frame(frame, self._feed_lbl)

    @pyqtSlot(str)
    def _on_cam_error(self, msg: str):
        self._sb.showMessage(f"Camera error: {msg}")
        self._on_disconnect()

    @pyqtSlot(float)
    def _on_uv_usage(self, seconds: float):
        max_secs = self.config.get("uv_max_on_minutes", 120) * 60
        self._uv_bar.setValue(int(seconds))
        mins = int(seconds / 60)
        max_mins = max_secs // 60
        self._uv_lbl.setText(f"{mins} mins / {max_mins} mins")
        
        if seconds >= max_secs:
            self._uv_bar.setObjectName("uv_bar_danger")
            self._uv_lbl.setStyleSheet("color:#ff3333; font-weight:bold;")
            self._uv_lbl.setText(f"⚠ OVERHEATING: {mins} mins - TURN OFF UV!")
            # Retrigger style update
            self._uv_bar.style().unpolish(self._uv_bar)
            self._uv_bar.style().polish(self._uv_bar)
        else:
            self._uv_bar.setObjectName("")
            self._uv_lbl.setStyleSheet("color:#aabbcc; font-weight:normal;")
            self._uv_bar.style().unpolish(self._uv_bar)
            self._uv_bar.style().polish(self._uv_bar)

    @pyqtSlot(int)
    @pyqtSlot()
    def _on_manual_capture(self):
        if self._last_frame is None:
            self._sb.showMessage("No camera frame available.")
            return

        self._set_phase("ANALYZING")
        self._overlay_lbl.setText("📷  ANALYZING…")
        self._overlay_lbl.setStyleSheet(
            "color:white;background:rgba(0,80,180,180);"
            "font-size:26px;font-weight:bold;border-radius:8px;padding:12px;"
        )
        self._overlay_lbl.setVisible(True)
        self._overlay_lbl.setGeometry(0, 0,
            self._feed_lbl.width(), self._feed_lbl.height())

        vin = self._vin_input.text().strip()
        if not vin:
            vin = f"MANUAL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._last_vin = vin
        self._vin_disp.setText(f"VIN: {vin}")
        self._stat_vin.setText(vin[:16] + "…" if len(vin) > 16 else vin)

        self._run_pipeline(self._last_frame.copy(), source="MANUAL")

    # ════════════════════════════════════════════════════════════════ #
    #   PIPELINE
    # ════════════════════════════════════════════════════════════════ #

    def _run_pipeline(self, frame: np.ndarray, source: str = "CAMERA"):
        if not self.pipeline.is_loaded:
            self._sb.showMessage("Models not ready yet — please wait")
            return
        if self._pipe_worker and self._pipe_worker.isRunning():
            return  # Already running

        self._pipe_worker = PipelineWorker(self.pipeline, frame)
        self._pipe_worker.result_ready.connect(
            lambda r: self._on_result(r, source)
        )
        self._pipe_worker.start()

    @pyqtSlot(dict)
    def _on_result(self, result: dict, source: str = "CAMERA"):
        self._last_result = result
        self._overlay_lbl.setVisible(False)

        # Only abort for genuine crash errors (not diagnostic notes)
        error_msg = result.get("error", "")
        if error_msg and result.get("result") not in ("NO_PART",):
            self._sb.showMessage(f"Pipeline error: {error_msg}")
            self._set_phase("ERROR")
            return

        res_str = result.get("result", "ERROR")
        if res_str == "NO_PART":
            debug = result.get("debug", "")
            yconf = result.get("yolo_conf", 0.0)
            hint  = "Part outside ROI zone — reposition car" if "outside ROI" in debug \
                    else "No part detected — check UV light and camera angle"
            self._big_result.setText("⚠  PART NOT DETECTED")
            self._big_result.setStyleSheet("color:#ffaa00;font-size:18px;font-weight:bold;")
            self._conf_lbl.setText(f"YOLO conf: {yconf:.3f}  |  {debug}")
            self._set_phase("NO PART DETECTED")
            self._sb.showMessage(f"{hint}   [{debug}]")
            return

        conf       = result.get("confidence", 0.0)
        crop       = result.get("crop")

        # ── Display big result ──
        if res_str == "NO LEAK":
            self._big_result.setText("✅  NO LEAK")
            self._big_result.setStyleSheet("color:#00ff88;font-size:26px;font-weight:bold;")
            self._set_phase("DONE — NO LEAK")
            is_leak = False
        elif res_str == "OIL LEAK":
            self._big_result.setText("🔴  OIL LEAK DETECTED")
            self._big_result.setStyleSheet("color:#ff3333;font-size:26px;font-weight:bold;")
            self._set_phase("DONE — LEAK DETECTED")
            is_leak = True
        else:
            self._big_result.setText("⚠️  PART NOT FOUND")
            self._big_result.setStyleSheet("color:#ffaa00;font-size:22px;font-weight:bold;")
            self._set_phase("NO PART DETECTED")
            is_leak = False

        self._conf_lbl.setText(f"Confidence: {conf:.1f}%")

        # ── Save images ──
        # full frame  → detected_images/   (used in UI panel + training data)
        # YOLO crop   → detected_images/crops/  (used in HTML daily report)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_vin  = _safe_filename(self._last_vin)
        base_name = f"{ts}_{safe_vin}"

        snap_path = None   # full frame path
        crop_path = None   # YOLO crop path

        # Always save the full frame (even when YOLO found nothing)
        full_frame = result.get("_full_frame") or self._last_frame
        if full_frame is not None:
            self._last_full_frame = full_frame.copy()
            snap_path = os.path.join(SNAP_DIR, f"{base_name}_full.jpg")
            try:
                if not cv2.imwrite(snap_path, full_frame):
                    snap_path = None
                else:
                    self._last_snap_path = snap_path
                    self._display_frame(full_frame, self._snap_lbl)
                    self._snap_info.setText(f"Full frame: {base_name}_full.jpg")
            except Exception as e:
                self._sb.showMessage(f"Warning: cannot save full frame: {str(e)[:50]}")
                snap_path = None

        # Save YOLO crop separately (for daily HTML report)
        if crop is not None and crop.size > 0:
            self._last_crop = crop
            crop_path = os.path.join(CROPS_DIR, f"{base_name}_crop.jpg")
            try:
                if not cv2.imwrite(crop_path, crop):
                    crop_path = None
                else:
                    self._last_crop_path = crop_path
            except Exception:
                crop_path = None
        else:
            self._last_crop = None
            self._last_crop_path = None
            if snap_path is None:
                self._snap_lbl.setText("No frame available")

        # ── Update stats ──
        self._total += 1
        if is_leak:
            self._failed += 1
        else:
            self._passed += 1
        self._update_stats()

        # ── Append to log + CSV ──
        _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        yolo_conf = result.get("yolo_conf", 0.0)
        self._log_result(
            vin=self._last_vin,
            timestamp=_ts,
            auto_result=res_str,
            confidence=f"{conf:.1f}%",
            yolo_conf=f"{yolo_conf:.3f}",
            full_frame_path=snap_path or "—",
            yolo_crop_path=crop_path or "—",
            manual_label="Pending",
            manual_image_path="—",
            source=source,
        )

        # ── Generate per-car HTML report (saved into reports/YYYY-MM-DD/) ──
        date_str, time_str = _ts.split(" ", 1)
        day_reports_dir = os.path.join(REPORTS_DIR, date_str)
        os.makedirs(day_reports_dir, exist_ok=True)
        try:
            generate_car_report(
                vin=self._last_vin,
                date_str=date_str,
                time_str=time_str,
                auto_result=res_str,
                manual_confirm="Pending",
                photo_path=crop_path,    # HTML report shows YOLO crop
                reports_dir=day_reports_dir,
                base_dir=BASE_DIR
            )
        except Exception as e:
            print(f"Failed to generate initial report: {e}")

        # ── Enable confirm buttons ──
        self._btn_confirm_noleak.setEnabled(True)
        self._btn_confirm_engine.setEnabled(True)
        self._btn_confirm_tmoil.setEnabled(True)
        self._btn_confirm_both.setEnabled(True)
        self._confirm_status.setText("")

        self._stat_last.setText(res_str)
        self._sb.showMessage(
            f"Result: {res_str}  ({conf:.1f}%)  —  VIN: {self._last_vin}"
        )

    # ════════════════════════════════════════════════════════════════ #
    #   MANUAL CONFIRM
    # ════════════════════════════════════════════════════════════════ #

    @pyqtSlot(str)
    def _on_confirm(self, label: str):
        """
        Save FULL FRAME (not crop) to training data folder.
        Update the CSV row and HTML report with manual label.
        """
        if self._last_full_frame is None:
            self._confirm_status.setText("No capture to save — run inspection first")
            return

        dest_dir = DATA_DIRS.get(label, DATA_DIRS["No Leak"])
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception as e:
            self._confirm_status.setText("Error: cannot create folder")
            self._sb.showMessage(f"Error creating data directory: {e}")
            return

        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_vin = _safe_filename(self._last_vin)
        fname    = f"{ts}_{safe_vin}_full.jpg"
        dest     = os.path.join(dest_dir, fname)

        try:
            # Save FULL FRAME for training (not the YOLO crop)
            if not cv2.imwrite(dest, self._last_full_frame):
                self._confirm_status.setText("Error: Failed to save image")
                self._sb.showMessage("Error saving full frame to data folder")
                return

            # ── Update session log table ──
            # Since new items are inserted at the top, the most recent car is at row 0
            row = 0
            if row < self._log.rowCount():
                self._log.setItem(row, 5, QTableWidgetItem(label))          # Manual Label
                self._log.setItem(row, 6, QTableWidgetItem(os.path.basename(dest)))  # Manual Image

            # ── Update CSV: rewrite last row with manual info ──
            self._update_csv_manual(label, dest)

            # ── Disable confirm buttons until next inspection ──
            self._btn_confirm_noleak.setEnabled(False)
            self._btn_confirm_engine.setEnabled(False)
            self._btn_confirm_tmoil.setEnabled(False)
            self._btn_confirm_both.setEnabled(False)
            self._confirm_status.setText(f"✅ Saved as: {label}")
            self._sb.showMessage(f"Manual confirm '{label}' — full frame saved → {dest}")

            # ── Regenerate HTML report with confirmed label ──
            try:
                row_idx = 0
                auto_result = "UNKNOWN"
                ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if row_idx < self._log.rowCount():
                    auto_result = self._log.item(row_idx, 3).text()
                    ts_str = self._log.item(row_idx, 2).text()
                date_str, time_str = ts_str.split(" ", 1)
                day_reports_dir = os.path.join(REPORTS_DIR, date_str)
                os.makedirs(day_reports_dir, exist_ok=True)
                generate_car_report(
                    vin=self._last_vin,
                    date_str=date_str,
                    time_str=time_str,
                    auto_result=auto_result,
                    manual_confirm=label,
                    photo_path=self._last_crop_path,   # report still shows YOLO crop
                    reports_dir=day_reports_dir,
                    base_dir=BASE_DIR
                )
            except Exception as e:
                print(f"Failed to regenerate report after confirm: {e}")

        except Exception as e:
            self._confirm_status.setText(f"Error: {str(e)[:40]}")
            self._sb.showMessage(f"Error saving manual confirm: {str(e)[:80]}")

    # ════════════════════════════════════════════════════════════════ #
    #   DEBUG TEST IMAGE
    # ════════════════════════════════════════════════════════════════ #

    @pyqtSlot()
    def _on_load_test_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Test Image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if not path:
            return
        self._test_img_path = path
        self._test_path_lbl.setText(os.path.basename(path))
        self._btn_run_test.setEnabled(True)

    @pyqtSlot()
    def _on_run_test(self):
        if not hasattr(self, "_test_img_path"):
            return
        frame = cv2.imread(self._test_img_path)
        if frame is None:
            QMessageBox.warning(self, "Load Error", "Cannot read image file.")
            return

        fname = os.path.basename(self._test_img_path)
        self._last_vin = f"TEST-{os.path.splitext(fname)[0][:20]}"
        self._vin_disp.setText(f"VIN: {self._last_vin}")
        self._stat_vin.setText(self._last_vin[:16])
        self._display_frame(frame, self._feed_lbl)
        self._set_phase("ANALYZING (TEST)")
        self._run_pipeline(frame, source="TEST_IMAGE")

    # ════════════════════════════════════════════════════════════════ #
    #   REPORTS
    # ════════════════════════════════════════════════════════════════ #

    @pyqtSlot()
    def _on_open_report(self):
        os.startfile(REPORTS_DIR)

    @pyqtSlot()
    def _on_generate_summary(self):
        if not os.path.exists(CSV_PATH):
            QMessageBox.information(self, "No Data", "No inspection log found. Inspect a part first.")
            return

        target_date_str = None
        if hasattr(self, "_summary_date"):
            target_date_str = self._summary_date.date().toString("yyyy-MM-dd")
            self._sb.showMessage(f"Generating summary report for {target_date_str}...")
        else:
            self._sb.showMessage("Generating summary report...")

        # Session summary goes into reports/sessions/YYYY-MM-DD/
        date_key = target_date_str or datetime.now().strftime("%Y-%m-%d")
        session_out_dir = os.path.join(SESSIONS_DIR, date_key)
        os.makedirs(session_out_dir, exist_ok=True)

        try:
            report_path = generate_summary_report(
                CSV_PATH, session_out_dir, BASE_DIR, target_date=target_date_str
            )
            if report_path and os.path.exists(report_path):
                os.startfile(report_path)
                self._sb.showMessage(f"Session summary saved → {report_path}")
            else:
                QMessageBox.information(self, "No Data",
                    f"No inspection data found for: {date_key}")
                self._sb.showMessage("No data found for selected date")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate report: {e}")
            self._sb.showMessage(f"Error: {e}")

    @pyqtSlot()
    def _on_open_data(self):
        os.startfile(os.path.join(BASE_DIR, "data"))

    # ════════════════════════════════════════════════════════════════ #
    #   TIMER TICK
    # ════════════════════════════════════════════════════════════════ #

    def _tick(self):
        self._clock_lbl.setText(datetime.now().strftime("  %Y-%m-%d   %H:%M:%S"))
        if self._cam_connected:
            self._led_blink = not self._led_blink
            dot = "●" if self._led_blink else "○"
            self._cam_badge.setText(f"{dot} CAM LIVE")
            color = "#00ff88" if self._led_blink else "#00aa55"
            self._cam_badge.setStyleSheet(
                f"color:{color};font-size:12px;font-weight:bold;"
                "background:#001a00;border:1px solid #005500;"
                "border-radius:4px;padding:3px 10px;"
            )

    # ════════════════════════════════════════════════════════════════ #
    #   HELPERS
    # ════════════════════════════════════════════════════════════════ #

    def _set_phase(self, phase: str):
        self._phase = phase
        self._phase_badge.setText(f"  {phase}  ")

        color_map = {
            "IDLE":              ("#778899", "#111122", "#223344"),
            "WAITING FOR CAR":  ("#557799", "#0a1525", "#1e3a5f"),
            "CAR DETECTED":     ("#ffaa00", "#1a1000", "#443300"),
            "ANALYZING":        ("#44aaff", "#001830", "#005080"),
            "ANALYZING (TEST)": ("#44aaff", "#001830", "#005080"),
            "DONE — NO LEAK":   ("#00ff88", "#001a00", "#005500"),
            "DONE — LEAK DETECTED": ("#ff3333", "#1a0000", "#550000"),
            "NO PART DETECTED": ("#ffaa00", "#1a1000", "#443300"),
            "ERROR":            ("#ff3333", "#1a0000", "#550000"),
        }
        color, bg, border = color_map.get(phase, ("#778899", "#111122", "#223344"))
        self._phase_badge.setStyleSheet(
            f"color:{color};font-size:12px;font-weight:bold;"
            f"background:{bg};border:1px solid {border};"
            "border-radius:4px;padding:3px 10px;"
        )

    def _display_frame(self, frame: np.ndarray, label: QLabel):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix)

    def _update_stats(self):
        self._stat_total.setText(str(self._total))
        self._stat_pass.setText(str(self._passed))
        self._stat_fail.setText(str(self._failed))
        if self._total > 0:
            rate = int(self._passed / self._total * 100)
            self._rate_lbl.setText(f"{rate} %")
            self._rate_bar.setValue(rate)
        else:
            self._rate_lbl.setText("— %")

    def _log_result(self, vin, timestamp, auto_result, confidence, yolo_conf,
                    full_frame_path, yolo_crop_path,
                    manual_label, manual_image_path, source):
        """Add one row to the session table and append to the CSV."""
        self._log.insertRow(0)

        # Table columns: #, VIN, Timestamp, Auto Result, Confidence, Manual Label, Manual Image
        vals = [
            str(self._total),
            vin,
            timestamp,
            auto_result,
            confidence,
            manual_label,
            os.path.basename(full_frame_path) if full_frame_path != "—" else "—"
        ]
        for col, val in enumerate(vals):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            if col == 3:
                if auto_result == "OIL LEAK":
                    item.setForeground(QColor("#ff3333"))
                elif auto_result == "NO LEAK":
                    item.setForeground(QColor("#00ff88"))
            self._log.setItem(0, col, item)
        self._log.scrollToTop()

        # ── CSV (expanded columns) ──
        try:
            write_header = not os.path.exists(CSV_PATH)
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "id", "vin_id", "timestamp",
                        "auto_result", "confidence_pct", "yolo_conf",
                        "full_frame_path", "yolo_crop_path",
                        "manual_label", "manual_image_path",
                        "source"
                    ])
                writer.writerow([
                    self._total, vin, timestamp,
                    auto_result, confidence, yolo_conf,
                    full_frame_path, yolo_crop_path,
                    manual_label, manual_image_path,
                    source
                ])
        except Exception as e:
            print(f"CSV write error: {e}")

    def _update_csv_manual(self, label: str, image_path: str):
        """Rewrite the last row in the CSV with manual confirm info."""
        if not os.path.exists(CSV_PATH):
            return
        try:
            with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                return
            # Update manual_label (col 8) and manual_image_path (col 9) in last data row
            last = rows[-1]
            while len(last) < 11:
                last.append("—")
            last[8] = label
            last[9] = image_path
            rows[-1] = last
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
        except Exception as e:
            print(f"CSV manual update error: {e}")

    # ── Resize handler for overlay ─────────────────────────────────── #

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_overlay_lbl") and self._overlay_lbl.isVisible():
            self._overlay_lbl.setGeometry(
                0, 0, self._feed_lbl.width(), self._feed_lbl.height()
            )

    # ── Clean exit ─────────────────────────────────────────────────── #

    def closeEvent(self, event):
        if self._cam_thread:
            self._cam_thread.stop()
        save_config(self.config)
        event.accept()
