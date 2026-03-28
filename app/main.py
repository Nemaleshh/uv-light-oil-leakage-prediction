"""
UV Engine Oil Leak Detection System
Entry point — launches the PyQt5 IoT dashboard.
"""

import sys
import os

# Suppress OpenCV FFmpeg log spam (like "mjpeg overread 8")
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# Make sure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from app.ui.dashboard import Dashboard


def _resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller EXE """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def load_stylesheet():
    qss_path = _resource_path(os.path.join("app", "ui", "styles.qss"))
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            return f.read()
    else:
        print(f"Warning: Stylesheet not found at {qss_path}")
    return ""


def main():
    # Enable high-DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("UV Oil Leak Detection System")
    app.setOrganizationName("Stellatis Engineering")
    app.setStyleSheet(load_stylesheet())

    window = Dashboard()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
