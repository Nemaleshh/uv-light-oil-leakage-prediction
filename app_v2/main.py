"""
main.py — UV Engine Oil Leak Detection System v2
Entry point: creates PyQt5 application and launches Dashboard.
"""
import sys
import os

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
import traceback
import datetime

from ui.dashboard import Dashboard

def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Catch all unhandled exceptions and show a dialog instead of crashing."""
    # Ignore KeyboardInterrupts
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Format the traceback
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print("CRITICAL UNHANDLED EXCEPTION:\n", tb_str)

    # Log to a crash file
    try:
        crash_log = os.path.join(os.path.dirname(__file__), "crash_log.txt")
        with open(crash_log, "a") as f:
            f.write(f"\n--- Crash Report: {datetime.datetime.now()} ---\n")
            f.write(tb_str)
    except Exception:
        pass

    # Show a dialog to the user
    error_box = QMessageBox()
    error_box.setIcon(QMessageBox.Critical)
    error_box.setWindowTitle("Application Error")
    error_box.setText("An unexpected error occurred!")
    error_box.setInformativeText("The application caught a critical error but recovered to prevent closing. Please check the 'crash_log.txt' file for details.")
    error_box.setDetailedText(tb_str)
    error_box.exec_()

def main():
    # Register the global exception handler
    sys.excepthook = global_exception_handler

    # HiDPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("UV Oil Leak Detection")
    app.setApplicationVersion("2.0")

    # Load stylesheet
    qss_path = os.path.join(os.path.dirname(__file__), "ui", "styles.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    window = Dashboard()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
