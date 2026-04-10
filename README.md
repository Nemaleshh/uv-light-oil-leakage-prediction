# UV Engine Oil Leak Detection System
**Industrial IoT — Assembly Line Quality Control**

This project is a computer vision application developed for **Stellantis**, designed to automate the visual inspection of vehicle engines for oil leaks on the assembly line. By utilizing UV lighting and fluorescent dye added to the engine fluids, this application detects leaks via bright light-blue/cyan glows on the engine block, oil pan, and surrounding chassis components.


<img width="1920" height="1080" alt="phott" src="https://github.com/user-attachments/assets/a9bcdde3-a3ef-40c2-a28e-10c442a724dd" />

## Features

- **High-Accuracy Dye Detection**: Custom-tuned OpenCV color isolation targeting the specific light-blue UV dye color (`#85b5f6`), demanding both high brightness and high saturation.
- **Advanced Noise Filtering**: Aggressive HSV tuning specifically ignores common industrial false positives:
  - Yellow/green quality-control stickers (Hue shifted filter).
  - Violet/purple diffuse glows from the black plastic oil pan and metal chassis under UV lighting (Hue & Saturation filter).
  - Small specular chrome edge glares (Area filtering > 800px²).
- **Dual Camera Architecture**: Supports standard USB Webcams (e.g. `0`, `1`) or networked IP Cameras via MJPEG/RTSP stream URLs.
- **Live Quality Control Dashboard**: 
  - Real-time video feed with dynamically drawn bounding boxes and text overlays highlighted in RED over detected leaks.
  - Live inspection counters: Total cars scanned, Pass (OK), Fail (Leak), Pass Rate %, and current vehicle WIN/VIN tracking.
- **Manual Image Verification**: Users can upload high-resolution static verification photos to run through the detection pipeline without starting the camera feed.
- **Automated Reporting**: Maintains an internal session log and automatically exports all inspection decisions to Excel (`.xlsx`) or CSV reports, saving high-quality annotated snapshots to disk for record-keeping.

## Technical Stack

- **Python 3.10+**
- **PyQt5**: Robust graphical UI framework, styled with a modern dark industrial theme (`styles.qss`).
- **OpenCV (`cv2`)**: Core computer vision library handling color space conversion (BGR to HSV), contour detection, noise erosion/dilation, and mask rendering.
- **Numpy**: Fast array processing for mask intersections and threshold generation.
- **Pandas**: Efficient export of tabular inspection data to Excel endpoints.

## Project Structure

```text
app/
├── core/
│   ├── detector.py      # Core OpenCV HSV filter logic and leak detection algorithm
│   └── reporter.py      # Pandas-based excel report generator for saving inspection results
├── ui/
│   ├── dashboard.py     # Main PyQt5 QtWidget definitions, stats layout, image rendering
│   ├── camera_thread.py # Async VideoCapture pulling frames to prevent UI blocking
│   └── styles.qss       # Dark industrial CSS injection for Qt styling
├── assets/              # Icons and general media resources
└── main.py              # Application entry point binding UI and Core
```

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Nemaleshh/uv-light-oil-leakage-prediction.git
   cd uv-light-oil-leakage-prediction
   ```

2. **Create a Python Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Core dependencies: `opencv-python`, `numpy`, `PyQt5`, `pandas`, `openpyxl`, `jupyter`)*

## How to Run

### Standard Mode
```bash
python app/main.py
```

### Development Mode (Windows)
```bash
run_dev.bat
```

## Usage Guide

1. **Connect Camera:** Select either "USB Web Camera" or "IP Camera" and enter the corresponding ID (e.g. `0`) or URL string. Click **Connect**.
2. **Setup Inspection:** Enter the engine's WIN or VIN number into the input field.
3. **Scan:** Click **Start Inspection**. The application will trigger a countdown, perform a single-frame capture analysis, and log the test. 
4. **Results:** If a leak is detected, an overlay will display "OIL LEAK DETECTED", a red bounding box will track the leak coordinates, and the "Failed" inspection counter increments.
5. **Get Report:** Click "Generate / Open Report" to open a spreadsheet mapping each scanned WIN to its Pass/Fail grade and time-stamped inspection image.

## Analysis & Data Science

The project includes Jupyter notebooks for data exploration and model analysis:
- **`analysis/analysis.ipynb`**: Primary analysis and model evaluation workflows
- **`analysissss/ana.ipynb`**: Additional experimental analysis and dataset exploration

Run analysis notebooks with:
```bash
jupyter notebook analysis/analysis.ipynb
```

## Output & Artifacts

- **`detected_images/`**: Stores high-resolution inspection snapshots with leak annotations.
- **`reports/`**: Contains auto-generated Excel and CSV inspection reports with timestamps and pass/fail records.
- **Build folder**: Compiled application binaries are generated here during build process.

## Build & Deployment

To build a standalone executable (Windows):
```bash
build.bat
```

The compiled application (`UV_OilLeak_Detection.exe`) will be available in the `dist/` folder.

## Configuration

Edit `config.json` to customize:
- Camera settings (resolution, frame rate, brightness)
- HSV color range thresholds for dye detection
- Detection sensitivity and area filtering parameters
- Report export formats and output paths

## Troubleshooting

- **Camera not detected**: Verify USB connection or IP camera URL accessibility.
- **False positives**: Adjust HSV thresholds in `config.json` or review dark mode lighting conditions.
- **Slow detection**: Reduce camera resolution or check system CPU/memory usage.
- **Report generation errors**: Ensure `reports/` directory exists and has write permissions.

## Contributing

This is a proprietary industrial quality control system developed for **Stellantis**. Contributions are welcome for bug fixes and performance improvements.

## License

This project is proprietary to **Stellantis** and should not be distributed without authorization.

---

**Last Updated:** April 2026  
**Version:** 1.0  
**Status:** Production-Ready
