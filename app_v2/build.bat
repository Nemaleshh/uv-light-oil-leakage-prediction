@echo off
REM ================================================================
REM  UV Oil Leak Detection v2 — PyInstaller Build Script
REM  Run from inside the app_v2\ folder on your LAPTOP.
REM  Output: dist\UVLeakDetector\   (copy this folder to factory PC)
REM ================================================================

setlocal
cd /d "%~dp0"

echo.
echo [1/5] Installing / upgrading PyInstaller...
pip install --quiet --upgrade pyinstaller
if errorlevel 1 ( echo ERROR: pip failed & pause & exit /b 1 )

echo [2/5] Cleaning old build artefacts...
if exist build\UVLeakDetector  rmdir /s /q build\UVLeakDetector
if exist dist\UVLeakDetector   rmdir /s /q dist\UVLeakDetector

echo [3/5] Running PyInstaller with spec file...
pyinstaller --noconfirm UVLeakDetector.spec
if errorlevel 1 ( echo ERROR: PyInstaller failed. See above. & pause & exit /b 1 )

echo [4/5] Copying runtime data into dist folder...

REM ── models (YOLO .pt + classifier .pkl) ──────────────────────────────
xcopy /I /Y /Q "models\*.*"  "dist\UVLeakDetector\models\"
if errorlevel 1 ( echo WARNING: models copy failed )

REM ── config.json — copy as-is (model_path / clf_path are always
REM    overridden at runtime by dashboard.py using BASE_DIR)
copy /Y "config.json" "dist\UVLeakDetector\config.json"
if errorlevel 1 ( echo WARNING: config.json copy failed )

REM ── ui assets (logo, stylesheet) ─────────────────────────────────────
xcopy /I /Y /Q "ui\*.*"  "dist\UVLeakDetector\ui\"
if errorlevel 1 ( echo WARNING: ui copy failed )

REM ── Pre-create writable runtime folders ─────────────────────────────
REM    These are created relative to the EXE location on the target PC.
mkdir "dist\UVLeakDetector\detected_images"        2>nul
mkdir "dist\UVLeakDetector\detected_images\crops"  2>nul
mkdir "dist\UVLeakDetector\reports"                2>nul
mkdir "dist\UVLeakDetector\reports\sessions"       2>nul
mkdir "dist\UVLeakDetector\data\no_leak"           2>nul
mkdir "dist\UVLeakDetector\data\engine_oil"        2>nul
mkdir "dist\UVLeakDetector\data\tm_oil_leak"       2>nul
mkdir "dist\UVLeakDetector\data\both_leaks"        2>nul

echo [5/5] Build complete!
echo.
echo =================================================================
echo  Standalone app ready at:
echo    %~dp0dist\UVLeakDetector\
echo.
echo  Copy the ENTIRE "UVLeakDetector" folder to the factory PC.
echo  Double-click UVLeakDetector.exe to run — no Python needed.
echo =================================================================
echo.
pause
