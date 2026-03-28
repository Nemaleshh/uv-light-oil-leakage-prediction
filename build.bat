@echo off
echo ============================================
echo   UV OIL LEAK DETECTION - EXE BUILD SCRIPT
echo ============================================
echo.

echo [1/2] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python is installed.
    pause
    exit /b 1
)

echo.
echo [2/2] Building EXE with PyInstaller...
pyinstaller --clean UV_OilLeak.spec
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   BUILD COMPLETE!
echo   EXE Location: dist\UV_OilLeak_Detection.exe
echo ============================================
echo.
pause
