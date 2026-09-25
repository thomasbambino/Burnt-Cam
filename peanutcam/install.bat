@echo off
rem One-time setup: creates a local Python environment and installs PeanutCam's packages.
cd /d "%~dp0"
if not exist .venv (
    py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
    echo Could not create a Python environment. Install Python 3.11 or 3.12 from python.org
    echo and tick "Add python.exe to PATH" during setup, then run this again.
    pause
    exit /b 1
)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
echo.
echo Done! Double-click run.bat to start PeanutCam.
pause
