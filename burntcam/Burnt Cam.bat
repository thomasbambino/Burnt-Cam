@echo off
title Burnt Cam
cd /d "%~dp0"
rem Sets everything up the first time (Python, OBS, packages), then just checks.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
    echo.
    echo Setup did not finish - see the message above. Run Burnt Cam again to retry.
    pause
    exit /b 1
)
"%LOCALAPPDATA%\BurntCam\python\python.exe" "%~dp0burntcam.py" %*
if errorlevel 1 pause
