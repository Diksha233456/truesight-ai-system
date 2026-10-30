@echo off
setlocal
cd /d "%~dp0"
title TRUSIGHT Unified Platform

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo   TRUSIGHT MULTIMODAL FORENSIC INTELLIGENCE SUITE
echo   Starting Unified Services...
echo ============================================================

if not defined TRUSIGHT_PYTHON (
    if exist "%LOCALAPPDATA%\Python\pythoncore-3.11-64\python.exe" (
        set "TRUSIGHT_PYTHON=%LOCALAPPDATA%\Python\pythoncore-3.11-64\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "TRUSIGHT_PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    )
)

if defined TRUSIGHT_PYTHON (
    "%TRUSIGHT_PYTHON%" trusight_launcher.py
) else (
    py -3.11 trusight_launcher.py 2>nul || python trusight_launcher.py
)

endlocal
