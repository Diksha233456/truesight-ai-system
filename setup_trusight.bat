@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Setup TRUSIGHT Environment

echo ============================================================
echo   TRUSIGHT Platform Environment Setup
echo ============================================================

echo.
echo [1/2] Installing Gateway and Face Analysis dependencies...
if not defined TRUSIGHT_PYTHON (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "TRUSIGHT_PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    )
)
if defined TRUSIGHT_PYTHON (
    "%TRUSIGHT_PYTHON%" -m pip install -r requirements.txt
) else (
    python -m pip install -r requirements.txt
)

echo.
echo [2/2] Setting up Voice Service dedicated environment...
cd truesight-voice-analysis\voice_service
if not exist ".venv" (
    echo Creating dedicated .venv for voice service...
    python -m venv .venv
)

echo Activating .venv and installing voice dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
call deactivate
cd ..\..

echo.
echo ============================================================
echo   Setup completed successfully!
echo   To launch Trusight: run .\run_trusight.bat
echo ============================================================
pause
