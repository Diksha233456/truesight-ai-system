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

python trusight_launcher.py

endlocal
