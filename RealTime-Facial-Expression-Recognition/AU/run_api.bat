@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
) else if exist "..\.venv\Scripts\python.exe" (
    set "PY_EXE=..\.venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)
if "%API_PORT%"=="" set "API_PORT=8002"
echo Starting Live Facial Cues FastAPI Server on port %API_PORT%...
"%PY_EXE%" -m facial_cues.api.server
endlocal
