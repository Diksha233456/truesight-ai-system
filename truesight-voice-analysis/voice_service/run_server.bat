@echo off
setlocal
cd /d "%~dp0"
title TRUSIGHT Voice Analysis Service

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not "%~1"=="" set "VOICE_PORT=%~1"
if "%VOICE_PORT%"=="" set "VOICE_PORT=8000"

echo ============================================================
echo   TRUSIGHT Voice Analysis Service (Standalone)
echo   Local URL:    http://127.0.0.1:%VOICE_PORT%
echo   Localhost:    http://localhost:%VOICE_PORT%
echo   Swagger Docs: http://127.0.0.1:%VOICE_PORT%/docs
echo ============================================================
echo.
echo Launching server on port %VOICE_PORT%...
echo (Opening browser window in 2 seconds...)
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:%VOICE_PORT%"

call .venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port %VOICE_PORT%

pause
endlocal
