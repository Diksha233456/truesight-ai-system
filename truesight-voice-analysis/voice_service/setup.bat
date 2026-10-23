@echo off
REM setup.bat — One-click environment setup for TRUSIGHT Voice Analysis Service (Windows)
REM Run from the voice_service/ project root with:  .\setup.bat
REM
REM IMPORTANT: This script deliberately skips Anaconda Python builds because they
REM            ship an experimental MINGW-built numpy that causes runtime crashes
REM            on Windows. It finds the official CPython installer build instead.

setlocal EnableDelayedExpansion

echo ============================================================
echo  TRUSIGHT Voice Analysis Service -- Environment Setup
echo ============================================================
echo.

REM ---- Find a valid CPython 3.11/3.12/3.13 installation ----
REM (CPython installer puts python.exe under AppData\Local\Programs\Python\PythonXXX)

SET PYTHON_EXE=
SET PYTHON_CANDIDATES=

FOR %%V IN (313 312 311) DO (
    SET "_P=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python%%V\python.exe"
    IF EXIST "!_P!" (
        IF NOT DEFINED PYTHON_EXE (
            SET PYTHON_EXE=!_P!
        )
    )
)

IF NOT DEFINED PYTHON_EXE (
    where python.exe >nul 2>&1
    IF NOT ERRORLEVEL 1 (
        FOR /F "tokens=*" %%I IN ('where python.exe') DO (
            IF NOT DEFINED PYTHON_EXE SET PYTHON_EXE=%%I
        )
    )
)

IF NOT DEFINED PYTHON_EXE (
    echo [ERROR] Could not find a CPython 3.11/3.12/3.13 installation under:
    echo           %%USERPROFILE%%\AppData\Local\Programs\Python\ or on PATH.
    echo.
    echo         Please install Python 3.11+ from https://python.org
    echo         Use the "Windows installer (64-bit)" option.
    echo         Do NOT use Anaconda or MINGW builds for this project.
    exit /b 1
)

echo [INFO] Found Python: %PYTHON_EXE%
"%PYTHON_EXE%" --version
echo.

REM ---- Create virtual environment ----
IF EXIST .venv\Scripts\python.exe (
    REM Check if it's the same Python (MSC build, not MINGW)
    .venv\Scripts\python.exe -c "import sys; print(sys.version)" | findstr /C:"MINGW" >nul
    IF NOT ERRORLEVEL 1 (
        echo [WARN] Existing .venv uses MINGW Python. Recreating with CPython...
        rmdir /S /Q .venv
    )
)

IF NOT EXIST .venv\Scripts\python.exe (
    echo [INFO] Creating virtual environment in .venv...
    "%PYTHON_EXE%" -m venv .venv
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

REM ---- Install dependencies ----
echo [INFO] Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip -q

echo [INFO] Installing dependencies from requirements.txt...
.venv\Scripts\pip install -r requirements.txt
IF ERRORLEVEL 1 (
    echo [ERROR] pip install failed. See errors above.
    exit /b 1
)
echo.

REM ---- Setup FFmpeg binary for audio decoding ----
.venv\Scripts\python.exe -c "import imageio_ffmpeg, shutil, os; src = imageio_ffmpeg.get_ffmpeg_exe(); dst = os.path.join(os.path.dirname(src), 'ffmpeg.exe'); shutil.copy2(src, dst) if not os.path.exists(dst) else None; dst2 = os.path.abspath(os.path.join(os.path.dirname(src), '..', '..', '..', 'Scripts', 'ffmpeg.exe')); shutil.copy2(src, dst2) if not os.path.exists(dst2) else None" >nul 2>&1

REM ---- Pretrained model resolution & caching ----
echo [INFO] Resolving and verifying pretrained model...
.venv\Scripts\python.exe -m app.model_loader
IF ERRORLEVEL 1 (
    echo [WARN] Pretrained model download could not be completed during setup.
    echo        The application will attempt to resolve/cache the model on launch,
    echo        or you can set TRUSIGHT_MODEL_PATH to point to a local model file.
) ELSE (
    echo [INFO] Pretrained model is cached and ready for inference.
)

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To run the service:
echo.
echo    .venv\Scripts\activate
echo    uvicorn app.main:app --reload
echo.
echo  Or without activating:
echo    .venv\Scripts\uvicorn app.main:app --reload
echo.
echo  Then open: http://127.0.0.1:8000/docs
echo ============================================================

endlocal
