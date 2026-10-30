"""
TRUSIGHT Unified Process Launcher
Launches and orchestrates:
1. TRUSIGHT Voice Engine (port 8001) in its dedicated virtualenv
2. TRUSIGHT Face Engine (port 8002) in system Python
3. TRUSIGHT Gemini Evidence Verification Engine (port 8003)
4. TRUSIGHT Unified Gateway (port 8000)
Monitors health, launches default browser, and cleanly shuts down all processes on Ctrl+C.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Service Paths
VOICE_DIR = ROOT_DIR / "truesight-voice-analysis" / "voice_service"
VOICE_VENV_PY = VOICE_DIR / ".venv" / "Scripts" / "python.exe"
if not VOICE_VENV_PY.exists():
    raise RuntimeError(
        f"Voice Engine dedicated virtual environment not found at: {VOICE_VENV_PY}\n"
        "Please ensure the voice service .venv is created and populated."
    )
VOICE_PY = str(VOICE_VENV_PY)

FACE_DIR = ROOT_DIR / "RealTime-Facial-Expression-Recognition" / "AU"
GEMINI_DIR = ROOT_DIR / "TruSight_Gemini_API" / "TruSight_Gemini_API"
if not GEMINI_DIR.exists():
    GEMINI_DIR = ROOT_DIR / "Trusight_Gemini_API" / "TruSight_Gemini_API"
GATEWAY_DIR = ROOT_DIR


def resolve_python() -> str:
    """
    Resolve the Python 3.11 interpreter required by Face, Gemini, and Gateway services.
    Checks:
    1. TRUSIGHT_PYTHON environment variable
    2. Python launcher 'py -3.11'
    3. Standard and user Python 3.11 install locations on Windows
    4. Current sys.executable if it is Python 3.11
    Validates required dependencies (mediapipe, fitz, google.genai, groq, cv2, torch) before returning.
    """
    candidates = []
    if "TRUSIGHT_PYTHON" in os.environ:
        candidates.append(os.environ["TRUSIGHT_PYTHON"])

    try:
        py311_res = subprocess.run(
            ["py", "-3.11", "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if py311_res.returncode == 0:
            py311_path = py311_res.stdout.strip()
            if py311_path and Path(py311_path).exists():
                candidates.append(py311_path)
    except Exception:
        pass

    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        candidates.append(str(Path(local_app_data) / "Python" / "pythoncore-3.11-64" / "python.exe"))
        candidates.append(str(Path(local_app_data) / "Programs" / "Python" / "Python311" / "python.exe"))

    # Also check if current sys.executable is 3.11
    if sys.version_info.major == 3 and sys.version_info.minor == 11:
        candidates.append(sys.executable)

    selected_py = None
    for cand in candidates:
        if cand and Path(cand).exists():
            selected_py = cand
            break

    if not selected_py:
        raise RuntimeError(
            "TRUSIGHT requires a compatible Python 3.11 environment with mediapipe, PyMuPDF, groq, and google-genai.\n"
            "None of the candidate paths exist. Set the TRUSIGHT_PYTHON environment variable to your Python 3.11 executable."
        )

    # Preflight check: verify required imports in the selected interpreter
    preflight_script = (
        "import cv2, mediapipe, torch, fitz, google.genai, groq, fastapi, uvicorn;"
        "print('PREFLIGHT_OK')"
    )
    res = subprocess.run(
        [selected_py, "-c", preflight_script],
        capture_output=True,
        text=True,
    )
    if "PREFLIGHT_OK" not in res.stdout:
        err_msg = res.stderr.strip() or res.stdout.strip()
        raise RuntimeError(
            f"Selected Python interpreter ({selected_py}) failed preflight dependency check:\n{err_msg}\n"
            "Please ensure mediapipe, PyMuPDF, google-genai, groq, opencv-python, and torch are installed in this environment."
        )

    return selected_py


SHARED_PY = resolve_python()
FACE_PY = SHARED_PY
GEMINI_PY = SHARED_PY
GATEWAY_PY = SHARED_PY

VOICE_PORT = 8001
FACE_PORT = 8002
GEMINI_PORT = 8003
GATEWAY_PORT = 8000

processes: list[subprocess.Popen] = []


def check_url(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def clean_ports(ports: list[int]):
    """Cleanly terminate any stale processes holding the target ports on Windows."""
    for port in ports:
        try:
            res = subprocess.run(
                f'netstat -ano | findstr LISTENING | findstr :{port}',
                shell=True,
                capture_output=True,
                text=True,
            )
            for line in res.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[1].endswith(f":{port}"):
                    pid = parts[-1]
                    if pid.isdigit() and int(pid) != 0 and int(pid) != os.getpid():
                        subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, capture_output=True)
        except Exception:
            pass
    time.sleep(1.0)


def cleanup(signum=None, frame=None):
    print("\n[TRUSIGHT] Shutting down services cleanly...")
    for p in processes:
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    clean_ports([VOICE_PORT, FACE_PORT, GEMINI_PORT, GATEWAY_PORT])
    print("[TRUSIGHT] All services stopped.")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 65)
    print("      TRUSIGHT MULTIMODAL FORENSIC INTELLIGENCE PLATFORM")
    print("      Unified Face & Voice Analysis Dashboard")
    print("=" * 65)

    # Free ports from any previous session
    clean_ports([VOICE_PORT, FACE_PORT, GEMINI_PORT, GATEWAY_PORT])

    # 1. Start Voice Service on port 8001
    voice_env = os.environ.copy()
    voice_env["PYTHONUTF8"] = "1"
    voice_env["PYTHONIOENCODING"] = "utf-8"
    voice_env["VOICE_PORT"] = str(VOICE_PORT)

    print(f"\n[1/4] Starting Voice Analysis Engine on port {VOICE_PORT}...")
    p_voice = subprocess.Popen(
        [
            VOICE_PY,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(VOICE_PORT),
        ],
        cwd=str(VOICE_DIR),
        env=voice_env,
    )
    processes.append(p_voice)

    # 2. Start Face Service on port 8002
    face_env = os.environ.copy()
    face_env["PYTHONUTF8"] = "1"
    face_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    face_env["API_PORT"] = str(FACE_PORT)
    face_env["API_HOST"] = "127.0.0.1"

    print(f"[2/4] Starting Face Analysis Engine on port {FACE_PORT}...")
    p_face = subprocess.Popen(
        [
            FACE_PY,
            "-m",
            "facial_cues.api.server",
        ],
        cwd=str(FACE_DIR),
        env=face_env,
    )
    processes.append(p_face)

    # 3. Start Gemini RAG & Statement Verification Engine on port 8003
    gemini_env = os.environ.copy()
    gemini_env["PYTHONUTF8"] = "1"
    gemini_env["PORT"] = str(GEMINI_PORT)

    print(f"[3/4] Starting Gemini Evidence Verification Engine on port {GEMINI_PORT}...")
    p_gemini = subprocess.Popen(
        [
            GEMINI_PY,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(GEMINI_PORT),
        ],
        cwd=str(GEMINI_DIR),
        env=gemini_env,
    )
    processes.append(p_gemini)

    # 4. Start Gateway Server on port 8000
    gw_env = os.environ.copy()
    gw_env["PORT"] = str(GATEWAY_PORT)
    gw_env["VOICE_BACKEND"] = f"http://127.0.0.1:{VOICE_PORT}"
    gw_env["FACE_BACKEND"] = f"http://127.0.0.1:{FACE_PORT}"
    gw_env["GEMINI_BACKEND"] = f"http://127.0.0.1:{GEMINI_PORT}"

    print(f"[4/4] Starting Unified Gateway on port {GATEWAY_PORT}...")
    p_gw = subprocess.Popen(
        [
            GATEWAY_PY,
            str(ROOT_DIR / "trusight_server.py"),
        ],
        cwd=str(GATEWAY_DIR),
        env=gw_env,
    )
    processes.append(p_gw)

    dashboard_url = f"http://localhost:{GATEWAY_PORT}"
    print(f"\n[TRUSIGHT] Waiting for services to initialize...")

    # Wait for gateway and backends to become ready
    max_wait = 25
    start_time = time.time()
    gw_ready = False

    while time.time() - start_time < max_wait:
        if check_url(f"{dashboard_url}/api/health", timeout=0.8):
            gw_ready = True
            break
        time.sleep(0.5)

    print("\n" + "=" * 65)
    print("  TRUSIGHT PLATFORM READY!")
    print(f"  Unified Dashboard : {dashboard_url}")
    print(f"  Voice Service     : http://127.0.0.1:{VOICE_PORT}")
    print(f"  Face Service      : http://127.0.0.1:{FACE_PORT}")
    print(f"  Gemini RAG Engine : http://127.0.0.1:{GEMINI_PORT}")
    print("=" * 65)
    print("  Opening dashboard in your web browser...")
    print("  Press Ctrl+C to terminate all services.\n")

    try:
        webbrowser.open(dashboard_url)
    except Exception:
        pass

    warned = set()
    try:
        while True:
            # Check if any process exited unexpectedly
            for name, p in [("Voice Engine", p_voice), ("Face Engine", p_face), ("Gemini Engine", p_gemini), ("Gateway", p_gw)]:
                if p.poll() is not None and name not in warned:
                    print(f"\n[TRUSIGHT WARNING] {name} exited with code {p.returncode}")
                    warned.add(name)
            time.sleep(1.0)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
