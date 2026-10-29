"""
Live scores for a frontend. Does not replace run.bat.

Start from the AU folder:
    python -m facial_cues.api.server
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

AU_ROOT = Path(__file__).resolve().parents[2]
os.chdir(AU_ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv

load_dotenv(AU_ROOT / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from facial_cues.config.schema import load_config
from facial_cues.pipeline.orchestrator import PipelineOrchestrator

_LINE = re.compile(r"^([A-Za-z]+)\s*(?:\(([0-9.]+)\))?\s*$")

pipeline: PipelineOrchestrator | None = None
_worker: threading.Thread | None = None


def _setting(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


def _url(path: str) -> str:
    base = _setting("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _split_line(text: str) -> dict:
    text = (text or "").strip()
    match = _LINE.match(text)
    if not match:
        return {"line": text, "label": "", "score": None}
    score = match.group(2)
    return {
        "line": text,
        "label": match.group(1).upper(),
        "score": float(score) if score is not None else None,
    }


import time
from fastapi.responses import StreamingResponse, JSONResponse

def snapshot(now: float | None = None) -> dict:
    if now is None:
        now = time.time()

    camera_running = bool((_worker is not None and _worker.is_alive()) or (pipeline is not None and getattr(pipeline, "_is_running", False)))

    if pipeline is None:
        return {
            "ready": False,
            "camera_running": camera_running,
            "faces_detected": 0,
            "face_detected": False,
            "face_present": False,
            "deception": None,
            "emotion": None,
            "fps": 0.0,
            "face_cues": {},
            "timestamp": None,
        }

    # Extract detected faces from latest frame
    viz_data = getattr(pipeline, "_last_viz_data", {}) or {}
    faces = viz_data.get("faces", [])
    faces_detected = len(faces)
    face_detected = faces_detected > 0

    # Face presence tracking with single-frame jitter/flicker tolerance (1.0s window)
    if face_detected:
        pipeline._last_face_detected_time = now
        face_present = True
    else:
        last_seen = getattr(pipeline, "_last_face_detected_time", None)
        face_present = bool(last_seen is not None and (now - last_seen) <= 1.0)

    # Deception prediction tracking
    deception_text = getattr(pipeline, "_last_deception_text", "") or ""
    reasons = list(getattr(pipeline, "_last_deception_reasons", []) or [])
    emotion_text = getattr(pipeline, "_last_emotion_text", "") or ""
    emo_time = getattr(pipeline, "_last_emotion_time", 0.0)
    stored = getattr(pipeline, "_last_deception", {}) or {}
    latest = None
    if stored:
        latest = next(reversed(stored.values()))

    deception = _split_line(deception_text)
    deception["reasons"] = reasons
    deception["comment"] = reasons[0] if reasons else ""
    deception["p_lie"] = None if latest is None else latest.get("p_lie")
    deception["model_label"] = None if latest is None else latest.get("label")

    # Prediction timestamp and freshness validation
    pred_time = getattr(pipeline, "_last_prediction_time", 0.0)
    has_valid_prediction = bool(pred_time and pred_time > 0.0)

    # If face is lost (>1.0s without detection) or prediction is older than 4.5s, do NOT represent old prediction as fresh
    is_prediction_fresh = False
    pred_timestamp = None
    if has_valid_prediction and face_present:
        pred_age = max(0.0, now - pred_time)
        if pred_age <= 4.5:
            is_prediction_fresh = True
            pred_timestamp = float(pred_time)

    deception["timestamp"] = pred_timestamp

    if not is_prediction_fresh:
        if not face_present:
            deception["label"] = "WAITING"
            deception["p_lie"] = None
            deception["model_label"] = None
            deception["reasons"] = ["Awaiting face detection"] if not faces_detected else []
        elif not has_valid_prediction:
            deception["label"] = "WAITING"
            deception["p_lie"] = None
            deception["model_label"] = None
            deception["reasons"] = ["Calibrating facial cues baseline..."]

    # Extract facial cues telemetry
    face_cues = {}
    if faces:
        p_face = faces[0]
        hp = p_face.get("head_pose")
        head_pose = None
        if hp is not None:
            head_pose = {
                "pitch": round(float(getattr(hp, "pitch", 0.0)), 1),
                "yaw": round(float(getattr(hp, "yaw", 0.0)), 1),
                "roll": round(float(getattr(hp, "roll", 0.0)), 1),
            }
        face_cues = {
            "face_id": p_face.get("face_id", 0),
            "head_pose": head_pose,
            "action_units": {k: round(float(v), 3) for k, v in list(p_face.get("action_units", {}).items())[:8]},
            "events": [
                {
                    "type": e.get("event_type", ""),
                    "intensity": round(float(e.get("intensity", 0.0)), 2),
                    "aus": e.get("involved_aus", [])
                }
                for e in viz_data.get("events", [])[-3:]
            ]
        }

    # Emotion prediction tracking with freshness check
    has_valid_emotion = bool(emo_time and emo_time > 0.0)
    is_emotion_fresh = False
    emo_timestamp = None
    if has_valid_emotion and face_present:
        emo_age = max(0.0, now - emo_time)
        if emo_age <= 4.5:
            is_emotion_fresh = True
            emo_timestamp = float(emo_time)

    if is_emotion_fresh:
        emotion_obj = _split_line(emotion_text)
        emotion_obj["timestamp"] = emo_timestamp
    else:
        emotion_obj = {"line": "", "label": "", "score": None, "timestamp": None}

    # Ready when camera is actively running, face is present, and either deception or emotion has evaluated fresh
    ready_flag = bool(face_present and (is_prediction_fresh or is_emotion_fresh))

    # Overall snapshot timestamp
    snapshot_timestamp = pred_timestamp or (emo_timestamp if (is_emotion_fresh and face_present) else (float(now) if face_present else None))

    return {
        "ready": ready_flag,
        "camera_running": camera_running,
        "faces_detected": faces_detected,
        "face_detected": face_detected,
        "face_present": face_present,
        "deception": deception,
        "emotion": emotion_obj,
        "fps": round(getattr(pipeline, "_last_fps", 0.0), 1),
        "face_cues": face_cues,
        "timestamp": snapshot_timestamp,
    }


def _run_camera() -> None:
    assert pipeline is not None
    try:
        pipeline.run(visualize=True, show_gui_window=False)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Camera loop stopped")


app = FastAPI(title="Facial cues live scores", version="1.0")

_origins = [item.strip() for item in _setting("CORS_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global pipeline, _worker
    config = load_config(None)
    config.deception.enabled = True
    config.visualization.enabled = True
    config.output.write_per_frame = False
    pipeline = PipelineOrchestrator(config)
    pipeline.initialize()
    _worker = threading.Thread(target=_run_camera, name="facial-cues-camera", daemon=True)
    _worker.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    global pipeline, _worker
    if pipeline is not None:
        pipeline.shutdown()


def _index():
    return {
        "deception": _url(_setting("ENDPOINT_DECEPTION", "/deception")),
        "emotion": _url(_setting("ENDPOINT_EMOTION", "/emotion")),
        "live": _url(_setting("ENDPOINT_LIVE", "/live")),
        "health": _url(_setting("ENDPOINT_HEALTH", "/health")),
        "video_feed": _url("/video_feed"),
    }


def _health():
    alive = _worker is not None and _worker.is_alive()
    return {"status": "ok" if alive else "starting", "camera_running": alive}


def _live():
    return snapshot()


def _deception():
    body = snapshot()
    return {"ready": body["ready"], **(body["deception"] or {})}


def _emotion():
    body = snapshot()
    return {"ready": body["ready"], **(body["emotion"] or {})}


import asyncio
from fastapi import Request

@app.get("/video_feed")
async def video_feed(request: Request):
    """MJPEG streaming endpoint for the live webcam feed with full overlays."""
    async def frame_gen():
        last_sent_id = -1
        while True:
            if await request.is_disconnected():
                break
            current_id = getattr(pipeline, "_jpeg_frame_id", 0)
            if current_id != last_sent_id:
                frame_bytes = getattr(pipeline, "_last_jpeg", None)
                if frame_bytes is not None:
                    last_sent_id = current_id
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
            await asyncio.sleep(0.02)
    return StreamingResponse(frame_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/camera/start")
def camera_start():
    global pipeline, _worker
    if _worker is not None and _worker.is_alive():
        return {"status": "already_running", "camera_running": True}
    config = load_config(None)
    config.deception.enabled = True
    config.visualization.enabled = True
    config.output.write_per_frame = False
    pipeline = PipelineOrchestrator(config)
    pipeline.initialize()
    _worker = threading.Thread(target=_run_camera, name="facial-cues-camera", daemon=True)
    _worker.start()
    return {"status": "started", "camera_running": True}


@app.post("/camera/stop")
def camera_stop():
    global pipeline, _worker
    if pipeline is not None and pipeline._is_running:
        pipeline._is_running = False
        pipeline.shutdown()
    if _worker is not None:
        _worker.join(timeout=2.0)
    return {"status": "stopped", "camera_running": False}


@app.get("/camera/status")
def camera_status():
    alive = _worker is not None and _worker.is_alive()
    return {"camera_running": alive, "ready": snapshot()["ready"]}


app.add_api_route(_setting("ENDPOINT_INDEX", "/"), _index, methods=["GET"])
app.add_api_route(_setting("ENDPOINT_HEALTH", "/health"), _health, methods=["GET"])
app.add_api_route(_setting("ENDPOINT_LIVE", "/live"), _live, methods=["GET"])
app.add_api_route(_setting("ENDPOINT_DECEPTION", "/deception"), _deception, methods=["GET"])
app.add_api_route(_setting("ENDPOINT_EMOTION", "/emotion"), _emotion, methods=["GET"])


def main() -> None:
    import uvicorn

    uvicorn.run(
        "facial_cues.api.server:app",
        host=_setting("API_HOST", "0.0.0.0"),
        port=int(_setting("API_PORT", "8002")),
        reload=False,
    )


if __name__ == "__main__":
    main()
