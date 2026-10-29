"""
main.py

FastAPI entrypoint for the TRUSIGHT Voice Analysis service.

This module is intentionally thin: it wires together utils (I/O),
feature_extractor (acoustic features), predictor (ML inference), and
explanation (human-readable reasoning) behind two HTTP endpoints.

Run with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model import AnalyzeResponse, HealthResponse, ErrorResponse, StructuredExplanation
from app.utils import (
    load_audio_from_bytes,
    check_voice_presence,
    evaluate_audio_speech_quality,
    SPEECH_STATUS_VOICE_DETECTED,
    SPEECH_STATUS_VOICE_NOT_DETECTED,
    SPEECH_STATUS_AUDIO_TOO_SHORT,
    SPEECH_STATUS_AUDIO_QUALITY_TOO_LOW,
    MIN_DURATION_SECONDS,
    AudioValidationError,
    AudioProcessingError,
)
from app.feature_extractor import FeatureExtractor, FeatureExtractionError
from app.predictor import Predictor, ModelNotLoadedError
from app.explanation import generate_explanations, generate_structured_explanation

# ── Live analysis imports ────────────────────────────────────────────────────
from app.live_manager import LiveSessionManager
from app.live_audio import VALID_WINDOWS, DEFAULT_WINDOW
from app.live_analyzer import SegmentResult
import app.live_analyzer as _live_analyzer
import time as _time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("voice_service.main")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

app = FastAPI(
    title="TRUSIGHT Voice Analysis Service",
    description=(
        "Standalone voice-analysis module for the TRUSIGHT multimodal truth "
        "verification system. Estimates deception likelihood from vocal "
        "acoustic cues only. This module does NOT make a final truth/lie "
        "decision on its own — see README for scope and limitations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons, constructed once at import time (i.e. at process startup).
feature_extractor = FeatureExtractor()
predictor = Predictor()

# ── Live session manager singleton ───────────────────────────────────────────
live_manager = LiveSessionManager()

# Inject singletons into live_analyzer so it never re-loads models.
_live_analyzer.set_singletons(feature_extractor, predictor, generate_explanations)


_STATIC_DIR = Path(__file__).parent / "static"

# Mount static files directory (CSS/JS assets if added later)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.on_event("startup")
def print_startup_banner():
    # Warm up Librosa & Numba JIT compilation in background so first live segment evaluates in <200ms
    try:
        import numpy as _np
        from app.utils import LoadedAudio as _LA
        _dummy = _np.zeros(16000, dtype=_np.float32)
        _la = _LA(samples=_dummy, sample_rate=16000, duration_seconds=1.0, source_filename="warmup.wav")
        feature_extractor.extract(_la)
        logger.info("[Voice Engine] Numba JIT & feature extractor warmed up successfully.")
    except Exception as _w_err:
        logger.debug("[Voice Engine] Warmup note: %s", _w_err)

    banner = (
        "\n"
        + "=" * 65 + "\n"
        + "  TRUSIGHT Voice Service is LIVE and READY!\n\n"
        + "  * OPEN IN BROWSER: http://localhost:8000\n"
        + "  * DIRECT IP LINK : http://127.0.0.1:8000\n"
        + "  * API DOCS       : http://localhost:8000/docs\n"
        + "=" * 65 + "\n"
    )
    try:
        print(banner, flush=True)
    except UnicodeEncodeError:
        print(banner.encode("ascii", "replace").decode("ascii"), flush=True)


@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the TRUSIGHT web UI."""
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HealthResponse(service="Voice Analysis Service", status="running")


@app.get("/health", response_model=HealthResponse, tags=["health"])
@app.get("/api/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """Liveness check for load balancers and clients."""
    model_version = predictor.metadata.get("model_version") if (predictor.metadata and hasattr(predictor, "metadata")) else None
    return HealthResponse(
        service="Voice Analysis Service",
        status="running",
        model_loaded=predictor.is_ready(),
        model_version=model_version,
    )


@app.post("/reload-model", tags=["admin"])
def reload_model() -> dict:
    """
    Hot-reload the pretrained model from cache or disk without restarting the service.
    """
    predictor.reload()
    model_version = predictor.metadata.get("model_version") if (predictor.metadata and hasattr(predictor, "metadata")) else None
    return {
        "model_loaded": predictor.is_ready(),
        "model_version": model_version,
        "model_path": predictor.model_path,
    }


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or missing audio file"},
        422: {"model": ErrorResponse, "description": "Audio failed acoustic processing"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["analysis"],
)
async def analyze_audio(
    audio: UploadFile = File(..., description="Audio recording (wav/flac/ogg/mp3/m4a/mpeg/mpg)"),
    include_features: bool = Query(
        False, description="If true, include raw extracted acoustic features in the response."
    ),
) -> AnalyzeResponse:
    """
    Analyze an uploaded audio recording and return a deception-likelihood
    estimate based on vocal acoustic cues.

    This endpoint is one modality of the larger TRUSIGHT system and its
    output is not a standalone lie-detection verdict.
    """
    if audio is None:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    file_bytes = await audio.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )

    filename = audio.filename or "upload.wav"

    # --- Load & preprocess ---
    try:
        loaded_audio = load_audio_from_bytes(file_bytes, filename, allow_silence=True)
    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AudioProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- Check voice presence before running ML prediction ---
    has_speech, speech_note, speech_status = evaluate_audio_speech_quality(loaded_audio)
    if not has_speech:
        verdict = "Inconclusive" if speech_status == SPEECH_STATUS_AUDIO_QUALITY_TOO_LOW else "No Speech Detected"
        raw_warnings = getattr(loaded_audio, "quality_warnings", []) or []
        quality_notes = [speech_note] + raw_warnings
        return AnalyzeResponse(
            verdict=verdict,
            score=None,
            confidence=None,
            speech_status=speech_status,
            speech_detected=False,
            explanation=[
                f"{verdict}: {speech_note}",
                "Silence, non-vocal background noise, and microphone hiss are never assigned deception or truth scores.",
                "Please provide a clear recording containing spoken human voice.",
            ],
            structured_explanation=StructuredExplanation(
                acoustic_observations=[],
                quality_warnings=quality_notes,
                prediction_context=(
                    f"Prediction was withheld ({speech_status}). Silence and poor-quality audio "
                    "are never classified as truthful or deceptive."
                ),
                limitations=[
                    "This experimental model cannot determine whether a person is telling the truth.",
                    "Voice analysis requires clear spoken human voice. Background noise, typing, and silence cannot be evaluated for truthfulness.",
                    "Indian-English deception detection has not been validated.",
                ],
            ),
            features=None,
            quality_warnings=quality_notes,
        )

    # --- Feature extraction (only on verified speech) ---
    try:
        if predictor.is_multimodal:
            # Upgraded pipeline: extract 814-dim (Acoustic + Wav2Vec2 + Verbal)
            from app.deep_feature_extractor import extract_deep_features
            features_obj, _ = feature_extractor.extract(loaded_audio)
            vector = extract_deep_features(loaded_audio)
            features = features_obj
        else:
            # Legacy pipeline: extract 42-dim acoustic features only
            features, vector = feature_extractor.extract(loaded_audio)
    except FeatureExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Double check voice presence using extracted features
    has_speech_feat, feat_note, _ = evaluate_audio_speech_quality(loaded_audio, features)
    if not has_speech_feat:
        return AnalyzeResponse(
            verdict="No Speech Detected",
            score=None,
            confidence=None,
            speech_status=SPEECH_STATUS_VOICE_NOT_DETECTED,
            speech_detected=False,
            explanation=[
                f"No Speech Detected: {feat_note}",
                "Signal level is consistent with ambient room silence or background noise.",
                "Please upload a recording containing clear spoken voice.",
            ],
            structured_explanation=StructuredExplanation(
                quality_warnings=[feat_note],
                prediction_context="Prediction was withheld because vocal phonation was insufficient.",
                limitations=[
                    "This experimental model cannot determine whether a person is telling the truth.",
                    "Indian-English deception detection has not been validated.",
                ],
            ),
            features=features if include_features else None,
        )

    # --- Prediction ---
    try:
        verdict, score, confidence = predictor.predict(vector)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed unexpectedly")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Build explanations (both legacy list and structured evidence-based format)
    raw_warnings = getattr(loaded_audio, "quality_warnings", []) or []
    explanation = generate_explanations(features, score, quality_warnings=raw_warnings)
    structured_expl = generate_structured_explanation(
        features=features,
        score=score,
        confidence=confidence,
        quality_warnings=raw_warnings,
    )

    return AnalyzeResponse(
        verdict=verdict,
        score=round(score, 4),
        confidence=round(confidence, 4),
        speech_status=SPEECH_STATUS_VOICE_DETECTED,
        speech_detected=True,
        explanation=explanation,
        features=features if include_features else None,
        structured_explanation=structured_expl,
        quality_warnings=raw_warnings if raw_warnings else None,
    )


# ============================================================================
# Live Microphone Endpoints
# ============================================================================

class LiveStartRequest(BaseModel):
    """Request body for POST /live/start."""

    window_seconds: int = Field(
        DEFAULT_WINDOW,
        description=f"Recording window duration in seconds. Allowed: {sorted(VALID_WINDOWS)}.",
    )


@app.post("/live/start", tags=["live"])
def live_start(body: LiveStartRequest) -> dict:
    """
    Start continuous live microphone recording and analysis.

    Each completed recording window is automatically analysed and the
    result is broadcast to all connected WebSocket clients at /ws/live.

    Returns 409 if a session is already running.
    Returns 503 if the microphone or model is unavailable.
    """
    if not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Deception model is not loaded. Pretrained model could not be loaded from cache. "
                "Run 'python -m app.model_loader' to inspect/cache it, or set TRUSIGHT_MODEL_PATH, "
                "then POST /reload-model."
            ),
        )

    try:
        started = live_manager.start(window_seconds=body.window_seconds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not started:
        raise HTTPException(
            status_code=409,
            detail="A live session is already running. POST /live/stop first.",
        )

    return {
        "message": "Live session started.",
        "window_seconds": body.window_seconds,
    }


@app.post("/live/stop", tags=["live"])
def live_stop() -> dict:
    """
    Stop the running live recording session gracefully.

    Waits for the current analysis window to complete before stopping.
    Safe to call even if no session is running.
    """
    live_manager.stop()
    summary = live_manager.get_session_summary()
    return {
        "message": "Live session stopped.",
        "summary": summary,
    }


@app.get("/live/status", tags=["live"])
def live_status() -> dict:
    """
    Get the current live session status.

    Returns running state, window size, segments processed,
    elapsed time, and human-readable status string.
    """
    return live_manager.get_status()


@app.get("/live/results", tags=["live"])
def live_results() -> list:
    """
    Get all analysis results from the current (or most recent) session.

    Results are appended as each window completes; they are never overwritten.
    """
    return live_manager.get_results()


@app.get("/live/summary", tags=["live"])
def live_summary() -> dict:
    """
    Get the aggregate summary for the current (or most recent) session.

    Includes average scores, highest/lowest risk segments, and common
    acoustic indicators.
    """
    return live_manager.get_session_summary()


@app.post(
    "/live/analyze",
    response_model=None,
    tags=["live"],
)
async def live_analyze_segment(
    audio: UploadFile = File(
        ...,
        description=(
            "A single audio segment recorded by the browser via MediaRecorder. "
            "Accepted formats: webm, wav, ogg, flac. "
            "The backend decodes and resamples to 16 kHz mono automatically."
        ),
    ),
) -> JSONResponse:
    """
    Analyse a single browser-recorded audio segment from the Live Microphone feature.

    This endpoint is the server-side counterpart of the browser's MediaRecorder.
    The browser captures each N-second window via getUserMedia, creates a Blob,
    and POSTs it here.  We run the SAME pipeline as POST /analyze:

        decode audio → extract features → run classifier → generate explanations

    The result is stored in the live session and broadcast via WebSocket so the
    frontend dashboard updates in real time.

    Returns the same fields as POST /analyze plus session-level metadata.
    """
    if not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Deception model is not loaded. Pretrained model could not be loaded from cache. "
                "Run 'python -m app.model_loader' to inspect/cache it, or set TRUSIGHT_MODEL_PATH, "
                "then POST /reload-model."
            ),
        )

    # ── Read uploaded audio blob ──────────────────────────────────────────
    file_bytes = await audio.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Audio segment is empty.")

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Segment too large (>{MAX_UPLOAD_BYTES // (1024*1024)}MB).",
        )

    # Determine filename/extension from the upload.  MediaRecorder typically
    # sends 'audio/webm' or 'audio/ogg'; we map these to proper extensions.
    content_type = (audio.content_type or "").lower()
    original_name = audio.filename or "segment.webm"

    # Normalise the extension so load_audio_from_bytes can route correctly.
    if original_name.endswith(".webm") or "webm" in content_type:
        segment_filename = "live_segment.webm"
    elif original_name.endswith(".ogg") or "ogg" in content_type:
        segment_filename = "live_segment.ogg"
    elif original_name.endswith(".wav") or "wav" in content_type:
        segment_filename = "live_segment.wav"
    else:
        # Default to webm — the most common MediaRecorder output in Chrome.
        segment_filename = "live_segment.webm"

    logger.info(
        "[TRUSIGHT] Live segment received: %d bytes, content_type=%s → treating as %s",
        len(file_bytes), content_type, segment_filename,
    )

    # ── Decode & preprocess ───────────────────────────────────────────────
    try:
        loaded_audio = load_audio_from_bytes(file_bytes, segment_filename, allow_silence=True, min_duration=0.1)
    except AudioValidationError as e:
        logger.warning("[TRUSIGHT-DIAG] Validation failed for %s (%d bytes): %s (min_dur=0.1s)", segment_filename, len(file_bytes), e)
        raise HTTPException(status_code=400, detail=str(e))
    except AudioProcessingError as e:
        logger.warning("[TRUSIGHT-DIAG] Decode failed for %s (%d bytes): %s", segment_filename, len(file_bytes), e)
        raise HTTPException(status_code=422, detail=str(e))

    # ── Diagnostic Logging: Section 2 ─────────────────────────────────────
    is_silent = (loaded_audio.raw_peak < 0.015) or (loaded_audio.raw_rms < 0.0025 and loaded_audio.raw_peak < 0.025)
    logger.info(
        "[TRUSIGHT-DIAG] mime=%s size=%d decode=OK sr=%d ch=1 dur=%.2fs samples=%d rms=%.5f peak=%.4f silent=%s",
        content_type, len(file_bytes), loaded_audio.sample_rate, loaded_audio.duration_seconds,
        len(loaded_audio.samples), loaded_audio.raw_rms, loaded_audio.raw_peak, is_silent,
    )

    # ── Check voice activity and rolling buffer accumulation ──────────────
    # Handle short live chunks: if non-silent and under minimum duration, accumulate into rolling buffer
    if loaded_audio.duration_seconds < MIN_DURATION_SECONDS:
        if is_silent:
            live_manager.record_silence_chunk("Audio is below ambient silence floor.")
            logger.info("[TRUSIGHT-DIAG] chunk_decision=REJECTED status=VOICE NOT DETECTED reason=Ambient silence")
            return JSONResponse(content={
                "segment": None,
                "verdict": "No Speech Detected",
                "score": None,
                "confidence": None,
                "speech_detected": False,
                "speech_status": SPEECH_STATUS_VOICE_NOT_DETECTED,
                "explanation": [
                    "Voice not detected — speak into your microphone.",
                    f"Short silent chunk ({loaded_audio.duration_seconds:.2f}s).",
                ],
                "duration": round(loaded_audio.duration_seconds, 2),
                "timestamp": _time.time(),            })

        # Non-silent short chunk: accumulate into rolling buffer
        rolling_audio = live_manager.append_audio_samples(loaded_audio.samples, loaded_audio.sample_rate)
        if rolling_audio.duration_seconds < MIN_DURATION_SECONDS:
            logger.info(
                "[TRUSIGHT-DIAG] chunk_decision=ACCUMULATING status=AUDIO TOO SHORT rolling_dur=%.2fs min_dur=%.1fs",
                rolling_audio.duration_seconds, MIN_DURATION_SECONDS,
            )
            return JSONResponse(content={
                "segment": None,
                "verdict": "Waiting for speech",
                "score": None,
                "confidence": None,
                "speech_detected": False,
                "speech_status": SPEECH_STATUS_AUDIO_TOO_SHORT,
                "explanation": [
                    "Waiting for speech — speak clearly into your microphone.",
                    f"Accumulated {rolling_audio.duration_seconds:.2f}s (minimum {MIN_DURATION_SECONDS:.1f}s required for feature extraction).",
                ],
                "duration": round(loaded_audio.duration_seconds, 2),
                "timestamp": _time.time(),            })
        chunk_to_evaluate = rolling_audio
    else:
        chunk_to_evaluate = loaded_audio

    chunk_has_speech, chunk_note, chunk_status = evaluate_audio_speech_quality(chunk_to_evaluate)
    vad_stats = getattr(chunk_to_evaluate, "vad_stats", {}) or {}
    logger.info(
        "[TRUSIGHT-DIAG] vad_speech=%s status=%s voiced_frames=%s voiced_ratio=%s hnr=%s min_dur=%.1fs reason=%s",
        chunk_has_speech, chunk_status,
        vad_stats.get("voiced_frames"), vad_stats.get("voiced_ratio"), vad_stats.get("hnr_mean"),
        MIN_DURATION_SECONDS, chunk_note,
    )

    if not chunk_has_speech:
        logger.info("[TRUSIGHT-DIAG] chunk_decision=REJECTED status=%s reason=%s", chunk_status, chunk_note)
        live_manager.record_silence_chunk(chunk_note)
        verdict = "Inconclusive" if chunk_status in (SPEECH_STATUS_AUDIO_QUALITY_TOO_LOW, SPEECH_STATUS_AUDIO_TOO_SHORT) else "No Speech Detected"
        return JSONResponse(content={
            "segment": None,
            "verdict": verdict,
            "score": None,
            "confidence": None,
            "speech_detected": False,
            "speech_status": chunk_status,
            "explanation": [
                "Voice not detected — speak into your microphone." if verdict == "No Speech Detected" else "Audio inconclusive.",
                chunk_note,
            ],
            "duration": round(loaded_audio.duration_seconds, 2),
                "timestamp": _time.time(),        })

    # User spoke! Append chunk to rolling buffer if not already appended
    if chunk_to_evaluate is loaded_audio:
        rolling_audio = live_manager.append_audio_samples(loaded_audio.samples, loaded_audio.sample_rate)
    analysis_audio = rolling_audio if rolling_audio.duration_seconds >= MIN_DURATION_SECONDS else loaded_audio
    logger.info("[TRUSIGHT-DIAG] chunk_decision=ACCEPTED status=%s rolling_dur=%.2fs analysis_dur=%.2fs", chunk_status, rolling_audio.duration_seconds, analysis_audio.duration_seconds)

    # ── Feature extraction on stabilized analysis audio ──────────────────
    try:
        if predictor.is_multimodal:
            from app.deep_feature_extractor import extract_deep_features
            features_obj, _ = feature_extractor.extract(analysis_audio)
            vector = extract_deep_features(analysis_audio)
        else:
            features_obj, vector = feature_extractor.extract(analysis_audio)
    except FeatureExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── Double check voice presence using extracted features ─────────────
    feat_has_speech, feat_note, feat_status = evaluate_audio_speech_quality(analysis_audio, features=features_obj, min_duration=0.5)
    if not feat_has_speech:
        logger.info("[TRUSIGHT-DIAG] Post-feature voice check rejected: status=%s reason=%s", feat_status, feat_note)
        live_manager.record_silence_chunk(feat_note)
        return JSONResponse(content={
            "segment": None,
            "verdict": "No Speech Detected",
            "score": None,
            "confidence": None,
            "speech_detected": False,
            "speech_status": feat_status,
            "explanation": [
                "Voice not detected — speak into your microphone.",
                feat_note,
            ],
            "duration": round(loaded_audio.duration_seconds, 2),
                "timestamp": _time.time(),        })

    # ── Prediction using verified feature vector ──────────────────────────
    try:
        verdict, score, confidence = predictor.predict(vector)
    except Exception as e:
        logger.exception("Live segment prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    raw_warnings = getattr(loaded_audio, "quality_warnings", []) or []
    explanation = generate_explanations(features_obj, score, quality_warnings=raw_warnings)

    logger.info(
        "[TRUSIGHT] Segment result (window=%.1fs): %s score=%.3f conf=%.3f",
        analysis_audio.duration_seconds, verdict, score, confidence,
    )

    # ── Store in session manager & broadcast via WebSocket ────────────────
    segment_index = live_manager.get_next_segment_index()
    start_seconds = _time.monotonic() - live_manager.get_session_start_time()
    end_seconds = start_seconds + loaded_audio.duration_seconds

    now_wall_ts = _time.time()
    seg_result = SegmentResult(
        segment=segment_index,
        start_seconds=max(0.0, start_seconds - loaded_audio.duration_seconds),
        end_seconds=end_seconds,
        verdict=verdict,
        score=round(score, 4),
        confidence=round(confidence, 4),
        explanation=explanation,
        features=features_obj,
        speech_detected=True,
        speech_status=SPEECH_STATUS_VOICE_DETECTED,
        timestamp=now_wall_ts,
    )

    live_manager.submit_segment(seg_result)

    structured_expl = generate_structured_explanation(
        features=features_obj,
        score=score,
        confidence=confidence,
        quality_warnings=raw_warnings,
    )

    return JSONResponse(content={
        "segment": segment_index,
        "verdict": verdict,
        "score": round(score, 4),
        "confidence": round(confidence, 4),
        "speech_detected": True,
        "speech_status": SPEECH_STATUS_VOICE_DETECTED,
        "explanation": explanation,
        "structured_explanation": structured_expl.dict() if structured_expl else None,
        "features": features_obj.dict() if features_obj else None,
        "duration": round(loaded_audio.duration_seconds, 2),
        "timestamp": now_wall_ts,
    })



# ============================================================================
# WebSocket endpoint — push updates to the frontend
# ============================================================================

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """
    WebSocket endpoint for live analysis push updates.

    Connect before or after starting a session. Every time a recording
    window is analysed the server pushes a JSON message of the form:

        {"type": "segment_result", "data": {...}, "status": {...}}

    When the session stops:

        {"type": "session_stopped", "status": {...}, "summary": {...}}
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    live_manager.register_ws_client(client_id, queue, loop)
    logger.info("WebSocket client connected: %s", client_id)

    # Send current status immediately so the frontend can sync.
    try:
        await websocket.send_text(
            json.dumps({"type": "connected", "status": live_manager.get_status()})
        )
    except Exception:
        pass

    try:
        while True:
            # Wait for a message from the manager (segment result or stop signal).
            # Also handle incoming ping/pong from the client.
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=5.0)
                await websocket.send_text(json.dumps(payload))
            except asyncio.TimeoutError:
                # Send a heartbeat ping so the browser keeps the connection alive.
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", client_id)
    except Exception as exc:
        logger.warning("WebSocket error for %s: %s", client_id, exc)
    finally:
        live_manager.unregister_ws_client(client_id)


# ============================================================================
# Global exception handler
# ============================================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """Catch-all so unexpected errors return structured JSON, not a raw 500 HTML page."""
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
