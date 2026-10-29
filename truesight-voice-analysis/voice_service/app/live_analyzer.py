"""
live_analyzer.py

Bridge between the live microphone capture layer and the existing ML pipeline.

Responsibilities
----------------
- Accept a LoadedAudio object produced by live_audio.record_window().
- Run it through the EXACT SAME pipeline used by POST /analyze:
    FeatureExtractor → (optional Wav2Vec2 deep features) → Predictor → Explanations
- Return a SegmentResult dataclass that the manager can store and broadcast.

This module contains ZERO ML logic. It delegates entirely to the existing
singletons created at application startup in main.py. The singletons are
injected via set_singletons() called once during startup — no re-loading
of any model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.utils import LoadedAudio
from app.model import VoiceFeatures

logger = logging.getLogger("voice_service.live_analyzer")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    """
    Result for a single continuous recording window.

    All fields match the shape of AnalyzeResponse so the frontend
    can render the same UI components for live and upload modes.
    """

    segment: Optional[int]         # 1-based segment index, or None if no speech
    start_seconds: float           # seconds from session start
    end_seconds: float             # seconds from session start
    verdict: str                   # "Truthful" | "Deceptive" | "No Speech Detected" | "Inconclusive"
    score: Optional[float] = None  # deception probability [0, 1] or None
    confidence: Optional[float] = None # model confidence [0, 1] or None
    explanation: List[str] = field(default_factory=list) # 2-5 human-readable notes
    features: Optional[VoiceFeatures] = None
    error: Optional[str] = None    # set if analysis failed for this segment
    speech_detected: bool = True
    speech_status: str = "VOICE DETECTED"
    timestamp: float = field(default_factory=time.time)

    # ── Computed helpers ─────────────────────────────────────────────────────

    @property
    def start_label(self) -> str:
        """Format start_seconds as MM:SS."""
        return _fmt_seconds(self.start_seconds)

    @property
    def end_label(self) -> str:
        """Format end_seconds as MM:SS."""
        return _fmt_seconds(self.end_seconds)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for the API response."""
        return {
            "segment": self.segment,
            "start": self.start_label,
            "end": self.end_label,
            "verdict": self.verdict,
            "score": round(self.score, 4) if self.score is not None else None,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "explanation": self.explanation,
            "features": self.features.dict() if self.features and hasattr(self.features, "dict") else (self.features if isinstance(self.features, dict) else None),
            "error": self.error,
            "speech_detected": self.speech_detected,
            "speech_status": self.speech_status,
            "timestamp": round(float(self.timestamp), 4) if self.timestamp is not None else None,
        }


def _fmt_seconds(total: float) -> str:
    mins = int(total) // 60
    secs = int(total) % 60
    return f"{mins:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# Singleton references (injected from main.py at startup)
# ---------------------------------------------------------------------------

_feature_extractor = None
_predictor = None
_generate_explanations = None


def set_singletons(feature_extractor, predictor, generate_explanations_fn) -> None:
    """
    Called once during application startup by main.py.

    Injects the already-constructed FeatureExtractor, Predictor, and
    generate_explanations function so this module never loads models itself.
    """
    global _feature_extractor, _predictor, _generate_explanations
    _feature_extractor = feature_extractor
    _predictor = predictor
    _generate_explanations = generate_explanations_fn
    logger.info("LiveAnalyzer singletons registered.")


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze_segment(
    audio: LoadedAudio,
    segment_index: int,
    start_seconds: float,
) -> SegmentResult:
    """
    Run the full existing pipeline on a single live audio window.

    Parameters
    ----------
    audio:
        A LoadedAudio object produced by live_audio.record_window().
    segment_index:
        1-based index of this segment in the current session.
    start_seconds:
        Session-relative timestamp at which this window started.

    Returns
    -------
    SegmentResult
        A fully populated result; on pipeline error, verdict/score are
        set to safe defaults and SegmentResult.error carries the message.
    """
    now_ts = time.time()
    if _feature_extractor is None or _predictor is None:
        return SegmentResult(
            segment=segment_index,
            start_seconds=start_seconds,
            end_seconds=start_seconds + audio.duration_seconds,
            verdict="Unknown",
            score=0.0,
            confidence=0.0,
            explanation=["Analysis pipeline not ready."],
            error="Singletons not injected — call set_singletons() first.",
            timestamp=now_ts,
        )

    end_seconds = start_seconds + audio.duration_seconds
    t0 = time.monotonic()

    # ── Feature extraction ────────────────────────────────────────────────
    try:
        if _predictor.is_multimodal:
            # 814-dim: acoustic + Wav2Vec2 deep embedding + verbal features
            from app.deep_feature_extractor import extract_deep_features
            features_obj, _ = _feature_extractor.extract(audio)
            vector: np.ndarray = extract_deep_features(audio)
        else:
            # 42-dim: acoustic only (legacy pipeline)
            features_obj, vector = _feature_extractor.extract(audio)
    except Exception as exc:
        logger.warning("Feature extraction failed for segment %d: %s", segment_index, exc)
        return SegmentResult(
            segment=segment_index,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            verdict="Unknown",
            score=0.0,
            confidence=0.0,
            explanation=["Feature extraction failed for this segment."],
            error=str(exc),
            timestamp=now_ts,
        )

    # ── Prediction ────────────────────────────────────────────────────────
    try:
        verdict, score, confidence = _predictor.predict(vector)
    except Exception as exc:
        logger.warning("Prediction failed for segment %d: %s", segment_index, exc)
        return SegmentResult(
            segment=segment_index,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            verdict="Unknown",
            score=0.0,
            confidence=0.0,
            explanation=["Prediction failed for this segment."],
            error=str(exc),
            timestamp=now_ts,
        )

    # ── Explanations ──────────────────────────────────────────────────────
    try:
        explanation = _generate_explanations(features_obj, score)
    except Exception as exc:
        logger.warning("Explanation generation failed for segment %d: %s", segment_index, exc)
        explanation = ["Explanation unavailable."]

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "Segment %d analysed in %.0f ms: %s (score=%.3f, conf=%.3f)",
        segment_index,
        elapsed_ms,
        verdict,
        score,
        confidence,
    )

    return SegmentResult(
        segment=segment_index,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        verdict=verdict,
        score=round(score, 4),
        confidence=round(confidence, 4),
        explanation=explanation,
        features=features_obj,
        timestamp=now_ts,
    )

