"""
model.py

Pydantic schemas for the Voice Analysis API. Kept separate from
predictor.py (the ML model wrapper) to avoid the naming clash between
"data model" and "ML model" causing confusion.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class VoiceFeatures(BaseModel):
    """
    All acoustic features extracted from a single audio sample.
    Returned internally to the predictor and exposed to the caller only
    when `include_features=true` is passed, to keep the default response
    payload small.
    """

    # --- Pitch (F0) ---
    pitch_mean: float = Field(..., description="Mean fundamental frequency (Hz)")
    pitch_std: float = Field(..., description="Standard deviation of F0 (Hz)")
    pitch_min: float = Field(..., description="Minimum voiced F0 (Hz)")
    pitch_max: float = Field(..., description="Maximum voiced F0 (Hz)")

    # --- Voice quality (Parselmouth) ---
    jitter_local: float = Field(..., description="Local jitter (cycle-to-cycle F0 perturbation)")
    shimmer_local: float = Field(..., description="Local shimmer (cycle-to-cycle amplitude perturbation)")

    # --- Timing / tempo ---
    speech_tempo: float = Field(..., description="Estimated tempo in beats/onsets per second")
    pause_ratio: float = Field(..., description="Fraction of total duration classified as silence/pause")
    pause_count: int = Field(..., description="Number of detected pause segments")
    silence_duration: float = Field(..., description="Total silent duration in seconds")

    # --- Energy ---
    rms_energy_mean: float = Field(..., description="Mean RMS energy")
    rms_energy_std: float = Field(..., description="Standard deviation of RMS energy")

    # --- Spectral ---
    zero_crossing_rate: float = Field(..., description="Mean zero crossing rate")
    spectral_centroid: float = Field(..., description="Mean spectral centroid (Hz)")
    spectral_bandwidth: float = Field(..., description="Mean spectral bandwidth (Hz)")
    spectral_rolloff: float = Field(..., description="Mean spectral roll-off (Hz)")

    # --- MFCCs (13 coefficients, mean over time) ---
    mfcc_mean: List[float] = Field(..., description="Mean of each of the 13 MFCC coefficients")
    mfcc_std: List[float] = Field(..., description="Std deviation of each of the 13 MFCC coefficients")

    # --- Reliability & Quality Flags (Stage 2) ---
    is_pitch_reliable: bool = Field(True, description="Whether fundamental frequency had sufficient voiced frames to be reliable")
    is_jitter_reliable: bool = Field(True, description="Whether glottal pulses were sufficient for reliable jitter computation")
    is_shimmer_reliable: bool = Field(True, description="Whether glottal pulses were sufficient for reliable shimmer computation")
    voiced_ratio: float = Field(0.0, description="Fraction of frames identified as voiced phonation")

    class Config:
        json_schema_extra = {
            "example": {
                "pitch_mean": 178.4,
                "pitch_std": 24.1,
                "pitch_min": 110.2,
                "pitch_max": 260.7,
                "jitter_local": 0.012,
                "shimmer_local": 0.045,
                "speech_tempo": 2.3,
                "pause_ratio": 0.18,
                "pause_count": 4,
                "silence_duration": 1.6,
                "rms_energy_mean": 0.084,
                "rms_energy_std": 0.021,
                "zero_crossing_rate": 0.071,
                "spectral_centroid": 1820.5,
                "spectral_bandwidth": 1590.2,
                "spectral_rolloff": 3400.9,
                "mfcc_mean": [0.0] * 13,
                "mfcc_std": [0.0] * 13,
            }
        }


class StructuredExplanation(BaseModel):
    """
    Structured evidence-based breakdown separating acoustic observations,
    audio quality flags, model prediction context, and scientific caveats.
    """
    acoustic_observations: List[str] = Field(default_factory=list, description="Verifiable physical measurements (pitch, tempo, pauses, jitter, shimmer)")
    quality_warnings: List[str] = Field(default_factory=list, description="Audio degradation flags (clipping, high noise floor, short phonation)")
    prediction_context: str = Field("", description="Explanation of model output vs probability calibration")
    limitations: List[str] = Field(default_factory=list, description="Scientific disclaimers regarding stress vs deception")


class AnalyzeResponse(BaseModel):
    """Response body for POST /analyze"""

    verdict: str = Field(..., description="'Truthful' or 'Deceptive' or 'Inconclusive' or 'No Speech Detected'")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Deception likelihood, 0=truthful, 1=deceptive. None if speech is absent or inconclusive.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model confidence in the verdict. None if speech is absent or inconclusive.")
    explanation: List[str] = Field(..., description="2-5 human-readable acoustic observations")
    features: Optional[VoiceFeatures] = Field(
        None, description="Raw extracted features (only present if include_features=true)"
    )
    structured_explanation: Optional[StructuredExplanation] = Field(
        None, description="Structured breakdown of observations, warnings, and caveats"
    )
    quality_warnings: Optional[List[str]] = Field(
        None, description="Diagnostic audio quality warnings (clipping, SNR, duration)"
    )
    speech_status: Optional[str] = Field(
        "VOICE DETECTED",
        description="Detailed speech presence category: VOICE DETECTED, VOICE NOT DETECTED, AUDIO TOO SHORT, AUDIO QUALITY TOO LOW"
    )
    speech_detected: Optional[bool] = Field(
        True,
        description="Whether genuine human vocal speech was detected and analyzed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "verdict": "Truthful",
                "score": 0.34,
                "confidence": 0.66,
                "explanation": [
                    "Stable pitch variation",
                    "Normal speech tempo",
                    "Few hesitation pauses",
                ],
                "features": None,
                "structured_explanation": None,
                "quality_warnings": None,
            }
        }


class HealthResponse(BaseModel):
    service: str
    status: str
    model_loaded: Optional[bool] = Field(None, description="Whether the deception ML model is loaded and ready")
    model_version: Optional[str] = Field(None, description="Active pretrained model version")


class ErrorResponse(BaseModel):
    detail: str
