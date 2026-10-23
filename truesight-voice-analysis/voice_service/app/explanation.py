"""
explanation.py

Turns extracted VoiceFeatures into a short list of human-readable
observations. This is a deterministic, rule-based layer (not part of the
ML model itself) so the reasoning behind a verdict is always inspectable,
independent of what the Random Forest actually weighted internally.

Thresholds below are heuristic prototype defaults based on commonly cited
ranges in vocal-analysis literature for adult conversational speech. They
are intentionally conservative and should be recalibrated against real
labeled data before any production use.
"""

from __future__ import annotations

from __future__ import annotations

from typing import List, Optional

from app.model import VoiceFeatures, StructuredExplanation

# --- Heuristic thresholds (prototype defaults) ---
PITCH_STD_HIGH = 35.0          # Hz; above this = notably variable pitch
PITCH_STD_LOW = 12.0           # Hz; below this = unusually flat/monotone
PAUSE_RATIO_HIGH = 0.30        # fraction of duration spent in silence
PAUSE_RATIO_LOW = 0.05
TEMPO_SLOW = 1.2               # onsets/sec
TEMPO_FAST = 3.5
JITTER_HIGH = 0.02             # local jitter ratio
SHIMMER_HIGH = 0.08            # local shimmer ratio
RMS_STD_HIGH = 0.05            # energy instability


def generate_structured_explanation(
    features: Optional[VoiceFeatures],
    score: Optional[float],
    confidence: Optional[float],
    quality_warnings: Optional[List[str]] = None,
) -> StructuredExplanation:
    """
    Generate an honest, evidence-based breakdown separating:
    1. Measured Acoustic Observations (only when reliable)
    2. Audio Quality Warnings (clipping, low SNR, short phonation)
    3. Prediction Context (uncalibrated probability vs verified evaluation)
    4. Limitations / Scientific Caveats
    """
    observations: List[str] = []
    warnings: List[str] = list(quality_warnings or [])
    limitations: List[str] = []

    # 1. Pitch measurements (only if features are present and pitch is reliable)
    if features is not None and getattr(features, "is_pitch_reliable", True) and features.pitch_mean > 0:
        if features.pitch_std >= PITCH_STD_HIGH:
            observations.append(
                f"Elevated fundamental frequency variability (F0 std: {features.pitch_std:.1f} Hz, mean: {features.pitch_mean:.1f} Hz), "
                "reflecting wide pitch range or expressive intonational prominence (does not indicate stress or deception)."
            )
        elif features.pitch_std <= PITCH_STD_LOW and features.pitch_std > 0:
            observations.append(
                f"Low fundamental frequency variability (F0 std: {features.pitch_std:.1f} Hz, mean: {features.pitch_mean:.1f} Hz), "
                "reflecting relatively narrow intonational range."
            )
        else:
            observations.append(
                f"Fundamental frequency variation within typical conversational range (F0 mean: {features.pitch_mean:.1f} Hz, std: {features.pitch_std:.1f} Hz)."
            )
    elif features is not None:
        warnings.append("Fundamental frequency (pitch) measurement is inconclusive due to insufficient voiced frames.")

    # 2. Timing & pause measurements
    if features is not None:
        if features.pause_ratio >= PAUSE_RATIO_HIGH:
            observations.append(
                f"Extended silent intervals detected ({features.pause_ratio * 100:.1f}% silent duration across {features.pause_count} pauses); "
                "pause duration reflects natural phrasing, task structure, or formulation pacing (does not indicate deception)."
            )
        elif features.pause_ratio <= PAUSE_RATIO_LOW:
            observations.append(
                f"Continuous delivery with minimal detected silence ({features.pause_ratio * 100:.1f}% silent duration)."
            )
        else:
            observations.append(f"Moderate pause distribution ({features.pause_ratio * 100:.1f}% silent duration across {features.pause_count} pauses).")

        # 3. Speech tempo (onset rate)
        if features.speech_tempo <= TEMPO_SLOW:
            observations.append(f"Speech onset rate measured at {features.speech_tempo:.1f} onsets/sec (slower pacing).")
        elif features.speech_tempo >= TEMPO_FAST:
            observations.append(
                f"Speech onset rate measured at {features.speech_tempo:.1f} onsets/sec; "
                "onset rate reflects rhythm, syllable density, and articulation cadence (does not indicate urgency, stress, or deception)."
            )
        else:
            observations.append(f"Speech onset rate within typical conversational pace ({features.speech_tempo:.1f} onsets/sec).")

        # 4. Voice quality: jitter & shimmer (strictly gated on reliability)
        if getattr(features, "is_jitter_reliable", False):
            if features.jitter_local >= JITTER_HIGH:
                observations.append(
                    f"Cycle-to-cycle frequency variation measured at {features.jitter_local * 100:.2f}%. "
                    "Note: Perturbation metrics on connected speech are influenced by phonetic transitions and boundary effects, and do not indicate vocal distress or deception."
                )
            else:
                observations.append(f"Cycle-to-cycle frequency variation measured at {features.jitter_local * 100:.2f}% (stable vocal periodicity).")
        else:
            warnings.append("Jitter measurement is inconclusive (< 25 stable glottal cycles in signal).")

        if getattr(features, "is_shimmer_reliable", False):
            if features.shimmer_local >= SHIMMER_HIGH:
                observations.append(
                    f"Cycle-to-cycle amplitude variation measured at {features.shimmer_local * 100:.2f}%. "
                    "Note: Amplitude perturbation on connected speech is affected by acoustic environment and consonants, and does not indicate vocal distress or deception."
                )
            else:
                observations.append(f"Cycle-to-cycle amplitude variation measured at {features.shimmer_local * 100:.2f}% (stable amplitude periodicity).")
        else:
            warnings.append("Shimmer measurement is inconclusive (< 25 stable glottal cycles in signal).")

    # 5. Prediction Context (Honest distinction between raw model score and calibrated confidence)
    if score is not None:
        pred_context = (
            f"Raw tree ensemble probability: {score:.1%}. Note: This score represents an uncalibrated "
            f"model probability (raw ensemble decision surface), not an empirical statistical certainty. "
            "Benchmark testing on unseen speakers yields ~58-66% accuracy."
        )
    else:
        pred_context = (
            "Prediction was withheld. Silence, background noise, or unvoiced audio is never "
            "assigned a truth or deception score."
        )

    # 6. Scientific Limitations & Disclaimers
    limitations.append(
        "This experimental model cannot determine whether a person is telling the truth. Voice patterns alone do not prove deception."
    )
    limitations.append(
        "Indian-English deception detection has not been empirically validated due to the absence of ground-truth labelled datasets."
    )
    limitations.append(
        "Acoustic measurements reflect individual anatomy, speaking style, prosody, linguistic cadence, or recording conditions; "
        "they do not prove deception, stress, or nervousness."
    )
    limitations.append(
        "Accent, dialect, pitch modulation, and speaking tempo vary naturally across regional populations (including Indian-accented English) "
        "and provide no evidence of veracity or deceit."
    )
    limitations.append(
        "Clinical voice-quality thresholds for sustained vowels do not transfer directly to continuous connected speech without specialized phonetic segmentation."
    )

    return StructuredExplanation(
        acoustic_observations=observations,
        quality_warnings=warnings,
        prediction_context=pred_context,
        limitations=limitations,
    )


def generate_explanations(
    features: Optional[VoiceFeatures],
    score: Optional[float] = None,
    quality_warnings: Optional[List[str]] = None,
) -> List[str]:
    """
    Build a 2-5 item list of plain-language observations from the extracted
    features. Preserves 100% backward compatibility with legacy API clients
    and UI renderers.
    """
    if features is None or score is None:
        notes: List[str] = []
        if quality_warnings:
            notes.append(f"Audio Quality Note: {quality_warnings[0]}")
        notes.append("Voice not detected — no human vocal activity detected in this recording.")
        notes.append(
            "Note: Silence, ambient noise, and microphone hiss are never assigned deception or truth scores."
        )
        return notes

    notes: List[str] = []

    # Include any severe audio quality warning first
    if quality_warnings:
        notes.append(f"Audio Quality Note: {quality_warnings[0]}")

    # Pitch variability
    if getattr(features, "is_pitch_reliable", True) and features.pitch_mean > 0:
        if features.pitch_std >= PITCH_STD_HIGH:
            notes.append("High pitch variability detected across utterance (reflects intonational range).")
        elif features.pitch_std <= PITCH_STD_LOW and features.pitch_std > 0:
            notes.append("Narrow pitch variability detected across utterance.")
        else:
            notes.append("Stable pitch variation.")
    else:
        notes.append("Pitch tracking inconclusive due to short or noisy phonation.")

    # Pauses / phrasing
    if features.pause_ratio >= PAUSE_RATIO_HIGH:
        notes.append("Extended pausing detected across segment.")
    elif features.pause_ratio <= PAUSE_RATIO_LOW:
        notes.append("Continuous delivery with few detected pauses.")
    else:
        notes.append("Moderate amount of pausing.")

    # Tempo / rhythm
    if features.speech_tempo <= TEMPO_SLOW:
        notes.append(f"Lower speech onset rate measured ({features.speech_tempo:.1f} onsets/sec).")
    elif features.speech_tempo >= TEMPO_FAST:
        notes.append(f"Higher speech onset rate measured ({features.speech_tempo:.1f} onsets/sec; reflects syllable rhythm).")
    else:
        notes.append("Normal speech tempo.")

    # Voice quality: jitter (only interpret if verified reliable)
    if getattr(features, "is_jitter_reliable", False):
        if features.jitter_local >= JITTER_HIGH:
            notes.append(f"Acoustic frequency perturbation measured (local jitter {features.jitter_local * 100:.2f}%).")
    else:
        notes.append("Jitter measurement inconclusive (<25 glottal cycles).")

    # Voice quality: shimmer (only interpret if verified reliable)
    if getattr(features, "is_shimmer_reliable", False):
        if features.shimmer_local >= SHIMMER_HIGH:
            notes.append(f"Acoustic amplitude perturbation measured (local shimmer {features.shimmer_local * 100:.2f}%).")

    # Trim to 4 items and append scientific caveat
    if len(notes) > 4:
        notes = notes[:4]
    if len(notes) < 1:
        notes.append("Overall acoustic profile within typical conversational range.")

    notes.append(
        "Note: Acoustic indicators reflect natural prosody, cadence, or recording environment, NOT definitive deception."
    )

    return notes
