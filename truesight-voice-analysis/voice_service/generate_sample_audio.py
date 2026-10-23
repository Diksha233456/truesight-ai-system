"""
generate_sample_audio.py

Generates synthetic WAV sample files in sample_audio/ for smoke-testing
the /analyze endpoint immediately after cloning. These are NOT real speech.

Run from the voice_service/ project root:
    python generate_sample_audio.py

Requirements: numpy, soundfile (already in requirements.txt)
"""

from __future__ import annotations

import os
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_audio")


def _make_voiced_segment(
    duration: float,
    f0: float,
    sr: int,
    jitter: float = 0.005,
    shimmer: float = 0.02,
) -> np.ndarray:
    """
    Synthesize a voiced (speech-like) segment using an additive harmonics model
    with slight perturbation to mimic jitter (F0 wobble) and shimmer (amplitude wobble).
    """
    n = int(duration * sr)
    t = np.arange(n) / sr

    # Add jitter: cycle-to-cycle perturbation of fundamental frequency
    phase = 2 * np.pi * f0 * t
    jitter_noise = np.cumsum(np.random.normal(0, jitter * f0, n)) / sr
    phase = phase + 2 * np.pi * jitter_noise

    # Sum first 5 harmonics
    signal = np.zeros(n)
    for k in range(1, 6):
        amp = 1.0 / k  # Natural harmonic rolloff
        signal += amp * np.sin(k * phase)

    # Add shimmer: cycle-to-cycle amplitude perturbation
    shimmer_env = 1.0 + np.random.normal(0, shimmer, n)
    signal *= shimmer_env

    # Apply a gentle amplitude envelope (attack + decay on edges)
    fade_len = min(int(0.02 * sr), n // 4)
    envelope = np.ones(n)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    signal *= envelope

    return signal.astype(np.float32)


def _make_silence(duration: float, sr: int) -> np.ndarray:
    """Synthesize a near-silent pause segment (very low noise floor)."""
    n = int(duration * sr)
    return (np.random.normal(0, 1e-5, n)).astype(np.float32)


def make_truthful_speech(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Synthesize a ~5s 'truthful-like' speech pattern:
    - Steady F0 (~160 Hz, typical female/male overlap)
    - Moderate speech tempo
    - Few pauses, short silence intervals
    - Low jitter/shimmer
    """
    np.random.seed(42)
    segments = []

    # Sentence 1: 1.2s speech
    segments.append(_make_voiced_segment(1.2, f0=155.0, sr=sr, jitter=0.003, shimmer=0.015))
    # Short pause: 0.15s
    segments.append(_make_silence(0.15, sr))
    # Sentence 2: 1.4s speech
    segments.append(_make_voiced_segment(1.4, f0=160.0, sr=sr, jitter=0.004, shimmer=0.018))
    # Short pause: 0.12s
    segments.append(_make_silence(0.12, sr))
    # Sentence 3: 1.5s speech
    segments.append(_make_voiced_segment(1.5, f0=158.0, sr=sr, jitter=0.003, shimmer=0.016))
    # Very short pause: 0.1s
    segments.append(_make_silence(0.10, sr))
    # Final phrase: 0.8s speech
    segments.append(_make_voiced_segment(0.8, f0=152.0, sr=sr, jitter=0.004, shimmer=0.014))

    signal = np.concatenate(segments)
    # Peak-normalize
    peak = np.max(np.abs(signal))
    if peak > 1e-8:
        signal = signal / peak * 0.9
    return signal


def make_deceptive_speech(sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Synthesize a ~5s 'deceptive-like' speech pattern with acoustic markers
    commonly discussed in voice-stress literature (though with weak empirical
    support in the real world — this is only for prototype demonstration):
    - Higher pitch variability / F0 jumps
    - Longer and more frequent pauses (hesitations)
    - Elevated jitter and shimmer
    - Slightly slower pace
    """
    np.random.seed(7)
    segments = []

    # Phrase 1: 0.9s speech with higher jitter
    segments.append(_make_voiced_segment(0.9, f0=170.0, sr=sr, jitter=0.012, shimmer=0.055))
    # Longer hesitation pause: 0.35s
    segments.append(_make_silence(0.35, sr))
    # Phrase 2: 0.7s speech — pitch jumps higher
    segments.append(_make_voiced_segment(0.7, f0=195.0, sr=sr, jitter=0.015, shimmer=0.065))
    # Hesitation: 0.30s
    segments.append(_make_silence(0.30, sr))
    # Phrase 3: 0.8s speech — elevated shimmer
    segments.append(_make_voiced_segment(0.8, f0=162.0, sr=sr, jitter=0.014, shimmer=0.072))
    # Another pause: 0.40s
    segments.append(_make_silence(0.40, sr))
    # Phrase 4: 0.6s speech — higher pitch again
    segments.append(_make_voiced_segment(0.6, f0=188.0, sr=sr, jitter=0.016, shimmer=0.060))
    # Short pause: 0.20s
    segments.append(_make_silence(0.20, sr))
    # Phrase 5: 0.8s speech
    segments.append(_make_voiced_segment(0.8, f0=175.0, sr=sr, jitter=0.013, shimmer=0.058))

    signal = np.concatenate(segments)
    peak = np.max(np.abs(signal))
    if peak > 1e-8:
        signal = signal / peak * 0.9
    return signal


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    truthful_path = os.path.join(OUTPUT_DIR, "truthful.wav")
    deceptive_path = os.path.join(OUTPUT_DIR, "deceptive.wav")

    print(f"Generating {truthful_path} ...")
    sf.write(truthful_path, make_truthful_speech(), SAMPLE_RATE, subtype="PCM_16")

    print(f"Generating {deceptive_path} ...")
    sf.write(deceptive_path, make_deceptive_speech(), SAMPLE_RATE, subtype="PCM_16")

    print("Done. Sample files written to sample_audio/")
    print()
    print("NOTE: These are synthetic audio signals, NOT real speech recordings.")
    print("They are intended only to smoke-test the /analyze API endpoint.")
    print("For meaningful predictions, retrain the model on real labeled speech data.")


if __name__ == "__main__":
    main()
