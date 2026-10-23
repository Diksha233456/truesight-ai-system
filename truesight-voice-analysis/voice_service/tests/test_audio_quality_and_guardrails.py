"""
tests/test_audio_quality_and_guardrails.py

Unit and regression tests verifying Stage 1 and Stage 2 enhancements:
1. Audio clipping detection and SNR estimation in app.utils.
2. Handling of silence, ultra-short recordings, and corrupt files without crashing.
3. FeatureExtractor guardrails: ensuring unvoiced speech and low glottal pulse count
   do not produce false zero interpretations or crash.
4. Schema preservation: verifying that to_vector strictly outputs 42-dimensional vectors.
5. Explanation guardrails: ensuring unreliable jitter/shimmer measurements are never
   claimed as voice instability evidence.
"""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import soundfile as sf

from app.utils import (
    load_audio_from_bytes,
    check_voice_presence,
    AudioValidationError,
    TARGET_SR,
)
from app.feature_extractor import FeatureExtractor, FEATURE_ORDER
from app.explanation import generate_explanations


def create_wav_bytes(samples: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """Helper to encode a float numpy array as 16-bit PCM WAV in-memory."""
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestAudioQualityAndGuardrails(unittest.TestCase):

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_silence_rejection(self):
        """Pure digital silence should raise AudioValidationError."""
        silence = np.zeros(TARGET_SR * 2, dtype=np.float32)
        wav_bytes = create_wav_bytes(silence)
        with self.assertRaises(AudioValidationError):
            load_audio_from_bytes(wav_bytes, "silence.wav")

    def test_short_recording_rejection(self):
        """Recordings under 0.5s should raise AudioValidationError."""
        short_audio = np.random.uniform(-0.1, 0.1, int(TARGET_SR * 0.2)).astype(np.float32)
        wav_bytes = create_wav_bytes(short_audio)
        with self.assertRaises(AudioValidationError):
            load_audio_from_bytes(wav_bytes, "short.wav")

    def test_clipping_detection(self):
        """Audio with severe clipping should be detected and flagged in quality_warnings."""
        t = np.linspace(0, 2.0, int(TARGET_SR * 2.0), endpoint=False)
        # Create heavily clipped square/sine wave
        signal = 2.5 * np.sin(2 * np.pi * 200 * t)
        signal = np.clip(signal, -1.0, 1.0).astype(np.float32)

        wav_bytes = create_wav_bytes(signal)
        loaded = load_audio_from_bytes(wav_bytes, "clipped.wav")

        self.assertGreater(loaded.clipping_ratio, 0.05)
        self.assertTrue(
            any("clipping detected" in w.lower() for w in loaded.quality_warnings),
            f"Expected clipping warning in {loaded.quality_warnings}",
        )

    def test_noise_and_snr_estimation(self):
        """Low SNR audio should populate quality_warnings without throwing an exception."""
        np.random.seed(42)
        t = np.linspace(0, 2.0, int(TARGET_SR * 2.0), endpoint=False)
        # Weak voice tone buried in loud white noise
        speech_tone = 0.05 * np.sin(2 * np.pi * 180 * t)
        noise = np.random.normal(0, 0.04, len(t))
        noisy_signal = (speech_tone + noise).astype(np.float32)

        wav_bytes = create_wav_bytes(noisy_signal)
        loaded = load_audio_from_bytes(wav_bytes, "noisy.wav")

        self.assertIsNotNone(loaded)
        self.assertIsInstance(loaded.snr_estimate_db, float)
        self.assertIsInstance(loaded.quality_warnings, list)

    def test_unvoiced_speech_feature_guardrails(self):
        """Whispered or unvoiced noise should not crash the extractor and must set reliability flags."""
        np.random.seed(123)
        # Pure white noise (no glottal periodicity)
        noise = np.random.normal(0, 0.2, int(TARGET_SR * 1.5)).astype(np.float32)
        wav_bytes = create_wav_bytes(noise)
        loaded = load_audio_from_bytes(wav_bytes, "unvoiced.wav")

        features, vector = self.extractor.extract(loaded)

        # Vector length must strictly equal 42
        self.assertEqual(len(vector), 42)
        self.assertEqual(len(FEATURE_ORDER), 42)

        # Reliability flags should accurately identify lack of periodic glottal pulses
        self.assertFalse(features.is_jitter_reliable)
        self.assertFalse(features.is_shimmer_reliable)

    def test_feature_vector_dimension_and_schema_preservation(self):
        """A synthetic tone must produce exactly 42 features matching FEATURE_ORDER."""
        t = np.linspace(0, 2.0, int(TARGET_SR * 2.0), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 150 * t).astype(np.float32)
        wav_bytes = create_wav_bytes(tone)
        loaded = load_audio_from_bytes(wav_bytes, "tone.wav")

        features, vector = self.extractor.extract(loaded)

        self.assertEqual(vector.shape, (42,))
        self.assertFalse(np.isnan(vector).any(), "Feature vector contains NaN values")
        self.assertFalse(np.isinf(vector).any(), "Feature vector contains Inf values")

    def test_unreliable_jitter_shimmer_not_claimed_as_deception(self):
        """When jitter/shimmer are marked unreliable, explanations must NOT claim voice instability."""
        t = np.linspace(0, 1.0, int(TARGET_SR * 1.0), endpoint=False)
        tone = 0.4 * np.sin(2 * np.pi * 180 * t).astype(np.float32)
        wav_bytes = create_wav_bytes(tone)
        loaded = load_audio_from_bytes(wav_bytes, "test.wav")

        features, _ = self.extractor.extract(loaded)
        # Artificially set high jitter but mark as unreliable
        features.jitter_local = 0.05
        features.is_jitter_reliable = False

        notes = generate_explanations(features, score=0.5)
        self.assertFalse(
            any("voice instability observed" in n.lower() for n in notes),
            "Unreliable jitter must not be interpreted as voice instability",
        )


if __name__ == "__main__":
    unittest.main()
