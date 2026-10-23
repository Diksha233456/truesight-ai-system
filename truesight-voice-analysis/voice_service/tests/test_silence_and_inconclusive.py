"""
tests/test_silence_and_inconclusive.py

Comprehensive regression test suite for TRUESIGHT silence detection, pause handling,
inconclusive result handling, and neutral labeling verification:

1. Completely silent uploaded audio -> returns No Speech Detected, score is None, confidence is None.
2. Near-silent uploaded audio -> returns No Speech Detected, score is None.
3. Background non-speech noise (Gaussian noise / clicks) -> rejected as No Speech Detected or Inconclusive.
4. Very short audio below minimum -> rejected with proper error or status.
5. Valid speech audio -> continues through analysis pipeline with speech_detected=True and valid score.
6. Live silence/pause -> returns speech_detected=False, does not inflate segment history or summary counts.
7. Silence never returns Truthful or Deceptive verdict.
8. API response compatibility (AnalyzeResponse maintains backward compatibility with optional score/confidence).
9. UI index.html contains neutral scientific disclosures and disclaimers.
10. Stage 8 fairness protections remain intact.
"""

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app, live_manager
from app.model import VoiceFeatures
from app.utils import (
    load_audio_from_bytes,
    evaluate_audio_speech_quality,
    check_voice_presence,
    TARGET_SR,
    AudioValidationError,
    SPEECH_STATUS_VOICE_DETECTED,
    SPEECH_STATUS_VOICE_NOT_DETECTED,
    SPEECH_STATUS_AUDIO_TOO_SHORT,
    SPEECH_STATUS_AUDIO_QUALITY_TOO_LOW,
)
from app.feature_extractor import FeatureExtractor, FEATURE_ORDER
from app.explanation import generate_structured_explanation, generate_explanations


def create_wav_bytes(samples: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """Encode float numpy array as 16-bit PCM WAV in memory."""
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def create_synthetic_speech(duration: float = 2.0, sr: int = TARGET_SR, f0: float = 160.0) -> np.ndarray:
    """Synthesize clean harmonic voiced speech."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    signal = np.zeros(n, dtype=np.float32)
    for k in range(1, 5):
        signal += (0.25 / k) * np.sin(2 * np.pi * (k * f0) * t)
    # Modulation and gentle windowing
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    signal *= envelope
    fade = int(0.04 * sr)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)
    return signal.astype(np.float32)


class TestSilenceAndInconclusive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.extractor = FeatureExtractor()

    def setUp(self):
        live_manager.stop()

    def tearDown(self):
        live_manager.stop()

    # -------------------------------------------------------------------------
    # 1. Completely silent uploaded audio
    # -------------------------------------------------------------------------
    def test_uploaded_complete_silence(self):
        """Pure zeros audio uploaded to /analyze should return No Speech Detected with score=None."""
        silence = np.zeros(TARGET_SR * 2, dtype=np.float32)
        wav_bytes = create_wav_bytes(silence)

        response = self.client.post(
            "/analyze",
            files={"audio": ("silence.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertFalse(data["speech_detected"])
        self.assertEqual(data["speech_status"], SPEECH_STATUS_VOICE_NOT_DETECTED)
        self.assertEqual(data["verdict"], "No Speech Detected")
        self.assertIsNone(data["score"])
        self.assertIsNone(data["confidence"])
        self.assertTrue(any("silence" in exp.lower() or "voice" in exp.lower() for exp in data["explanation"]))

    # -------------------------------------------------------------------------
    # 2. Near-silent audio
    # -------------------------------------------------------------------------
    def test_uploaded_near_silence(self):
        """Audio with RMS < 0.003 should be detected as VOICE NOT DETECTED with score=None."""
        near_silence = np.random.uniform(-0.0005, 0.0005, TARGET_SR * 2).astype(np.float32)
        wav_bytes = create_wav_bytes(near_silence)

        response = self.client.post(
            "/analyze",
            files={"audio": ("near_silence.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertFalse(data["speech_detected"])
        self.assertEqual(data["speech_status"], SPEECH_STATUS_VOICE_NOT_DETECTED)
        self.assertIsNone(data["score"])
        self.assertIsNone(data["confidence"])
        self.assertIn("No Speech Detected", data["verdict"])

    # -------------------------------------------------------------------------
    # 3. Background noise without meaningful speech
    # -------------------------------------------------------------------------
    def test_background_noise_without_speech(self):
        """Random unvoiced noise (e.g. fan hiss or clicks) must not receive deception scores."""
        # Gaussian unvoiced noise with no harmonic structure (HNR < 3.5 dB)
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, TARGET_SR * 2).astype(np.float32)
        noise = np.clip(noise, -0.9, 0.9)
        wav_bytes = create_wav_bytes(noise)

        response = self.client.post(
            "/analyze",
            files={"audio": ("fan_noise.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertFalse(data["speech_detected"])
        self.assertIn(data["speech_status"], [SPEECH_STATUS_VOICE_NOT_DETECTED, SPEECH_STATUS_AUDIO_QUALITY_TOO_LOW])
        self.assertIsNone(data["score"])
        self.assertIsNone(data["confidence"])
        self.assertNotIn(data["verdict"], ["Truthful", "Deceptive"])

    # -------------------------------------------------------------------------
    # 4. Very short audio below minimum
    # -------------------------------------------------------------------------
    def test_ultra_short_audio_handling(self):
        """Recordings under 0.5s are rejected or marked AUDIO TOO SHORT without scores."""
        short_audio = create_synthetic_speech(duration=0.2)
        wav_bytes = create_wav_bytes(short_audio)

        response = self.client.post(
            "/analyze",
            files={"audio": ("short.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        # Should reject with 400/422 validation error or return 200 with AUDIO TOO SHORT
        if response.status_code == 200:
            data = response.json()
            self.assertFalse(data["speech_detected"])
            self.assertEqual(data["speech_status"], SPEECH_STATUS_AUDIO_TOO_SHORT)
            self.assertIsNone(data["score"])
        else:
            self.assertIn(response.status_code, (400, 422))

    # -------------------------------------------------------------------------
    # 5. Valid speech continues through analysis pipeline
    # -------------------------------------------------------------------------
    def test_valid_speech_analysis(self):
        """Valid voiced harmonic speech continues through full pipeline with valid prediction."""
        speech = create_synthetic_speech(duration=2.0, f0=150.0)
        wav_bytes = create_wav_bytes(speech)

        response = self.client.post(
            "/analyze",
            files={"audio": ("valid_speech.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["speech_detected"])
        self.assertEqual(data["speech_status"], SPEECH_STATUS_VOICE_DETECTED)
        self.assertIn(data["verdict"], ["Truthful", "Deceptive"])
        self.assertIsNotNone(data["score"])
        self.assertIsNotNone(data["confidence"])
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)
        self.assertIn("structured_explanation", data)
        self.assertIn("acoustic_observations", data["structured_explanation"])

    # -------------------------------------------------------------------------
    # 6. Live recording pause/silence does not create prediction or inflate counts
    # -------------------------------------------------------------------------
    def test_live_silence_does_not_inflate_counts_or_create_prediction(self):
        """Live recording silence chunks must not create scores, must not append to history,
        and must not inflate session segment counts."""
        self.client.post("/live/start", json={"window_seconds": 15})

        silence = np.zeros(int(TARGET_SR * 1.5), dtype=np.float32)
        silence_bytes = create_wav_bytes(silence)

        # Send multiple consecutive silence chunks (simulating user staying silent / pausing)
        for i in range(3):
            resp = self.client.post(
                "/live/analyze",
                files={"audio": (f"live_silence_{i}.wav", io.BytesIO(silence_bytes), "audio/wav")},
            )
            self.assertEqual(resp.status_code, 200)
            res_data = resp.json()

            self.assertFalse(res_data["speech_detected"])
            self.assertEqual(res_data["speech_status"], SPEECH_STATUS_VOICE_NOT_DETECTED)
            self.assertIsNone(res_data["score"])
            self.assertIsNone(res_data["confidence"])
            self.assertIsNone(res_data["segment"])
            self.assertEqual(res_data["verdict"], "No Speech Detected")

        # Check /live/results -> should have 0 segments
        res_history = self.client.get("/live/results").json()
        self.assertEqual(len(res_history), 0)

        # Check /live/summary -> total_segments must be 0
        summary_resp = self.client.get("/live/summary").json()
        self.assertEqual(summary_resp["total_segments"], 0)
        self.assertIn(summary_resp["overall_verdict"], ["No speech segments analyzed", "Voice not detected", "No Speech Detected"])
        self.assertIsNone(summary_resp["avg_score"])

    # -------------------------------------------------------------------------
    # 7. Resuming speech after pause in live session
    # -------------------------------------------------------------------------
    def test_live_speech_resumes_after_pause(self):
        """When speech returns after a pause, a valid segment result is produced and recorded."""
        self.client.post("/live/start", json={"window_seconds": 15})

        # Silence chunk first
        silence_bytes = create_wav_bytes(np.zeros(int(TARGET_SR * 1.5), dtype=np.float32))
        r1 = self.client.post("/live/analyze", files={"audio": ("s.wav", io.BytesIO(silence_bytes), "audio/wav")})
        self.assertFalse(r1.json()["speech_detected"])

        # Real speech arrives
        speech_bytes = create_wav_bytes(create_synthetic_speech(duration=1.5, f0=180.0))
        r2 = self.client.post("/live/analyze", files={"audio": ("speech.wav", io.BytesIO(speech_bytes), "audio/wav")})
        d2 = r2.json()

        self.assertTrue(d2["speech_detected"])
        self.assertEqual(d2["speech_status"], SPEECH_STATUS_VOICE_DETECTED)
        self.assertIsNotNone(d2["score"])
        self.assertEqual(d2["segment"], 1)

        # Results should have exactly 1 segment
        results = self.client.get("/live/results").json()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["segment"], 1)

        # Summary should have total_segments == 1
        summary = self.client.get("/live/summary").json()
        self.assertEqual(summary["total_segments"], 1)
        self.assertIsNotNone(summary["avg_score"])

    # -------------------------------------------------------------------------
    # 8. API schema compatibility
    # -------------------------------------------------------------------------
    def test_api_schema_compatibility(self):
        """AnalyzeResponse schema preserves all 42-feature schema fields, reliability, and structured explanations."""
        speech = create_synthetic_speech(duration=2.0, f0=160.0)
        wav_bytes = create_wav_bytes(speech)

        resp = self.client.post(
            "/analyze?include_features=true",
            files={"audio": ("test_compat.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        expected_fields = [
            "verdict", "score", "confidence", "explanation", "features",
            "structured_explanation", "speech_detected", "speech_status"
        ]
        for field in expected_fields:
            self.assertIn(field, data, f"Missing expected field: {field}")

        feats = data["features"]
        self.assertIsNotNone(feats)
        self.assertEqual(len(feats["mfcc_mean"]), 13)
        self.assertEqual(len(feats["mfcc_std"]), 13)
        vf_obj = VoiceFeatures(**feats)
        vec = self.extractor.to_vector(vf_obj)
        self.assertEqual(len(vec), 42)

    # -------------------------------------------------------------------------
    # 9. UI uses neutral labels and disclaimers
    # -------------------------------------------------------------------------
    def test_ui_neutral_labels_and_disclaimers(self):
        """index.html must display scientific disclaimers and neutral model terminology."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Verify neutral labels
        self.assertIn("Raw Model Score", html)
        self.assertIn("Uncalibrated Model Score", html)
        self.assertIn("MODEL OUTPUT — NOT VERIFIED", html)
        self.assertIn("Scientific Disclaimer", html)
        self.assertIn("Voice patterns alone do not prove deception", html)
        self.assertIn("Indian-English deception detection has not been validated", html)

    # -------------------------------------------------------------------------
    # 10. Stage 8 fairness protections remain intact
    # -------------------------------------------------------------------------
    def test_stage8_fairness_protections_intact(self):
        """Structured explanations must not claim pitch, tempo, or accent prove deception."""
        from app.model import VoiceFeatures
        feat_dict = {f: 0.0 for f in FEATURE_ORDER if f not in ("mfcc_mean", "mfcc_std")}
        feat_dict["mfcc_mean"] = [0.0] * 13
        feat_dict["mfcc_std"] = [0.0] * 13
        feat_dict["syllable_rate"] = 6.2  # Fast tempo common in Indian English
        feat_dict["f0_mean"] = 230.0
        feat_dict["hnr_mean"] = 18.0
        vf = VoiceFeatures(**feat_dict)

        expl = generate_structured_explanation(features=vf, score=0.72, confidence=0.72)
        for note in expl.acoustic_observations:
            self.assertNotIn("proves lying", note.lower())
            self.assertNotIn("guilty", note.lower())
            self.assertNotIn("nervousness indicates deception", note.lower())
        self.assertTrue(len(expl.limitations) > 0)

    # -------------------------------------------------------------------------
    # 11. Regression: Valid speech passes live speech-detection guard
    # -------------------------------------------------------------------------
    def test_live_valid_speech_passes_detection_guard(self):
        """Audible conversational speech must pass live speech guard and reach prediction pipeline."""
        self.client.post("/live/start", json={"window_seconds": 15})
        speech = create_synthetic_speech(duration=1.8, f0=160.0)
        wav_bytes = create_wav_bytes(speech)

        resp = self.client.post(
            "/live/analyze",
            files={"audio": ("live_user_speech.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertTrue(data["speech_detected"])
        self.assertEqual(data["speech_status"], SPEECH_STATUS_VOICE_DETECTED)
        self.assertIn(data["verdict"], ["Truthful", "Deceptive"])
        self.assertIsNotNone(data["score"])
        self.assertIsNotNone(data["confidence"])
        self.assertEqual(data["segment"], 1)

    # -------------------------------------------------------------------------
    # 12. Regression: Genuine silence rejected without prediction
    # -------------------------------------------------------------------------
    def test_live_genuine_silence_rejected_without_prediction(self):
        """Zero amplitude digital silence must not generate deception scores or segment count."""
        self.client.post("/live/start", json={"window_seconds": 15})
        silence = np.zeros(TARGET_SR * 2, dtype=np.float32)
        wav_bytes = create_wav_bytes(silence)

        resp = self.client.post(
            "/live/analyze",
            files={"audio": ("live_silence.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertFalse(data["speech_detected"])
        self.assertEqual(data["speech_status"], SPEECH_STATUS_VOICE_NOT_DETECTED)
        self.assertIsNone(data["score"])
        self.assertIsNone(data["confidence"])
        self.assertIsNone(data["segment"])
        self.assertEqual(data["verdict"], "No Speech Detected")

    # -------------------------------------------------------------------------
    # 13. Regression: Near-silence and background noise not generating scores
    # -------------------------------------------------------------------------
    def test_live_near_silence_and_noise_no_scores(self):
        """Ambient room noise and unvoiced noise floor must never generate scores."""
        self.client.post("/live/start", json={"window_seconds": 15})
        np.random.seed(123)
        ambient_noise = np.random.normal(0, 0.0005, TARGET_SR * 2).astype(np.float32)
        wav_bytes = create_wav_bytes(ambient_noise)

        resp = self.client.post(
            "/live/analyze",
            files={"audio": ("ambient_room.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["speech_detected"])
        self.assertIsNone(data["score"])
        self.assertIsNone(data["segment"])

        # Also test louder unvoiced noise (e.g. mic hiss or typing)
        fan_noise = np.random.normal(0, 0.04, TARGET_SR * 2).astype(np.float32)
        fan_bytes = create_wav_bytes(fan_noise)
        resp2 = self.client.post(
            "/live/analyze",
            files={"audio": ("fan_noise.wav", io.BytesIO(fan_bytes), "audio/wav")},
        )
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertFalse(data2["speech_detected"])
        self.assertIsNone(data2["score"])

    # -------------------------------------------------------------------------
    # 14. Regression: Short live chunks accumulating correctly in rolling buffer
    # -------------------------------------------------------------------------
    def test_short_live_chunks_accumulate_in_rolling_buffer(self):
        """Short live audio chunks accumulate across submissions until sufficient duration is reached."""
        self.client.post("/live/start", json={"window_seconds": 15})

        # Chunk 1: 0.3s of speech -> under 0.5s floor -> returns Waiting for speech / AUDIO TOO SHORT
        chunk1 = create_synthetic_speech(duration=0.3, f0=150.0)
        resp1 = self.client.post(
            "/live/analyze",
            files={"audio": ("chunk1.wav", io.BytesIO(create_wav_bytes(chunk1)), "audio/wav")},
        )
        self.assertEqual(resp1.status_code, 200)
        d1 = resp1.json()
        self.assertFalse(d1["speech_detected"])
        self.assertEqual(d1["speech_status"], SPEECH_STATUS_AUDIO_TOO_SHORT)
        self.assertIsNone(d1["score"])

        # Chunk 2: 0.4s of speech -> accumulated buffer is 0.7s >= 0.5s -> completes analysis
        chunk2 = create_synthetic_speech(duration=0.4, f0=150.0)
        resp2 = self.client.post(
            "/live/analyze",
            files={"audio": ("chunk2.wav", io.BytesIO(create_wav_bytes(chunk2)), "audio/wav")},
        )
        self.assertEqual(resp2.status_code, 200)
        d2 = resp2.json()
        self.assertTrue(d2["speech_detected"])
        self.assertEqual(d2["speech_status"], SPEECH_STATUS_VOICE_DETECTED)
        self.assertIsNotNone(d2["score"])
        self.assertEqual(d2["segment"], 1)

    # -------------------------------------------------------------------------
    # 15. Regression: Supported browser audio formats decoding correctly
    # -------------------------------------------------------------------------
    def test_supported_audio_formats_decoding(self):
        """WAV, OGG, and FLAC containers decode without errors and route to VAD."""
        speech = create_synthetic_speech(duration=1.2, f0=170.0)

        # OGG container
        buf_ogg = io.BytesIO()
        sf.write(buf_ogg, speech, TARGET_SR, format="OGG")
        resp_ogg = self.client.post(
            "/analyze",
            files={"audio": ("speech.ogg", io.BytesIO(buf_ogg.getvalue()), "audio/ogg")},
        )
        self.assertEqual(resp_ogg.status_code, 200)
        self.assertTrue(resp_ogg.json()["speech_detected"])

        # FLAC container
        buf_flac = io.BytesIO()
        sf.write(buf_flac, speech, TARGET_SR, format="FLAC")
        resp_flac = self.client.post(
            "/analyze",
            files={"audio": ("speech.flac", io.BytesIO(buf_flac.getvalue()), "audio/flac")},
        )
        self.assertEqual(resp_flac.status_code, 200)
        self.assertTrue(resp_flac.json()["speech_detected"])

    # -------------------------------------------------------------------------
    # 16. Regression: Invalid or incomplete chunks producing diagnostic status
    # -------------------------------------------------------------------------
    def test_invalid_or_incomplete_chunks_diagnostic_status(self):
        """Incomplete / empty / corrupt chunks produce explicit diagnostic status."""
        self.client.post("/live/start", json={"window_seconds": 15})

        # Empty chunk
        resp_empty = self.client.post(
            "/live/analyze",
            files={"audio": ("empty.wav", io.BytesIO(b""), "audio/wav")},
        )
        self.assertEqual(resp_empty.status_code, 400)
        self.assertIn("empty", resp_empty.json()["detail"].lower())

        # Corrupted non-audio chunk
        resp_corrupt = self.client.post(
            "/live/analyze",
            files={"audio": ("corrupt.wav", io.BytesIO(b"NOT_A_WAV_HEADER_DATA"), "audio/wav")},
        )
        self.assertIn(resp_corrupt.status_code, (400, 422))

    # -------------------------------------------------------------------------
    # 17. Regression: Live status and response fields backward compatible
    # -------------------------------------------------------------------------
    def test_live_status_and_response_backward_compatibility(self):
        """Live endpoints maintain all existing schema keys and types."""
        self.client.post("/live/start", json={"window_seconds": 15})

        status = self.client.get("/live/status").json()
        expected_status_keys = {"running", "window", "segments_processed", "elapsed_time", "current_status"}
        for k in expected_status_keys:
            self.assertIn(k, status)

        speech = create_synthetic_speech(duration=1.5, f0=160.0)
        resp = self.client.post(
            "/live/analyze",
            files={"audio": ("segment.wav", io.BytesIO(create_wav_bytes(speech)), "audio/wav")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        expected_response_keys = {
            "segment", "verdict", "score", "confidence",
            "speech_detected", "speech_status", "explanation", "duration"
        }
        for k in expected_response_keys:
            self.assertIn(k, data)

    # -------------------------------------------------------------------------
    # 18. Regression: 15-second live silence with fan hum / room noise never predicts
    # -------------------------------------------------------------------------
    def test_15s_live_silence_with_fan_hum_never_predicts(self):
        """A full 15s live window of ambient room silence with fan hum must return
        speech_detected=False, score=None, and never generate a deception/truth score."""
        self.client.post("/live/start", json={"window_seconds": 15})

        n = int(15.0 * TARGET_SR)
        t = np.arange(n) / TARGET_SR
        fan = 0.015 * np.sin(2 * np.pi * 120 * t) + 0.008 * np.sin(2 * np.pi * 240 * t)
        noise = np.random.normal(0, 0.008, n)
        ambient = (fan + noise).astype(np.float32)

        resp = self.client.post(
            "/live/analyze",
            files={"audio": ("live_15s_ambient.wav", io.BytesIO(create_wav_bytes(ambient)), "audio/wav")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertFalse(data["speech_detected"])
        self.assertEqual(data["speech_status"], SPEECH_STATUS_VOICE_NOT_DETECTED)
        self.assertIsNone(data["score"])
        self.assertIsNone(data["confidence"])
        self.assertIsNone(data["segment"])
        self.assertEqual(data["verdict"], "No Speech Detected")

        # Must not be recorded into live history
        results = self.client.get("/live/results").json()
        self.assertEqual(len(results), 0)

        summary = self.client.get("/live/summary").json()
        self.assertEqual(summary["total_segments"], 0)
        self.assertIsNone(summary["avg_score"])

    # -------------------------------------------------------------------------
    # 19. Regression: 15-second uploaded silence with fan hum never predicts
    # -------------------------------------------------------------------------
    def test_15s_uploaded_silence_with_fan_hum_never_predicts(self):
        """A 15s uploaded audio file containing room noise and fan hum must return
        speech_detected=False, score=None, and No Speech Detected."""
        n = int(15.0 * TARGET_SR)
        t = np.arange(n) / TARGET_SR
        fan = 0.015 * np.sin(2 * np.pi * 120 * t) + 0.008 * np.sin(2 * np.pi * 240 * t)
        noise = np.random.normal(0, 0.008, n)
        ambient = (fan + noise).astype(np.float32)

        resp = self.client.post(
            "/analyze",
            files={"audio": ("uploaded_15s_ambient.wav", io.BytesIO(create_wav_bytes(ambient)), "audio/wav")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertFalse(data["speech_detected"])
        self.assertEqual(data["speech_status"], SPEECH_STATUS_VOICE_NOT_DETECTED)
        self.assertIsNone(data["score"])
        self.assertIsNone(data["confidence"])
        self.assertEqual(data["verdict"], "No Speech Detected")

    # -------------------------------------------------------------------------
    # 20. Regression: Empty live session shows "No speech segments analyzed"
    # -------------------------------------------------------------------------
    def test_empty_session_summary_shows_no_speech_analyzed(self):
        """When a live session has no valid speech predictions, overall_verdict
        must be 'No speech segments analyzed' rather than inventing an overall verdict."""
        self.client.post("/live/start", json={"window_seconds": 15})

        # Submit pure silence
        silence = np.zeros(TARGET_SR * 2, dtype=np.float32)
        self.client.post(
            "/live/analyze",
            files={"audio": ("silence.wav", io.BytesIO(create_wav_bytes(silence)), "audio/wav")},
        )

        # Stop session
        stop_resp = self.client.post("/live/stop").json()
        summary = stop_resp.get("summary", {})

        self.assertEqual(summary["total_segments"], 0)
        self.assertEqual(summary["overall_verdict"], "No speech segments analyzed")
        self.assertIsNone(summary["avg_score"])
        self.assertIsNone(summary["avg_confidence"])
        self.assertIsNone(summary["highest_risk_segment"])
        self.assertIsNone(summary["lowest_risk_segment"])

    # -------------------------------------------------------------------------
    # 21. Regression: Session summary with speech uses neutral Experimental label
    # -------------------------------------------------------------------------
    def test_speech_session_summary_uses_neutral_model_output_label(self):
        """Summary of analyzed speech must describe results as Experimental Model Output,
        not as verified honesty or deception."""
        self.client.post("/live/start", json={"window_seconds": 15})

        speech = create_synthetic_speech(duration=2.0, f0=160.0)
        resp = self.client.post(
            "/live/analyze",
            files={"audio": ("speech.wav", io.BytesIO(create_wav_bytes(speech)), "audio/wav")},
        )
        self.assertEqual(resp.status_code, 200)

        summary = self.client.get("/live/summary").json()
        self.assertEqual(summary["total_segments"], 1)
        self.assertIsNotNone(summary["avg_score"])
        self.assertTrue(
            summary["overall_verdict"].startswith("Experimental Model Output"),
            f"Expected overall_verdict to start with 'Experimental Model Output', got: {summary['overall_verdict']}"
        )

    # -------------------------------------------------------------------------
    # 22. Regression: UI HTML contains neutral labels and accurate terminology
    # -------------------------------------------------------------------------
    def test_ui_html_neutral_labels_and_accurate_terminology(self):
        """index.html must display neutral terminology: Experimental Model Output,
        Overall Model Output, Raw Model Score, and Avg Uncalibrated Model Score."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        self.assertIn("Experimental Model Output", html)
        self.assertIn("Overall Model Output", html)
        self.assertIn("Raw Model Score", html)
        self.assertIn("Uncalibrated Model Score", html)
        self.assertIn("Avg Raw Model Score", html)
        self.assertIn("Avg Uncalibrated Model Score", html)
        self.assertIn("Highest Model Score Segment", html)

    # -------------------------------------------------------------------------
    # 23. Regression: Silence following speech preserves speech history without inflation
    # -------------------------------------------------------------------------
    def test_silence_after_speech_preserves_history_and_no_extra_segments(self):
        """When silence arrives after valid speech, valid speech is preserved in history,
        silent windows return score=None with speech_detected=False, and segment count remains 1."""
        self.client.post("/live/start", json={"window_seconds": 15})

        # 1. Valid speech
        speech = create_synthetic_speech(duration=1.8, f0=175.0)
        r_speech = self.client.post(
            "/live/analyze",
            files={"audio": ("speech1.wav", io.BytesIO(create_wav_bytes(speech)), "audio/wav")},
        ).json()
        self.assertTrue(r_speech["speech_detected"])
        self.assertEqual(r_speech["segment"], 1)

        # 2. Silence arrives
        silence = np.zeros(TARGET_SR * 2, dtype=np.float32)
        r_silence1 = self.client.post(
            "/live/analyze",
            files={"audio": ("silence1.wav", io.BytesIO(create_wav_bytes(silence)), "audio/wav")},
        ).json()
        self.assertFalse(r_silence1["speech_detected"])
        self.assertIsNone(r_silence1["score"])
        self.assertIsNone(r_silence1["segment"])

        r_silence2 = self.client.post(
            "/live/analyze",
            files={"audio": ("silence2.wav", io.BytesIO(create_wav_bytes(silence)), "audio/wav")},
        ).json()
        self.assertFalse(r_silence2["speech_detected"])
        self.assertIsNone(r_silence2["score"])

        # 3. Check history: must still contain exactly 1 segment
        history = self.client.get("/live/results").json()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["segment"], 1)

        # 4. Check summary: total_segments must be 1, not 3
        summary = self.client.get("/live/summary").json()
        self.assertEqual(summary["total_segments"], 1)


if __name__ == "__main__":
    unittest.main()


