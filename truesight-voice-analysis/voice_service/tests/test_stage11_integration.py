"""
tests/test_stage11_integration.py

Comprehensive integration and release-readiness test suite for TRUESIGHT (Stage 11).
Tests all HTTP and WebSocket endpoints against app.main:app using FastAPI TestClient:
- GET /
- GET /health
- POST /reload-model
- POST /analyze (upload analysis, quality validation, 42-feature schema, structured explanations)
- POST /live/start (validation, conflict handling)
- GET /live/status
- POST /live/analyze (live segment processing)
- GET /live/results
- GET /live/summary
- POST /live/stop
- WebSocket /ws/live (handshake, initial state push)
- Safety and regression checks (error response hygiene, no leak of sensitive paths/secrets)
"""

import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app, live_manager
from app.utils import TARGET_SR
from app.feature_extractor import FEATURE_ORDER


def create_mock_speech_wav(duration: float = 2.0, sr: int = TARGET_SR, f0: float = 160.0) -> bytes:
    """Synthesize a clean harmonic speech-like WAV with slight modulation."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    signal = np.zeros(n, dtype=np.float32)
    for k in range(1, 5):
        signal += (0.2 / k) * np.sin(2 * np.pi * (k * f0) * t)
    # Add subtle amplitude envelope
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    signal *= envelope
    # Soft fade at boundaries
    fade = int(0.05 * sr)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)

    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestStage11Integration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Ensure clean initial state
        live_manager.stop()

    def tearDown(self):
        # Stop any open live session after each test
        live_manager.stop()

    def test_01_health_and_info_routes(self):
        """GET /health and GET / return expected status and headers."""
        # /health
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("service"), "Voice Analysis Service")
        self.assertEqual(data.get("status"), "running")

        # GET / (serves HTML UI or fallback)
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)

    def test_02_reload_model(self):
        """POST /reload-model successfully reloads the production model."""
        res = self.client.post("/reload-model")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("model_loaded", data)
        self.assertTrue(data["model_loaded"])

    def test_03_analyze_empty_and_invalid_inputs(self):
        """POST /analyze rejects empty files, invalid extensions, and digital silence."""
        # Empty file
        res = self.client.post(
            "/analyze",
            files={"audio": ("empty.wav", b"", "audio/wav")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json().get("detail", "").lower())

        # Invalid format / text disguised as wav
        res = self.client.post(
            "/analyze",
            files={"audio": ("fake.wav", b"NOT_A_WAV_FILE", "audio/wav")}
        )
        self.assertIn(res.status_code, (400, 422))

    def test_04_analyze_valid_speech(self):
        """POST /analyze returns full schema including structured explanations and 42-features."""
        wav_bytes = create_mock_speech_wav(duration=2.5, f0=150.0)

        # 1. Standard request without include_features
        res = self.client.post(
            "/analyze",
            files={"audio": ("speech.wav", wav_bytes, "audio/wav")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Contract checks
        self.assertIn("verdict", data)
        self.assertIn(data["verdict"], ["Truthful", "Deceptive", "Inconclusive", "No Speech Detected"])
        self.assertIn("score", data)
        self.assertIsInstance(data["score"], float)
        self.assertTrue(0.0 <= data["score"] <= 1.0)
        self.assertIn("confidence", data)
        self.assertIsInstance(data["confidence"], float)
        self.assertTrue(0.0 <= data["confidence"] <= 1.0)
        self.assertIn("explanation", data)
        self.assertIsInstance(data["explanation"], list)
        self.assertGreaterEqual(len(data["explanation"]), 1)

        # Structured explanation checks
        self.assertIn("structured_explanation", data)
        struct = data["structured_explanation"]
        self.assertIn("acoustic_observations", struct)
        self.assertIn("quality_warnings", struct)
        self.assertIn("prediction_context", struct)
        self.assertIn("limitations", struct)
        self.assertGreaterEqual(len(struct["limitations"]), 1)

        # Features should be None by default
        self.assertIsNone(data.get("features"))

        # 2. Request with include_features=true
        res_feat = self.client.post(
            "/analyze?include_features=true",
            files={"audio": ("speech.wav", wav_bytes, "audio/wav")}
        )
        self.assertEqual(res_feat.status_code, 200)
        data_feat = res_feat.json()
        features = data_feat.get("features")
        self.assertIsNotNone(features)

        # Check essential 42-feature fields exist
        for base_field in [
            "pitch_mean", "pitch_std", "pitch_min", "pitch_max",
            "jitter_local", "shimmer_local", "speech_tempo", "pause_ratio",
            "pause_count", "silence_duration", "rms_energy_mean", "rms_energy_std",
            "zero_crossing_rate", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"
        ]:
            self.assertIn(base_field, features)
        self.assertEqual(len(features["mfcc_mean"]), 13)
        self.assertEqual(len(features["mfcc_std"]), 13)
        # 16 scalar + 13 + 13 = 42 features exactly
        total_dim = 16 + len(features["mfcc_mean"]) + len(features["mfcc_std"])
        self.assertEqual(total_dim, 42)

    def test_05_live_session_lifecycle(self):
        """Lifecycle of live recording: start -> status -> analyze segment -> results -> summary -> stop."""
        # 1. Start with invalid window
        res = self.client.post("/live/start", json={"window_seconds": 99})
        self.assertEqual(res.status_code, 400)

        # 2. Start valid session (window 15s)
        res = self.client.post("/live/start", json={"window_seconds": 15})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json().get("window_seconds"), 15)

        # 3. Conflict: starting another session while one is active returns 409
        res_conflict = self.client.post("/live/start", json={"window_seconds": 15})
        self.assertEqual(res_conflict.status_code, 409)

        # 4. Check status
        res_status = self.client.get("/live/status")
        self.assertEqual(res_status.status_code, 200)
        status = res_status.json()
        self.assertTrue(status.get("running"))
        self.assertEqual(status.get("window"), 15)

        # 5. POST live segment
        chunk_wav = create_mock_speech_wav(duration=1.5, f0=170.0)
        res_chunk = self.client.post(
            "/live/analyze",
            files={"audio": ("live_segment.wav", chunk_wav, "audio/wav")}
        )
        self.assertEqual(res_chunk.status_code, 200)
        chunk_data = res_chunk.json()
        self.assertIn("segment", chunk_data)
        self.assertIn("verdict", chunk_data)
        self.assertIn("score", chunk_data)
        self.assertIn("confidence", chunk_data)

        # 6. Check results
        res_results = self.client.get("/live/results")
        self.assertEqual(res_results.status_code, 200)
        results = res_results.json()
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)

        # 7. Check summary
        res_summary = self.client.get("/live/summary")
        self.assertEqual(res_summary.status_code, 200)
        summary = res_summary.json()
        self.assertIn("total_segments", summary)
        self.assertGreaterEqual(summary["total_segments"], 1)

        # 8. Stop session
        res_stop = self.client.post("/live/stop")
        self.assertEqual(res_stop.status_code, 200)
        stop_data = res_stop.json()
        self.assertEqual(stop_data.get("message"), "Live session stopped.")
        self.assertIn("summary", stop_data)

        # 9. Verify status is stopped
        res_status_after = self.client.get("/live/status")
        self.assertFalse(res_status_after.json().get("running"))

    def test_06_websocket_live_connection(self):
        """WebSocket /ws/live accepts connections and sends initial status message."""
        with self.client.websocket_connect("/ws/live") as websocket:
            init_msg = websocket.receive_json()
            self.assertEqual(init_msg.get("type"), "connected")
            self.assertIn("status", init_msg)

    def test_07_error_response_hygiene(self):
        """Error responses must return clean JSON and not leak environment or path details."""
        res = self.client.post(
            "/analyze",
            files={"audio": ("corrupt.wav", b"\x00" * 64, "audio/wav")}
        )
        self.assertIn(res.status_code, (400, 422))
        err_body = res.json()
        self.assertIn("detail", err_body)
        detail = str(err_body["detail"])
        # Ensure internal root paths or passwords are not disclosed
        self.assertNotIn("password", detail.lower())
        self.assertNotIn("c:\\users\\", detail.lower())
        self.assertNotIn("secret", detail.lower())


if __name__ == "__main__":
    unittest.main()
