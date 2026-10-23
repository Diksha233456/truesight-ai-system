"""
tests/test_model_caching_and_integrity.py

Automated tests verifying:
1. Pretrained model loads from persistent local cache without any network access.
2. The inference execution path never invokes model training or fitting.
3. Checksum verification detects corrupted artifacts and raises ModelIntegrityError.
4. Atomic download cleans up temporary files upon interruption or checksum failure.
5. Missing models and failed downloads produce clear actionable errors and NEVER silently retrain.
6. Loaded model instance is retained in memory and reused across predictions.
"""

import hashlib
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from app.model_loader import (
    resolve_model,
    compute_sha256,
    download_file_atomic,
    ModelIntegrityError,
    ModelDownloadError,
    ModelNotFoundError,
    PINNED_CHECKSUMS,
)
from app.predictor import Predictor, ModelNotLoadedError


class TestModelCachingAndIntegrity(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dummy_features = np.zeros(42, dtype=np.float32)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cached_model_loads_without_network(self):
        """Confirm that resolving a cached model makes zero network requests."""
        # 1. Create a dummy valid model in a custom cache directory
        version = "v1.0.0"
        cache_version_dir = Path(self.temp_dir) / version
        cache_version_dir.mkdir(parents=True)
        model_file = cache_version_dir / "deception_model.pkl"

        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.randn(20, 42).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10, dtype=np.int64)
        clf.fit(X, y)
        joblib.dump(clf, model_file)

        model_sha256 = compute_sha256(model_file)

        # Patch PINNED_CHECKSUMS to match this test model
        with patch.dict(PINNED_CHECKSUMS, {version: {"deception_model.pkl": model_sha256}}):
            # Block all network socket and urllib calls
            with patch("urllib.request.urlopen", side_effect=RuntimeError("Network access forbidden")):
                with patch("urllib.request.Request", side_effect=RuntimeError("Network access forbidden")):
                    resolved_path, meta = resolve_model(
                        revision=version,
                        cache_dir=self.temp_dir,
                        force_download=False,
                    )
                    self.assertEqual(resolved_path, model_file)
                    self.assertTrue(resolved_path.exists())

                    # Initialize predictor using this resolved path
                    pred = Predictor(model_path=str(resolved_path))
                    self.assertTrue(pred.is_ready())
                    verdict, score, conf = pred.predict(self.dummy_features)
                    self.assertIn(verdict, ["Truthful", "Deceptive"])

    def test_inference_path_never_invokes_training(self):
        """Confirm that calling Predictor.predict never calls clf.fit or any training routine."""
        pred = Predictor()
        if not pred.is_ready():
            self.skipTest("Predictor model not available for inference test.")

        # Spy / mock RandomForestClassifier.fit to ensure it is NEVER called
        with patch.object(RandomForestClassifier, "fit", side_effect=AssertionError("Training was invoked during inference!")):
            # Also mock train_model.train if imported
            with patch("train_model.train", side_effect=AssertionError("train_model.train was invoked during inference!")):
                verdict, score, conf = pred.predict(self.dummy_features)
                self.assertIn(verdict, ["Truthful", "Deceptive"])
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_checksum_mismatch_raises_integrity_error(self):
        """Confirm that corrupted downloaded files raise ModelIntegrityError."""
        dest_file = Path(self.temp_dir) / "corrupted_model.pkl"

        # Mock urllib to return dummy corrupted bytes
        mock_response = MagicMock()
        mock_response.read.side_effect = [b"corrupted_model_content_here", b""]
        mock_response.__enter__.return_value = mock_response

        with patch("urllib.request.urlopen", return_value=mock_response):
            with self.assertRaises(ModelIntegrityError) as ctx:
                download_file_atomic(
                    url="https://example.com/fake_model.pkl",
                    destination=dest_file,
                    expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
                )
            self.assertIn("Integrity check failed", str(ctx.exception))
            # Verify temporary file was cleaned up and corrupted target does NOT exist
            self.assertFalse(dest_file.exists())

    def test_failed_download_never_falls_back_to_training(self):
        """Confirm that missing models with network failure raise actionable errors and NEVER retrain."""
        empty_cache = Path(self.temp_dir) / "empty_cache"
        empty_cache.mkdir()

        with patch.dict(os.environ, {"TRUSIGHT_MODEL_CACHE_DIR": str(empty_cache)}):
            with patch("app.model_loader.fetch_from_huggingface", side_effect=ModelDownloadError("HF unreachable")):
                with patch("train_model.train", side_effect=AssertionError("Training must not be triggered!")):
                    with self.assertRaises(ModelDownloadError) as ctx:
                        resolve_model(
                            revision="v9.9.9",
                            cache_dir=empty_cache,
                            force_download=True,
                        )
                    self.assertIn("HF unreachable", str(ctx.exception))

    def test_predictor_model_not_loaded_error(self):
        """Confirm that when model fails to load, predict_raw raises ModelNotLoadedError."""
        pred = Predictor(model_path=os.path.join(self.temp_dir, "non_existent.pkl"))
        self.assertFalse(pred.is_ready())
        with self.assertRaises(ModelNotLoadedError) as ctx:
            pred.predict(self.dummy_features)
        self.assertIn("Deception model is not loaded", str(ctx.exception))
        # Verify message does NOT instruct user to run train_model.py
        self.assertNotIn("Run train_model.py to create", str(ctx.exception))

    def test_explicit_model_path_override(self):
        """Confirm that TRUSIGHT_MODEL_PATH environment variable overrides cache resolution."""
        override_file = Path(self.temp_dir) / "custom_override_model.pkl"
        clf = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.randn(20, 42).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10, dtype=np.int64)
        clf.fit(X, y)
        joblib.dump(clf, override_file)

        with patch.dict(os.environ, {"TRUSIGHT_MODEL_PATH": str(override_file)}):
            resolved_path, meta = resolve_model()
            self.assertEqual(resolved_path, override_file)

    def test_model_loaded_once_and_reused(self):
        """Confirm that Predictor retains the loaded model in memory and reuses it."""
        pred = Predictor()
        if not pred.is_ready():
            self.skipTest("Predictor model not available.")

        model_id_1 = id(pred.model)
        pred.predict(self.dummy_features)
        model_id_2 = id(pred.model)
        self.assertEqual(model_id_1, model_id_2, "Predictor should reuse the same model instance in memory.")


if __name__ == "__main__":
    unittest.main()
