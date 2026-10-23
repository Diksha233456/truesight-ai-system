"""
verify_setup.py

Quick smoke-test that verifies the entire TRUSIGHT Voice Analysis pipeline
works end-to-end without starting the HTTP server:

1. Imports all app modules
2. Loads the trained model
3. Runs feature extraction on a synthetic audio clip
4. Runs prediction and explanation generation
5. Validates the output shape matches the API contract

Run from the voice_service/ project root:
    python verify_setup.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

from __future__ import annotations

import sys
import os
import traceback

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASSED = []
FAILED = []


def check(name: str, fn):
    try:
        result = fn()
        PASSED.append(name)
        print(f"  [PASS] {name}")
        return result
    except Exception as e:
        FAILED.append(name)
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print(" TRUSIGHT Voice Analysis -- Setup Verification")
    print("=" * 60)
    print()

    # 1. Core imports
    print("-- 1. Package imports --")
    numpy = check("import numpy", lambda: __import__("numpy"))
    check("import librosa", lambda: __import__("librosa"))
    check("import parselmouth", lambda: __import__("parselmouth"))
    check("import sklearn", lambda: __import__("sklearn"))
    check("import joblib", lambda: __import__("joblib"))
    check("import soundfile", lambda: __import__("soundfile"))
    check("import fastapi", lambda: __import__("fastapi"))
    print()

    # 2. App module imports
    print("-- 2. App module imports --")
    utils_mod = check("from app.utils import load_audio_from_bytes",
                      lambda: __import__("app.utils", fromlist=["load_audio_from_bytes"]))
    fe_mod = check("from app.feature_extractor import FeatureExtractor",
                   lambda: __import__("app.feature_extractor", fromlist=["FeatureExtractor"]))
    pred_mod = check("from app.predictor import Predictor",
                     lambda: __import__("app.predictor", fromlist=["Predictor"]))
    expl_mod = check("from app.explanation import generate_explanations",
                     lambda: __import__("app.explanation", fromlist=["generate_explanations"]))
    print()

    # 3. Synthesize a short audio clip in-memory
    print("-- 3. Synthetic audio generation --")
    audio_obj = None
    if numpy is not None and utils_mod is not None:
        def make_audio():
            import numpy as np
            import soundfile as sf
            import io
            from app.utils import load_audio_from_bytes

            sr = 16000
            t = np.linspace(0, 2.0, int(2.0 * sr))
            # Voiced-like signal: sum of first 4 harmonics at ~160 Hz
            signal = (np.sin(2 * np.pi * 160 * t) +
                      0.5 * np.sin(2 * np.pi * 320 * t) +
                      0.25 * np.sin(2 * np.pi * 480 * t) +
                      0.125 * np.sin(2 * np.pi * 640 * t)).astype(np.float32)
            # Normalize
            signal = signal / np.max(np.abs(signal)) * 0.9

            buf = io.BytesIO()
            sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
            audio_bytes = buf.getvalue()
            return load_audio_from_bytes(audio_bytes, "test.wav")

        audio_obj = check("synthesize + load 2s audio", make_audio)
    print()

    # 4. Feature extraction
    print("-- 4. Feature extraction --")
    features = None
    vector = None
    if fe_mod is not None and audio_obj is not None:
        def run_extraction():
            from app.feature_extractor import FeatureExtractor
            ext = FeatureExtractor()
            f, v = ext.extract(audio_obj)
            assert len(v) == 42, f"Expected 42-dim vector, got {len(v)}"
            assert len(f.mfcc_mean) == 13, "Expected 13 MFCC means"
            return f, v

        result = check("FeatureExtractor.extract (42-dim vector)", run_extraction)
        if result:
            features, vector = result
    print()

    # 5. Model loading
    print("-- 5. Model loading --")
    predictor_obj = None
    if pred_mod is not None:
        def load_model():
            from app.predictor import Predictor
            p = Predictor()
            if not p.is_ready():
                raise RuntimeError(
                    "Model not loaded. Ensure pretrained model is in cache or run: python -m app.model_loader"
                )
            return p

        predictor_obj = check("Predictor loaded deception_model.pkl", load_model)
    print()

    # 6. Prediction
    print("-- 6. Prediction --")
    if predictor_obj is not None and vector is not None:
        def run_predict():
            verdict, score, confidence = predictor_obj.predict(vector)
            assert verdict in ("Truthful", "Deceptive"), f"Bad verdict: {verdict}"
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
            assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"
            return verdict, score, confidence

        pred_result = check("Predictor.predict returns valid (verdict, score, confidence)", run_predict)
        if pred_result:
            verdict, score, confidence = pred_result
            print(f"         -> verdict={verdict}, score={score:.4f}, confidence={confidence:.4f}")
    print()

    # 7. Explanation generation
    print("-- 7. Explanation generation --")
    if expl_mod is not None and features is not None:
        def run_explain():
            from app.explanation import generate_explanations
            notes = generate_explanations(features, 0.4)
            assert 2 <= len(notes) <= 5, f"Expected 2-5 notes, got {len(notes)}: {notes}"
            for note in notes:
                assert isinstance(note, str) and len(note) > 0
            return notes

        expl_result = check("generate_explanations returns 2-5 strings", run_explain)
        if expl_result:
            for note in expl_result:
                print(f"         -> {note}")
    print()

    # 8. FastAPI app import
    print("-- 8. FastAPI app --")
    check("from app.main import app (FastAPI app importable)", lambda: __import__("app.main", fromlist=["app"]))
    print()

    # Summary
    print("=" * 60)
    print(f" RESULTS: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print()
        print(" Failed checks:")
        for f in FAILED:
            print(f"   • {f}")
        print()
        print(" Run 'python -m app.model_loader' if model resolution or downloading failed.")
        print("=" * 60)
        sys.exit(1)
    else:
        print()
        print(" All checks passed! Start the server with:")
        print("   uvicorn app.main:app --reload")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
