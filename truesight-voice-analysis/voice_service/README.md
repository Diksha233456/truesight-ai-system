# TRUSIGHT — Voice Analysis Service

Standalone FastAPI microservice implementing the **Voice Analysis module** of TRUSIGHT, a multimodal truth-verification system.

This service estimates vocal deception likelihood and acoustic indicators from speech audio using a **pretrained, versioned machine learning model**. Training is completely decoupled from inference: users and developers can clone, install, and run inference immediately without retraining the model.

---

## Architecture: Decoupled Training & Inference

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MAINTAINER WORKFLOW                             │
│                                                                        │
│  [DOLOS / Real Dataset] ──> python train_model.py ──> artifacts/       │
│                                (42 acoustic feats)    ├─ model.pkl     │
│                                                       └─ metadata.json │
│                                                              │         │
│                        python publish_model.py <─────────────┘         │
│                                  │ (HF_TOKEN)                          │
│                                  ▼                                     │
│               [Hugging Face Hub: Diksha233456/...]                     │
│                           (Revision: v1.0.0)                           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   │ (Public download, no auth required)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          USER / INFERENCE FLOW                         │
│                                                                        │
│  1. uvicorn app.main:app (or setup.bat)                                │
│  2. app.model_loader: Checks local cache (~/.cache/truesight/models/)  │
│     ├─ If cached & SHA-256 valid: Loads model directly (Offline)       │
│     └─ If not in cache: Downloads pinned revision, verifies checksum,  │
│        and writes atomically to local cache.                           │
│  3. Normal API execution: POST /analyze runs inference in <100ms.      │
│     ⚠️ TRAINING IS NEVER TRIGGERED AUTOMATICALLY.                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart for New Users (Inference Only)

No model training is required. The service automatically resolves, downloads, and caches the pinned pretrained model on first boot.

### 1. Prerequisites
- **Python 3.11, 3.12, or 3.13** (from [python.org](https://python.org)).
- On Linux/Debian/Ubuntu, install audio decoding libraries:
  ```bash
  sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg
  ```

### 2. Windows One-Click Setup
```bat
cd voice_service
.\setup.bat
```
This creates `.venv`, installs dependencies, and verifies the pretrained model cache.

### 3. Manual Installation (Any OS)
```bash
cd voice_service

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the Service
```bash
uvicorn app.main:app --reload
```
Open **http://127.0.0.1:8000** for the web interface or **http://127.0.0.1:8000/docs** for the interactive Swagger API docs.

---

## Pretrained Model Details & Hosting

- **Hosting Service**: [Hugging Face Hub](https://huggingface.co/)
- **Repository ID**: `Diksha233456/truesight-voice-deception`
- **Pinned Revision**: `v1.0.0`
- **Artifacts**:
  - `deception_model.pkl` (~828 KB, SHA-256: `f2ad274377a33c5511ed1e01452113f75874f912818faeea95776cdd934f0b88`)
  - `model_metadata.json` (architecture, schema, performance metrics, and timestamps)
- **Authentication**: **None required**. Model artifacts are publicly hosted and can be downloaded by any user without providing maintainer API keys or tokens.
- **Model Architecture**: Balanced `RandomForestClassifier` (200 estimators) trained on 42 Praat and Librosa acoustic features.
- **Evaluation Performance**:
  - 5-Fold Stratified Cross-Validation Accuracy: `0.643 ± 0.075`
  - Held-out Test Accuracy: `0.588`
  - Held-out ROC AUC: `0.696`

---

## Local Caching & Offline Execution

### Default Cache Location
Downloaded models are stored in:
- **Linux/macOS**: `~/.cache/truesight/models/<version>/`
- **Windows**: `C:\Users\<username>\.cache\truesight\models\<version>\`

### Atomic File Downloads & Verification
The loader writes all downloads to temporary files (`.tmp.<uuid>`) and computes the SHA-256 digest before moving them to the cache. If an interrupted download or invalid hash is detected, the temporary file is removed immediately to prevent partial/corrupted models.

### Environment Variables
You can customize model loading using environment variables:

| Variable | Default | Description |
|---|---|---|
| `TRUSIGHT_MODEL_REPO` | `Diksha233456/truesight-voice-deception` | Remote Hugging Face model repository ID. |
| `TRUSIGHT_MODEL_REVISION` | `v1.0.0` | Pinned model version or Git tag. |
| `TRUSIGHT_MODEL_CACHE_DIR` | `~/.cache/truesight/models` | Root directory for local model storage. |
| `TRUSIGHT_MODEL_PATH` | _None_ | Direct path to a local `.pkl` file (bypasses remote checks, ideal for air-gapped environments). |
| `TRUSIGHT_AUTO_DOWNLOAD` | `true` | Set to `false` to disable network downloads and enforce local-only execution. |

### Cache Management CLI
Inspect or pre-download the model:
```bash
python -m app.model_loader
```

Clear the local cache:
```bash
python -m app.model_loader --clear-cache
```

---

## Maintainer Workflow: Training & Publishing

Training is an explicit operation that only maintainers run when creating or releasing a new model version. Normal application execution **never** calls training.

### 1. Train and Export Versioned Artifacts
```bash
# Train on the bundled DOLOS dataset and export to artifacts/
python train_model.py --version v1.0.0 --export-dir artifacts

# Or train on a custom dataset directory with truthful/ and deceptive/ subfolders:
python train_model.py --data-dir path/to/dataset --version v1.1.0
```
This generates:
- `artifacts/deception_model.pkl`
- `artifacts/model_metadata.json` (with real cross-validation and test metrics and SHA-256 hash)

### 2. Verify Artifacts Locally (Dry Run)
```bash
python publish_model.py --dry-run
```

### 3. Publish to Hugging Face Hub
Maintainers need a Hugging Face write token (`HF_TOKEN`):
```bash
# Set your Hugging Face write token
export HF_TOKEN="hf_your_write_token"      # Linux / macOS
set HF_TOKEN=hf_your_write_token          # Windows cmd
$env:HF_TOKEN="hf_your_write_token"       # Windows PowerShell

# Upload artifacts, create the model card, and pin the git tag
python publish_model.py --version v1.0.0
```

---

## Running Verification & Tests

### End-to-End Pipeline Smoke Test
```bash
python verify_setup.py
```
Validates acoustic feature extraction, model cache resolution, and prediction formats.

### Automated Unit & Integrity Tests
```bash
pytest -o pythonpath=. tests/test_model_caching_and_integrity.py
```
Verifies:
- Cached models load offline with zero network calls.
- Inference never triggers `fit()` or training routines.
- SHA-256 checksum mismatches raise `ModelIntegrityError`.
- Interrupted or failed downloads clean up temporary files safely.
- Failed downloads produce clear errors without retraining.

---

## API Reference

### `GET /health`
Liveness check returning service and model status:
```json
{
  "service": "Voice Analysis Service",
  "status": "running",
  "model_loaded": true,
  "model_version": "v1.0.0"
}
```

### `POST /analyze`
Accepts an audio recording (`.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`) and returns acoustic deception metrics:
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "audio=@sample_audio/truthful.wav"
```

Response format:
```json
{
  "verdict": "Truthful",
  "score": 0.475,
  "confidence": 0.525,
  "speech_status": "VOICE DETECTED",
  "speech_detected": true,
  "explanation": [
    "Narrow pitch variability detected across utterance.",
    "Continuous delivery with few detected pauses.",
    "Higher speech onset rate measured."
  ],
  "structured_explanation": {
    "acoustic_observations": [...],
    "quality_warnings": [],
    "prediction_context": "Vocal prosody analyzed.",
    "limitations": [...]
  }
}
```

---

## Error Handling & Reliability

- **Model Unavailable (HTTP 503)**: If the local cache is empty and network access is unavailable, the API returns a clear 503 error instructing the administrator how to download the model or set `TRUSIGHT_MODEL_PATH`. **The service never crashes or attempts to retrain in production.**
- **No Speech Detected**: Audio containing silence, background noise, or microphone hiss is not scored; an informative response is returned with `verdict: "No Speech Detected"`.

---

## Docker Deployment

Build and run the container:
```bash
docker build -t trusight-voice-service .
docker run -p 8000:8000 trusight-voice-service
```
The Docker image pre-caches the model during build or downloads it on first container boot without retraining.

---

## Dataset Attribution & Licensing

- **Code License**: MIT License.
- **Dataset**: Trained on the DOLOS Audio-Visual Deception Detection corpus (ICCV 2023) and Real-Life Trial dataset.
- **Scientific Caveat**: Acoustic cues alone are not a definitive lie detector. Vocal cues reflect physiological arousal, pitch cadence, or audio recording conditions. This module is intended as one component in a multimodal verification system.
