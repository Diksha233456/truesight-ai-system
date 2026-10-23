# TRUSIGHT Voice Analysis Service — Implementation & Model Details

This document provides a comprehensive overview of everything implemented in this repository, including system architecture, feature extraction pipelines, AI/ML models used, datasets processed, and API/UI features.

---

## 1. Executive Summary

**TRUSIGHT Voice Analysis Service** is a standalone, multimodal-ready FastAPI microservice for voice deception analysis. Given an audio recording (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`), it extracts acoustic signals, deep neural audio embeddings, and verbal speech cues to estimate deception likelihood, confidence level, and human-readable acoustic observations.

---

## 2. Implemented Features & Architecture

```
                                  ┌───────────────────────────────┐
                                  │   Uploaded Audio (.wav, etc)  │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │  Validation & Preprocessing   │
                                  │   (16kHz Mono, Peak Norm)     │
                                  └───────────────┬───────────────┘
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                ▼                                 ▼                                 ▼
   ┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
   │    Acoustic Cues (42)   │       │ Deep Audio Vector (768) │       │   Verbal Features (4)   │
   │ Pitch, Jitter, Shimmer, │       │  Wav2Vec2 Hidden State  │       │  ASR Transcript via CTC │
   │ Pause, Energy, MFCCs    │       │     Transformer Embed   │       │  Hedges, Pronouns, Len  │
   └────────────┬────────────┘       └────────────┬────────────┘       └────────────┬────────────┘
                │                                 │                                 │
                └─────────────────────────────────┼─────────────────────────────────┘
                                                  │ (814-dim concatenated)
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │      StandardScaler (SKLearn) │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │ SelectKBest (ANOVA Top 50 k) │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │   Soft Voting Ensemble Model  │
                                  │ (RandomForest + GradBoost)    │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │ Verdict, Score, Confidence,   │
                                  │  & Acoustic Explanations      │
                                  └───────────────────────────────┘
```

---

## 3. Models Used in this Project

### A. Pre-trained Neural Models (Feature Extractors & ASR)
* **Model Name:** `facebook/wav2vec2-base-960h` (Meta AI / Hugging Face Transformers)
* **Components Used:**
  1. **Wav2Vec2Model (`_base_model`):** Extracts a 768-dimensional deep acoustic embedding by mean-pooling the last hidden layer across time frames.
  2. **Wav2Vec2ForCTC (`_ctc_model`):** Performs Automatic Speech Recognition (ASR) to transcribe speech into text without requiring an external speech-to-text API.
  3. **Wav2Vec2Processor (`_tokenizer`):** Handles raw audio sample tokenization and transcript decoding.

### B. Machine Learning Classifiers & Pipelines

#### 1. Baseline Model (`train_model.py`)
* **Algorithm:** `scikit-learn` `RandomForestClassifier`
* **Configuration:** 200 trees, `max_depth=10`, `class_weight='balanced'`, `random_state=42`.
* **Input Dimension:** 42 acoustic features.
* **Output:** Deception probability ($[0.0, 1.0]$).

#### 2. Upgraded Multimodal Ensemble Pipeline (`train_optimized_multimodal_model.py`)
* **Architecture:** `sklearn.pipeline.make_pipeline`
* **Input Dimension:** 814 total features (42 Acoustic + 768 Wav2Vec2 + 4 Verbal).
* **Pipeline Steps:**
  1. `StandardScaler()`: Feature normalization.
  2. `SelectKBest(score_func=f_classif, k=50)`: Selects top 50 discriminative features based on ANOVA F-value.
  3. `VotingClassifier(voting="soft")`:
     * **RandomForestClassifier:** 300 estimators, `max_depth=8`, `min_samples_leaf=2`.
     * **GradientBoostingClassifier:** 200 estimators, `max_depth=3`, `learning_rate=0.03`.
* **Cross-Validation:** Evaluated using 5-Fold Stratified Cross-Validation.

#### 3. Other Machine Learning Models Evaluated (`test_deep_learning_pipeline.py`)
* **XGBoost Classifier (`XGBClassifier`)** (`n_estimators=100`, `max_depth=4`, `learning_rate=0.05`).
* **Gradient Boosting Classifier (`GradientBoostingClassifier`)**.

---

## 4. Feature Extraction Breakdown (814 Total Features)

| Category | Count | Technology / Library | Description |
|---|---|---|---|
| **Pitch & Fundamental Frequency (F0)** | 4 | Praat-Parselmouth | Mean pitch, Pitch Std, Min pitch, Max pitch |
| **Voice Perturbation / Quality** | 2 | Praat-Parselmouth | Jitter (local pitch variability %), Shimmer (local amplitude variability %) |
| **Temporal & Pauses** | 4 | Praat-Parselmouth & Librosa | Total duration, Speaking duration, Speech-to-pause ratio, Hesitation pause count |
| **Energy & Spectral Cues** | 6 | Librosa | RMS energy (mean & std), Spectral Centroid (mean & std), Spectral Flux (mean & std) |
| **MFCCs** | 26 | Librosa | 13 MFCC means + 13 MFCC standard deviations |
| **Deep Audio Embeddings** | 768 | PyTorch / Hugging Face Wav2Vec2 | Deep transformer hidden-layer embeddings representing voice timbral and phonetic patterns |
| **Verbal Deception Stats** | 4 | Regex / Lexicon rules on Wav2Vec2 ASR | 1st-person pronoun density (`i`, `we`, `my`), Hedge word density (`maybe`, `guess`, `think`), Avg word length, Total word count |

---

## 5. Datasets Integrated & Supporting Pipelines

1. **Synthetic Benchmark Dataset (`generate_sample_audio.py`)**
   * Generates synthetic `truthful.wav` and `deceptive.wav` audio clips for instant pipeline smoke testing.
2. **DOLOS Audio-Visual Deception Dataset (`download_and_train.py`)**
   * Implements an automated YouTube audio extraction pipeline (`yt-dlp` + `ffmpeg`) with clip trimming based on ICCV 2023 DOLOS timestamps.
3. **Real-Life Trial Deception Dataset (`process_reallife_and_retrain.py`)**
   * Processes court trial video clips (`.mp4`), extracts 16kHz mono WAV audio, merges them into the training corpus, retrains the model, and hot-reloads the API.

---

## 6. Microservice & Web Interface Details

### FastAPI Backend Endpoints (`app/main.py`)
* `POST /analyze`: Main endpoint for audio upload analysis. Returns verdict (`Truthful` / `Deceptive`), probability score, confidence score, and human-readable acoustic explanations (`app/explanation.py`).
* `POST /reload-model`: Dynamically reloads the updated `.pkl` model file from disk into memory without requiring server restart.
* `GET /health` & `GET /`: Health check and web UI route.

### Frontend Dashboard (`app/static/index.html`)
* Modern, responsive web dashboard built with HTML5, CSS3 glassmorphism design, real-time Web Audio API waveform visualization, interactive risk gauges, and dynamic score breakdowns.

---

## 7. File Map

| File Path | Description |
|---|---|
| [app/main.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/app/main.py) | FastAPI service entry point & API endpoints |
| [app/model_loader.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/app/model_loader.py) | Robust pretrained model resolver, automatic HF downloader, persistent caching, and integrity validation |
| [app/predictor.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/app/predictor.py) | In-memory singleton inference wrapper with zero-retraining guarantees |
| [publish_model.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/publish_model.py) | Maintainer utility to publish trained model artifacts to Hugging Face Hub |
| [artifacts/](file:///c:/Users/diksh/Downloads/voice_service/voice_service/artifacts/) | Exported model weights (`deception_model.pkl`) and versioned metadata (`model_metadata.json`) |
| [app/feature_extractor.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/app/feature_extractor.py) | Praat + Librosa 42-dim acoustic feature extractor |
| [app/deep_feature_extractor.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/app/deep_feature_extractor.py) | 814-dim multimodal extractor (Acoustic + Wav2Vec2 + Verbal ASR) |
| [app/explanation.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/app/explanation.py) | Rule-based generator for human-readable acoustic insights |
| [app/utils.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/app/utils.py) | Audio decoding, resampling, normalization, and validation |
| [train_model.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/train_model.py) | Maintainer script to train and export versioned model artifacts with checksums |
| [train_optimized_multimodal_model.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/train_optimized_multimodal_model.py) | 814-feature Multimodal Scaler + SelectKBest + Soft Voting Ensemble trainer script |
| [download_and_train.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/download_and_train.py) | Automated DOLOS dataset YouTube downloader & clip trimmer |
| [process_reallife_and_retrain.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/process_reallife_and_retrain.py) | Real-Life Trial dataset audio extractor and model merger script |
| [generate_sample_audio.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/generate_sample_audio.py) | Synthetic test audio generator script |
| [verify_setup.py](file:///c:/Users/diksh/Downloads/voice_service/voice_service/verify_setup.py) | End-to-end setup verification test script |
| [app/static/index.html](file:///c:/Users/diksh/Downloads/voice_service/voice_service/app/static/index.html) | Modern Web UI for live audio analysis |
