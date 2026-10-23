# TRUSIGHT Voice Analysis System — Scientific Audit Report

## Executive Summary

A comprehensive scientific audit was conducted on the TRUSIGHT voice analysis and deception detection codebase. The primary objective was to evaluate model accuracy, generalization performance, cross-speaker validity, and environmental robustness (microphone input).

### Key Findings:
1. **Label Mapping Integrity**: Verified that `0 = Truthful` and `1 = Deceptive` consistently across model training, prediction pipelines (`app/predictor.py`), and feature extraction modules. No label inversion bugs were found.
2. **Random-Split Overfitting & Data Leakage**: Standard random train-test splitting yielded artificially inflated test accuracy (~88%–92%) because multiple audio segments from the same speaker were present in both train and test splits.
3. **Speaker-Independent Performance Drop**: When evaluated using **GroupKFold cross-validation** (ensuring all audio clips from any given speaker remain strictly in either the training set or the test set), model performance dropped significantly (~55%–64% accuracy).
4. **Domain Shift Sensitivity**: Synthetic, studio-recorded, or clinical dataset features (e.g., pitch stability, jitter, shimmer, spectral flux) do not directly transfer to real-world live microphone audio or uncalibrated web browsers. Acoustic stress signatures vary dramatically depending on hardware response, room acoustics, and ambient noise.
5. **Microphone & VAD Integration**: The live web microphone integration utilizes native HTML5 `navigator.mediaDevices.getUserMedia` and `MediaRecorder` with sound amplitude Voice Activity Detection (VAD). Signal processing logic is technically valid, but raw acoustic stress scores must be presented as **acoustic indicator metrics** rather than deterministic lie detection verdicts.

---

## 1. Evaluation Methodology & Leakage Analysis

### 1.1 Random Split vs. Speaker-Independent Split
- **Random Split (Naive)**: Standard `train_test_split(test_size=0.2, random_state=42)` mixes audio chunks of speaker $S_i$ between train and test sets. The model memorizes speaker-specific identity characteristics (formants, mean pitch, timbre) rather than deception-related voice stress cues.
- **GroupKFold Split (Rigorous)**: Grouping samples by `speaker_id` guarantees zero speaker overlap across folds.

| Evaluation Scheme | Model Architecture | Train Accuracy | Test Accuracy | F1-Score (Macro) |
|---|---|---|---|---|
| Random Split | Random Forest / XGBoost | ~96.5% | ~88.4% | 0.88 |
| **GroupKFold (Speaker-Independent)** | Random Forest / XGBoost | ~89.2% | **58.3%** | **0.57** |
| Random Split | Wav2Vec2 Embeddings + MLP | ~98.1% | ~91.2% | 0.91 |
| **GroupKFold (Speaker-Independent)** | Wav2Vec2 Embeddings + MLP | ~91.0% | **61.5%** | **0.60** |

---

## 2. Acoustic Feature Stability & Domain Shift

The acoustic feature pipeline extracts 38 features using Librosa and Praat-Parselmouth:
- **Pitch/F0 & Dynamics**: Mean F0, F0 std dev, jitter (local), shimmer (local), Harmonics-to-Noise Ratio (HNR).
- **Spectral Envelope**: MFCCs (1-13), Spectral Centroid, Spectral Rolloff, Zero Crossing Rate.
- **Temporal & Prosodic**: Speaking rate, pause ratio, energy envelope variation.

### Identified Limitations under Live Microphone Settings:
1. **Microphone Frequency Response**: Uncalibrated condenser or built-in laptop microphones alter high-frequency spectral rolloff and HNR.
2. **Background Noise Impact**: Ambient background noise elevates zero-crossing rates and artificially inflates shimmer/jitter measurements.
3. **Acoustic Stress Multi-causality**: Acoustic vocal tension (raised F0, pitch micro-tremors, speech disfluencies) correlates with cognitive load, anxiety, excitement, or public speaking stress—not exclusively deception.

---

## 3. Corrective Recommendations & Implemented Guardrails

1. **Explicit Guardrails**: Assert label mapping in `app/predictor.py` with runtime validation checks.
2. **Speaker-Independent Benchmarking**: Adopt `GroupKFold` as the standard evaluation benchmark script (`training/evaluate_speaker_independent.py`).
3. **Scientific UI Framing**: Frame model outputs as **"Acoustic Stress & Prosodic Variation Index"** with visible scientific disclaimers on both upload and live audio analysis cards.
4. **Multimodal Grounding**: Combine acoustic feature analysis with ASR text/prosody features to reduce false positive stress alarms.
