# Model Card: TRUSIGHT Acoustic Voice Stress & Analysis Engine

## Model Details

- **Model Name**: TRUSIGHT Acoustic Classifier (Random Forest / XGBoost Baseline & Multimodal Wav2Vec2 Pipeline)
- **Model Type**: Ensemble Tree Classifier / Deep Feature Extractor
- **Input Modality**: Audio Signal (Waveform / Acoustic Feature Vector)
- **Output**: Class Probabilities ($P(\text{Truthful})$, $P(\text{Deceptive/Stressed})$) & Acoustic Stress Score ($0.0 - 1.0$)

---

## Intended Use

- **Primary Application**: Academic research, voice stress visualization, acoustic prosodic auditing, and interactive audio analysis demo.
- **Out of Scope Applications**: Formal lie detection, legal evidence, employment screening, security clearance checks, or forensic lie detection.

---

## Technical Specifications

### Feature Extraction Pipeline
1. **Parselmouth / Praat**: Fundamental Frequency ($F_0$), Local Jitter, Local Shimmer, Harmonics-to-Noise Ratio (HNR).
2. **Librosa**: 13 Mel-Frequency Cepstral Coefficients (MFCCs), Spectral Centroid, Spectral Rolloff, Zero Crossing Rate (ZCR), RMS Energy.
3. **Temporal Features**: Speech duration, silence/pause duration ratio, speech rate.

### Model Benchmarks

| Metric | Random Split (Naive) | GroupKFold (Speaker-Independent) |
|---|---|---|
| Accuracy | 88.4% | 58.3% |
| Precision | 0.89 | 0.59 |
| Recall | 0.87 | 0.56 |
| F1-Score | 0.88 | 0.57 |

---

## Caveats and Limitations

1. **Acoustic Stress Multi-causality**: High stress scores indicate physiological/vocal acoustic tension, which can be caused by nervousness, background noise, microphone distortion, or task difficulty—not exclusively deception.
2. **Domain Shift**: Performance degrades on raw live microphone web inputs due to acoustic reverberation and mic hardware differences.
