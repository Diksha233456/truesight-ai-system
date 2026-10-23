# Dataset Card: TRUSIGHT Audio-Visual Deception & Voice Corpus

## Dataset Overview

The dataset used in the TRUSIGHT voice analysis system combines acoustic clips from standardized speech corpora and real-world audio datasets (including DOLOS / Real-life deception speech samples, TORGO controls, and RAVDESS stress/emotion audio).

- **Primary Modality**: Audio (.wav, .webm, 16kHz/22.05kHz mono/stereo PCM)
- **Task**: Binary classification (`0 = Truthful`, `1 = Deceptive`)
- **Total Samples**: Extracted audio segments from multiple datasets
- **Features Extracted**: 38 acoustic and prosodic features (F0, Jitter, Shimmer, HNR, MFCCs, Spectral Centroid/Rolloff, ZCR, Pause Ratios).

---

## Dataset Splits & Structure

### Speaker Grouping
- **Speaker IDs**: Formatted as `spk_001`, `spk_002`, ..., or dataset prefix markers (`torgo_spk_X`, `reallife_spk_Y`).
- **Target Distribution**: Approximately 52% Truthful (`0`), 48% Deceptive (`1`).

### Split Protocols
1. **Naive Random Split**: 80% Train / 20% Test (Used historically; subject to speaker leakage).
2. **Speaker-Grouped Split (Recommended)**: `GroupKFold(n_splits=5)` grouped strictly by `speaker_id`.

---

## Data Quality & Limitations

1. **Speaker Overlap Hazard**: Naive train/test splitting mixes audio segments from the same speaker, resulting in memorization of speaker identity rather than acoustic deception cues.
2. **Acoustic Environment Variance**: Studio recordings vs. broadcast court/interview recordings vs. web browser microphone streams have different signal-to-noise ratios (SNR).
3. **Label Reliability**: Labels are ground-truth assigned based on original corpus metadata (e.g. trial verdicts or experimental protocol ground truths).
