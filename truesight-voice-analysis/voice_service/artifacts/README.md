---
language:
- en
license: mit
tags:
- audio
- voice-analysis
- deception-detection
- acoustic-features
- parselmouth
- librosa
- scikit-learn
metrics:
- accuracy
- precision
- recall
- f1
model-index:
- name: trusight-voice-deception
  results:
  - task:
      type: audio-classification
      name: Vocal Deception Likelihood Estimation
    metrics:
    - name: 5-Fold Cross-Validation Accuracy
      type: accuracy
      value: 0.6431
    - name: Held-out Test Accuracy
      type: accuracy
      value: 0.5882
    - name: Held-out Test F1-Score
      type: f1
      value: 0.5882
---

# trusight-voice-deception (v1.0.0)

Pretrained acoustic classifier for the TRUSIGHT voice analysis service.

## Model Summary
- **Architecture**: `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)`
- **Framework**: `scikit-learn 1.9.0`
- **Input Dimension**: 42 acoustic features (pitch, jitter, shimmer, tempo, pauses, spectral, MFCCs)
- **Output Classes**: `0` = Truthful, `1` = Deceptive
- **Decision Threshold**: 0.5

## Performance Metrics
- **5-Fold Cross-Validation Accuracy**: 0.643 ± 0.075
- **Held-out Test Accuracy**: 0.588
- **Held-out Test Precision**: 0.588
- **Held-out Test Recall**: 0.588
- **Held-out Test F1**: 0.588
- **Held-out ROC AUC**: 0.6955

## Automatic Download & Usage
This model is automatically downloaded and cached by the TRUSIGHT voice service upon first run:
```bash
uvicorn app.main:app --reload
```

To load manually with Python:
```python
import joblib
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="sofi233/truesight-voice-deception",
    filename="deception_model.pkl",
    revision="v1.0.0"
)
model = joblib.load(model_path)
```

## Intended Use & Ethical Disclaimers
This model estimates deception likelihood from vocal acoustic cues only as part of an experimental multimodal truth verification system. It does NOT make definitive lie-detection determinations on its own. Indian-English and non-native accents have not been validated for deception detection.
