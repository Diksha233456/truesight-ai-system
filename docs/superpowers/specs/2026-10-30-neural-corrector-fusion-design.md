# Technical Specification: Neural Corrector DL Model & Equal Module Weighting for Multimodal Fusion

## Overview
This specification details the implementation of a lightweight PyTorch neural corrector deep learning model for the TRUSIGHT multimodal fusion system (`assessment_coordinator.py`). It enforces equal weighting across active modules ($1/N$) and applies a learned residual neural correction layer to produce a robust, calibrated cross-modal assessment score.

## Architecture & Components

### 1. Neural Corrector PyTorch Model (`fusion_service/neural_corrector.py`)
- **Inputs:** A fixed-size feature vector representing normalized probability scores and confidence metrics from the 3 modalities:
  1. Documentary Evidence (`evidence_finding`)
  2. Facial Observation (`facial_observation`)
  3. Acoustic Observation (`acoustic_observation`)
- **Equal Weight Initialization:** The first linear layer weights are initialized to equal proportions ($w_i = 1/N$ for active modalities).
- **Architecture:** 
  - Input Layer: 6 dimensions (3 probability scores + 3 confidence values).
  - Hidden Layer: Linear(6, 16) -> ReLU -> Dropout(0.1) -> Linear(16, 8) -> ReLU.
  - Output Layer: Linear(8, 1) with Sigmoid activation (or residual offset).
- **Storage:** Model weights saved at `fusion_service/models/neural_corrector.pth`.

### 2. Dynamic Masking & Modality Weighting
- When one or more modalities are inactive, missing, or excluded, their input vector components are zero-masked.
- The active modality count $N$ dynamically determines the baseline equal weights ($1/N$).

### 3. Integration into `AssessmentCoordinator`
- Replaces `experimental_heuristic_score` with `neural_corrected_fusion_score` in `FusionAssessment`.
- Graceful fallback: If `neural_corrector.pth` is missing or PyTorch inference errors out, the system automatically falls back to analytical equal-weighted averaging.

### 4. Testing & Verification
- Unit tests in `tests/test_multimodal_fusion.py` covering:
  - Equal weight allocation with 1, 2, and 3 active modalities.
  - Neural corrector forward pass with mock telemetry.
  - Fallback behavior when model file is absent.
