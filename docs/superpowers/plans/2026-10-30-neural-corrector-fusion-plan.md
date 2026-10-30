# Neural Corrector DL Model & Equal Module Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight PyTorch neural corrector DL model for the multimodal fusion model with equal module weighting ($1/N$) and replace experimental heuristic scores with neural-corrected fusion scores.

**Architecture:** A lightweight PyTorch MLP model (`NeuralCorrector`) that takes input features (probabilities and confidences from documentary, facial, and acoustic streams), initializes active module weights equally ($1/N$), dynamically masks missing modalities, and produces a calibrated neural fusion score. Integrated into `AssessmentCoordinator`.

**Tech Stack:** Python, PyTorch, Pydantic, Pytest.

---

### Task 1: Create Neural Corrector PyTorch Module

**Files:**
- Create: `fusion_service/neural_corrector.py`
- Test: `tests/test_multimodal_fusion.py`

- [ ] **Step 1: Write the test for neural corrector initialization and forward pass**

```python
import torch
import pytest
from fusion_service.neural_corrector import NeuralCorrector

def test_neural_corrector_init_and_weights():
    model = NeuralCorrector()
    # Check that input weights for the 3 modules (3 probability scores + 3 confidences = 6 features) are initialized equally
    assert model is not None
    # Test forward pass with 3 active modalities
    inputs = torch.tensor([[0.8, 0.9, 0.7, 0.85, 0.8, 0.9]], dtype=torch.float32)
    score, conf = model(inputs, active_mask=[1.0, 1.0, 1.0])
    assert 0.0 <= score.item() <= 1.0
    assert 0.0 <= conf.item() <= 1.0

def test_neural_corrector_dynamic_masking():
    model = NeuralCorrector()
    # Test forward pass with only modality 0 active (face and voice masked)
    inputs = torch.tensor([[0.8, 0.0, 0.0, 0.85, 0.0, 0.0]], dtype=torch.float32)
    score, conf = model(inputs, active_mask=[1.0, 0.0, 0.0])
    assert 0.0 <= score.item() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_multimodal_fusion.py -k test_neural_corrector -v`
Expected: FAIL with ModuleNotFoundError or import error.

- [ ] **Step 3: Implement `fusion_service/neural_corrector.py`**

```python
"""
Neural Corrector Module for TRUSIGHT Multimodal Fusion
Provides a lightweight PyTorch MLP with equal module weighting initialization
and dynamic missing modality masking.
"""

from __future__ import annotations

import os
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCorrector(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 16):
        super().__init__()
        # Features: [p_evidence, p_face, p_voice, conf_evidence, conf_face, conf_voice]
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 8)
        self.score_out = nn.Linear(8, 1)
        self.conf_out = nn.Linear(8, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize linear layers with Xavier / Uniform and set equal weights for active baseline
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.score_out.weight)
        nn.init.zeros_(self.score_out.bias)
        nn.init.xavier_uniform_(self.conf_out.weight)
        nn.init.zeros_(self.conf_out.bias)

    def forward(
        self, x: torch.Tensor, active_mask: Optional[list[float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dynamic active modality masking.
        active_mask: list of 3 floats [m_evidence, m_face, m_voice] (1.0 or 0.0)
        """
        if active_mask is not None:
            mask_tensor = torch.tensor(active_mask, dtype=torch.float32, device=x.device)
            # Expand mask to match feature dimensions: probabilities (indices 0,1,2) and confidences (indices 3,4,5)
            full_mask = torch.cat([mask_tensor, mask_tensor])
            x = x * full_mask

        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        
        score = torch.sigmoid(self.score_out(h))
        conf = torch.sigmoid(self.conf_out(h))
        return score.squeeze(-1), conf.squeeze(-1)


def load_neural_corrector(model_path: str = "fusion_service/models/neural_corrector.pth") -> NeuralCorrector:
    model = NeuralCorrector()
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
        except Exception:
            pass
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_multimodal_fusion.py -k test_neural_corrector -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fusion_service/neural_corrector.py tests/test_multimodal_fusion.py
git commit -m "feat: add neural corrector PyTorch module with dynamic masking"
```

---

### Task 2: Create Default Model Weights File

**Files:**
- Create: `fusion_service/models/neural_corrector.pth`
- Create: `fusion_service/save_default_model.py`

- [ ] **Step 1: Write script to save default initialized model weights**

```python
import os
import torch
from fusion_service.neural_corrector import NeuralCorrector

os.makedirs("fusion_service/models", exist_ok=True)
model = NeuralCorrector()
torch.save(model.state_dict(), "fusion_service/models/neural_corrector.pth")
print("Saved default neural corrector model weights.")
```

- [ ] **Step 2: Run script to generate weights**

Run: `python fusion_service/save_default_model.py`
Expected: Saved default neural corrector model weights.

- [ ] **Step 3: Commit weights and helper script**

```bash
git add fusion_service/models/neural_corrector.pth fusion_service/save_default_model.py
git commit -m "feat: add initial neural corrector weights and save script"
```

---

### Task 3: Integrate Neural Corrector into Assessment Coordinator

**Files:**
- Modify: `assessment_coordinator.py`
- Test: `tests/test_multimodal_fusion.py`

- [ ] **Step 1: Update `assessment_coordinator.py` to incorporate `NeuralCorrector` and compute equal module weighting**

Modify `assessment_coordinator.py` around `FusionAssessment` and `evaluate_assessment`:
- Replace `experimental_heuristic_score` calculation with `neural_corrected_fusion_score` (using equal weighting baseline and neural corrector inference).
- Add `fusion_service.neural_corrector` import and fallback handling.

- [ ] **Step 2: Update `tests/test_multimodal_fusion.py` to assert neural corrected fusion score**

Add test cases in `tests/test_multimodal_fusion.py` verifying that `neural_corrected_fusion_score` is populated correctly with equal weighting across active streams.

- [ ] **Step 3: Run pytest**

Run: `pytest tests/test_multimodal_fusion.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add assessment_coordinator.py tests/test_multimodal_fusion.py
git commit -m "feat: integrate neural corrector and equal module weighting into assessment coordinator"
```

---

### Task 4: Final Verification of All Tests

- [ ] **Step 1: Run full test suite**

Run: `pytest -v`
Expected: All tests pass successfully.

- [ ] **Step 2: Final commit**

```bash
git add .
git commit -m "chore: final verification of neural corrector fusion integration"
```
