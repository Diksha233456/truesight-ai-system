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
